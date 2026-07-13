# src/tmdc_optics_tools/reference/loader.py

import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Controlled vocabulary for spectroscopy types
# ---------------------------------------------------------------------------

SPECTROSCOPY_TYPES = {
    "PL":    "Photoluminescence",
    "R":     "Reflectance",
    "T":     "Transmission",
    "A":     "Absorption",
    "Raman": "Raman scattering",
    "TRPL":  "Time-resolved photoluminescence",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """
    Standardised container for a single spectrum.

    Attributes
    ----------
    energy          : photon energy axis (eV)
    intensity       : raw intensity counts
    label           : human-readable label, e.g. "E=+300, 40.0 µW"
    spectroscopy    : measurement type, one of SPECTROSCOPY_TYPES
    energy_unit     : unit of energy axis (default "eV")
    intensity_unit  : unit of intensity axis (default "counts")
    parameter_value : value of the sweep parameter this spectrum was taken at
    is_default      : whether this is the canonical spectrum for its sweep
    """
    energy:          np.ndarray
    intensity:       np.ndarray
    label:           str            = ""
    spectroscopy:    str            = "PL"
    energy_unit:     str            = "eV"
    intensity_unit:  str            = "counts"
    parameter_value: Optional[float] = None
    is_default:      bool           = False

    def normalised(self) -> np.ndarray:
        """Return intensity normalised to [0, 1]."""
        s = self.intensity
        return (s - s.min()) / (s.max() - s.min())


@dataclass
class SweepSeries:
    """
    A set of spectra taken while sweeping one parameter.

    Attributes
    ----------
    parameter_name : e.g. "electric_field", "power"
    parameter_unit : e.g. "mV/nm", "uW"
    spectra        : {parameter_value: Spectrum} sorted by parameter value
    default_value  : parameter_value of the canonical spectrum
    """
    parameter_name: str
    parameter_unit: str
    spectra:        dict   = field(default_factory=dict)
    default_value:  Optional[float] = None

    def default(self) -> Spectrum:
        """Return the canonical spectrum for this sweep."""
        if self.default_value is None:
            raise ValueError(f"No default set for sweep '{self.parameter_name}'")
        return self.spectra[self.default_value]

    def values(self) -> list:
        """Return parameter values in sorted order."""
        return sorted(self.spectra.keys())

    def all_spectra(self) -> list:
        """Return all Spectrum objects in sorted parameter order."""
        return [self.spectra[v] for v in self.values()]

    def __getitem__(self, param_value: float) -> Spectrum:
        return self.spectra[param_value]


@dataclass
class FieldCondition:
    """
    One condition of an outer sweep (e.g. one electric field value),
    containing an energy axis, metadata, and an inner SweepSeries.

    Attributes
    ----------
    parameter_value : value of the outer sweep parameter, e.g. 300.0
    parameter_unit  : unit of the outer sweep parameter, e.g. "mV/nm"
    energy          : photon energy axis (eV), trimmed per authors' choice
    energy_unit     : unit of energy axis
    is_default      : whether this is the default outer condition
    sweeps          : {sweep_name: SweepSeries} — inner sweeps, e.g. "power"
    """
    parameter_value: float
    parameter_unit:  str
    energy:          np.ndarray
    energy_unit:     str  = "eV"
    is_default:      bool = False
    sweeps:          dict = field(default_factory=dict)

    def default_spectrum(self, sweep_name: str) -> Spectrum:
        """Return the default spectrum for a named inner sweep."""
        return self.sweeps[sweep_name].default()

    def __getitem__(self, sweep_name: str) -> SweepSeries:
        return self.sweeps[sweep_name]


@dataclass
class ReferenceDataset:
    """
    All data from one paper for one material.

    Attributes
    ----------
    material      : e.g. "WSe2_bilayer"
    source        : e.g. "Tagarelli2023"
    doi           : publication DOI
    zenodo_doi    : Zenodo record DOI
    title         : full paper title
    about       : any notes added in the registry
    spectroscopy  : measurement type, one of SPECTROSCOPY_TYPES
    sweeps        : {sweep_name: SweepSeries} for flat datasets, or
                    outer-level sweep for nested datasets
    """
    material:     str
    source:       str
    doi:          str  = ""
    zenodo_doi:   str  = ""
    title:        str  = ""
    about:        str  = ""
    spectroscopy: str  = "PL"
    sweeps:       dict = field(default_factory=dict)

    def default_spectrum(self, sweep_name: str) -> Spectrum:
        """Return the default spectrum for a named sweep."""
        return self.sweeps[sweep_name].default()

    def __getitem__(self, sweep_name: str):
        return self.sweeps[sweep_name]

    def __repr__(self):
        sweep_summary = {k: len(v.spectra) for k, v in self.sweeps.items()}
        return (
            f"ReferenceDataset("
            f"material={self.material!r}, "
            f"source={self.source!r}, "
            f"spectroscopy={self.spectroscopy!r}, "
            f"sweeps={sweep_summary})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _read_sweep(group: h5py.Group, spectroscopy: str) -> SweepSeries:
    """
    Recursively read a SweepSeries from an HDF5 group.
    Handles both flat sweeps (spectrum datasets at leaf level)
    and nested sweeps (inner SweepSeries inside each condition).
    """
    parameter_name = group.attrs.get("parameter_name", group.name.split("/")[-1])
    parameter_unit = group.attrs.get("parameter_unit", "")
    default_value  = group.attrs.get("default_value", None)

    spectra = {}

    for key in group.keys():
        subgroup = group[key]

        if isinstance(subgroup, h5py.Dataset):
            continue

        # Leaf node — contains an actual spectrum dataset
        if "spectrum" in subgroup:
            energy_ds = subgroup.parent.parent.get("energy") or subgroup.get("energy")
            energy = energy_ds[:] if energy_ds is not None else np.array([])

            param_val = float(subgroup.attrs.get("parameter_value", float(key)))
            spectrum  = Spectrum(
                energy          = energy,
                intensity       = subgroup["spectrum"][:],
                label           = key,
                spectroscopy    = spectroscopy,
                energy_unit     = subgroup.parent.parent.attrs.get("energy_unit", "eV"),
                intensity_unit  = subgroup.attrs.get("spectrum_unit", "counts"),
                parameter_value = param_val,
                is_default      = bool(subgroup.attrs.get("is_default", False)),
            )
            spectra[param_val] = spectrum

        # Non-leaf — this is a FieldCondition group containing an inner sweep
        else:
            inner_sweeps = {}
            for inner_key in subgroup.keys():
                if inner_key == "energy":
                    continue
                inner_sweeps[inner_key] = _read_sweep(subgroup[inner_key], spectroscopy)

            param_val = float(subgroup.attrs.get("parameter_value", float(key)))
            energy    = subgroup["energy"][:] if "energy" in subgroup else np.array([])

            condition = FieldCondition(
                parameter_value = param_val,
                parameter_unit  = parameter_unit,
                energy          = energy,
                energy_unit     = subgroup.attrs.get("energy_unit", "eV"),
                is_default      = bool(subgroup.attrs.get("is_default", False)),
                sweeps          = inner_sweeps,
            )
            spectra[param_val] = condition

    return SweepSeries(
        parameter_name = parameter_name,
        parameter_unit = parameter_unit,
        spectra        = spectra,
        default_value  = float(default_value) if default_value is not None else None,
    )


def load_reference(material: str, source: str) -> ReferenceDataset:
    """
    Load a processed reference dataset from its HDF5 file.

    Parameters
    ----------
    material : str — e.g. "WSe2_bilayer"
    source   : str — e.g. "Tagarelli2023"

    Returns
    -------
    ReferenceDataset

    Raises
    ------
    FileNotFoundError if the .h5 file doesn't exist — run registry.py first.

    Examples
    --------
    >>> ref = load_reference("WSe2_bilayer", "Tagarelli2023")
    >>> sp  = ref.default_spectrum("electric_field")
    >>> ax.plot(sp.energy, sp.normalised())

    >>> for val in ref["electric_field"].values():
    ...     cond = ref["electric_field"][val]
    ...     sp   = cond.default_spectrum("power")
    ...     ax.plot(sp.energy, sp.normalised(), label=f"E={val} mV/nm")
    """
    path = DATA_DIR / f"{material}__{source}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for {material!r} / {source!r}.\n"
            f"Expected: {path}\n"
            f"Run `python -m tmdc_optics_tools.reference.registry` to download and process."
        )

    with h5py.File(path, "r") as hf:
        dataset = ReferenceDataset(
            material     = hf.attrs.get("material",     material),
            source       = hf.attrs.get("source",       source),
            doi          = hf.attrs.get("doi",          ""),
            zenodo_doi   = hf.attrs.get("zenodo_doi",   ""),
            title        = hf.attrs.get("title",        ""),
            about        = hf.attrs.get("about",      ""),
            spectroscopy = hf.attrs.get("spectroscopy", "PL"),
        )

        for sweep_name in hf.keys():
            dataset.sweeps[sweep_name] = _read_sweep(hf[sweep_name], dataset.spectroscopy)

    return dataset