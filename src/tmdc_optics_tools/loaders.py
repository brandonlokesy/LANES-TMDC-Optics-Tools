# tmdc_optics_tools/loaders.py
"""
Data loaders and device geometry for TMD heterostructure measurements.

Classes
-------
DeviceGeometry
    Encodes the physical geometry and dielectric constants of a vdW stack.
AttoCubePLScan
    Parses and holds data from a gate-dependent PL scan taken on the
    AttoCube cryogenic confocal setup.
AttoCubePLScanRealSpace
    Loads a sequence of real-space PL image CSVs swept over gate voltage.
_AttoCubeImage
    Internal base class shared by AttoCubeSampleImage and
    AttoCubeLaserReferenceImage.
AttoCubeSampleImage
    White-light reference image of the sample.
AttoCubeLaserReferenceImage
    Laser-spot reference image with fitted 1/e² radius.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat, disk
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
import matplotlib.patches as patches

from .constants import (
    EPS_HBN,
    EPS_TMDC,
    HC_EV_NM,
)

from . import processing
from .processing import _draw_region_box, jacobian_correction_wvl2E, subtract_background

# ---------------------------------------------------------------------------
# StackLayer — one material slab in the heterostructure
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

@_dataclass
class StackLayer:
    """
    One TMDC (or generic dielectric) slab in a vdW heterostructure stack.

    Parameters
    ----------
    material : str
        Material name, e.g. ``"WS2"``, ``"MoSe2"``.  Used to look up
        default thickness and dielectric constant from ``constants.py``
        when *d_monolayer* or *eps* are not supplied explicitly.
    n_layers : int
        Number of monolayers of this material.  Default 1.
    d_monolayer : float, optional
        Monolayer thickness in **nm**.  If ``None``, looked up from
        :data:`~tmdc_optics_tools.constants.T_MONOLAYER`.
    eps : float, optional
        Out-of-plane dielectric constant.  If ``None``, looked up from
        :data:`~tmdc_optics_tools.constants.EPS_TMDC`.

    Examples
    --------
    >>> StackLayer("MoSe2")               # 1 ML, defaults from constants
    >>> StackLayer("WSe2", n_layers=2)    # 2 ML WSe2
    >>> StackLayer("WS2", d_monolayer=0.65, eps=7.2)   # explicit override
    """
    material    : str
    n_layers    : int   = 1
    d_monolayer : float = None   # resolved in __post_init__
    eps         : float = None   # resolved in __post_init__

    def __post_init__(self):
        from .constants import T_MONOLAYER, EPS_TMDC

        if self.d_monolayer is None:
            if self.material not in T_MONOLAYER:
                raise ValueError(
                    f"No monolayer thickness for '{self.material}' in T_MONOLAYER. "
                    f"Pass d_monolayer explicitly."
                )
            self.d_monolayer = T_MONOLAYER[self.material]

        if self.eps is None:
            if self.material not in EPS_TMDC:
                raise ValueError(
                    f"No dielectric constant for '{self.material}' in EPS_TMDC. "
                    f"Pass eps explicitly."
                )
            self.eps = EPS_TMDC[self.material]

    @property
    def thickness(self) -> float:
        """Total thickness of this slab in nm (n_layers × d_monolayer)."""
        return self.n_layers * self.d_monolayer

    def __repr__(self) -> str:
        return (
            f"StackLayer({self.material}, n_layers={self.n_layers}, "
            f"d={self.thickness:.3f} nm, ε={self.eps})"
        )


# ---------------------------------------------------------------------------
# DeviceGeometry
# ---------------------------------------------------------------------------

class DeviceGeometry:
    """
    Physical geometry and dielectric constants of a vdW heterostructure.

    The heterostructure is modelled as a vertical stack of dielectric slabs
    in series (series-capacitor model).  The effective dielectric constant
    and optical thickness are computed from the general formula:

        d_total / ε_eff = Σ_i  d_i / ε_i

    where the sum runs over every slab (hBN top, TMDC layers, hBN bottom).

    Parameters
    ----------
    tmdc_stack : list of StackLayer
        Ordered list of TMDC (or other dielectric) slabs between the two
        hBN layers.  For a simple monolayer use
        ``[StackLayer("WS2")]``; for a heterostructure use e.g.
        ``[StackLayer("MoSe2"), StackLayer("WSe2")]``.
    d_hbn_top : float or None
        Top hBN thickness in nm.  Pass ``None`` for a device without a
        top hBN encapsulation layer (e.g. no top gate dielectric).
    d_hbn_bottom : float or None
        Bottom hBN thickness in nm.  Pass ``None`` likewise.
    eps_hbn : float
        Out-of-plane hBN dielectric constant.  Defaults to
        :data:`~tmdc_optics_tools.constants.EPS_HBN`.
    label : str, optional
        Human-readable description of the stack, e.g.
        ``"hBN/MoSe2/WSe2/hBN"``.  For record-keeping only.

    Class methods
    -------------
    from_single(tmdc, d_hbn_top, d_hbn_bottom, ...)
        Convenience constructor for single-material stacks — preserves
        the old interface so existing code does not need to change.

    Examples
    --------
    **Simple monolayer (old-style, via classmethod):**

    >>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)

    **Heterobilayer MoSe2/WSe2:**

    >>> geom = DeviceGeometry(
    ...     tmdc_stack   = [StackLayer("MoSe2"), StackLayer("WSe2")],
    ...     d_hbn_top    = 53,
    ...     d_hbn_bottom = 46,
    ...     label        = "hBN/MoSe2/WSe2/hBN",
    ... )

    **Trilayer with non-default thicknesses:**

    >>> geom = DeviceGeometry(
    ...     tmdc_stack   = [
    ...         StackLayer("WS2",   n_layers=2),
    ...         StackLayer("MoS2",  n_layers=1),
    ...     ],
    ...     d_hbn_top    = 30,
    ...     d_hbn_bottom = 40,
    ... )

    **No top hBN (single-gated device):**

    >>> geom = DeviceGeometry(
    ...     tmdc_stack   = [StackLayer("WSe2")],
    ...     d_hbn_top    = None,
    ...     d_hbn_bottom = 50,
    ... )
    """

    def __init__(
        self,
        tmdc_stack   : list,          # list[StackLayer]
        d_hbn_top    : float = None,
        d_hbn_bottom : float = None,
        eps_hbn      : float = EPS_HBN,
        label        : str   = None,
    ):
        if not tmdc_stack:
            raise ValueError("tmdc_stack must contain at least one StackLayer.")

        self.tmdc_stack   = list(tmdc_stack)
        self.d_hbn_top    = d_hbn_top
        self.d_hbn_bottom = d_hbn_bottom
        self.eps_hbn      = eps_hbn
        self.label        = label
        self.slabs = self._slabs()  # precompute for efficiency

    # --- Classmethod for backward compatibility ----------------------------

    @classmethod
    def from_single(
        cls,
        tmdc         : str,
        d_hbn_top    : float = None,
        d_hbn_bottom : float = None,
        n_layers     : int   = 1,
        d_monolayer  : float = None,
        eps_tmdc     : float = None,
        eps_hbn      : float = EPS_HBN,
        label        : str   = None,
    ) -> "DeviceGeometry":
        """
        Convenience constructor for a single-material TMDC stack.

        Mirrors the old ``DeviceGeometry(tmdc=..., layers=..., ...)`` interface
        so existing code requires only minimal changes:

            # old
            DeviceGeometry(d_hbn_top=53, d_hbn_bottom=46, tmdc="WS2", layers=2)
            # new (equivalent)
            DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46, n_layers=2)

        Parameters
        ----------
        tmdc : str
            Material name, e.g. ``"WS2"``.
        d_hbn_top, d_hbn_bottom : float or None
            hBN thicknesses in nm.
        n_layers : int
            Number of monolayers.
        d_monolayer : float, optional
            Monolayer thickness in nm.  Looked up from constants if ``None``.
        eps_tmdc : float, optional
            Dielectric constant.  Looked up from constants if ``None``.
        eps_hbn : float
            hBN dielectric constant.
        label : str, optional
            Stack description string.
        """
        layer = StackLayer(tmdc, n_layers=n_layers,
                           d_monolayer=d_monolayer, eps=eps_tmdc)
        return cls(
            tmdc_stack   = [layer],
            d_hbn_top    = d_hbn_top,
            d_hbn_bottom = d_hbn_bottom,
            eps_hbn      = eps_hbn,
            label        = label,
        )

    # --- Internal: build the ordered slab list ----------------------------

    def _slabs(self) -> list:
        """
        Return an ordered list of ``(thickness_nm, epsilon)`` tuples for
        every slab in the stack, including the hBN layers if present.
        """
        slabs = []
        if self.d_hbn_top is not None:
            slabs.append((self.d_hbn_top, self.eps_hbn))
        for layer in self.tmdc_stack:
            slabs.append((layer.thickness, layer.eps))
        if self.d_hbn_bottom is not None:
            slabs.append((self.d_hbn_bottom, self.eps_hbn))
        return slabs

    # --- Derived quantities ------------------------------------------------

    @property
    def eps_hs(self) -> float:
        """
        Effective out-of-plane dielectric constant of the heterostructure (TMDCs and hBN)
        computed with the series-capacitor (harmonic-mean) model:

            d_total / ε_eff = Σ_i  d_i / ε_i

        This accounts for every slab in the stack — top hBN, each TMDC layer,
        and bottom hBN — with their individual thicknesses and dielectric
        constants.
        """
        slabs   = self._slabs()
        d_hs = sum(layer.thickness for layer in slabs)
        return d_hs / sum(layer.thickness / layer.eps for layer in slabs)
    
    @property
    def eps_2d(self) -> float:
        """
        Effective out-of-plane dielectric constant of the TMDC layers
        computed with the series-capacitor (harmonic-mean) model:

            d_total / ε_eff = Σ_i  d_i / ε_i

        This accounts for only TMDC layers in the stack with their individual thicknesses and dielectric
        constants.
        """
        tmdc_slabs   = self.tmdc_stack
        d_2d = sum(layer.thickness for layer in tmdc_slabs)
        return d_2d / sum(layer.thickness / layer.eps for layer in tmdc_slabs)
    

    @property
    def optical_thickness(self) -> float:
        """
        Effective optical thickness of the full heterostructure in nm:

            d_opt = d_total × ε_eff
        """
        slabs   = self.slabs
        d_2d = sum(d for d, _ in slabs)
        return d_2d * self.eps_2d

    @property
    def heterostructure_thickness(self) -> float:
        """
        Returns the thickness of the heterostructure consisting of TMDC layers and the hBN layers
        """
        slabs = self.slabs
        d_hs = sum(d for d, _ in slabs)
        return d_hs
    
    @property
    def stack_label(self) -> str:
        """
        Human-readable description of the stack, e.g. ``"hBN/MoSe2/WSe2/hBN"``.
        If a custom label was provided at initialization, it is returned
        instead.
        """
        if self.label:
            return self.label
        parts = []
        if self.d_hbn_top is not None:
            parts.append(f"hBN({self.d_hbn_top:.0f} nm)")
        for layer in self.tmdc_stack:
            parts.append(f"{layer.material}({layer.n_layers} ML)")
        if self.d_hbn_bottom is not None:
            parts.append(f"hBN({self.d_hbn_bottom:.0f} nm)")
        return " / ".join(parts)

    def electric_field(
        self, v_top: np.ndarray, v_bot: np.ndarray
    ) -> np.ndarray:
        """
        Displacement field at the TMDC in mV/nm from gate voltages.

        Uses the parallel-plate capacitor model for the full dielectric stack.
        Requires at least one hBN layer to be defined.

        Parameters
        ----------
        v_top : array-like
            Top gate voltages in V.
        v_bot : array-like
            Bottom gate voltages in V.

        Returns
        -------
        np.ndarray
            Electric displacement field in mV/nm.

        Raises
        ------
        ValueError
            If neither hBN layer is set (optical_thickness is not meaningful).
        """
        if self.d_hbn_top is None and self.d_hbn_bottom is None:
            raise ValueError(
                "Cannot compute electric_field: both d_hbn_top and "
                "d_hbn_bottom are None. At least one hBN layer is required."
            )
        vdiff = np.asarray(v_bot) - np.asarray(v_top)
        return 1000.0 * (vdiff/ self.heterostructure_thickness) * (self.eps_hbn / self.eps_2d)

    # --- Dunder methods ----------------------------------------------------

    def __repr__(self) -> str:
        hbn_top_str = (f"{self.d_hbn_top} nm" if self.d_hbn_top is not None
                       else "None")
        hbn_bot_str = (f"{self.d_hbn_bottom} nm" if self.d_hbn_bottom is not None
                       else "None")
        stack_str   = " / ".join(repr(s) for s in self.tmdc_stack)
        label_str   = f"\n  Label         : {self.label}" if self.label else ""
        return (
            f"DeviceGeometry\n"
            f"  hBN top       : {hbn_top_str}\n"
            f"  TMDC stack    : {stack_str}\n"
            f"  hBN bottom    : {hbn_bot_str}\n"
            f"  ε_HS         : {self.eps_hs:.4f}\n"
            f"  HS thickness  : {self.heterostructure_thickness:.2f} nm"
            f"{label_str}"
        )


# ---------------------------------------------------------------------------
# AttoCubePLScan
# ---------------------------------------------------------------------------

class AttoCubePLVabScan:
    """
    Parser for gate-dependent PL scans from the AttoCube cryogenic confocal.

    The AttoCube software exports a CSV where:

    * The **first column** is a row label (parameter name, e.g. ``"V_A"``).
    * Every **sweep point** occupies four consecutive columns:
      ``[Par, Wavelength, ExpROI1, ExpROI2]``.
    * The file is padded with empty columns beyond the last sweep point.

    The class reads the raw file, strips padding, extracts voltages,
    and (if a :class:`DeviceGeometry` is supplied) computes the
    displacement field axis automatically.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` file.
    geometry : DeviceGeometry, optional
        Device geometry used to convert gate voltages to a displacement
        field. If not supplied, the :attr:`ef` attribute is ``None``.
    bg_region_nm : tuple of (wl_min, wl_max), optional
        Wavelength range in **nm** used to estimate the background level.
        The mean counts in this window are subtracted from every sweep
        point *before* any Jacobian correction is applied, which is the
        correct order of operations.  Mutually exclusive with
        *bg_region_eV*; passing both raises ``ValueError``.
    bg_region_eV : tuple of (E_min, E_max), optional
        Same as *bg_region_nm* but specified in **eV**.  Internally
        converted to a wavelength range (with the order flipped, since
        energy and wavelength are inversely related) before subtraction.
        Mutually exclusive with *bg_region_nm*.
    apply_jacobian : bool
        If ``True`` (default), the Jacobian correction
        ``dλ/dE = λ²/(hc)`` is applied when building the energy-axis
        spectra, so that integrated intensity is conserved under the
        wavelength → energy change of variables.  Set to ``False`` to
        skip the correction (useful when only peak *positions* are
        needed and the density distortion is undesirable).
    top_gate_label : str, optional
        Override the CSV row label for the top-gate voltage channel.
        ``None`` (default) uses the :attr:`_CURATED` registry value (``"V_A"``).
    bot_gate_label : str, optional
        Override the bottom-gate voltage row label.
        ``None`` (default) uses the registry value (``"V_B"``).
    power_label : str, optional
        Override the excitation power row label.
        ``None`` (default) uses the registry value (``"Excitation Power"``).
    power_scale : float, optional
        Override the multiplicative factor that converts raw power to µW.
        ``None`` (default) uses the registry value ``0.303e6`` (calibrated by CdG).
    ich1_label, ich2_label : str, optional
        Override the current-channel row labels (registry defaults ``"I_A"`` /
        ``"I_B"``, both scaled to nA).
    roi : {1, 2}
        Which spectrometer ROI to load. Default ``1``.

    Attributes
    ----------
    wavelength : np.ndarray, shape (n_pixels,)
        Spectrometer wavelength axis in nm (original, ascending order).
    energy : np.ndarray, shape (n_pixels,)
        Photon energy axis in eV (ascending order).
    spectra : np.ndarray, shape (n_pixels, n_sweeps)
        Raw PL counts in wavelength space. Never modified after loading.
    energy_spectra : np.ndarray, shape (n_pixels, n_sweeps)
        Spectra remapped to the energy axis.  Jacobian correction applied
        if *apply_jacobian* is ``True``.  No background subtraction.
    energy_spectra_pre_jacobian : np.ndarray, shape (n_pixels, n_sweeps)
        Spectra remapped to the energy axis with **no** Jacobian correction,
        regardless of *apply_jacobian*.  Useful for comparing raw counts
        on the energy axis or for peak-position fitting where the density
        correction is undesirable.  No background subtraction.
    energy_spectra_bg : np.ndarray or None, shape (n_pixels, n_sweeps)
        Background-subtracted version of *energy_spectra*.  Background is
        removed in wavelength space *before* the Jacobian is applied, so
        the correction does not amplify the residual baseline.  ``None``
        when no *bg_region_nm* / *bg_region_eV* was supplied.
    bg_region_nm : tuple or None
        The background window actually used, always in nm (even if the
        caller supplied *bg_region_eV*).
    apply_jacobian : bool
        Whether the Jacobian correction was applied.
    v_top, v_bot : np.ndarray, shape (n_sweeps,)
        Top / bottom gate voltages in V.  Read-only properties (scaled views
        into :attr:`parameters`).
    power : np.ndarray, shape (n_sweeps,)
        Excitation power in µW.  Read-only property (scaled view).
    Ich1, Ich2 : np.ndarray, shape (n_sweeps,)
        Gate-channel currents in nA.  Read-only properties (scaled views).
    scanner_x, scanner_y : np.ndarray, shape (n_sweeps,)
        Sample-stage X / Y position, assumed µm (scale 1.0 until the raw
        AttoCube unit is confirmed).  Read-only properties (views).
    ef : np.ndarray or None, shape (n_sweeps,)
        Displacement field in mV/nm, or ``None`` if no geometry supplied.
        Read-only property computed from :attr:`v_top` / :attr:`v_bot`.
    curated_parameters : dict[str, tuple]
        Mapping ``attr -> (csv_label, scale, unit)`` documenting which rows are
        promoted to the curated properties above, the scale applied, and the
        resulting unit.  Configurable via the class-level :attr:`_CURATED`
        registry and the constructor ``*_label`` / ``power_scale`` overrides.
    parameters : dict[str, np.ndarray]
        Every labeled parameter row from the CSV, mapped to its per-sweep
        array in the file's **raw** units (one entry per row, shape
        ``(n_sweeps,)``).  Exposes the full instrument state — e.g.
        ``"Scanner X"``, ``"Scanner Y"``, ``"Galvo_X"``, ``"Galvo_Y"``,
        ``"Excitation Power"``, ``"T"`` — beyond the curated attributes above.
        The curated attributes (:attr:`v_top`, :attr:`power`, …) are scaled
        views into this store; the values here are unscaled.
    geometry : DeviceGeometry or None
    path : str

    Notes
    -----
    Use :attr:`parameter_labels` to list every available row name and
    :meth:`get_parameter` (or ``scan["label"]``) to pull any one of them, with
    an optional ``scale`` factor for unit conversion.

    Beware the raw-vs-scaled distinction: :attr:`parameters` holds the file's
    **raw** values, whereas the curated properties apply a scale — e.g.
    ``scan.power`` (µW) differs from ``scan.parameters["Excitation Power"]``
    (raw).  See :attr:`curated_parameters` for the exact label/scale/unit map.

    Examples
    --------
    >>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)

    >>> # No background subtraction, Jacobian applied (default)
    >>> scan = AttoCubePLVabScan("myscan.csv", geometry=geom)

    >>> # Discover and pull any instrument parameter
    >>> scan.parameter_labels                 # ['Excitation Power', 'Galvo_X', ...]
    >>> scan.get_parameter("Scanner X")       # (n_sweeps,) raw units
    >>> scan["Galvo_Y"]                        # sugar for get_parameter

    >>> # Background from a wavelength window, Jacobian applied
    >>> scan = AttoCubePLVabScan("myscan.csv", geometry=geom, bg_region_nm=(930, 960))

    >>> # Background from an energy window, no Jacobian correction
    >>> scan = AttoCubePLVabScan(
    ...     "myscan.csv", geometry=geom,
    ...     bg_region_eV=(1.28, 1.32),
    ...     apply_jacobian=False,
    ... )
    """

    _COL_PAR  = 0
    _COL_WL   = 1
    _COL_ROI1 = 2
    _COL_ROI2 = 3

    # Canonical curated parameters: attribute name -> (CSV row label, scale, unit).
    # These are the analysis-primary quantities promoted to first-class
    # properties (scaled views into :attr:`parameters`).  The label and/or scale
    # of any entry can be overridden per-instance via the matching constructor
    # argument; everything else in the file is reached through the generic
    # :attr:`parameters` store.
    _CURATED = {
        "v_top":     ("V_A",              1.0,      "V"),
        "v_bot":     ("V_B",              1.0,      "V"),
        "power":     ("Excitation Power", 0.303e6,  "µW"),
        "Ich1":      ("I_A",              1e9,      "nA"),
        "Ich2":      ("I_B",              1e9,      "nA"),
        # Sample-stage position.  Unit assumed µm with scale 1.0 until the raw
        # AttoCube units are confirmed (adjust the scale here if not microns).
        "scanner_x": ("Scanner X",        1.0,      "µm"),
        "scanner_y": ("Scanner Y",        1.0,      "µm"),
    }

    def __init__(
        self,
        path            : str,
        geometry        : DeviceGeometry = None,
        bg_region_nm    : tuple = None,
        bg_region_eV    : tuple = None,
        apply_jacobian  : bool  = False,
        top_gate_label  : str   = None,
        bot_gate_label  : str   = None,
        power_label     : str   = None,
        power_scale     : float = None,
        ich1_label      : str   = None,
        ich2_label      : str   = None,
        roi             : int   = 1,
    ):
        if roi not in (1, 2):
            raise ValueError("roi must be 1 or 2.")
        if bg_region_nm is not None and bg_region_eV is not None:
            raise ValueError(
                "Provide at most one of bg_region_nm or bg_region_eV, not both."
            )

        self.path           = str(path)
        self.geometry       = geometry
        self.apply_jacobian = apply_jacobian
        self._roi           = roi

        # --- Resolve background window to nm (always work in wavelength space) ---
        if bg_region_eV is not None:
            # E and λ are inversely related: higher E → shorter λ, so the
            # nm interval is (λ(E_max), λ(E_min)) — order flips.
            wl_lo = HC_EV_NM / bg_region_eV[1]   # E_max → λ_min
            wl_hi = HC_EV_NM / bg_region_eV[0]   # E_min → λ_max
            self.bg_region_nm = (wl_lo, wl_hi)
        else:
            self.bg_region_nm = bg_region_nm      # may be None

        # --- Load raw CSV ---
        raw = self._load_raw(path)
        self._row_labels = list(raw.index)

        d = raw.to_numpy(dtype=float)
        valid = ~np.all(np.isnan(d), axis=0)
        d = d[:, valid]

        n_cols = d.shape[1]
        if n_cols % 4 != 0:
            raise ValueError(
                f"After stripping padding, got {n_cols} columns which is "
                f"not divisible by 4. Check the CSV format."
            )

        par_cols  = np.arange(0, n_cols, 4)
        wl_cols   = np.arange(1, n_cols, 4)
        roi1_cols = np.arange(2, n_cols, 4)
        roi2_cols = np.arange(3, n_cols, 4)
        spec_cols = roi1_cols if roi == 1 else roi2_cols

        # Generic parameter store: every labeled row -> per-sweep array, in the
        # file's raw units.  This exposes the full instrument state (Scanner X/Y,
        # Galvo X/Y, temperature, laser settings, …), not just the curated few.
        # The Par column of each sweep block carries the scalar for that row.
        self.parameters = {
            str(label): d[i, par_cols]
            for i, label in enumerate(self._row_labels)
            if not pd.isna(label) and str(label).strip()
        }

        # Resolve the curated registry: class defaults, overridden by any
        # explicit constructor arguments.  The curated quantities are exposed as
        # scaled @property views into self.parameters (single source of truth).
        self._curated = {name: list(cfg) for name, cfg in self._CURATED.items()}
        for name, override in (
            ("v_top", top_gate_label), ("v_bot", bot_gate_label),
            ("power", power_label),    ("Ich1", ich1_label), ("Ich2", ich2_label),
        ):
            if override is not None:
                self._curated[name][0] = override
        if power_scale is not None:
            self._curated["power"][1] = power_scale
        self._curated = {name: tuple(cfg) for name, cfg in self._curated.items()}

        # Fail fast (as before) if a curated row is missing from this file.
        for name, (label, _, _) in self._curated.items():
            if label not in self.parameters:
                raise KeyError(
                    f"Curated parameter '{name}' expects CSV row '{label}', "
                    f"which is not present. Available: {self.parameter_labels}"
                )

        wl_raw   = d[:, wl_cols[0]]
        valid_px = np.isfinite(wl_raw)
        self.wavelength = wl_raw[valid_px]                          # nm, ascending
        self.spectra    = d[valid_px][:, spec_cols]                 # (n_pixels, n_sweeps), raw counts

        # --- Build energy axis and energy-space spectra ---
        self.energy       = HC_EV_NM / self.wavelength              # eV, descending at this point
        _sort_idx         = np.argsort(self.energy)                 # ascending energy sort index
        self.energy       = self.energy[_sort_idx]                  # eV, ascending

        # energy_spectra: Jacobian applied (or not), no background subtraction
        self.energy_spectra = self._build_energy_spectra(
            self.spectra, self.wavelength, _sort_idx, apply_jacobian
        )

        # energy_spectra_pre_jacobian: always no Jacobian, no background subtraction.
        # Identical to energy_spectra when apply_jacobian=False; a separate array
        # when apply_jacobian=True so both representations are always available.
        if apply_jacobian:
            self.energy_spectra_pre_jacobian = self._build_energy_spectra(
                self.spectra, self.wavelength, _sort_idx, apply_jacobian=False
            )
        else:
            self.energy_spectra_pre_jacobian = self.energy_spectra

        # energy_spectra_bg: background subtracted in wavelength space first,
        # then Jacobian applied (or not). None if no bg_region supplied.
        if self.bg_region_nm is not None:
            spectra_bg = subtract_background(
                self.spectra,
                bg_region = self.bg_region_nm,
                x         = self.wavelength,
                axis      = 0,
            )
            self.energy_spectra_bg = self._build_energy_spectra(
                spectra_bg, self.wavelength, _sort_idx, apply_jacobian
            )
        else:
            self.energy_spectra_bg = None

    # --- Private helpers ---------------------------------------------------

    @staticmethod
    def _build_energy_spectra(
        spectra        : np.ndarray,
        wavelength_nm  : np.ndarray,
        sort_idx       : np.ndarray,
        apply_jacobian : bool,
    ) -> np.ndarray:
        """
        Convert raw wavelength-space spectra to an energy-axis array.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_pixels, n_sweeps)
            Spectra in wavelength space (may already have BG subtracted).
        wavelength_nm : np.ndarray, shape (n_pixels,)
            Wavelength axis in nm, matching ``spectra`` row order.
        sort_idx : np.ndarray
            Argsort indices that put the energy axis in ascending order.
        apply_jacobian : bool
            Whether to apply the ``λ²/hc`` density correction.

        Returns
        -------
        np.ndarray, shape (n_pixels, n_sweeps)
            Spectra on the ascending energy axis.
        """
        if apply_jacobian:
            out = jacobian_correction_wvl2E(spectra, wavelength_nm, axis=0)
        else:
            out = spectra.copy()
        return out[sort_idx, :]

    @staticmethod
    def _load_raw(path: str) -> pd.DataFrame:
        return pd.read_csv(path, header=0, index_col=0, low_memory=False)

    def get_parameter(self, label: str, scale: float = 1.0) -> np.ndarray:
        """
        Return the per-sweep values for *label* as a NumPy array.

        Parameters
        ----------
        label : str
            Exact row label as it appears in the CSV, e.g. ``"Galvo_X"``,
            ``"Scanner Y"``, ``"Excitation Power"``.  See
            :attr:`parameter_labels` for the available names.
        scale : float
            Multiplicative factor applied to the raw values (e.g. unit
            conversion).  Default ``1.0`` (raw units, as stored in the file).

        Returns
        -------
        np.ndarray, shape (n_sweeps,)

        Raises
        ------
        KeyError
            If *label* is not present, with the available labels listed.
        """
        if label not in self.parameters:
            raise KeyError(
                f"Parameter '{label}' not found in CSV rows. "
                f"Available: {self.parameter_labels}"
            )
        return self.parameters[label] * scale

    def _curated_value(self, name: str) -> np.ndarray:
        """Return a curated quantity as a scaled view into :attr:`parameters`."""
        label, scale, _ = self._curated[name]
        return self.get_parameter(label, scale)

    # --- Curated parameter properties (scaled views into self.parameters) ---

    @property
    def v_top(self) -> np.ndarray:
        """Top gate voltage in V (per sweep)."""
        return self._curated_value("v_top")

    @property
    def v_bot(self) -> np.ndarray:
        """Bottom gate voltage in V (per sweep)."""
        return self._curated_value("v_bot")

    @property
    def power(self) -> np.ndarray:
        """Excitation power in µW (per sweep)."""
        return self._curated_value("power")

    @property
    def Ich1(self) -> np.ndarray:
        """Channel-1 current in nA (per sweep)."""
        return self._curated_value("Ich1")

    @property
    def Ich2(self) -> np.ndarray:
        """Channel-2 current in nA (per sweep)."""
        return self._curated_value("Ich2")

    @property
    def scanner_x(self) -> np.ndarray:
        """Sample-stage X position in µm (per sweep). Unit assumed; see _CURATED."""
        return self._curated_value("scanner_x")

    @property
    def scanner_y(self) -> np.ndarray:
        """Sample-stage Y position in µm (per sweep). Unit assumed; see _CURATED."""
        return self._curated_value("scanner_y")

    @property
    def ef(self) -> np.ndarray:
        """
        Displacement field in mV/nm (per sweep), or ``None`` if no
        :class:`DeviceGeometry` was supplied.  Computed from the curated
        :attr:`v_top` / :attr:`v_bot`.
        """
        if self.geometry is None:
            return None
        return self.geometry.electric_field(self.v_top, self.v_bot)

    @property
    def curated_parameters(self) -> dict:
        """
        Mapping ``attr -> (csv_label, scale, unit)`` for the curated quantities.

        Documents which CSV rows are promoted to first-class properties, the
        scale applied (e.g. raw amps → nA), and the resulting unit — making the
        raw-vs-scaled distinction with :attr:`parameters` explicit.
        """
        return dict(self._curated)

    # --- Convenience properties --------------------------------------------

    @property
    def parameter_labels(self) -> list:
        """Sorted list of every parameter label available via :attr:`parameters`."""
        return sorted(self.parameters)

    @property
    def n_sweeps(self) -> int:
        """Number of gate voltage sweep points."""
        return self.spectra.shape[1]

    @property
    def n_pixels(self) -> int:
        """Number of spectrometer pixels."""
        return self.spectra.shape[0]

    @property
    def gate_axis(self) -> np.ndarray:
        """Returns :attr:`ef` if a geometry is set, otherwise :attr:`v_top`."""
        return self.ef if self.ef is not None else self.v_top

    @property
    def gate_axis_label(self) -> str:
        """Matching axis label string for :attr:`gate_axis`."""
        return r"$E_F$ (mV/nm)" if self.ef is not None else r"$V_\mathrm{top}$ (V)"

    @property
    def best_energy_spectra(self) -> np.ndarray:
        """
        Return the best available energy-axis spectra.

        Yields :attr:`energy_spectra_bg` when a background region was
        supplied at construction time, otherwise :attr:`energy_spectra`.
        Use this in downstream code (fitting, plotting) to automatically
        benefit from background correction without needing to know whether
        it was configured.
        """
        return (self.energy_spectra_bg
                if self.energy_spectra_bg is not None
                else self.energy_spectra)

    # --- Dunder methods ----------------------------------------------------

    def __getitem__(self, label: str) -> np.ndarray:
        """Sugar for :meth:`get_parameter` — ``scan["Galvo_X"]``."""
        return self.get_parameter(label)

    def __repr__(self) -> str:
        ef_str = (
            f"  {'E_F':<10}: {self.ef.min():.1f} → {self.ef.max():.1f} mV/nm\n"
            if self.ef is not None else ""
        )

        bg_str = (
            f"  {'BG region':<10}: "
            f"{self.bg_region_nm[0]:.1f} – {self.bg_region_nm[1]:.1f} nm\n"
            if self.bg_region_nm is not None else ""
        )

        jac_str = (
            f"  {'Jacobian':<10}: "
            f"{'applied' if self.apply_jacobian else 'not applied'}\n"
        )

        return (
            f"AttoCubePLVabScan — "
            f"{self.n_sweeps} sweeps × {self.n_pixels} pixels\n"
            f"  {'File':<10}: {self.path}\n"
            f"  {'λ range':<10}: "
            f"{self.wavelength.min():.1f} – {self.wavelength.max():.1f} nm\n"
            f"  {'Energy range':<10}: "
            f"{self.energy.min():.3f} – {self.energy.max():.3f} eV\n"
            f"  {'V_top':<10}: "
            f"{self.v_top.min():.1f} → {self.v_top.max():.1f} V\n"
            f"  {'V_bot':<10}: "
            f"{self.v_bot.min():.1f} → {self.v_bot.max():.1f} V\n"
            f"  {'Power':<10}: "
            f"{self.power.min():.1f} → {self.power.max():.1f} µW\n"
            f"{ef_str}"
            f"{bg_str}"
            f"{jac_str}"
        )
    
# ---------------------------------------------------------------------------
# SingleSpectrum
# ---------------------------------------------------------------------------

class SingleSpectrum:
    """
    Single PL spectrum loaded from a 2-row CSV.

    The file must contain exactly two comma-separated rows:

    * **Row 0** : wavelength axis in nm (ascending).
    * **Row 1** : counts (PL intensity).

    Attribute names mirror :class:`AttoCubePLVabScan` so the same plotting
    helpers (e.g. :func:`~tmdc_optics_tools.plotting.plot_single_spectrum`,
    :func:`~tmdc_optics_tools.plotting._resolve_x_axis`) work unchanged.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` file.
    apply_jacobian : bool
        If ``True``, apply the ``dλ/dE = λ²/hc`` density correction when
        building :attr:`energy_spectra`, conserving integrated intensity
        under the wavelength → energy change of variables. Default ``False``.
    bg_region_nm : tuple of (wl_min, wl_max), optional
        Wavelength range in **nm** used to estimate the background level.
        The mean counts in this window are subtracted in wavelength space
        *before* any Jacobian correction is applied (the correct order of
        operations). Mutually exclusive with *bg_region_eV*.
    bg_region_eV : tuple of (E_min, E_max), optional
        Same as *bg_region_nm* but specified in **eV** (internally converted
        to a wavelength range, with the order flipped). Mutually exclusive
        with *bg_region_nm*.

    Attributes
    ----------
    wavelength : np.ndarray, shape (n_pixels,)
        Wavelength axis in nm (ascending).
    spectra : np.ndarray, shape (n_pixels,)
        Raw counts in wavelength space. Never modified after loading.
    spectra_bg : np.ndarray or None, shape (n_pixels,)
        Background-subtracted counts in wavelength space, or ``None`` when no
        background region was supplied.
    energy : np.ndarray, shape (n_pixels,)
        Photon energy axis in eV (ascending).
    energy_spectra : np.ndarray, shape (n_pixels,)
        Raw counts remapped to the energy axis, Jacobian-corrected if
        *apply_jacobian* is ``True``. No background subtraction.
    energy_spectra_bg : np.ndarray or None, shape (n_pixels,)
        Background-subtracted version of :attr:`energy_spectra` (background
        removed in wavelength space before the Jacobian). ``None`` when no
        background region was supplied.
    bg_region_nm : tuple or None
        The background window actually used, always in nm.
    apply_jacobian : bool
    path : str

    Notes
    -----
    Use :attr:`best_spectra` / :attr:`best_energy_spectra` in downstream code
    to automatically pick the background-corrected array when one is available
    and fall back to the raw array otherwise.
    """

    def __init__(
        self,
        path           : str,
        apply_jacobian : bool  = False,
        bg_region_nm   : tuple = None,
        bg_region_eV   : tuple = None,
    ):
        if bg_region_nm is not None and bg_region_eV is not None:
            raise ValueError(
                "Provide at most one of bg_region_nm or bg_region_eV, not both."
            )

        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError(
                f"Expected a 2-row CSV [wavelength; counts], got array of "
                f"shape {arr.shape} from '{path}'."
            )

        self.path           = str(path)
        self.apply_jacobian = apply_jacobian

        # Resolve background window to nm (always work in wavelength space).
        # E and λ are inversely related, so the nm interval order flips.
        if bg_region_eV is not None:
            self.bg_region_nm = (HC_EV_NM / bg_region_eV[1],   # E_max → λ_min
                                 HC_EV_NM / bg_region_eV[0])   # E_min → λ_max
        else:
            self.bg_region_nm = bg_region_nm                   # may be None

        self.wavelength = arr[0]                      # nm, ascending
        self.spectra    = arr[1].astype(float)        # raw counts, wavelength space

        # Energy axis (ascending) and matching energy-space spectra.
        energy        = HC_EV_NM / self.wavelength    # descending
        self._sort_idx = np.argsort(energy)
        self.energy   = energy[self._sort_idx]

        self.energy_spectra = self._to_energy(self.spectra)

        # Background subtracted in wavelength space first, then to energy.
        if self.bg_region_nm is not None:
            self.spectra_bg = subtract_background(
                self.spectra, bg_region=self.bg_region_nm,
                x=self.wavelength, axis=0,
            )
            self.energy_spectra_bg = self._to_energy(self.spectra_bg)
        else:
            self.spectra_bg        = None
            self.energy_spectra_bg = None

    # --- Private helpers ---------------------------------------------------

    def _to_energy(self, spectra: np.ndarray) -> np.ndarray:
        """Apply the Jacobian (if enabled) and reorder onto the energy axis."""
        y = (jacobian_correction_wvl2E(spectra, self.wavelength, axis=0)
             if self.apply_jacobian else spectra)
        return y[self._sort_idx]

    # --- Convenience properties --------------------------------------------

    @property
    def n_pixels(self) -> int:
        """Number of spectrometer pixels."""
        return self.spectra.shape[0]

    @property
    def best_spectra(self) -> np.ndarray:
        """Background-subtracted wavelength-space counts if available, else raw."""
        return self.spectra_bg if self.spectra_bg is not None else self.spectra

    @property
    def best_energy_spectra(self) -> np.ndarray:
        """Background-subtracted energy-space counts if available, else raw."""
        return (self.energy_spectra_bg
                if self.energy_spectra_bg is not None
                else self.energy_spectra)

    def __repr__(self) -> str:
        bg_str = (
            f"  BG region: {self.bg_region_nm[0]:.1f} – {self.bg_region_nm[1]:.1f} nm\n"
            if self.bg_region_nm is not None else ""
        )
        return (
            f"SingleSpectrum — {self.n_pixels} pixels\n"
            f"  File     : {self.path}\n"
            f"  λ range  : {self.wavelength.min():.1f} – {self.wavelength.max():.1f} nm"
            f"  ({self.energy.min():.3f} – {self.energy.max():.3f} eV)\n"
            f"{bg_str}"
            f"  Jacobian : {'applied' if self.apply_jacobian else 'not applied'}\n"
        )


# ---------------------------------------------------------------------------
# AttoCubePLScanRealSpace
# ---------------------------------------------------------------------------

class AttoCubePLScanRealSpace:
    """
    Loader for a gate-dependent sequence of real-space PL images from the
    AttoCube cryogenic confocal.

    Files must be pure numeric CSVs (no header row) matching the pattern
    ``{prefix}*.csv`` in *path*. Files that contain a text header (e.g. a
    spectral scan file) are automatically excluded.

    Parameters
    ----------
    path : str or Path
        Directory containing the ``.csv`` files.
    prefix : str
        Common filename prefix, e.g. ``"PLdualgatesweep_iter_"``.
    geometry : DeviceGeometry, optional
        Device geometry. Stored for reference but not currently used to
        compute a field axis (gate voltages are not embedded in these files).
    laser_ref : AttoCubeLaserReferenceImage, optional
        Laser spot reference, used for annotation in animations.
    """

    def __init__(
        self,
        path      : str,
        prefix    : str,
        geometry  : DeviceGeometry = None,
        laser_ref : "AttoCubeLaserReferenceImage" = None,
        bg_region : tuple = None,
        bg_stat   : str   = "median",
    ):
        self.path      = str(path)
        self.geometry  = geometry
        self.laser_ref = laser_ref
        self.bg_region = bg_region      # (row_slice, col_slice) in pixel space
        self.bg_stat   = bg_stat

        candidates = sorted(Path(path).glob(f"{prefix}*.csv"))
        files = [f for f in candidates if self._is_image_csv(f)]
        if not files:
            raise ValueError(
                f"No real-space image CSV files found with prefix '{prefix}' in '{path}'. "
                f"Found {len(candidates)} candidate(s) but none passed the numeric-grid check "
                f"(spectral scan files with header rows are excluded automatically)."
            )
        self.files = files

    @staticmethod
    def _is_image_csv(path: Path) -> bool:
        """
        Return True if the first line of *path* is parseable as floats.
        Spectral scan files begin with a text header and return False.
        """
        try:
            with open(path, "r") as fh:
                first_line = fh.readline()
            float(first_line.strip().split(",")[0])
            return True
        except (ValueError, OSError):
            return False

    def load_frame(self, idx: int) -> np.ndarray:
        """Load and return a single frame as a 2-D NumPy array."""
        return np.loadtxt(self.files[idx], delimiter=",")

    @property
    def n_frames(self) -> int:
        """Number of frames loaded."""
        return len(self.files)

    def preview_image(self, idx: int, cmap = "gray") -> tuple:
        """Plot a single frame and return (fig, ax)."""
        fig, ax = plt.subplots()
        ax.imshow(self.load_frame(idx), cmap)
        ax.axis("off")
        return fig, ax


# ---------------------------------------------------------------------------
# Shared base for single-image classes
# ---------------------------------------------------------------------------

class _AttoCubeImage:
    """
    Base class for single grayscale images loaded from a CSV.

    Provides :meth:`load_image`, :meth:`show_image`, and the shared laser
    circle annotation logic so subclasses do not duplicate it.
    """

    def __init__(self, path: str, laser_ref: "AttoCubeLaserReferenceImage" = None, bg_region : tuple = None, bg_stat : str = "median"):
        self.path      = str(path)
        self.laser_ref = laser_ref
        self.bg_region = bg_region
        self.bg_stat   = bg_stat
        self.img_raw       = np.loadtxt(self.path, delimiter=",")
        self.img = (
            processing._apply_bg_region(self.img_raw, bg_region, bg_stat)
            if bg_region is not None else self.img_raw
        )

    # --- Internal helpers --------------------------------------------------

    @staticmethod
    def _add_laser_circle(
        ax,
        laser_ref : "AttoCubeLaserReferenceImage",
        linewidth : float = 1,
        legend    : bool  = False,
    ) -> patches.Circle:
        """
        Draw the 1/e² laser boundary circle on *ax* and optionally add a legend.
        Returns the Circle artist.
        """
        circle = patches.Circle(
            (laser_ref.center_x, laser_ref.center_y),
            radius    = laser_ref.radius,
            edgecolor = "red",
            facecolor = "none",
            linewidth = linewidth,
            linestyle = "--",
            label     = f"$1/e^2$ Radius ({laser_ref.radius:.1f} px)",
        )
        ax.add_patch(circle)
        if legend:
            ax.legend(handles=[circle], loc="upper right")
        return circle

    def __array__(self, dtype=None):
        return np.asarray(self.img, dtype=dtype)


    # --- Public interface --------------------------------------------------

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """
        Return the image data as a NumPy array.

        Parameters
        ----------
        copy : bool
            If True, return a copy of the image data. If False, return the
            underlying array directly.
        """
        return self.img.copy() if copy else self.img

    def show_image(
        self,
        img              = None,
        laser_annotation : bool = False,
        legend           : bool = False,
        normalise        : bool = False,
        show_bg_region   : bool = False,
        bg_region_color  : str  = "orange",
    ) -> tuple:
        """
        Display the image and return (fig, ax).

        Parameters
        ----------
        img : np.ndarray, optional
            Image to display. Uses ``self.img`` if ``None``.
        laser_annotation : bool
            Overlay the 1/e² laser spot boundary if a ``laser_ref`` is set.
        legend : bool
            Show a legend for the laser circle.
        normalise : bool
            Rescale intensity to [0, 1] before display.

        Returns
        -------
        fig, ax
        """
        display = self.img if img is None else img
        if normalise:
            display = rescale_intensity(display, in_range="image", out_range=(0, 1))

        fig, ax = plt.subplots()
        ax.imshow(display, cmap="gray")
        ax.axis("off")

        if show_bg_region:
            processing._draw_region_box(ax, self.bg_region, bg_region_color, label="bg region")
            legend = True
        if laser_annotation and self.laser_ref is not None:
            self._add_laser_circle(ax, self.laser_ref, legend=legend)

        return fig, ax


# ---------------------------------------------------------------------------
# AttoCubeSampleImage
# ---------------------------------------------------------------------------

class AttoCubeSampleImage(_AttoCubeImage):
    """
    White-light reference image of the sample taken on the AttoCube confocal.

    Use in conjunction with :class:`AttoCubeLaserReferenceImage` to locate
    the laser spot on the sample.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file.
    laser_ref : AttoCubeLaserReferenceImage, optional
        Laser spot reference for annotation.
    """

    def __init__(self, path: str, laser_ref: "AttoCubeLaserReferenceImage" = None):
        super().__init__(path, laser_ref)


# ---------------------------------------------------------------------------
# AttoCubeLaserReferenceImage
# ---------------------------------------------------------------------------

class AttoCubeLaserReferenceImage(_AttoCubeImage):
    """
    Laser-spot reference image taken on the AttoCube cryogenic confocal.

    On construction the laser-spot centre and 1/e² radius are extracted with a
    pipeline that is robust to white-light illumination (where the spot sits on
    a structured background of flake contrast and reflectivity gradients):

    1. Median filter to despeckle hot pixels.
    2. **White top-hat** background suppression (optional, on by default): the
       laser PSF is a compact bright blob while white light is large-scale
       background.  A structuring element larger than the spot keeps the laser
       and removes the broad illumination/flake structure.
    3. Robust threshold + connected components → pick the brightest *compact,
       round* region (rejecting elongated flake edges) → intensity-weighted
       centroid as the seed.
    4. **2-D Gaussian fit with a tilted-plane baseline** on a local window
       around the seed → precise centre and σ (radius = 2σ).
    5. If the fit fails or returns an implausible σ, fall back to the weighted
       centroid and a second-moment radius estimate.

    Set ``white_light=False`` to skip the top-hat step for clean dark-background
    images.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file.
    expected_radius_px : float
        Approximate 1/e² laser radius in pixels.  Seeds the fit and sizes the
        top-hat structuring element and fit window.  Default ``10.0``.
    white_light : bool
        Apply white top-hat background suppression before fitting.  Default
        ``True``; set ``False`` for clean dark-background reference images.
    tophat_radius : float, optional
        Override the top-hat structuring-element radius in pixels.  ``None``
        (default) uses ``2.5 * expected_radius_px`` (must exceed the spot size).

    Attributes
    ----------
    center_x, center_y : float
        Fitted laser-spot centre (column, row) in pixels.
    radius : float
        Fitted 1/e² radius in pixels (``2σ``).
    """

    def __init__(
        self,
        path               : str,
        expected_radius_px : float = 10.0,
        white_light        : bool  = True,
        tophat_radius      : float = None,
    ):
        super().__init__(path, laser_ref=None)   # no external ref needed
        self.expected_radius_px = float(expected_radius_px)
        self.white_light        = bool(white_light)
        self.tophat_radius      = tophat_radius
        self.center_x, self.center_y, self.radius = self._fit_laser_spot()

    # --- Preprocessing -----------------------------------------------------

    def _preprocess(self) -> np.ndarray:
        """
        Despeckle and (optionally) remove the white-light background.

        Returns a non-negative image in which the laser spot dominates.
        """
        img = ndi.median_filter(np.nan_to_num(self.img.astype(float)), size=3)
        if self.white_light:
            r = self.tophat_radius
            if r is None:
                r = max(3, int(round(2.5 * self.expected_radius_px)))
            img = white_tophat(img, footprint=disk(int(r)))
        return img - img.min()           # ensure non-negative for weighting

    # --- Seed detection ----------------------------------------------------

    def _seed(self, proc: np.ndarray) -> tuple:
        """
        Robust (x0, y0) seed for the laser centre.

        Thresholds the preprocessed image, labels connected components, and
        picks the brightest *compact, round* region (rejecting elongated flake
        edges), returning its intensity-weighted centroid.  Falls back to the
        global intensity centroid if no region qualifies.
        """
        ny, nx = proc.shape
        if not np.any(proc > 0):
            return nx / 2.0, ny / 2.0
        try:
            thr = threshold_otsu(proc)
        except Exception:                       # noqa: BLE001 - degenerate image
            thr = float(np.median(proc) + 3 * np.std(proc))

        mask = proc > thr
        if not mask.any():
            cy, cx = ndi.center_of_mass(proc)
            return float(cx), float(cy)

        regions = regionprops(label(mask), intensity_image=proc)
        exp_area = np.pi * self.expected_radius_px ** 2
        good = [
            r for r in regions
            if r.eccentricity < 0.85 and 0.1 * exp_area <= r.area <= 10 * exp_area
        ]
        pool = good or regions
        best = max(pool, key=lambda r: r.intensity_mean * r.area)  # total intensity
        cy, cx = best.centroid_weighted                            # (row, col)
        return float(cx), float(cy)

    # --- Gaussian fitting --------------------------------------------------

    @staticmethod
    def _gaussian2d_plane(coords, A, x0, y0, sigma, c0, cx, cy):
        """Isotropic 2-D Gaussian plus a tilted-plane baseline."""
        x, y = coords
        g = A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))
        return (g + c0 + cx * x + cy * y).ravel()

    def _window(self, proc: np.ndarray, x0: float, y0: float) -> tuple:
        """Return (sub, X, Y) for a local window of ±3·expected_radius."""
        ny, nx = proc.shape
        half = int(round(3 * self.expected_radius_px)) + 1
        x_lo, x_hi = max(0, int(x0) - half), min(nx, int(x0) + half + 1)
        y_lo, y_hi = max(0, int(y0) - half), min(ny, int(y0) + half + 1)
        sub = proc[y_lo:y_hi, x_lo:x_hi]
        X, Y = np.meshgrid(np.arange(x_lo, x_hi), np.arange(y_lo, y_hi))
        return sub, X, Y

    def _fit_2d(self, proc: np.ndarray, x0: float, y0: float) -> tuple:
        """2-D Gaussian + plane fit on a local window. Returns (cx, cy, sigma)."""
        sub, X, Y = self._window(proc, x0, y0)
        base = float(np.median(sub))
        p0 = [float(sub.max() - base), x0, y0, self.expected_radius_px,
              base, 0.0, 0.0]
        lo = [0.0, X.min(), Y.min(), 0.5, -np.inf, -np.inf, -np.inf]
        hi = [np.inf, X.max(), Y.max(),
              max(2 * self.expected_radius_px, (X.max() - X.min()) + 1),
              np.inf, np.inf, np.inf]
        popt, _ = curve_fit(
            self._gaussian2d_plane, (X.ravel(), Y.ravel()), sub.ravel(),
            p0=p0, bounds=(lo, hi), maxfev=10000,
        )
        return float(popt[1]), float(popt[2]), abs(float(popt[3]))

    def _fallback(self, proc: np.ndarray, x0: float, y0: float) -> tuple:
        """Intensity-weighted centroid + second-moment radius on a window."""
        sub, X, Y = self._window(proc, x0, y0)
        sub = np.clip(sub - np.median(sub), 0, None)
        total = sub.sum()
        if total <= 0:
            return x0, y0, 2.0 * self.expected_radius_px
        cx = float((sub * X).sum() / total)
        cy = float((sub * Y).sum() / total)
        var = float((sub * ((X - cx) ** 2 + (Y - cy) ** 2)).sum() / total)
        sigma = np.sqrt(var / 2.0)          # <r²> = 2σ² for a 2-D Gaussian
        return cx, cy, 2.0 * sigma

    def _fit_laser_spot(self) -> tuple:
        """Return (center_x, center_y, 1/e² radius) using the robust pipeline."""
        proc = self._preprocess()
        x0, y0 = self._seed(proc)
        ny, nx = proc.shape
        try:
            cx, cy, sigma = self._fit_2d(proc, x0, y0)
            plausible = (0 <= cx < nx and 0 <= cy < ny
                         and 0.5 <= sigma <= 0.5 * max(ny, nx))
            if not plausible:
                raise RuntimeError("implausible fit result")
            return cx, cy, 2.0 * sigma
        except Exception:                       # noqa: BLE001 - graceful fallback
            return self._fallback(proc, x0, y0)

    # --- Display -----------------------------------------------------------

    def show_image(
        self,
        laser_annotation : bool = False,
        legend           : bool = False,
        normalise        : bool = False,
    ) -> tuple:
        """
        Display the laser reference image and return (fig, ax).

        Parameters
        ----------
        laser_annotation : bool
            Overlay the fitted 1/e² boundary circle.
        legend : bool
            Show a legend for the circle.
        normalise : bool
            Rescale intensity to [0, 1] before display.

        Returns
        -------
        fig, ax
        """
        # Provide self as own laser_ref so the base helper can draw the circle
        self.laser_ref = self
        fig, ax = super().show_image(
            laser_annotation=laser_annotation,
            legend=legend,
            normalise=normalise,
        )
        self.laser_ref = None   # reset — this class has no external reference
        return fig, ax

    # --- Dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AttoCubeLaserReferenceImage\n"
            f"  File                  : {self.path}\n"
            f"  Center                : ({self.center_x:.1f}, {self.center_y:.1f}) px\n"
            f"  Estimated 1/e² Radius : {self.radius:.1f} px\n"
            f"  Estimated 1/e² Diameter: {2 * self.radius:.1f} px"
        )


# ---------------------------------------------------------------------------
# SingleImage
# ---------------------------------------------------------------------------

class SingleImage(_AttoCubeImage):
    """
    Generic single 2-D image loaded from a numeric CSV grid.

    Exposes the image as :attr:`img`, so it can be displayed in grayscale via
    the inherited :meth:`show_image` or with a colormap (and optional colorbar)
    via :func:`~tmdc_optics_tools.plotting.plot_image`.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file (numeric grid, comma-delimited).
    """

    def __init__(self, path: str):
        super().__init__(path, laser_ref=None)

    def __repr__(self) -> str:
        return (
            f"SingleImage — {self.img.shape[0]} × {self.img.shape[1]} px\n"
            f"  File : {self.path}\n"
        )

class AttoCubePLImage(_AttoCubeImage):
    """A single real-space PL frame — same handling as AttoCubePLScanRealSpace,
    for spot-checking one file without building a full sequence."""
    def __init__(self, path, laser_ref=None, bg_region=None, bg_stat="median"):
        super().__init__(path, laser_ref=laser_ref, bg_region=bg_region, bg_stat=bg_stat)