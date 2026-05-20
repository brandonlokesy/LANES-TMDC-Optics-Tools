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
from scipy.optimize import curve_fit
import matplotlib.patches as patches

from .constants import (
    EPS_HBN,
    EPS_TMDC,
    HC_EV_NM,
)

from . import processing
from .processing import jacobian_correction_wvl2E, subtract_background

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
    def eps_2d(self) -> float:
        """
        Effective out-of-plane dielectric constant of the full heterostructure
        computed with the series-capacitor (harmonic-mean) model:

            d_total / ε_eff = Σ_i  d_i / ε_i

        This accounts for every slab in the stack — top hBN, each TMDC layer,
        and bottom hBN — with their individual thicknesses and dielectric
        constants.
        """
        tmdc_slabs   = self.tmdc_stack
        d_2d = sum(layer.thickness for layer in tmdc_slabs)
        return d_2d / sum(layer.thickness / layer.eps for layer in tmdc_slabs)
    
    @property
    def eps_hs(self) -> float:
        """
        Effective dielectric constant of the heterostructure region (TMDC layers
        only, excluding hBN) computed with the series-capacitor model.
        """
        slabs   = self._slabs()
        d_hs = sum(layer.thickness for layer in slabs)
        return d_hs / sum(layer.thickness / layer.eps for layer in slabs)

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
    top_gate_label : str
        Row label for the top-gate voltage channel. Default ``"V_A"``.
    bot_gate_label : str
        Row label for the bottom-gate voltage channel. Default ``"V_B"``.
    power_label : str
        Row label for the excitation power channel.
        Default ``"Excitation Power"``.
    power_scale : float
        Multiplicative factor applied to the raw power values to convert
        to µW. Default ``0.303e6`` (calibrated by CdG).
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
    v_top : np.ndarray, shape (n_sweeps,)
        Top gate voltages in V.
    v_bot : np.ndarray, shape (n_sweeps,)
        Bottom gate voltages in V.
    power : np.ndarray, shape (n_sweeps,)
        Excitation power in µW.
    ef : np.ndarray or None, shape (n_sweeps,)
        Displacement field in mV/nm, or ``None`` if no geometry supplied.
    geometry : DeviceGeometry or None
    path : str

    Examples
    --------
    >>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)

    >>> # No background subtraction, Jacobian applied (default)
    >>> scan = AttoCubePLVabScan("myscan.csv", geometry=geom)

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

    def __init__(
        self,
        path            : str,
        geometry        : DeviceGeometry = None,
        bg_region_nm    : tuple = None,
        bg_region_eV    : tuple = None,
        apply_jacobian  : bool  = True,
        top_gate_label  : str   = "V_A",
        bot_gate_label  : str   = "V_B",
        power_label     : str   = "Excitation Power",
        power_scale     : float = 0.303e6,
        ich1_label      : str   = "I_A",
        ich2_label      : str   = "I_B",
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

        self.v_top = self._get_row(d, par_cols, top_gate_label)
        self.v_bot = self._get_row(d, par_cols, bot_gate_label)
        self.power = self._get_row(d, par_cols, power_label) * power_scale
        self.Ich1  = self._get_row(d, par_cols, ich1_label) * 1e9   # → nA
        self.Ich2  = self._get_row(d, par_cols, ich2_label) * 1e9   # → nA

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

        self.ef = geometry.electric_field(self.v_top, self.v_bot) if geometry else None

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

    def _get_row(
        self, d: np.ndarray, col_idx: np.ndarray, label: str
    ) -> np.ndarray:
        if label not in self._row_labels:
            raise KeyError(
                f"Label '{label}' not found in CSV rows. "
                f"Available: {self._row_labels}"
            )
        return d[self._row_labels.index(label), col_idx]

    # --- Convenience properties --------------------------------------------

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

    def __repr__(self) -> str:
        ef_str = (
            f"  E_F        : {self.ef.min():.1f} → {self.ef.max():.1f} mV/nm\n"
            if self.ef is not None else ""
        )
        bg_str = (
            f"  BG region  : {self.bg_region_nm[0]:.1f} – {self.bg_region_nm[1]:.1f} nm\n"
            if self.bg_region_nm is not None else ""
        )
        jac_str = f"  Jacobian   : {'applied' if self.apply_jacobian else 'not applied'}\n"
        return (
            f"AttoCubePLVabScan — {self.n_sweeps} sweeps × {self.n_pixels} pixels\n"
            f"  File       : {self.path}\n"
            f"  λ range    : {self.wavelength.min():.1f} – {self.wavelength.max():.1f} nm"
            f"  ({self.energy.min():.3f} – {self.energy.max():.3f} eV)\n"
            f"  V_top      : {self.v_top.min():.1f} → {self.v_top.max():.1f} V\n"
            f"  V_bot      : {self.v_bot.min():.1f} → {self.v_bot.max():.1f} V\n"
            f"{ef_str}"
            f"{bg_str}"
            f"{jac_str}"
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
    ):
        self.path      = str(path)
        self.geometry  = geometry
        self.laser_ref = laser_ref

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

    def preview_image(self, idx: int) -> tuple:
        """Plot a single frame and return (fig, ax)."""
        fig, ax = plt.subplots()
        ax.imshow(self.load_frame(idx), cmap="gray")
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

    def __init__(self, path: str, laser_ref: "AttoCubeLaserReferenceImage" = None):
        self.path      = str(path)
        self.laser_ref = laser_ref
        self.img       = np.loadtxt(self.path, delimiter=",")

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

    # --- Public interface --------------------------------------------------

    def show_image(
        self,
        img              = None,
        laser_annotation : bool = False,
        legend           : bool = False,
        normalise        : bool = False,
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

    On construction the laser spot centre and 1/e² radius are extracted
    by fitting a 1-D Gaussian to the row- and column-summed intensity
    profiles.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file.
    """

    def __init__(self, path: str):
        super().__init__(path, laser_ref=None)   # no external ref needed
        self.center_x, self.center_y, self.radius = self._fit_laser_spot()

    # --- Gaussian fitting --------------------------------------------------

    @staticmethod
    def _gaussian_1d(x, A, x0, y0, sigma):
        """1-D Gaussian: y0 + A·exp(−(x−x0)²/(2σ²))."""
        return y0 + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    def _fit_profile(self, axis: int) -> tuple:
        """
        Sum the image along *axis*, fit a Gaussian, return (center, sigma).
        """
        x       = np.arange(self.img.shape[axis])
        profile = self.img.sum(axis=axis)
        p0 = [
            profile.max() - profile.min(),   # amplitude
            float(np.argmax(profile)),        # center
            float(profile.min()),             # baseline
            10.0,                             # sigma
        ]
        popt, _ = curve_fit(self._gaussian_1d, x, profile, p0=p0)
        center, sigma = popt[1], abs(popt[3])
        return center, sigma

    def _fit_laser_spot(self) -> tuple:
        """Return (center_x, center_y, avg_1e2_radius) from 2-D Gaussian fits."""
        center_x, sigma_x = self._fit_profile(axis=0)
        center_y, sigma_y = self._fit_profile(axis=1)
        radius = (2 * sigma_x + 2 * sigma_y) / 2.0   # average 1/e² radius
        return center_x, center_y, radius

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