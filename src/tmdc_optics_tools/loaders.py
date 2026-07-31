# tmdc_optics_tools/loaders.py
"""
Data loaders and device geometry for TMD heterostructure measurements.

Classes
-------
DeviceGeometry
    Encodes the physical geometry and dielectric constants of a vdW stack.
AttoCubeSpectralSweep
    A sweep of spectra taken on the AttoCube cryogenic confocal — any
    measurement type, over any scanned parameter.  Reads the raw CSV export or
    an HDF5 file written by its own :meth:`AttoCubeSpectralSweep.to_hdf5`.
AttoCubePLVabScan
    Deprecated pre-rename name for the above, fixed to PL gate sweeps.
SingleSpectrum
    Single spectrum from a 2-row CSV.
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

import re
import warnings
from pathlib import Path
from typing import NamedTuple

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
    SIGNAL_LABELS,
    SPECTROSCOPY_TYPES,
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
    is computed from the general formula:

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
    def eps_stack(self) -> float:
        r"""
        Effective out-of-plane dielectric constant of the **whole gate stack**
        (TMDC layers *and* both hBN layers), from the series-capacitor
        (thickness-weighted harmonic-mean) model:

        .. math :: \frac{d_{\text{total}}}{\epsilon_{\text{stack}}} = \sum_{i}\frac{d_{i}}{\epsilon_{i}}

        This accounts for every slab in the stack — top hBN, each TMDC layer,
        and bottom hBN — with their individual thicknesses and dielectric
        constants.

        The mean is harmonic rather than arithmetic because, with no free
        charge between the gates, the displacement field is continuous, so
        :math:`\epsilon_i E_i` is constant and it is the *voltage drops*
        :math:`d_i/\epsilon_i` that add in series.

        See Also
        --------
        eps_2d : the same harmonic mean over the TMDC layers only.
        """
        slabs = self._slabs()  # slabs is a list of (thickness_nm, epsilon)
        d_total = sum(d for d, _ in slabs)
        return d_total / sum(d / eps for d, eps in slabs)
    
    @property
    def eps_2d(self) -> float:
        r"""
        Effective out-of-plane dielectric constant of the TMDC layers
        computed with the series-capacitor (harmonic-mean) model:

        .. math :: \frac{d_{\text{2D}}}{\epsilon_{\text{2D}}} = \sum_{i}\frac{d_{i}}{\epsilon_{i}}

        This accounts for only TMDC layers in the stack with their individual thicknesses and dielectric
        constants.  For a stack of a single material it returns that material's
        :data:`~tmdc_optics_tools.constants.EPS_TMDC` value unchanged; it only
        does work for a genuine heterostructure.

        See Also
        --------
        eps_stack : the same harmonic mean over every slab, hBN included.
        """
        # Thickness-weighted harmonic mean over the TMDC layers only.
        return self.d_2d / sum(
            layer.thickness / layer.eps for layer in self.tmdc_stack
        )

    @property
    def d_2d(self) -> float:
        """
        Total physical thickness of the TMDC layers in nm, :math:`d_{2D}`,
        excluding hBN.

        See Also
        --------
        d_stack : the same total over every slab, hBN included.
        """
        return sum(layer.thickness for layer in self.tmdc_stack)

    @property
    def d_stack(self) -> float:
        """
        Total physical thickness of the whole gate stack in nm, :math:`d_{TOT}`
        — every TMDC layer plus both hBN layers.

        See Also
        --------
        d_2d : the TMDC layers alone.
        """
        return sum(d for d, _ in self.slabs)

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
        r"""
        Electrostatic field inside the TMDC layers, in mV/nm, from gate voltages.

        This is the field that enters the quantum-confined Stark shift, so its
        accuracy sets the accuracy of any dipole length extracted from one.
        Exact within the series-capacitor model:

        .. math ::
            \epsilon_{2D} E_{2D} = \epsilon_{\text{stack}} E_{\text{stack}},
            \qquad E_{\text{stack}} = \frac{V_{BG} - V_{TG}}{d_{TOT}}

        With no free charge between the gates the displacement field is
        continuous, so :math:`\epsilon_i E_i` is the same in every slab.
        :attr:`eps_stack` is by construction the dielectric constant of the
        homogeneous slab of thickness :math:`d_{TOT}` with the same capacitance
        as the real stack, so :math:`E_{\text{stack}}` is that slab's uniform
        field and the identity above is D-continuity restated.

        Parameters
        ----------
        v_top : array-like
            Top gate voltages in V.
        v_bot : array-like
            Bottom gate voltages in V.

        Returns
        -------
        np.ndarray
            Electrostatic field in the TMDC in mV/nm.  Positive when
            ``v_bot > v_top``.

        Raises
        ------
        ValueError
            If neither hBN layer is set, i.e. the device has no gate dielectric.

        Notes
        -----
        **Do not "simplify"** :attr:`eps_stack` to ``eps_hbn`` here.  That form,

        .. math :: E_{2D} \approx \frac{V_{BG}-V_{TG}}{d_{TOT}}
                   \cdot \frac{\epsilon_{hBN}}{\epsilon_{2D}}

        is the thin-TMDC approximation used by the group's earlier MATLAB
        scripts and by this function before 2026-07-30.  It is low by
        :math:`(d_{2D}/d_{hBN})(1 - \epsilon_{hBN}/\epsilon_{2D})` — 0.59 % for
        53/46 nm hBN around a MoSe2/WSe2 bilayer, growing for thicker TMDC
        stacks or thinner hBN — so fields from this function are ~0.6 % higher
        than pre-2026-07-30 results.  Note also that repairing that form by
        substituting :attr:`eps_stack` for ``eps_hbn`` *in the denominator*
        rather than the numerator is wrong by a factor of ~1.8; the arrangement
        above is the one to keep.

        The sign convention depends on which physical gate each voltage came
        from, which is per-session wiring and is not yet recorded per scan.
        Callers are responsible for mapping their source channels to *v_top* and
        *v_bot* correctly; transposing them mirrors the field axis and flips the
        sign of any extracted dipole.
        """
        if self.d_hbn_top is None and self.d_hbn_bottom is None:
            raise ValueError(
                "Cannot compute electric_field: both d_hbn_top and "
                "d_hbn_bottom are None. At least one hBN layer is required."
            )
        vdiff = np.asarray(v_bot) - np.asarray(v_top)
        # eps_2d * E_2d = eps_stack * E_stack, with E_stack = vdiff / d_TOT the
        # uniform field of the equivalent homogeneous slab. 1000 -> mV/nm.
        return 1000.0 * (vdiff / self.d_stack) * (self.eps_stack / self.eps_2d)

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
            f"  ε_stack       : {self.eps_stack:.4f}\n"
            f"  ε_2D          : {self.eps_2d:.4f}\n"
            f"  d_stack       : {self.d_stack:.2f} nm\n"
            f"  d_2D          : {self.d_2d:.2f} nm"
            f"{label_str}"
        )


# ---------------------------------------------------------------------------
# Sweep-axis registry
# ---------------------------------------------------------------------------

# Recognised sweep types: key -> (curated attribute carrying the axis, axis
# label, unit).  A ``None`` source means "no physical axis recorded", so the
# sweep index is used.  Anything *not* in this table is looked up as a raw CSV
# row label instead, which is why an unforeseen swept quantity ("Galvo_Y",
# "T", …) needs no change here to be usable as a sweep axis.
_SWEEP_TYPES = {
    "index"          : (None,        "Sweep index",        ""),
    "electric_field" : ("ef",        r"$E_F$",             "mV/nm"),
    "top_voltage"    : ("v_top",     r"$V_\mathrm{top}$",  "V"),
    "bottom_voltage" : ("v_bot",     r"$V_\mathrm{bot}$",  "V"),
    "power"          : ("power",     "Power",              "µW"),
    "position_x"     : ("scanner_x", r"$x$",               "µm"),
    "position_y"     : ("scanner_y", r"$y$",               "µm"),
}

# Which curated rows each sweep type depends on.  Checked at load time so a
# declared sweep fails immediately, with the available labels listed, rather
# than at the first plot or fit.
_SWEEP_REQUIRES = {
    "electric_field" : ("v_top", "v_bot"),
    "top_voltage"    : ("v_top",),
    "bottom_voltage" : ("v_bot",),
    "power"          : ("power",),
    "position_x"     : ("scanner_x",),
    "position_y"     : ("scanner_y",),
}

# Recognised input formats, dispatched on the file suffix.
_CSV_SUFFIXES  = (".csv",)
_HDF5_SUFFIXES = (".h5", ".hdf5", ".he5")


# ---------------------------------------------------------------------------
# Export block layout
# ---------------------------------------------------------------------------

# The AttoCube export writes one column *block* per sweep point.  Two block
# shapes exist, told apart by the header's field names:
#
#   spectral  [Par_i, Wavelength{i}, ExpROI1_{i}, ExpROI2_{i}]   PL, R, RC …
#   temporal  [Par_i, Wavelength{i}, Exp_{i}]                    TRPL
#
# In the temporal layout the column *named* "Wavelength" actually holds **time**
# — an acquisition-software misnomer, not something to correct on read.
#
# The names are the only honest source for the layout: the block count cannot be
# recovered from the column count, because the exporter over-allocates the header
# and then fills the surplus blocks with zeros (see _drop_unwritten_blocks).
_BLOCK_LAYOUTS = {
    ("Par", "Wavelength", "ExpROI1", "ExpROI2"): "spectral",
    ("Par", "Wavelength", "Exp")               : "temporal",
}

# Which loader reads which layout.  Names rather than classes, since this table
# sits above their definitions; used only to point a mis-aimed load at the right
# class by name.
_CLASS_FOR_KIND = {
    "spectral": "AttoCubeSpectralSweep",
    "temporal": "AttoCubeTRPLSweep",
}

# Trailing digits, with or without a separating underscore: "Par_0" -> "Par",
# "Wavelength12" -> "Wavelength".  "ExpROI1_0" -> "ExpROI1" — the ROI number is
# kept because the underscore-and-index suffix is what gets stripped.
_BLOCK_FIELD_INDEX = re.compile(r"_?\d+$")

# A block's first column, e.g. "Par_0".  Matching the *indexed* form matters: the
# header's own leading label column is "Parameters Labels", which a bare
# startswith("Par") would mistake for a block start.
_BLOCK_START = re.compile(r"^Par_?\d+$")


def _read_block_layout(path) -> dict:
    """
    Determine an export's block layout from its **header line alone**.

    Reading one line keeps this cheap on the large exports (a real reflectance
    raster is 314 MB), and makes it usable as a file *classifier* — which is how
    a directory of TRPL files is sorted into decays and their metadata companion
    without parsing any of them.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        ``kind`` : ``"spectral"`` or ``"temporal"``.
        ``block_width`` : columns per sweep point.
        ``n_blocks`` : sweep points *declared* by the header — not necessarily
            the number actually written; see :func:`_drop_unwritten_blocks`.
        ``roles`` : the de-indexed field names of one block, in order.

    Raises
    ------
    ValueError
        If the file has no ``Par`` columns (not an AttoCube spectral export at
        all — a real-space image CSV, for instance), or if the block's field
        names match no known layout.
    """
    with open(path, "r") as fh:
        header = fh.readline()
    names = [name.strip() for name in header.split(",")]

    # Every block starts with a Par column, so the Par positions give both the
    # block count and the stride — no arithmetic on the total column count.
    par_at = [i for i, name in enumerate(names) if _BLOCK_START.match(name)]
    if not par_at:
        raise ValueError(
            f"'{path}' has no 'Par' columns in its header, so it is not an "
            f"AttoCube spectral export. Real-space image CSVs are bare numeric "
            f"grids with no header — load those with AttoCubePLScanRealSpace."
        )

    block_width = (par_at[1] - par_at[0]) if len(par_at) > 1 else (
        # A single-block file: the block runs to the last non-empty field.
        sum(1 for name in names[par_at[0]:] if name) )

    roles = tuple(
        _BLOCK_FIELD_INDEX.sub("", name)
        for name in names[par_at[0]:par_at[0] + block_width]
    )
    if roles not in _BLOCK_LAYOUTS:
        raise ValueError(
            f"'{path}' has an unrecognised block layout {roles}. Known layouts:\n"
            + "\n".join(f"  {kind:<8} {fields}"
                        for fields, kind in _BLOCK_LAYOUTS.items())
        )

    return {
        "kind"        : _BLOCK_LAYOUTS[roles],
        "block_width" : block_width,
        "n_blocks"    : len(par_at),
        "roles"       : roles,
    }


def _drop_unwritten_blocks(axis_col: np.ndarray, blocks: dict, path) -> tuple:
    """
    Strip blocks the exporter declared and zero-filled but never wrote.

    The acquisition software over-allocates: a 2091-point reflectance raster is
    exported with 4182 declared blocks, the surplus half filled with literal
    ``0.0`` for *every* field.  Those columns are numeric, not empty, so the
    all-NaN padding strip does not touch them — and keeping them fabricates
    thousands of measurements that were never taken.  Removing them is decoding,
    not a correction: there is no data there to preserve.

    The sentinel is the **axis column being identically zero**.  Sound for both
    layouts: a spectrometer axis never contains zeros, and a time axis has only
    its first bin at zero, never the whole column.

    Parameters
    ----------
    axis_col : np.ndarray, shape (n_axis, n_blocks)
        The Wavelength column of every block.
    blocks : dict
        Mapping of field role -> ``(n_axis, n_blocks)`` array, sliced in place.
    path : str or Path
        Used in messages only.

    Returns
    -------
    keep : np.ndarray of bool, shape (n_blocks,)
        Blocks to retain.
    n_declared : int
    axis_block : int
        Index of the first block holding a real axis.  Returned separately from
        *keep* because in the interleaved case every block is kept, and block 0
        may then be a zero-filled one — taking the axis from it would give an
        all-zero wavelength axis and infinite energies.

    Warns
    ------
    UserWarning
        If the unwritten blocks are *interleaved* rather than strictly trailing.
        Nothing is dropped in that case: interleaving would mean this model of
        the export format is wrong, and guessing which columns to discard could
        silently misalign every sweep point against its parameters.
    """
    n_declared = axis_col.shape[1]
    # A block counts as written if its axis column holds any real non-zero value.
    # The finiteness test is not decoration: exports carry a trailing NaN row, and
    # `NaN != 0` is True, so a bare `!= 0` marks every block as written.
    written = np.any(np.isfinite(axis_col) & (axis_col != 0), axis=0)  # (n_blocks,)
    axis_block = int(np.flatnonzero(written)[0]) if written.any() else 0

    if written.all():
        return written, n_declared, axis_block

    # Strictly trailing means: once written stops, it never resumes.
    n_written = int(written.sum())
    if not written[:n_written].all():
        warnings.warn(
            f"'{path}' has {n_declared - n_written} zero-filled block(s) "
            f"*interleaved* with real ones (first unwritten at index "
            f"{int(np.flatnonzero(~written)[0])}, but written blocks continue "
            f"after it). Expected any unwritten blocks to be trailing, so this "
            f"file does not match the known export layout. Keeping all "
            f"{n_declared} blocks rather than guessing which to drop — check "
            f"the file before trusting the sweep axis.",
            UserWarning, stacklevel=3,
        )
        return np.ones(n_declared, dtype=bool), n_declared, axis_block

    return written, n_declared, axis_block


# ---------------------------------------------------------------------------
# _AttoCubeSweep — shared machinery
# ---------------------------------------------------------------------------

class SweepGrid(NamedTuple):
    """A detected 2-D raster: the inner (fast) axis and the outer (slow) one."""
    inner_label : str
    n_inner     : int
    outer_label : str
    n_outer     : int

    def __str__(self) -> str:
        return (f"{self.inner_label} ({self.n_inner}) × "
                f"{self.outer_label} ({self.n_outer}) = "
                f"{self.n_inner * self.n_outer}")


class _AttoCubeSweep:
    """
    Shared machinery for AttoCube sweeps, whatever the measured axis.

    Not constructed directly — see :class:`AttoCubeSpectralSweep` (wavelength /
    energy) and :class:`AttoCubeTRPLSweep` (time).  Everything here is
    independent of what the spectral axis *is*: the parameter store, the curated
    registry, sweep-axis resolution, measurement-type metadata, and HDF5 export.

    Subclasses declare which export layout they read (``_LAYOUT_KIND``) and where
    their axis and signal arrays live (``_AXIS_ATTR``, ``_SIGNAL_ATTR``), then
    drive construction explicitly:

    .. code-block:: python

        payload = self._decode_and_describe(path, spectra_type=..., ...)
        # ... set the axis and signal arrays, apply corrections ...
        self._bind_sweep_axis(sweep, sweep_label, sweep_unit)

    That sequence is spelled out in each subclass rather than hidden behind a
    template method, because the ordering matters — the sweep axis cannot be
    resolved until ``n_sweeps`` is known — and a reader should be able to see it.
    """

    # Set by subclasses.
    _LAYOUT_KIND = None     # "spectral" | "temporal": which export it accepts
    _AXIS_ATTR   = None     # attribute holding the (n_points,) axis
    _SIGNAL_ATTR = None     # attribute holding the (n_points, n_sweeps) signal

    # Canonical curated parameters: attribute name -> (CSV row label, scale, unit).
    # These are the analysis-primary quantities promoted to first-class
    # properties (scaled views into :attr:`parameters`).  The label and/or scale
    # of any entry can be overridden per-instance via ``curated_labels`` /
    # ``curated_scales``; everything else in the file is reached through the
    # generic :attr:`parameters` store.  A row listed here that a given file does
    # not contain is not an error — the property raises only if accessed.
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

    # --- Construction helpers ----------------------------------------------

    def _decode_and_describe(
        self,
        path,
        *,
        spectra_type   : str  = None,
        geometry              = None,
        curated_labels : dict = None,
        curated_scales : dict = None,
    ) -> dict:
        """
        Decode *path* and settle everything that does not depend on the axis.

        Returns the payload so the subclass can pull its axis and signal arrays
        out of it.  See the class docstring for where this sits in construction.
        """
        self.path = str(path)

        payload = self._decode(path)
        meta    = payload["metadata"]
        self.source_metadata = dict(meta)

        # Every labeled row -> per-sweep array in the file's raw units.  Exposes
        # the full instrument state (Scanner X/Y, Galvo X/Y, T, laser settings,
        # …), not just the curated few.
        self.parameters = payload["parameters"]
        # Sweep points the header declared; exceeds n_sweeps when the exporter
        # over-allocated and zero-filled the surplus.  None -> no over-allocation.
        self._n_declared = payload.get("n_declared")

        self.spectra_type = self._resolve_spectra_type(spectra_type, meta)
        self.geometry     = geometry if geometry is not None else meta.get("geometry")

        # Curated registry: class defaults, then the file's recorded map, then
        # explicit constructor overrides.  Curated quantities are scaled @property
        # views into self.parameters (single source of truth).
        self._curated = {name: list(cfg) for name, cfg in self._CURATED.items()}
        for store, idx in ((meta.get("curated_labels"), 0),
                           (curated_labels,             0),
                           (meta.get("curated_scales"), 1),
                           (curated_scales,             1)):
            for name, value in (store or {}).items():
                if name not in self._curated:
                    raise ValueError(
                        f"'{name}' is not a curated parameter. "
                        f"Valid names: {sorted(self._CURATED)}."
                    )
                self._curated[name][idx] = value if idx == 0 else float(value)
        self._curated = {name: tuple(cfg) for name, cfg in self._curated.items()}

        return payload

    def _bind_sweep_axis(self, sweep, sweep_label, sweep_unit) -> None:
        """
        Resolve the sweep axis, falling back to what the file recorded.

        Called last in construction: resolving a sweep validates against
        ``n_sweeps``, which is not known until the signal array is in place.
        """
        meta = self.source_metadata
        (self.sweep_type, self._sweep_source,
         self._sweep_label, self._sweep_unit) = self._resolve_sweep(
            sweep       if sweep       is not None else meta.get("sweep"),
            sweep_label if sweep_label is not None else meta.get("sweep_label"),
            sweep_unit  if sweep_unit  is not None else meta.get("sweep_unit"),
        )

    def _validate_axis_and_signals(self, axis, signals: dict) -> None:
        """
        Check the decoded arrays are self-consistent before anything uses them.

        *signals* maps a display name (as it appears in the export header) to its
        ``(n_points, n_sweeps)`` array, so the error message can name the field
        the caller would recognise.
        """
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(
                f"'{self.path}' yielded an axis of shape {axis.shape}; "
                f"expected a non-empty 1-D array."
            )
        n_points = axis.size
        for name, arr in signals.items():
            if arr.ndim != 2 or arr.shape[0] != n_points:
                raise ValueError(
                    f"'{self.path}': {name} has shape {arr.shape}, which does "
                    f"not match the {n_points}-point axis. Expected "
                    f"(n_points, n_sweeps)."
                )
        shapes = {name: arr.shape for name, arr in signals.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"'{self.path}': mismatched signal shapes {shapes}.")

        n_sweeps = next(iter(signals.values())).shape[1]
        bad = {lbl: arr.shape for lbl, arr in self.parameters.items()
               if arr.shape != (n_sweeps,)}
        if bad:
            raise ValueError(
                f"'{self.path}': parameter rows {bad} do not have one value per "
                f"sweep point (expected shape ({n_sweeps},))."
            )

    # --- Decoding ----------------------------------------------------------

    def _decode(self, path) -> dict:
        """
        Read *path* into the payload every format decodes to.

        The payload carries the axis, the signal array(s), ``parameters`` and
        ``metadata``.  Only the decoders differ between formats; everything after
        this point is shared, which is why HDF5 input needs no second class.
        """
        path = Path(path)
        if path.is_dir():
            return self._decode_dir(path)
        suffix = path.suffix.lower()
        if suffix in _HDF5_SUFFIXES:
            from . import hdf5 as _hdf5      # local: h5py is only needed here
            payload = _hdf5.read_sweep(path)
            # An .h5 carries its axis kind, so the wrong class is caught here just
            # as a mismatched header is for a CSV.
            expected = _hdf5._AXIS_KIND_FOR_LAYOUT[self._LAYOUT_KIND]["name"]
            found    = payload.pop("axis_kind")
            if found != expected:
                raise ValueError(
                    f"'{path}' stores a '{found}' axis, but {type(self).__name__} "
                    f"reads '{expected}'. Use "
                    f"{_hdf5._CLASS_FOR_AXIS_KIND[found]} instead."
                )
            return payload
        if suffix in _CSV_SUFFIXES:
            return self._decode_csv(path)
        raise ValueError(
            f"Unrecognised input format '{suffix}' for '{path}'. Expected a raw "
            f"AttoCube export ({', '.join(_CSV_SUFFIXES)}) or a file written by "
            f"to_hdf5 ({', '.join(_HDF5_SUFFIXES)})."
        )

    @classmethod
    def _decode_csv(cls, path) -> dict:
        raise NotImplementedError(f"{cls.__name__} defines no CSV decoder.")

    def _decode_dir(self, path) -> dict:
        raise ValueError(
            f"'{path}' is a directory, but {type(self).__name__} reads a single "
            f"file. Only TRPL sweeps are stored as a directory of per-point files."
        )

    @staticmethod
    def _resolve_spectra_type(spectra_type: str, meta: dict) -> str:
        """
        Resolve and validate :attr:`spectra_type` from argument and file metadata.

        Required for a raw export, which records nothing.  Taken from the file
        otherwise; if both are present and disagree, the argument wins but the
        disagreement is not swallowed — a relabelled measurement is exactly the
        kind of error that survives into every downstream figure.
        """
        from_file = meta.get("spectra_type")
        if spectra_type is None:
            if from_file is None:
                raise ValueError(
                    "spectra_type is required: this file records no measurement "
                    "type, and it cannot be inferred from the data. Pass one of "
                    f"{sorted(SPECTROSCOPY_TYPES)} — e.g. spectra_type='PL'."
                )
            return from_file

        if spectra_type not in SPECTROSCOPY_TYPES:
            raise ValueError(
                f"spectra_type={spectra_type!r} is not a recognised measurement "
                f"type. Choose from {sorted(SPECTROSCOPY_TYPES)}."
            )
        if from_file is not None and from_file != spectra_type:
            warnings.warn(
                f"This file records spectra_type={from_file!r} but "
                f"spectra_type={spectra_type!r} was passed; using the argument. "
                f"Re-export with to_hdf5 to correct the stored metadata.",
                UserWarning, stacklevel=4,
            )
        return spectra_type

    def _resolve_sweep(self, sweep, label, unit) -> tuple:
        """
        Resolve the declared sweep to ``(key, source, label, unit)``.

        *source* is a ``(kind, name)`` pair consumed by :attr:`sweep_axis`:
        ``("curated", attr)`` for a registry entry, ``("row", label)`` for a raw
        CSV row, ``("index", None)`` when nothing was declared.
        """
        if sweep is None:
            sweep = "index"

        if sweep in _SWEEP_TYPES:
            source, default_label, default_unit = _SWEEP_TYPES[sweep]
            # Fail on the row the *declared* sweep needs, not on every curated
            # row: the requirement follows what the caller said was measured.
            for name in _SWEEP_REQUIRES.get(sweep, ()):
                csv_label = self._curated[name][0]
                if csv_label not in self.parameters:
                    raise KeyError(
                        f"sweep={sweep!r} needs the curated parameter '{name}' "
                        f"(row '{csv_label}'), which '{self.path}' does not "
                        f"contain. Available rows: {self.parameter_labels}. "
                        f"Pass curated_labels={{'{name}': '<row>'}} if it is "
                        f"under another name."
                    )
            if sweep == "electric_field" and self.geometry is None:
                raise ValueError(
                    "sweep='electric_field' needs a DeviceGeometry to convert "
                    "gate voltages into a field — pass geometry=. To sweep a "
                    "raw gate voltage instead, use sweep='top_voltage' or "
                    "'bottom_voltage'."
                )
            kind = ("index", None) if source is None else ("curated", source)
            return (sweep, kind,
                    label if label is not None else default_label,
                    unit  if unit  is not None else default_unit)

        if sweep in self.parameters:
            # A raw row used directly as the axis.  The file does not state its
            # unit, so the label defaults to the row name and the unit is blank
            # until the caller supplies one.
            return (sweep, ("row", sweep),
                    label if label is not None else sweep,
                    unit  if unit  is not None else "")

        raise ValueError(
            f"sweep={sweep!r} is neither a known sweep type nor a parameter row "
            f"in '{self.path}'.\n"
            f"  Sweep types : {sorted(_SWEEP_TYPES)}\n"
            f"  File rows   : {self.parameter_labels}"
        )

    # --- The parameter store -----------------------------------------------

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

    def _curated_or_none(self, name: str) -> np.ndarray:
        """A curated quantity, or ``None`` if its row is absent from this file."""
        label = self._curated[name][0]
        return self._curated_value(name) if label in self.parameters else None

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

    @property
    def parameter_labels(self) -> list:
        """Sorted list of every parameter label available via :attr:`parameters`."""
        return sorted(self.parameters)

    # --- Shape -------------------------------------------------------------

    @property
    def n_sweeps(self) -> int:
        """Number of sweep points."""
        return getattr(self, self._SIGNAL_ATTR).shape[1]

    @property
    def n_points(self) -> int:
        """Length of the measured axis — detector pixels, or time bins."""
        return getattr(self, self._AXIS_ATTR).size

    @property
    def n_declared_sweeps(self) -> int:
        """
        Sweep points the export *declared*.

        Exceeds :attr:`n_sweeps` when the acquisition software over-allocated its
        header and zero-filled the surplus blocks — see
        :func:`_drop_unwritten_blocks`.
        """
        return self._n_declared if self._n_declared is not None else self.n_sweeps

    # --- The sweep axis ----------------------------------------------------

    @property
    def sweep_axis(self) -> np.ndarray:
        """
        The declared sweep axis, shape ``(n_sweeps,)``.

        A registry sweep returns its curated quantity (converted, e.g. a field in
        mV/nm); a raw-row sweep returns that row in file units; an undeclared
        sweep returns ``arange(n_sweeps)`` — never a guess at which parameter was
        meant.  See :meth:`varying_parameters`.
        """
        kind, name = self._sweep_source
        if kind == "index":
            return np.arange(self.n_sweeps, dtype=float)
        if kind == "row":
            return self.get_parameter(name)
        return getattr(self, name)

    @property
    def sweep_axis_label(self) -> str:
        """Axis label for :attr:`sweep_axis`, with its unit when one is known."""
        if self._sweep_unit:
            return f"{self._sweep_label} ({self._sweep_unit})"
        return self._sweep_label

    @property
    def sweep_label(self) -> str:
        """Bare axis label for :attr:`sweep_axis`, without the unit."""
        return self._sweep_label

    @property
    def sweep_unit(self) -> str:
        """Unit of :attr:`sweep_axis`; empty when the file does not state one."""
        return self._sweep_unit

    # --- What the measurement is -------------------------------------------

    @property
    def signal_name(self) -> str:
        """Name of the measured signal, from :attr:`spectra_type`."""
        return SIGNAL_LABELS[self.spectra_type][0]

    @property
    def signal_label(self) -> str:
        """
        Y-axis label for the measured signal, e.g. ``"PL intensity (counts)"``.

        Use this rather than hardcoding "PL" in a plot that may one day be
        handed reflectance; the unit is omitted for dimensionless ratios.
        """
        name, unit = SIGNAL_LABELS[self.spectra_type]
        return f"{name} ({unit})" if unit else name

    @property
    def spectroscopy(self) -> str:
        """Human-readable :attr:`spectra_type`, e.g. ``"Photoluminescence"``."""
        return SPECTROSCOPY_TYPES[self.spectra_type]

    # --- What actually varied ----------------------------------------------

    def varying_parameters(self, rtol: float = 1e-3) -> dict:
        """
        Report which parameter rows actually changed across the sweep.

        Evidence for choosing a *sweep*, and the check for "was only one gate
        driven?".  A row counts as varying when its span exceeds *rtol* times
        its own mean magnitude, so instrument read-back jitter on a nominally
        static channel does not register.

        Parameters
        ----------
        rtol : float
            Relative span above which a row is reported.  Default ``1e-3``.

        Returns
        -------
        dict
            ``label -> (min, max, span)``, ordered by *relative* span,
            largest first.

        Notes
        -----
        The ordering ranks how much a channel moved relative to its own
        magnitude; it does **not** identify the swept quantity, and should not be
        read as doing so. A near-zero-mean channel outranks the real sweep axis
        for exactly that reason — on the example reflectance raster the leakage
        currents ``I_A``/``I_B`` (means ~1e-13 A) come above ``Scanner X``/``Y``.
        This returns the evidence; which axis was swept is the caller's to
        declare via ``sweep=``.

        Examples
        --------
        >>> scan.varying_parameters()
        {'V_A': (0.0, 1.0, 1.0), 'V_B': (-1.0, 0.0, 1.0), 'Scanner Y': ...}
        """
        found = []
        for label, arr in self.parameters.items():
            finite = arr[np.isfinite(arr)]
            if finite.size < 2:
                continue
            span  = float(np.ptp(finite))
            # Relative to the row's own magnitude: a 1 mV wobble on a 10 V gate
            # is noise, the same wobble on a 2 mV channel is a sweep.
            scale = max(abs(float(np.mean(finite))), np.finfo(float).tiny)
            if span > rtol * scale:
                found.append((span / scale, label,
                              (float(finite.min()), float(finite.max()), span)))
        found.sort(reverse=True)
        return {label: values for _, label, values in found}

    @property
    def gate_mode(self) -> str:
        """
        How the gates were driven, read off the data.

        ``None`` when the curated gate rows are absent from the file.  Purely
        descriptive: the correlation sign distinguishes an anti-symmetric
        (field-like) sweep from a symmetric (doping-like) one, but which physical
        electrode each channel drove is per-session wiring that no file records
        — see :meth:`DeviceGeometry.electric_field`.
        """
        labels = [self._curated[n][0] for n in ("v_top", "v_bot")]
        if any(lbl not in self.parameters for lbl in labels):
            return None

        varying = self.varying_parameters()
        top_var, bot_var = (lbl in varying for lbl in labels)

        if not (top_var or bot_var):
            return "gates static"
        if top_var and not bot_var:
            return "top-gate only"
        if bot_var and not top_var:
            return "bottom-gate only"

        if self.n_sweeps < 3:
            return "dual-gate"
        r = float(np.corrcoef(self.v_top, self.v_bot)[0, 1])
        if r < -0.95:
            return "dual-gate, anti-correlated (field-like)"
        if r > 0.95:
            return "dual-gate, correlated (doping-like)"
        return "dual-gate, independent"

    def sweep_grid(self, rtol: float = 1e-3) -> SweepGrid:
        """
        Detect a 2-D raster flattened into the sweep axis, or return ``None``.

        A spatial map arrives as one long sweep: the reflectance example is 41
        ``Scanner X`` positions nested inside 51 ``Scanner Y`` positions, 2091
        points in all.  This reports that shape so it is visible in
        :func:`repr`; it does **not** reshape anything, and the sweep axis is
        unaffected.

        The signature looked for is strong: two varying rows whose distinct-value
        counts multiply to exactly :attr:`n_sweeps`, with the inner row taking
        every one of its values before repeating.

        Returns
        -------
        SweepGrid or None
            ``(inner_label, n_inner, outer_label, n_outer)``, inner being the
            fast axis.  ``None`` when the sweep is not a raster — which is the
            normal case for a gate or power sweep.
        """
        n = self.n_sweeps
        if n < 4:
            return None

        counts = {}
        for label in self.varying_parameters(rtol=rtol):
            arr = self.parameters[label]
            n_unique = np.unique(arr[np.isfinite(arr)]).size
            if 1 < n_unique < n:
                counts[label] = n_unique

        for inner, n_inner in counts.items():
            for outer, n_outer in counts.items():
                if inner == outer or n_inner * n_outer != n:
                    continue
                values = self.parameters[inner]
                # The fast axis runs through all its values, then starts over.
                if (np.unique(values[:n_inner]).size == n_inner
                        and np.isclose(values[0], values[n_inner])):
                    return SweepGrid(inner, n_inner, outer, n_outer)
        return None

    # --- Export ------------------------------------------------------------

    def to_hdf5(self, path, **kwargs):
        """
        Write this sweep to a self-describing HDF5 file.

        Thin delegation to :func:`tmdc_optics_tools.hdf5.write_sweep`; see there
        for the layout and for what is deliberately *not* stored.

        Parameters
        ----------
        path : str or Path
            Destination ``.h5`` file.
        **kwargs
            Forwarded to :func:`~tmdc_optics_tools.hdf5.write_sweep`
            (``compression``, ``overwrite``).

        Returns
        -------
        pathlib.Path
            The file written.
        """
        from . import hdf5 as _hdf5
        return _hdf5.write_sweep(self, path, **kwargs)

    # --- Dunder methods ----------------------------------------------------

    def __getitem__(self, label: str) -> np.ndarray:
        """Sugar for :meth:`get_parameter` — ``scan["Galvo_X"]``."""
        return self.get_parameter(label)

    def _repr_axis_lines(self, w: int) -> list:
        """Subclass hook: lines describing the measured axis."""
        return []

    def _repr_extra_lines(self, w: int) -> list:
        """Subclass hook: lines describing corrections and source selection."""
        return []

    def __repr__(self) -> str:
        w = 12
        lines = [
            f"{type(self).__name__} — {self.n_sweeps} "
            f"{'sweep' if self.n_sweeps == 1 else 'sweeps'} × "
            f"{self.n_points} {self._POINT_NOUN}  [{self.spectroscopy}]",
            f"  {'File':<{w}}: {self.path}",
        ]
        lines += self._repr_axis_lines(w)

        if self.sweep_type == "index":
            lines.append(
                f"  {'Sweep':<{w}}: not declared — sweep index used as the axis"
            )
        else:
            axis = self.sweep_axis
            unit = f" {self._sweep_unit}" if self._sweep_unit else ""
            lines.append(
                f"  {'Sweep':<{w}}: {self.sweep_type} — "
                f"{axis.min():.4g} → {axis.max():.4g}{unit}"
            )

        # Gate wiring is per-session and unrecorded, so state which row was read
        # as which gate: transposing them flips the sign of any extracted dipole.
        if self.gate_mode is not None:
            lines.append(
                f"  {'Gates':<{w}}: {self.gate_mode}  "
                f"(top ← '{self._curated['v_top'][0]}', "
                f"bottom ← '{self._curated['v_bot'][0]}')"
            )

        # Context the sweep axis does not already show; skip E_F when the field
        # *is* the swept axis, which the Sweep line has just reported.
        extra = [("Power", self._curated_or_none("power"), "µW")]
        if self.sweep_type != "electric_field":
            extra.append(("E_F", self.ef, "mV/nm"))
        for label, arr, unit in extra:
            if arr is not None:
                lines.append(
                    f"  {label:<{w}}: {arr.min():.1f} → {arr.max():.1f} {unit}"
                )

        varying = [lbl for lbl in self.varying_parameters()][:4]
        if varying:
            lines.append(f"  {'Varying':<{w}}: {', '.join(varying)}")

        grid = self.sweep_grid()
        if grid is not None:
            lines.append(f"  {'Grid':<{w}}: {grid}")

        if self.n_declared_sweeps != self.n_sweeps:
            lines.append(
                f"  {'Blocks':<{w}}: {self.n_declared_sweeps} declared, "
                f"{self.n_declared_sweeps - self.n_sweeps} zero-filled and dropped"
            )
        lines += self._repr_extra_lines(w)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# AttoCubeSpectralSweep
# ---------------------------------------------------------------------------

class AttoCubeSpectralSweep(_AttoCubeSweep):
    """
    A sweep of spectra from the AttoCube cryogenic confocal.

    One spectrum per sweep point, over *any* scanned parameter — displacement
    field, a single gate, excitation power, sample position, or a raw
    instrument channel.  What was measured (:attr:`spectra_type`) and what was
    swept (:attr:`sweep_type`) are recorded metadata rather than assumptions
    baked into the parser, so the same class serves a Stark-shift map and a
    power series.

    Two input formats, dispatched on the file suffix:

    * ``.csv`` — the raw AttoCube export.  The **first column** is a row label
      (e.g. ``"V_A"``), every **sweep point** occupies four consecutive columns
      ``[Par, Wavelength, ExpROI1, ExpROI2]``, and the file is padded with
      empty columns beyond the last sweep point.
    * ``.h5`` / ``.hdf5`` — written by :meth:`to_hdf5`.  Carries the parameter
      rows verbatim plus the measurement metadata, so a re-read reproduces the
      object without the caller re-supplying ``spectra_type``, ``sweep``, or a
      :class:`DeviceGeometry`.

    .. note::
       The CSV parser is still the PL-shaped 4-column layout.  ``spectra_type``
       currently drives **labelling and metadata only** — it does not change how
       the file is decoded, because no reflectance or reflectance-contrast
       export has been characterised yet (see E9 in ``dev/audit-2026-07.md``).
       Loading an ``"R"`` file works only if it happens to share the PL layout.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` or ``.h5`` file.
    spectra_type : str
        What the spectra *are*, one of
        :data:`~tmdc_optics_tools.constants.SPECTROSCOPY_TYPES`
        (``"PL"``, ``"R"``, ``"RC"``, ``"T"``, ``"A"``, ``"Raman"``, ``"TRPL"``).
        Required for CSV input; optional when reading HDF5, where it is taken
        from the file unless given (a mismatch warns and the argument wins).
        Keyword-only and deliberately without a default: the value is written
        into exported metadata and trusted thereafter, so a guess would outlive
        the session that made it.
    sweep : str, optional
        What was scanned.  Either a key of :data:`_SWEEP_TYPES`
        (``"electric_field"``, ``"top_voltage"``, ``"bottom_voltage"``,
        ``"power"``, ``"position_x"``, ``"position_y"``) or any raw CSV row
        label (``"Galvo_Y"``, ``"T"``, …), which is then used in its file
        units.  ``None`` (default) means **no axis is assumed**: the sweep axis
        becomes the sweep index.  Use :meth:`varying_parameters` to see which
        rows actually changed before committing to one.
    sweep_label, sweep_unit : str, optional
        Override the axis label / unit for the resolved sweep.  Needed mainly
        for raw-row sweeps, whose units the file does not state.
    geometry : DeviceGeometry, optional
        Device geometry used to convert gate voltages to a displacement field.
        Without it :attr:`ef` is ``None``, and ``sweep="electric_field"``
        raises.
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
    bg_spectrum : path, SingleSpectrum or array, optional
        A separately **measured** background spectrum, subtracted from every
        sweep point.  Unlike *bg_region_nm*, which removes a scalar estimated
        from a window of the spectrum itself, this removes wavelength-dependent
        structure — a dark frame or stray-light reference.  Applies to PL and
        reflectance alike.  Must be on this scan's wavelength axis; a mismatch
        raises rather than resampling.
    reference : path, SingleSpectrum or array, optional
        Reference spectrum against which to form a **contrast** — for
        reflectance, the bare-substrate measurement.  Supplying it is what opts
        into :attr:`contrast` / :attr:`energy_contrast`; nothing is computed
        otherwise.  Background-corrected first if *bg_spectrum* is also given.
    reference_scale : float, optional
        Multiplies *reference* before the contrast is formed.  Needed when the
        reference was **not** acquired at the same integration time and
        excitation power as the sample, because for a reference scaled by *k*,
        ``(S − kR)/(kR)`` is a *biased* contrast rather than a rescaled one.  A
        2-row reference CSV carries no parameter rows, so the package cannot
        check this and cannot correct for it: matching the acquisition, or
        supplying the ratio here, is the caller's responsibility.
    contrast : {"contrast", "ratio"}
        Which contrast to form when *reference* is given — ``(S − R)/R`` or
        ``S/R``.  See :func:`~tmdc_optics_tools.processing.spectral_contrast`.
    apply_jacobian : bool
        If ``True``, the Jacobian correction ``dλ/dE = λ²/(hc)`` is applied
        when building the energy-axis spectra, so integrated intensity is
        conserved under the wavelength → energy change of variables.  Default
        ``False``: it redistributes spectral weight, so it is opt-in, and peak
        *positions* do not need it.  It is **never** applied to
        :attr:`energy_contrast`: the factor cancels identically in a ratio.
    curated_labels : dict, optional
        Override which CSV row backs a curated attribute, e.g.
        ``{"v_top": "V_B", "v_bot": "V_A"}`` for the opposite gate wiring.
        Keys are :attr:`_CURATED` names; unknown keys raise.
    curated_scales : dict, optional
        Override the scale factor of a curated attribute, e.g.
        ``{"power": 1.0}`` to keep raw power.  Same keys as *curated_labels*.
    roi : {1, 2}
        Which spectrometer ROI :attr:`spectra` points at.  Default ``1``.
        Both are always loaded — see :attr:`spectra_roi1` / :attr:`spectra_roi2`.

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
        when neither *bg_region_nm* / *bg_region_eV* nor *bg_spectrum* was given.
    contrast : np.ndarray or None, shape (n_pixels, n_sweeps)
        Contrast against *reference* in wavelength space, or ``None`` when no
        reference was supplied.  See :attr:`contrast_label`.
    energy_contrast : np.ndarray or None, shape (n_pixels, n_sweeps)
        :attr:`contrast` on the ascending energy axis, built with the Jacobian
        **off** regardless of *apply_jacobian* — ``(S·λ²/hc)/(R·λ²/hc) = S/R``,
        so it cancels exactly in a ratio.
    reference_guarded : np.ndarray or None, shape (n_pixels,)
        Which reference pixels were non-positive and so excluded from the
        contrast (``NaN`` there).  Returned so the gaps can be handled knowingly.
    bg_region_nm : tuple or None
        The background window actually used, always in nm (even if the
        caller supplied *bg_region_eV*).
    apply_jacobian : bool
        Whether the Jacobian correction was applied.
    spectra_roi1, spectra_roi2 : np.ndarray, shape (n_pixels, n_sweeps)
        Both spectrometer ROIs in wavelength space, always loaded.  ``ExpROI2``
        is frequently all zeros in PL exports; for a measurement that carries a
        reference channel it is where that channel would sit.  :attr:`spectra`
        is whichever of the two *roi* selected.
    sweep_axis : np.ndarray, shape (n_sweeps,)
        The declared sweep axis in its own units, or ``arange(n_sweeps)`` when
        no *sweep* was declared.  Read-only property.
    sweep_axis_label : str
        Matching axis label, e.g. ``"$E_F$ (mV/nm)"``.
    sweep_type : str
        The resolved sweep key — a :data:`_SWEEP_TYPES` name, a raw CSV row
        label, or ``"index"``.
    spectra_type : str
        The declared measurement type (see *spectra_type* above).
    signal_label : str
        Y-axis label for the measured signal, from :attr:`spectra_type` — e.g.
        ``"PL intensity (counts)"``, ``"$\\Delta R/R_0$"``.  Use this instead of
        hardcoding "PL" in a plot that may be handed reflectance.
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
    gate_mode : str or None
        Description of how the gates were driven — ``"dual-gate,
        anti-correlated (field-like)"``, ``"bottom-gate only"``, … — or ``None``
        when the gate rows are absent.  Descriptive, from the data; see
        :meth:`varying_parameters`.
    source_metadata : dict
        Metadata recorded in the source file, empty for CSV input.  Includes the
        ``apply_jacobian`` / ``bg_region_nm`` of the session that wrote an HDF5
        file — provenance only.  Those corrections are **not** replayed on read;
        the stored spectra are raw, and a correction stays the caller's decision.
    curated_parameters : dict[str, tuple]
        Mapping ``attr -> (csv_label, scale, unit)`` documenting which rows are
        promoted to the curated properties above, the scale applied, and the
        resulting unit.  Configurable via the class-level :attr:`_CURATED`
        registry and the constructor *curated_labels* / *curated_scales*
        overrides.  A curated row that this file does not contain is simply
        absent here, and its property raises only if accessed.
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

    **A displacement-field PL sweep** (what ``AttoCubePLVabScan`` used to be):

    >>> scan = AttoCubeSpectralSweep(
    ...     "myscan.csv", spectra_type="PL",
    ...     sweep="electric_field", geometry=geom,
    ... )
    >>> scan.sweep_axis_label
    '$E_F$ (mV/nm)'

    **A power series and a position line-scan — same class, same file layout:**

    >>> pwr = AttoCubeSpectralSweep("power.csv", spectra_type="PL", sweep="power")
    >>> pos = AttoCubeSpectralSweep("line.csv",  spectra_type="PL", sweep="position_y")

    **Don't know what was swept?  Load it and ask:**

    >>> scan = AttoCubeSpectralSweep("unknown.csv", spectra_type="PL")
    >>> scan.varying_parameters()             # {'V_A': (0.0, 1.0, 1.0), ...}
    >>> scan.gate_mode
    'dual-gate, anti-correlated (field-like)'

    **Opposite gate wiring**, recorded rather than assumed:

    >>> scan = AttoCubeSpectralSweep(
    ...     "myscan.csv", spectra_type="PL", sweep="electric_field",
    ...     geometry=geom, curated_labels={"v_top": "V_B", "v_bot": "V_A"},
    ... )

    **Any instrument parameter, and a raw row as the sweep axis:**

    >>> scan.parameter_labels                 # ['Excitation Power', 'Galvo_X', ...]
    >>> scan.get_parameter("Scanner X")       # (n_sweeps,) raw units
    >>> scan["Galvo_Y"]                       # sugar for get_parameter
    >>> AttoCubeSpectralSweep("s.csv", spectra_type="PL",
    ...                       sweep="Galvo_Y", sweep_unit="V")

    **Corrections, and a round trip through HDF5:**

    >>> scan = AttoCubeSpectralSweep(
    ...     "myscan.csv", spectra_type="PL", sweep="electric_field",
    ...     geometry=geom, bg_region_eV=(1.28, 1.32), apply_jacobian=True,
    ... )
    >>> scan.to_hdf5("myscan.h5")
    >>> again = AttoCubeSpectralSweep("myscan.h5")   # type, sweep, geometry restored

    See Also
    --------
    tmdc_optics_tools.hdf5 : the export/import format, and what it stores.
    """

    # One sweep point per 4-column block: [Par, Wavelength, ExpROI1, ExpROI2].
    # The block *shape* is read from the header (see _read_block_layout); this is
    # the layout this class is willing to accept.
    _LAYOUT_KIND = "spectral"
    _AXIS_ATTR   = "wavelength"
    _SIGNAL_ATTR = "spectra"
    _POINT_NOUN  = "pixels"

    def __init__(
        self,
        path           : str,
        *,
        spectra_type   : str   = None,
        sweep          : str   = None,
        sweep_label    : str   = None,
        sweep_unit     : str   = None,
        geometry        : DeviceGeometry = None,
        bg_region_nm    : tuple = None,
        bg_region_eV    : tuple = None,
        bg_spectrum            = None,
        reference              = None,
        reference_scale : float = None,
        contrast        : str   = "contrast",
        apply_jacobian  : bool  = False,
        curated_labels  : dict  = None,
        curated_scales  : dict  = None,
        roi             : int   = None,
    ):
        if bg_region_nm is not None and bg_region_eV is not None:
            raise ValueError(
                "Provide at most one of bg_region_nm or bg_region_eV, not both."
            )

        # --- Decode, and settle everything independent of the spectral axis ---
        payload = self._decode_and_describe(
            path, spectra_type=spectra_type, geometry=geometry,
            curated_labels=curated_labels, curated_scales=curated_scales,
        )
        meta = payload["metadata"]

        self.wavelength   = payload["wavelength"]      # nm, ascending
        self.spectra_roi1 = payload["roi1"]            # (n_pixels, n_sweeps)
        self.spectra_roi2 = payload["roi2"]
        self._validate_payload()

        roi = roi if roi is not None else meta.get("roi", 1)
        if roi not in (1, 2):
            raise ValueError(f"roi must be 1 or 2, got {roi!r}.")
        self._roi    = int(roi)
        self.spectra = self.spectra_roi1 if roi == 1 else self.spectra_roi2

        # ExpROI2 is only written for two-spot galvo scans (excitation spot plus
        # a remote, spatially-filtered spot).  A flat zero array is otherwise
        # indistinguishable from a valid dark measurement, so say so.
        if not self.spectra.any():
            hint = (
                "ExpROI2 is only populated for two-spot galvo scans (excitation "
                "spot plus a remote, spatially-filtered spot); every other "
                "measurement leaves it blank. Did you mean roi=1?"
                if self._roi == 2 else
                "ExpROI1 is the normal acquisition ROI, so an all-zero one "
                "suggests the spectrometer recorded nothing for this scan."
            )
            warnings.warn(
                f"roi={self._roi} (ExpROI{self._roi}) is identically zero in "
                f"'{self.path}'. {hint}",
                UserWarning, stacklevel=3,
            )

        self.apply_jacobian  = apply_jacobian
        self.contrast_mode   = contrast
        self.reference_scale = reference_scale

        # Auxiliary spectra, resolved onto this scan's own wavelength grid.
        self.bg_spectrum = self._resolve_aux_spectrum(bg_spectrum, "bg_spectrum")
        self.reference   = self._resolve_aux_spectrum(reference, "reference")
        if self.reference is not None and reference_scale is not None:
            self.reference = self.reference * float(reference_scale)

        # No curated row is mandatory: a file from a different instrument
        # configuration still loads, and each property raises only if accessed.
        # What *is* checked is the row the declared sweep needs — which is why
        # this comes after the signal array, since it validates against n_sweeps.
        self._bind_sweep_axis(sweep, sweep_label, sweep_unit)

        # --- Resolve background window to nm (always work in wavelength space) ---
        if bg_region_eV is not None:
            # E and λ are inversely related: higher E → shorter λ, so the
            # nm interval is (λ(E_max), λ(E_min)) — order flips.
            wl_lo = HC_EV_NM / bg_region_eV[1]   # E_max → λ_min
            wl_hi = HC_EV_NM / bg_region_eV[0]   # E_min → λ_max
            self.bg_region_nm = (wl_lo, wl_hi)
        else:
            self.bg_region_nm = bg_region_nm      # may be None

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

        # --- Wavelength-space corrections, in the order the physics requires ---
        # 1. the bg_region window mean, 2. a measured background spectrum. Both
        # must precede any ratio: a pedestal in either array biases a contrast
        # non-linearly, and both must precede the Jacobian (a flat pedestal B
        # becomes B·λ²/hc — curved, not flat — in energy space).
        corrected = self.spectra
        if self.bg_region_nm is not None:
            corrected = subtract_background(
                corrected,
                bg_region = self.bg_region_nm,
                x         = self.wavelength,
                axis      = 0,
            )
        if self.bg_spectrum is not None:
            corrected = processing.subtract_spectrum(
                corrected, self.bg_spectrum, axis=0)

        # energy_spectra_bg: background-corrected, then Jacobian applied (or not).
        # None when neither background mechanism was used.
        if corrected is not self.spectra:
            self.energy_spectra_bg = self._build_energy_spectra(
                corrected, self.wavelength, _sort_idx, apply_jacobian
            )
        else:
            self.energy_spectra_bg = None

        # --- Contrast against a reference spectrum -------------------------
        if self.reference is None:
            self.contrast = None
            self.energy_contrast = None
            self.reference_guarded = None
        else:
            self.contrast, self.reference_guarded = processing.spectral_contrast(
                corrected, self.reference, mode=contrast, axis=0,
            )
            # The Jacobian is NOT applied here, whatever apply_jacobian says:
            # (S·λ²/hc)/(R·λ²/hc) = S/R, so it cancels identically in a ratio.
            # Applying it to the numerator alone would distort the contrast.
            self.energy_contrast = self._build_energy_spectra(
                self.contrast, self.wavelength, _sort_idx, apply_jacobian=False
            )

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

    # --- Decoding: one payload contract, one builder, two formats ----------

    @classmethod
    def _decode_csv(cls, path) -> dict:
        """
        Decode a raw AttoCube spectral export.

        Each sweep point occupies a ``[Par, Wavelength, ExpROI1, ExpROI2]``
        block, so every field is read as a stride-``block_width`` column slice.
        The layout comes from the header rather than from arithmetic on the
        column count — see :func:`_read_block_layout`.
        """
        blocks = _read_block_layout(path)
        if blocks["kind"] != cls._LAYOUT_KIND:
            raise ValueError(
                f"'{path}' is a {blocks['kind']} export "
                f"({', '.join(blocks['roles'])}), but "
                f"{cls.__name__} reads {cls._LAYOUT_KIND} exports. "
                f"Use {_CLASS_FOR_KIND[blocks['kind']]} instead."
            )

        width, n_declared = blocks["block_width"], blocks["n_blocks"]
        raw = pd.read_csv(path, header=0, index_col=0, low_memory=False)
        row_labels = list(raw.index)

        # Keep only the declared block columns; the exporter appends as many
        # unnamed empty columns again after them.  Slicing here rather than via
        # read_csv(usecols=...) is deliberate and measured: on the 314 MB raster
        # usecols costs 13.8 s against 10.7 s for reading everything and slicing,
        # because filtering 16 728 of 33 457 columns during the parse is dearer
        # than parsing them and throwing them away.
        d = raw.to_numpy(dtype=float)[:, :n_declared * width]

        # One column index per sweep point, for each field of the block.
        cols = {role: np.arange(offset, d.shape[1], width)
                for offset, role in enumerate(blocks["roles"])}

        # Blocks the exporter declared but never wrote are zero-filled, not
        # empty, so they survive any NaN-based strip and must go before the
        # arrays are shaped.
        keep, n_declared, axis_block = _drop_unwritten_blocks(
            d[:, cols["Wavelength"]], cols, path)

        # Labeled scalar rows are overlaid on the leading pixel rows: row i's
        # value for sweep j sits in that block's Par column.
        parameters = {
            str(label): d[i, cols["Par"]][keep]
            for i, label in enumerate(row_labels)
            if not pd.isna(label) and str(label).strip()
        }

        # The axis is repeated once per sweep point; take the first written one
        # and use its finite entries to select the real pixel rows.  With nothing
        # written at all the axis is zeros, and the caller gets the diagnostic
        # from _validate_payload rather than an IndexError.
        axis_raw = d[:, cols["Wavelength"][axis_block]]
        valid_px = np.isfinite(axis_raw)

        payload = {
            "wavelength" : axis_raw[valid_px],
            "parameters" : parameters,
            "metadata"   : {},          # a raw export records none
            "n_declared" : n_declared,
        }
        for role, out in (("ExpROI1", "roi1"), ("ExpROI2", "roi2")):
            payload[out] = d[valid_px][:, cols[role][keep]]
        return payload

    def _resolve_aux_spectrum(self, spec, name: str) -> np.ndarray:
        """
        Resolve an auxiliary spectrum onto this scan's wavelength grid.

        Accepts a path, anything exposing ``wavelength`` plus ``best_spectra``
        (a :class:`SingleSpectrum`, or another sweep's single column), or a bare
        ``(n_pixels,)`` array — the same duck-typing the image loaders use for
        ``laser_ref``.

        A grid **mismatch raises** rather than interpolating: resampling changes
        the numbers and smooths the data, so it is a correction and cannot be a
        default.  A bare array is accepted precisely so a caller who has aligned
        the two axes themselves has a route in, with no extra API.
        """
        if spec is None:
            return None

        if isinstance(spec, (str, Path)):
            spec = SingleSpectrum(str(spec))

        wavelength = getattr(spec, "wavelength", None)
        if wavelength is None:
            values = np.asarray(spec, dtype=float)
            if values.ndim != 1 or values.size != self.wavelength.size:
                raise ValueError(
                    f"{name} was given as a bare array of shape {values.shape}, "
                    f"which does not match this scan's {self.wavelength.size}-pixel "
                    f"wavelength axis. Pass a SingleSpectrum or a path to let the "
                    f"axes be checked."
                )
            return values

        values = np.asarray(getattr(spec, "best_spectra", spec.spectra), dtype=float)
        wavelength = np.asarray(wavelength, dtype=float)
        if wavelength.shape != self.wavelength.shape or not np.allclose(
                wavelength, self.wavelength, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"{name} is on a different wavelength axis from this scan "
                f"({wavelength.size} points spanning "
                f"{wavelength.min():.3f}–{wavelength.max():.3f} nm, against "
                f"{self.wavelength.size} points spanning "
                f"{self.wavelength.min():.3f}–{self.wavelength.max():.3f} nm). "
                f"They are not resampled automatically, because interpolating "
                f"changes the numbers and smooths the data. Resample it yourself "
                f"and pass the resulting array as {name}=."
            )
        return values

    def _validate_payload(self) -> None:
        """Check the decoded arrays are self-consistent before anything uses them."""
        if self.spectra_roi1.ndim == 2 and self.spectra_roi1.shape[1] == 0:
            # Every block was zero-filled: the file carries a parameter table and
            # no measurement.  That is the metadata companion written alongside a
            # TRPL sweep, whose header is the spectral layout.
            raise ValueError(
                f"'{self.path}' declares {self.n_declared_sweeps} sweep point(s) "
                f"but contains no spectra — every block is zero-filled. This is "
                f"the metadata companion written alongside a TRPL sweep, not a "
                f"spectral export. Point AttoCubeTRPLSweep at the directory "
                f"instead; it reads this file for cross-checking."
            )
        self._validate_axis_and_signals(
            self.wavelength,
            {"ExpROI1": self.spectra_roi1, "ExpROI2": self.spectra_roi2},
        )

    @property
    def n_pixels(self) -> int:
        """Number of spectrometer pixels — :attr:`n_points` under its optical name."""
        return self.n_points

    @property
    def roi(self) -> int:
        """Which spectrometer ROI :attr:`spectra` points at (1 or 2)."""
        return self._roi

    # --- The sweep axis ----------------------------------------------------

    @property
    def gate_axis(self) -> np.ndarray:
        """Deprecated alias for :attr:`sweep_axis`."""
        return self.sweep_axis

    @property
    def gate_axis_label(self) -> str:
        """Deprecated alias for :attr:`sweep_axis_label`."""
        return self.sweep_axis_label

    # --- What the spectra are ----------------------------------------------

    @property
    def best_energy_spectra(self) -> np.ndarray:
        """
        Return the best available energy-axis spectra.

        Yields :attr:`energy_spectra_bg` when a background was supplied at
        construction time, otherwise :attr:`energy_spectra`.  Use this in
        downstream code (fitting, plotting) to automatically benefit from
        background correction without needing to know whether it was configured.

        **A contrast array is deliberately not returned here**, even when a
        *reference* was given.  "Best" means the same physical quantity, better
        corrected — not a different quantity.  Contrast is negative-going, so
        feeding it to :func:`~tmdc_optics_tools.fitting.fit_scan_peak`, whose peak
        models decay to zero in their wings, would give quietly meaningless fits;
        and a PL map's colour bar would silently start meaning ΔR/R₀.  Ask for
        :attr:`energy_contrast` explicitly, or ``spectra_source="contrast"`` in
        :mod:`~tmdc_optics_tools.plotting`.
        """
        return (self.energy_spectra_bg
                if self.energy_spectra_bg is not None
                else self.energy_spectra)

    @property
    def contrast_label(self) -> str:
        """
        Y-axis label for :attr:`contrast` / :attr:`energy_contrast`.

        Independent of :attr:`signal_label`, because the contrast is a *derived*
        quantity: a scan of ``spectra_type="R"`` keeps reporting "Reflectance
        (counts)" for its raw spectra while its contrast is labelled ΔR/R₀.
        """
        if self.contrast_mode == "ratio":
            return r"$R/R_0$"
        name, unit = SIGNAL_LABELS["RC"]
        return f"{name} ({unit})" if unit else name

    # --- Dunder methods ----------------------------------------------------

    def _repr_axis_lines(self, w: int) -> list:
        return [
            f"  {'λ range':<{w}}: "
            f"{self.wavelength.min():.1f} – {self.wavelength.max():.1f} nm",
            f"  {'Energy range':<{w}}: "
            f"{self.energy.min():.3f} – {self.energy.max():.3f} eV",
        ]

    def _repr_extra_lines(self, w: int) -> list:
        lines = [f"  {'ROI':<{w}}: ExpROI{self._roi}"]
        if self.bg_region_nm is not None:
            lines.append(
                f"  {'BG region':<{w}}: "
                f"{self.bg_region_nm[0]:.1f} – {self.bg_region_nm[1]:.1f} nm"
            )
        lines.append(
            f"  {'Jacobian':<{w}}: "
            f"{'applied' if self.apply_jacobian else 'not applied'}"
        )
        return lines


# ---------------------------------------------------------------------------
# AttoCubePLVabScan — deprecated pre-rename name
# ---------------------------------------------------------------------------

class AttoCubePLVabScan(AttoCubeSpectralSweep):
    """
    Deprecated alias for :class:`AttoCubeSpectralSweep`, PL gate sweeps only.

    Reproduces the pre-rename behaviour exactly: ``spectra_type="PL"``, and a
    sweep axis of displacement field when a :class:`DeviceGeometry` is supplied,
    top-gate voltage otherwise — which is what the old ``gate_axis`` did.  The
    old per-channel ``*_label`` / ``power_scale`` arguments are accepted and
    folded into the new *curated_labels* / *curated_scales* dictionaries.

    .. deprecated::
       Use :class:`AttoCubeSpectralSweep` with an explicit ``spectra_type=`` and
       ``sweep=``.  This subclass exists so that existing notebooks and scripts
       keep running; it adds no behaviour of its own.

    Notes
    -----
    The warning is a ``FutureWarning``, not a ``DeprecationWarning``: Python
    filters the latter out by default outside ``__main__``, so a library that
    raises one is warning nobody.
    """

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
        warnings.warn(
            "AttoCubePLVabScan is deprecated; use AttoCubeSpectralSweep with an "
            "explicit spectra_type= and sweep=, e.g. "
            "AttoCubeSpectralSweep(path, spectra_type='PL', "
            "sweep='electric_field', geometry=geom).",
            FutureWarning, stacklevel=2,
        )

        labels = {name: value for name, value in (
            ("v_top", top_gate_label), ("v_bot", bot_gate_label),
            ("power", power_label),
            ("Ich1",  ich1_label),     ("Ich2",  ich2_label),
        ) if value is not None}

        super().__init__(
            path,
            spectra_type   = "PL",
            # The old gate_axis was ef when a geometry was given, else v_top.
            sweep          = "electric_field" if geometry is not None else "top_voltage",
            geometry       = geometry,
            bg_region_nm   = bg_region_nm,
            bg_region_eV   = bg_region_eV,
            apply_jacobian = apply_jacobian,
            curated_labels = labels or None,
            curated_scales = {"power": power_scale} if power_scale is not None else None,
            roi            = roi,
        )


# ---------------------------------------------------------------------------
# AttoCubeTRPLSweep
# ---------------------------------------------------------------------------

# The TRPL export's time axis. 4 ps bins over ~12.8 ns, consistent with the
# Picoharp rows in the parameter list and a ~78 MHz repetition rate. Defined once
# so the unit is a single edit if it is ever found to be otherwise.
_TRPL_TIME_UNIT = "ns"

# The per-point index in a TRPL sweep's filenames, e.g. "..._iter_12.csv".
# Ordering must be numeric on this: plain lexicographic sorting puts iter_10
# before iter_2, which would silently pair every decay with the wrong parameters.
_ITER_INDEX = re.compile(r"_iter_(\d+)$", re.IGNORECASE)


class AttoCubeTRPLSweep(_AttoCubeSweep):
    """
    Time-resolved PL from the AttoCube cryogenic confocal.

    One TCSPC decay per sweep point.  A **single decay is one file**; a **sweep is
    a directory** of per-point files, one per iteration, written alongside a
    metadata companion — so this class accepts either.

    The export's block is ``[Par_i, Wavelength{i}, Exp_{i}]``, and the column
    *named* "Wavelength" actually holds **time**: an acquisition-software
    misnomer that this class reads as time and exposes as :attr:`time`.

    Why this is a separate class from :class:`AttoCubeSpectralSweep` rather than a
    mode of it: a single decay is simply ``n_sweeps == 1``, which the sweep
    machinery already handles, so that is not the dividing line.  The axis is.
    ``energy = hc/t`` is meaningless and divides by zero at ``t = 0``, so
    ``energy``, ``energy_spectra``, ``apply_jacobian`` and ``bg_region_eV`` do not
    exist here — and neither does ``spectra``, so handing a TRPL sweep to a
    spectral plot raises rather than drawing time as if it were wavelength.

    Parameters
    ----------
    path : str or Path
        A directory (a sweep) or a single ``.csv`` (one decay).  An ``.h5``
        written by :meth:`to_hdf5` also works.
    prefix : str, optional
        Filename filter applied within a directory, e.g. ``"TRPL_"``.  ``None``
        (default) considers every ``.csv`` and classifies by content.
    spectra_type : str
        Defaults to ``"TRPL"``.  Deliberately unlike
        :class:`AttoCubeSpectralSweep`, which requires an explicit type: the
        class name already declares the modality here, whereas a spectral sweep
        may be PL, R, RC, T or A and cannot know which.
    sweep, sweep_label, sweep_unit : str, optional
        As :class:`AttoCubeSpectralSweep`.  The example 3-point sweep is a
        displacement-field sweep, so ``sweep="electric_field"`` with a *geometry*
        gives a field axis.
    geometry : DeviceGeometry, optional
    bg_region_ns : tuple of (t_min, t_max), optional
        **Pre-pulse** time window whose mean is subtracted from every decay —
        the TCSPC equivalent of a spectral background window, and the dark/
        afterpulse floor before the rise.  ``None`` (default) subtracts nothing.
    curated_labels, curated_scales : dict, optional
    time_rtol : float
        Tolerance for agreement between the per-file time axes.  They are *not*
        bit-identical across files — the example sweep's bin width varies in its
        seventh figure — so this is a closeness test, not an equality test.

    Attributes
    ----------
    time : np.ndarray, shape (n_bins,)
        Time axis in ns, ascending.
    decays : np.ndarray, shape (n_bins, n_sweeps)
        Raw photon counts.  Never modified after load.
    decays_bg : np.ndarray or None
        Pre-pulse-subtracted decays, or ``None`` when no *bg_region_ns* was given.
    best_decays : np.ndarray
        :attr:`decays_bg` when available, else :attr:`decays`.
    n_bins : int
        Number of time bins.
    files : list of Path
        The per-point data files, in sweep order.  A single-file load gives one.
    metadata_file : Path or None
        The companion parameter table, when the directory contained one.
    declared_parameters : dict or None
        The companion's own parameter table, one value per declared sweep point.
        Provided for comparison against :attr:`parameters`; the two are
        independent read-backs and are not expected to agree on drifting
        channels — see :meth:`_cross_check_companion`.

    Notes
    -----
    Parameters come from **each data file's own snapshot**, not from the metadata
    companion.  Each snapshot is recorded at the moment its own decay was
    measured, and using them means a sweep still loads when the companion was
    never written or has been lost.  The companion is evidence: it supplies
    :attr:`n_declared_sweeps`, so a truncated sweep is visible, and its own table
    is exposed as :attr:`declared_parameters` for comparison.  Its values are not
    checked row by row — it is written after the last decay, so channels that
    drift genuinely disagree with the snapshots taken during acquisition.

    Examples
    --------
    >>> decay = AttoCubeTRPLSweep("TRPL_..._iter_0.csv")     # one decay
    >>> decay.n_sweeps
    1

    >>> geom  = DeviceGeometry.from_single("WSe2", d_hbn_top=53, d_hbn_bottom=46)
    >>> sweep = AttoCubeTRPLSweep(
    ...     "examples/data/TRPL/", sweep="electric_field", geometry=geom,
    ...     bg_region_ns=(0.0, 1.0),
    ... )
    >>> sweep.time.max(), sweep.n_sweeps
    (12.817, 3)
    """

    _LAYOUT_KIND = "temporal"
    _AXIS_ATTR   = "time"
    _SIGNAL_ATTR = "decays"
    _POINT_NOUN  = "time bins"

    # Picoharp channels, relevant to normalising decay counts.  Units unconfirmed
    # — the same standing question as Scanner X/Y and power_scale.
    _CURATED = {
        **_AttoCubeSweep._CURATED,
        "rep_rate":        ("Picoharp - RepRate",                 1.0, "Hz?"),
        "meas_time":       ("Picoharp - Actual Measurement Time", 1.0, "s?"),
        "picoharp_counts": ("Picoharp - Counts",                  1.0, "counts"),
    }

    def __init__(
        self,
        path,
        *,
        prefix         : str   = None,
        spectra_type   : str   = "TRPL",
        sweep          : str   = None,
        sweep_label    : str   = None,
        sweep_unit     : str   = None,
        geometry       : DeviceGeometry = None,
        bg_region_ns   : tuple = None,
        curated_labels : dict  = None,
        curated_scales : dict  = None,
        time_rtol      : float = 1e-4,
    ):
        self._prefix    = prefix
        self._time_rtol = time_rtol

        payload = self._decode_and_describe(
            path, spectra_type=spectra_type, geometry=geometry,
            curated_labels=curated_labels, curated_scales=curated_scales,
        )

        self.time   = payload["time"]                 # ns, ascending
        self.decays = payload["counts"]               # (n_bins, n_sweeps)
        self.files              = payload.get("files", [Path(self.path)])
        self.metadata_file      = payload.get("metadata_file")
        self.declared_parameters = payload.get("declared_parameters")
        self._validate_axis_and_signals(self.time, {"Exp": self.decays})

        self._bind_sweep_axis(sweep, sweep_label, sweep_unit)

        # Pre-pulse baseline.  processing.subtract_background is generic in x, so
        # a time window needs no separate implementation from a spectral one.
        self.bg_region_ns = bg_region_ns
        if bg_region_ns is not None:
            self.decays_bg = subtract_background(
                self.decays, bg_region=bg_region_ns, x=self.time, axis=0,
            )
        else:
            self.decays_bg = None

    # --- Decoding ----------------------------------------------------------

    @classmethod
    def _decode_csv(cls, path) -> dict:
        """Decode one TRPL export: a single decay, so ``n_sweeps == 1``."""
        blocks = _read_block_layout(path)
        if blocks["kind"] != cls._LAYOUT_KIND:
            raise ValueError(
                f"'{path}' is a {blocks['kind']} export "
                f"({', '.join(blocks['roles'])}), but {cls.__name__} reads "
                f"{cls._LAYOUT_KIND} exports. Use "
                f"{_CLASS_FOR_KIND[blocks['kind']]} instead."
            )
        return cls._decode_temporal_file(path, blocks)

    @staticmethod
    def _decode_temporal_file(path, blocks: dict) -> dict:
        """Shared per-file decode, used for one decay and for each of a sweep."""
        width      = blocks["block_width"]
        n_declared = blocks["n_blocks"]
        raw = pd.read_csv(path, header=0, index_col=0, low_memory=False)
        row_labels = list(raw.index)
        d = raw.to_numpy(dtype=float)[:, :n_declared * width]

        cols = {role: np.arange(offset, d.shape[1], width)
                for offset, role in enumerate(blocks["roles"])}
        keep, n_declared, axis_block = _drop_unwritten_blocks(
            d[:, cols["Wavelength"]], cols, path)

        parameters = {
            str(label): d[i, cols["Par"]][keep]
            for i, label in enumerate(row_labels)
            if not pd.isna(label) and str(label).strip()
        }

        # "Wavelength" holds time here — the exporter's misnomer, not ours.
        axis_raw  = d[:, cols["Wavelength"][axis_block]]
        valid_bin = np.isfinite(axis_raw)

        return {
            "time"       : axis_raw[valid_bin],
            "counts"     : d[valid_bin][:, cols["Exp"][keep]],
            "parameters" : parameters,
            "metadata"   : {},
            "n_declared" : n_declared,
        }

    def _decode_dir(self, path) -> dict:
        """
        Assemble a sweep from a directory of per-point files.

        Files are classified by **header alone**, so the 11 MB metadata companion
        is never parsed just to discover what it is.
        """
        pattern    = f"{self._prefix or ''}*.csv"
        candidates = sorted(Path(path).glob(pattern))
        data_files, companions, skipped = [], [], []
        for f in candidates:
            try:
                layout = _read_block_layout(f)
            except ValueError:
                skipped.append(f.name)
                continue
            if layout["kind"] == self._LAYOUT_KIND:
                data_files.append((f, layout))
            elif layout["kind"] == "spectral":
                # A spectral header in a TRPL directory is the parameter-table
                # companion; a real spectral sweep would not live here.
                companions.append((f, layout))
            else:
                skipped.append(f.name)

        if not data_files:
            raise ValueError(
                f"No TRPL data files found in '{path}' matching '{pattern}'. "
                f"Looked at {len(candidates)} .csv file(s): "
                f"{len(companions)} metadata companion(s), "
                f"{len(skipped)} unrecognised ({', '.join(skipped[:5])}). "
                f"A TRPL export has a [Par, Wavelength, Exp] block layout."
            )

        data_files = self._order_by_iter(data_files, path)
        payload    = self._assemble(data_files, path)

        if companions:
            self._cross_check_companion(payload, companions, path)
        return payload

    @staticmethod
    def _order_by_iter(data_files: list, path) -> list:
        """
        Sort per-point files by the integer in their ``_iter_N`` suffix.

        Lexicographic order would place ``iter_10`` before ``iter_2`` and pair
        every decay with the wrong parameter values.  A gap in the sequence is
        reported rather than closed silently: a missing point means an aborted
        sweep, and shifting the rest would misalign the whole axis.
        """
        indexed = []
        for f, layout in data_files:
            m = _ITER_INDEX.search(f.stem)
            indexed.append((int(m.group(1)) if m else None, f, layout))

        if any(idx is None for idx, _, _ in indexed):
            warnings.warn(
                f"Some files in '{path}' carry no '_iter_N' suffix "
                f"({', '.join(f.name for idx, f, _ in indexed if idx is None)}); "
                f"falling back to filename order, which may not be acquisition "
                f"order.",
                UserWarning, stacklevel=4,
            )
            return [(f, layout) for _, f, layout in indexed]

        indexed.sort(key=lambda item: item[0])
        seen    = [idx for idx, _, _ in indexed]
        missing = sorted(set(range(seen[0], seen[-1] + 1)) - set(seen))
        if missing:
            warnings.warn(
                f"'{path}' is missing iteration(s) {missing} between iter_"
                f"{seen[0]} and iter_{seen[-1]} — an aborted or partly copied "
                f"sweep. The {len(seen)} file(s) present are loaded in order; "
                f"sweep index i is not iteration i.",
                UserWarning, stacklevel=4,
            )
        return [(f, layout) for _, f, layout in indexed]

    def _assemble(self, data_files: list, path) -> dict:
        """
        Stack one decay per file into ``(n_bins, n_sweeps)``.

        The time axis is taken from the first file and the rest are checked
        against it; parameters come from each file's own ``Par_0`` snapshot,
        contemporaneous with the decay it belongs to.
        """
        per_file = [self._decode_temporal_file(f, layout)
                    for f, layout in data_files]

        time = per_file[0]["time"]
        rtol = self._time_rtol
        for (f, _), one in zip(data_files[1:], per_file[1:]):
            if one["time"].size != time.size:
                raise ValueError(
                    f"'{f.name}' has {one['time'].size} time bins but "
                    f"'{data_files[0][0].name}' has {time.size}. Every file in a "
                    f"sweep must share one time axis."
                )
            if not np.allclose(one["time"], time, rtol=rtol, atol=0.0):
                worst = int(np.argmax(np.abs(one["time"] - time)))
                raise ValueError(
                    f"'{f.name}' has a time axis that differs from "
                    f"'{data_files[0][0].name}' by more than rtol={rtol:g} "
                    f"(largest disagreement at bin {worst}: "
                    f"{one['time'][worst]:.9g} vs {time[worst]:.9g} "
                    f"{_TRPL_TIME_UNIT}). Raise time_rtol if this is only "
                    f"acquisition jitter."
                )

        # One column per file: each file holds a single decay, so its (n_bins, 1)
        # counts block becomes column i of the assembled sweep.
        counts = np.hstack([one["counts"] for one in per_file])

        # Each file carries its own full snapshot; keep only rows every file has,
        # so a parameter array can never be short of a sweep point.
        shared = set(per_file[0]["parameters"])
        for one in per_file[1:]:
            shared &= set(one["parameters"])
        parameters = {
            label: np.concatenate([one["parameters"][label] for one in per_file])
            for label in shared
        }

        return {
            "time"       : time,
            "counts"     : counts,
            "parameters" : parameters,
            "metadata"   : {},
            "files"      : [f for f, _ in data_files],
        }

    def _cross_check_companion(self, payload: dict, companions: list, path) -> None:
        """
        Use the metadata companion as evidence, never as the source.

        It supplies the declared sweep length, so a **truncated or aborted sweep
        is visible** — that is the check with real signal, and the only one made
        here.

        Its parameter *values* are exposed as :attr:`declared_parameters` but are
        deliberately **not** compared row by row.  The companion is written seconds
        after the last decay, so it and the per-file snapshots are independent
        read-backs of a moving instrument: on the example sweep the leakage
        currents (~1e-11 A) and ``Fianium_Select_A6`` (160/140/130 against
        190/190/140) differ genuinely, while the swept gates agree to seven
        figures.  Nothing in the file says which channels are stable, so a
        value check cannot separate "wrong companion" from "channel drifted", and
        would fire on every real sweep — which is how warnings get ignored.
        Both tables are returned; the comparison is the caller's to make.
        """
        if len(companions) > 1:
            warnings.warn(
                f"'{path}' holds {len(companions)} metadata companions "
                f"({', '.join(f.name for f, _ in companions)}); using "
                f"'{companions[0][0].name}'.",
                UserWarning, stacklevel=5,
            )
        companion, layout = companions[0]
        payload["metadata_file"]   = companion
        payload["n_declared"]      = layout["n_blocks"]

        raw = pd.read_csv(companion, header=0, index_col=0, low_memory=False)
        width = layout["block_width"]
        d = raw.to_numpy(dtype=float)[:, :layout["n_blocks"] * width]
        par_cols = np.arange(0, d.shape[1], width)
        declared = {
            str(label): d[i, par_cols]
            for i, label in enumerate(raw.index)
            if not pd.isna(label) and str(label).strip()
        }
        payload["declared_parameters"] = declared

        n_found = payload["counts"].shape[1]
        if layout["n_blocks"] != n_found:
            warnings.warn(
                f"'{companion.name}' declares {layout['n_blocks']} sweep point(s) "
                f"but {n_found} data file(s) were found in '{path}'. The sweep may "
                f"have been aborted, or files may be missing. The files present "
                f"are loaded; compare declared_parameters against parameters to "
                f"see which points they are.",
                UserWarning, stacklevel=5,
            )

    # --- Convenience -------------------------------------------------------

    @property
    def n_bins(self) -> int:
        """Number of time bins — :attr:`n_points` under its temporal name."""
        return self.n_points

    @property
    def axis_label(self) -> str:
        """Label for the measured axis, e.g. ``"Time (ns)"``."""
        return f"Time ({_TRPL_TIME_UNIT})"

    @property
    def best_decays(self) -> np.ndarray:
        """
        Pre-pulse-subtracted decays when available, else the raw counts.

        The temporal counterpart of
        :attr:`AttoCubeSpectralSweep.best_energy_spectra`.
        """
        return self.decays_bg if self.decays_bg is not None else self.decays

    def _repr_axis_lines(self, w: int) -> list:
        return [
            f"  {'Time range':<{w}}: {self.time.min():.4g} – "
            f"{self.time.max():.4g} {_TRPL_TIME_UNIT} "
            f"({(self.time[1] - self.time[0]) * 1e3:.3g} ps bins)"
            if self.n_bins > 1 else
            f"  {'Time':<{w}}: {self.time[0]:.4g} {_TRPL_TIME_UNIT}",
        ]

    def _repr_extra_lines(self, w: int) -> list:
        lines = []
        if len(self.files) > 1:
            lines.append(f"  {'Files':<{w}}: {len(self.files)} per-point files")
        if self.metadata_file is not None:
            lines.append(f"  {'Companion':<{w}}: {self.metadata_file.name}")
        lines.append(f"  {'Peak counts':<{w}}: {int(self.decays.max())}")
        if self.bg_region_ns is not None:
            lines.append(
                f"  {'Pre-pulse':<{w}}: {self.bg_region_ns[0]:.4g} – "
                f"{self.bg_region_ns[1]:.4g} {_TRPL_TIME_UNIT} subtracted"
            )
        return lines


# ---------------------------------------------------------------------------
# SingleSpectrum
# ---------------------------------------------------------------------------

class SingleSpectrum:
    """
    Single PL spectrum loaded from a 2-row CSV.

    The file must contain exactly two comma-separated rows:

    * **Row 0** : wavelength axis in nm (ascending).
    * **Row 1** : counts (PL intensity).

    Attribute names mirror :class:`AttoCubeSpectralSweep` so the same plotting
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