# tmdc_optics_tools/plotting.py
"""
Plotting helpers for TMD spectroscopy.

Provides a consistent Matplotlib style and convenience functions for the most
common plot types encountered when spectra are swept over a parameter — gate
voltage, displacement field, excitation power, or piezo position — together with
real-space image and diffusion-cloud plots.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from skimage.exposure import rescale_intensity

from . import fitting, processing
from . import diffusion as _diffusion
# The spectra-source registry names attributes on the loader classes, so it lives
# with them; imported here under its own name because this is where callers of
# ``spectra_source=`` are.
from .loaders import (
    _SPECTRA_SOURCES, _SPECTRA_SOURCE_LABELS, _resolve_spectra,
)

# Optional: cmcrameri diverging colormaps (pip install cmcrameri)
try:
    from cmcrameri import cm as cmc
    _HAS_CRAMERI = True
except ImportError:
    _HAS_CRAMERI = False

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style(context: str = "paper") -> None:
    """
    Apply a clean, publication-ready Matplotlib style.

    Parameters
    ----------
    context : {"paper", "talk", "poster"}
        Scales font sizes appropriately for the output medium.
    """
    base_size = {"paper": 8, "talk": 14, "poster": 18}.get(context, 8)

    plt.rcParams.update({
        "font.family"        : "sans-serif",
        "font.sans-serif"    : ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size"          : base_size,
        "axes.labelsize"     : base_size,
        "axes.titlesize"     : base_size,
        "xtick.labelsize"    : base_size - 1,
        "ytick.labelsize"    : base_size - 1,
        "legend.fontsize"    : base_size - 1,
        "axes.linewidth"     : 0.8,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "xtick.direction"    : "in",
        "ytick.direction"    : "in",
        "xtick.major.width"  : 0.8,
        "ytick.major.width"  : 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth"    : 1.2,
        "figure.dpi"         : 150,
        "savefig.dpi"        : 300,
        "savefig.bbox"       : "tight",
    })


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def get_cmap(name: str = "vik"):
    """
    Return a colormap by name, preferring cmcrameri if available.

    Parameters
    ----------
    name : str
        Any cmcrameri map (e.g. ``"vik"``, ``"roma"``) or standard
        Matplotlib map.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    if _HAS_CRAMERI and hasattr(cmc, name):
        return getattr(cmc, name)
    return plt.get_cmap(name)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_x_axis(scan, x_axis: str) -> tuple:
    """
    Return ``(x_array, xlabel_string)`` for a scan.

    Centralises the repeated ``"energy"`` / ``"wavelength"`` branching so
    every plotting function can call this instead of duplicating the logic.
    """
    if x_axis == "energy":
        return scan.energy, "Energy (eV)"
    elif x_axis == "wavelength":
        return scan.wavelength, "Wavelength (nm)"
    else:
        raise ValueError(
            f"x_axis must be 'energy' or 'wavelength', got '{x_axis}'."
        )


def _signal_name_unit(obj, source: str = None) -> tuple:
    """
    Return ``(quantity_name, unit)`` for the signal an object measures.

    *source* is a :data:`_SPECTRA_SOURCES` key.  A contrast source is a
    different physical quantity from the raw signal, and a dimensionless one,
    so it takes the contrast label and an empty unit.

    Objects that declare no measurement type fall back to a neutral
    "Intensity" / "counts" — a :class:`~tmdc_optics_tools.loaders.SingleSpectrum`
    is a 2-row CSV as likely to be a bare-substrate reflectance reference as PL.
    """
    if source is not None and source.startswith("contrast"):
        return getattr(obj, "contrast_label", r"$\Delta R/R_0$"), ""
    return (getattr(obj, "signal_name", "Intensity"),
            getattr(obj, "signal_unit", "counts"))


def _signal_label(obj, normalized: bool = False, source: str = None) -> str:
    """
    Compose the y-axis or colour-bar label for a measured signal.

    Only the calling plot function knows whether it rescaled the values, and a
    rescaled array has no unit left — hence *normalized*, which substitutes
    "norm." for whatever the native unit was.  A dimensionless quantity has no
    unit to substitute and is left alone: a ratio such as ΔR/R₀ already reads as
    normalised, so marking it again says the same thing twice.
    """
    name, unit = _signal_name_unit(obj, source)
    if normalized and unit:
        unit = "norm."
    return f"{name} ({unit})" if unit else name


# ---------------------------------------------------------------------------
# 2-D map plots
# ---------------------------------------------------------------------------

def plot_spectral_map(
    scan,
    ax             = None,
    figsize        : tuple = (6, 4),
    dpi            : int   = None,
    x_axis         : str   = "energy",
    cmap           : str   = "vik",
    median_kernel  : int   = 3,
    clim           : tuple = None,
    colorbar       : bool  = True,
    colorbar_label : str   = None,
    rescale_img    : bool  = False,
) -> tuple:
    """
    Plot every spectrum of a sweep as a 2-D map: spectral axis against sweep axis.

    Works for any sweep an
    :class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep` can hold — gate
    voltage, displacement field, excitation power, piezo position, or the bare
    sweep index.  The y-axis and its label come from
    :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.sweep_axis` and
    :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.sweep_axis_label`,
    so whatever was declared as ``sweep=`` at load time is what is plotted.

    Background subtraction and Jacobian correction are configured at
    load time on the scan object (via ``bg_region_nm``, ``bg_region_eV``,
    and ``apply_jacobian``).  This function always uses
    :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.best_energy_spectra`,
    which automatically returns the background-corrected array when one
    is available, and falls back to the uncorrected array otherwise.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    x_axis : {"energy", "wavelength"}
    cmap : str
        Colormap name passed to :func:`get_cmap`.
    median_kernel : int
        2-D median filter size. Set to 1 to disable.
    clim : tuple of (vmin, vmax), optional
        Colour axis limits. Auto-scaled if ``None``.
    colorbar : bool
    colorbar_label : str, optional
        Colour-bar label.  Derived from the scan's measurement type when
        ``None`` — a reflectance sweep is labelled as reflectance, and a
        dimensionless ratio gets no unit.  A string is used **verbatim**, so
        include the unit.
    rescale_img : bool
        Default is `False`. If `True`, rescales intensity to [0, 1] before plotting.

    Returns
    -------
    fig, ax, mesh
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    x, xlabel = _resolve_x_axis(scan, x_axis)
    y, ylabel  = scan.sweep_axis, scan.sweep_axis_label

    x_m = np.tile(x[:, np.newaxis], (1, scan.n_sweeps))
    y_m = np.tile(y[np.newaxis, :], (scan.n_pixels, 1))

    # Use best_energy_spectra (BG-corrected if available) for energy axis;
    # raw spectra for wavelength axis (BG correction is a loader concern).
    data = scan.best_energy_spectra.copy() if x_axis == "energy" else scan.spectra.copy()

    if median_kernel > 1:
        data = processing.smooth_median(data, kernel=median_kernel)

    if rescale_img:
        data = rescale_intensity(data, in_range="image", out_range=(0, 1))

    vmin, vmax = clim if clim is not None else (None, None)
    mesh = ax.pcolormesh(
        x_m, y_m, data,
        cmap=get_cmap(cmap), shading="auto",
        vmin=vmin, vmax=vmax,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label(colorbar_label if colorbar_label is not None
                     else _signal_label(scan, normalized=rescale_img))

    return fig, ax, mesh


def plot_pl_map_Vab_scan(*args, **kwargs) -> tuple:
    """
    Deprecated alias for :func:`plot_spectral_map`.

    Accepts and returns exactly what :func:`plot_spectral_map` does, and emits a
    ``FutureWarning``.  Use :func:`plot_spectral_map`: the map is not specific to
    a two-gate voltage sweep, nor to PL.

    .. deprecated::
       Call :func:`plot_spectral_map` instead.
    """
    # No explicit signature: the arguments are unchanged by the rename, so
    # forwarding cannot drift out of step with the function it delegates to.
    warnings.warn(
        "plot_pl_map_Vab_scan is deprecated; use plot_spectral_map, which takes "
        "the same arguments. The map is not specific to a V_a/V_b gate sweep — "
        "the y-axis is whatever was declared as sweep= at load time.",
        FutureWarning, stacklevel=2,
    )
    return plot_spectral_map(*args, **kwargs)


# ---------------------------------------------------------------------------
# Spectrum plots
# ---------------------------------------------------------------------------

def plot_spectrum(
    scan,
    sweep_index  : int,
    ax           = None,
    figsize      : tuple = (5, 3),
    dpi          : int   = None,
    x_axis       : str  = "energy",
    normalize    : bool = False,
    smooth_window       = None,
    smooth_poly  : int  = 3,
    label        : str  = None,
    ylabel       : str  = None,
    **line_kwargs,
) -> tuple:
    """
    Plot one spectrum from a sweep.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    sweep_index : int
        Index of the sweep point to plot.
    ax : matplotlib.axes.Axes, optional
    x_axis : {"energy", "wavelength"}
    normalize : bool
        Normalise spectrum to its own [0, 1] range.
    smooth_window, smooth_poly : int, optional
        Forwarded to :func:`~tmdc_optics_tools.processing.smooth_savgol`, run
        before *normalize* so both reflect the same smoothed data.
        ``smooth_window=None`` (default) skips smoothing.
    label : str, optional
        Legend label. Defaults to the gate voltage / field value.
    ylabel : str, optional
        Y-axis label.  Derived from the scan's measurement type when ``None``,
        so a reflectance sweep is not labelled as PL.  A string is used
        **verbatim**, so include the unit.
    **line_kwargs
        Passed directly to ``ax.plot``.

    Returns
    -------
    fig, ax, line
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    x, xlabel = _resolve_x_axis(scan, x_axis)
    if x_axis == "energy":
        y = scan.best_energy_spectra[:, sweep_index].astype(float)
    else:
        y = scan.spectra[:, sweep_index].astype(float)
    y = processing.maybe_smooth(y, smooth_window, smooth_poly)
    if normalize:
        y = processing.normalise_minmax(y)

    if label is None:
        # Fall back to whatever the scan says it swept rather than to a gate
        # voltage: a gate role needs a declared wiring, and the sweep axis is
        # already the scan's own answer to "what varied", labelled and in its own
        # units.  A field needs a geometry and two declared gates, so check the
        # device first — reading scan.ef without them raises.
        if scan.is_dual_gated and scan.ef is not None:
            label = f"$E_F$ = {scan.ef[sweep_index]:.1f} mV/nm"
        else:
            label = (f"{scan.sweep_axis_label} = "
                     f"{scan.sweep_axis[sweep_index]:.4g}")

    line, = ax.plot(x, y, label=label, **line_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None
                  else _signal_label(scan, normalized=normalize))

    return fig, ax, line


def plot_single_spectrum(
    spectrum,
    ax          = None,
    figsize     : tuple = (5, 3),
    dpi         : int   = None,
    x_axis      : str   = "wavelength",
    normalize   : bool  = False,
    label       : str   = None,
    ylabel      : str   = None,
    **line_kwargs,
) -> tuple:
    """
    Plot a spectrum held in a
    :class:`~tmdc_optics_tools.loaders.SingleSpectrum`.

    Parameters
    ----------
    spectrum : SingleSpectrum
        Any object exposing ``wavelength``, ``energy``, ``best_spectra`` and
        ``best_energy_spectra`` attributes. Background-corrected arrays are
        used automatically when a background region was set at load time.
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    x_axis : {"wavelength", "energy"}
        Axis to plot against. Default ``"wavelength"`` (the native axis).
    normalize : bool
        Normalise the spectrum to its own [0, 1] range.
    label : str, optional
        Legend label. A legend is shown only when a label is given.
    ylabel : str, optional
        Y-axis label.  A ``SingleSpectrum`` declares no measurement type, so
        this falls back to a neutral "Intensity (counts)" when ``None`` — a
        2-row CSV is as likely to be a bare-substrate reflectance reference as
        PL.  A string is used **verbatim**, so include the unit.
    **line_kwargs
        Passed directly to ``ax.plot``.

    Returns
    -------
    fig, ax, line
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    x, xlabel = _resolve_x_axis(spectrum, x_axis)
    y = (spectrum.best_energy_spectra if x_axis == "energy"
         else spectrum.best_spectra).astype(float)
    if normalize:
        y = processing.normalise_minmax(y)

    line, = ax.plot(x, y, label=label, **line_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None
                  else _signal_label(spectrum, normalized=normalize))
    if label:
        ax.legend(frameon=False)

    return fig, ax, line


def plot_spectra_overlay(
    entries           : dict,
    x                 = None,
    xlabel            : str  = None,
    ylabel            : str  = None,
    ylabel_normalized : str  = None,
    smooth_window           = None,
    smooth_poly       : int  = 3,
    figsize           : tuple = (10.5, 4.2),
) -> tuple:
    """
    Raw and min-max-normalized overlay of several spectra and/or fits.

    Each entry in *entries* is either a plain array of y-values (sharing
    *x*) or a :class:`~tmdc_optics_tools.fitting.FitResult` — decided per
    entry, not by a caller-chosen mode, so a set of points where only some
    were fit still draws in one call and one figure.

    A plain-array entry draws one line, normalized independently via
    :func:`~tmdc_optics_tools.processing.normalise_minmax` (there is no fit
    for it to visually agree or disagree with). A ``FitResult`` entry draws
    its data (recovered as ``result.residuals + result.y_fit`` — exact, that
    is the definition of the residual — on its own ``result.x_fit``; the
    shared *x* is unused for it) as points plus its fit-sum curve as a line,
    both normalized *together* using the data's own min/max rather than
    independently, so the fit's visual agreement with its data survives
    normalizing instead of being hidden by two curves put on disagreeing
    scales.

    Parameters
    ----------
    entries : dict of {label: array-like or FitResult}
        Resolving a coordinate to an entry is the caller's job, since it
        differs by loader — e.g. ``scan.get_spectrum_at(fast=, slow=)`` for a
        nested AttoCube sweep, or ``raman_map.spectrum_at(*raman_map.nearest_index(x, y))``
        for a :class:`~tmdc_optics_tools.loaders.RamanMap`. Pass a
        :func:`~tmdc_optics_tools.fitting.fit_multi_voigt` result (or a
        wrapper's, e.g. :func:`~tmdc_optics_tools.fitting.fit_raman_modes`)
        for any point that was fit.
    x : array-like, optional
        Shared x-axis for any plain-array entries. Unused by, and not
        required if *entries* holds only, ``FitResult`` entries.
    xlabel, ylabel, ylabel_normalized : str, optional
        Axis labels, left blank by default: this function has no way to know
        what domain the spectra are in, and *ylabel_normalized* separately
        from *ylabel* because normalizing substitutes the unit rather than
        appending to it (e.g. "PL intensity (counts)" becomes "PL intensity
        (norm.)", not "PL intensity (counts) (norm.)") and only a caller that
        knows the signal's name can build that string.
    smooth_window, smooth_poly : int, optional
        Forwarded to :func:`~tmdc_optics_tools.processing.maybe_smooth`, run
        once per plain-array entry before either panel is drawn — not
        applied to a ``FitResult`` entry's data, which was already the array
        the fit itself ran on. ``smooth_window=None`` (default) skips
        smoothing.
    figsize : tuple

    Returns
    -------
    fig, (ax_raw, ax_norm)
    """
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=figsize)
    has_plain_entry = False

    for label, entry in entries.items():
        if isinstance(entry, fitting.FitResult):
            x_i, y_fit = entry.x_fit, entry.y_fit
            y_i = entry.residuals + y_fit

            line, = ax_raw.plot(x_i, y_i, ".", ms=3, alpha=0.35)
            ax_raw.plot(x_i, y_fit, "-", color=line.get_color(), label=label)

            lo, hi = y_i.min(), y_i.max()
            span = hi - lo if hi > lo else 1.0
            ax_norm.plot(x_i, (y_i - lo) / span, ".", ms=3, alpha=0.35, color=line.get_color())
            ax_norm.plot(x_i, (y_fit - lo) / span, "-", color=line.get_color())
        else:
            has_plain_entry = True
            y_i = processing.maybe_smooth(np.asarray(entry, dtype=float), smooth_window, smooth_poly)
            ax_raw.plot(x, y_i, label=label)
            ax_norm.plot(x, processing.normalise_minmax(y_i), label=label)

    ax_raw.set_title("Raw")
    ax_norm.set_title("Normalized to [0, 1]")
    for ax, label in ((ax_raw, ylabel), (ax_norm, ylabel_normalized)):
        if xlabel:
            ax.set_xlabel(xlabel)
        if label:
            ax.set_ylabel(label)
    ax_raw.legend(fontsize=8)
    if has_plain_entry:
        # A FitResult entry labels only its fit-sum line, on ax_raw -- its
        # normalized-panel curves are unlabeled (same color already ties them
        # to ax_raw's legend), so ax_norm gets a legend only when a
        # plain-array entry actually labeled something there.
        ax_norm.legend(fontsize=8)

    fig.tight_layout()
    return fig, (ax_raw, ax_norm)


# ---------------------------------------------------------------------------
# Breakdown / leakage current monitor
# ---------------------------------------------------------------------------

def plot_current(
    scan,
    ax          = None,
    figsize     : tuple = (6, 3.5),
    dpi         : int   = None,
    ef_axis     : bool = True,
    color_ich1  : str  = "C0",
    color_ich2  : str  = "C1",
    color_power : str  = "C2",
) -> tuple:
    """
    Plot leakage currents and excitation power vs. electric field (or gate
    voltage) to check for dielectric breakdown.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    ax : matplotlib.axes.Axes, optional
        Must be a standard (non-twin) axes.
    ef_axis : bool
        Use displacement field on the x-axis when the scan can supply one — it
        needs both a :class:`DeviceGeometry` and a declared channel-to-gate
        mapping.  Otherwise the scan's declared sweep axis is used.
    color_ich1, color_ich2, color_power : str
        Matplotlib colours for the respective traces.

    Returns
    -------
    fig, ax_left, ax_right
    """
    if ax is None:
        fig, ax_left = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig     = ax.get_figure()
        ax_left = ax

    # A field needs a geometry *and* two declared gate electrodes, so check the
    # device first — reading scan.ef without them raises.  Otherwise use the scan's
    # own sweep axis: labelling this "$V_top$" would assert a wiring the scan may
    # not have been told, or a gate the device may not have.
    if ef_axis and scan.is_dual_gated and scan.ef is not None:
        x, xlabel = scan.ef, r"$E_F$ (mV/nm)"
    else:
        x, xlabel = scan.sweep_axis, scan.sweep_axis_label

    l1, = ax_left.plot(x, scan.Ich1, color=color_ich1, label=r"$I_\mathrm{ch1}$")
    l2, = ax_left.plot(x, scan.Ich2, color=color_ich2, label=r"$I_\mathrm{ch2}$")
    ax_left.axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel("Current (nA)")

    ax_right = ax_left.twinx()
    ax_right.spines["right"].set_visible(True)
    l3, = ax_right.plot(x, scan.power, color=color_power, linestyle="--", label="Power")
    ax_right.set_ylabel("Power (µW)")

    ax_left.legend(handles=[l1, l2, l3], loc="best", frameon=False)
    fig.tight_layout()
    return fig, ax_left, ax_right


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(
    fig,
    filename  : str  = None,
    directory : str  = ".",
    fmt       : str  = "png",
    dpi       : int  = 300,
    prompt    : bool = True,
) -> str:
    """
    Save a Matplotlib figure, optionally prompting for a filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str, optional
        Output filename without extension.
    directory : str
        Output directory. Created if it does not exist.
    fmt : str or list of str
        File format(s), e.g. ``"png"`` or ``["png", "pdf"]``.
    dpi : int
    prompt : bool
        Ask for a filename interactively if none is given.

    Returns
    -------
    str or list of str
        Full path(s) of the saved file(s).
    """
    import os

    if filename is None:
        if prompt:
            filename = input("Enter filename (without extension): ").strip()
            if not filename:
                raise ValueError("No filename provided.")
        else:
            raise ValueError("filename is None and prompt=False.")

    formats = [fmt] if isinstance(fmt, str) else list(fmt) 
    os.makedirs(directory, exist_ok=True)

    saved_paths = []
    for f in formats:
        path = os.path.join(directory, f"{filename}.{f}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")
        saved_paths.append(path)

    return saved_paths[0] if len(saved_paths) == 1 else saved_paths


# ---------------------------------------------------------------------------
# Real-space PL map
# ---------------------------------------------------------------------------

def plot_real_space_PL_map(
    scan,
    ax     = None,
    figsize : tuple = (6, 3.5),
    dpi     : int   = None,
    idx    : int = 0,
    xlabel : str = "x-axis (pixels)",
    ylabel : str = "y-axis (pixels)",
    cmap   : str = "cork"
) -> tuple:
    """
    Plot a single real-space PL map from an
    :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace`.

    Parameters
    ----------
    scan : AttoCubePLScanRealSpace
    ax : matplotlib.axes.Axes, optional
    idx : int
        Frame index to display.
    xlabel, ylabel : str
        Axis labels.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    ax.imshow(scan.load_frame(idx), cmap=get_cmap(cmap))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_image(
    image,
    ax             = None,
    figsize        : tuple = (6, 5),
    dpi            : int   = None,
    cmap           : str   = "vik",
    colorbar       : bool  = True,
    colorbar_label : str   = None,
    rescale_img    : bool  = False,
    clim           : tuple = None,
    xlabel         : str   = "x (px)",
    ylabel         : str   = "y (px)",
    show_axes      : bool  = True,
    extent         : tuple = None,
    origin         : str   = "upper",
) -> tuple:
    """
    Plot a single 2-D image with a colormap and an optional colorbar.

    Parameters
    ----------
    image : np.ndarray or object with ``.img``
        A 2-D array, or any object exposing a 2-D ``img`` attribute
        (e.g. :class:`~tmdc_optics_tools.loaders.SingleImage`,
        :class:`~tmdc_optics_tools.loaders.AttoCubeSampleImage`).
        ``NaN`` entries are masked (drawn as nothing) rather than colored at
        the scale's low end, so "not computed here" stays visually distinct
        from "computed and found to be small" — e.g. a
        :class:`~tmdc_optics_tools.loaders.RamanMap` mode fit only over part
        of the grid.
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    cmap : str
        Colormap name passed to :func:`get_cmap`.
    colorbar : bool
        Show a colorbar alongside the image.
    colorbar_label : str, optional
        Colour-bar label.  Defaults to "Intensity (counts)", or
        "Intensity (norm.)" when *rescale_img*; a plain 2-D array carries no
        measurement type to derive anything better from.  A string is used
        **verbatim**, so include the unit.
    rescale_img : bool
        Rescale intensity to [0, 1] before plotting.
    clim : tuple of (vmin, vmax), optional
        Colour axis limits. Auto-scaled if ``None``.
    xlabel, ylabel : str
        Axis labels (ignored when *show_axes* is ``False``).
    show_axes : bool
        Show axis ticks/labels. Set ``False`` to hide them entirely.
    extent : tuple of (left, right, bottom, top), optional
        Physical-coordinate extent for the array's edges, e.g.
        ``(x.min(), x.max(), y.min(), y.max())`` for a real-space map in µm
        — the array itself carries no unit, only pixel indices, so this is
        the caller's to supply. ``None`` (default) leaves the axes in
        pixel-index units, unchanged from before this parameter existed.
    origin : {"upper", "lower"}
        Row 0's position: "upper" (default, unchanged prior behaviour) for
        a camera/CCD frame read top row first; "lower" for a physical map
        whose Y should increase upward, matching a normal Cartesian axis.

    Returns
    -------
    fig, ax, im
    """
    img = image.img if hasattr(image, "img") else np.asarray(image, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    if rescale_img:
        img = rescale_intensity(img, in_range="image", out_range=(0, 1))

    img = np.ma.masked_invalid(img)
    vmin, vmax = clim if clim is not None else (None, None)
    im = ax.imshow(img, cmap=get_cmap(cmap), vmin=vmin, vmax=vmax,
                    extent=extent, origin=origin)

    if show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.axis("off")

    if colorbar:
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(colorbar_label if colorbar_label is not None
                     else ("Intensity (norm.)" if rescale_img
                           else "Intensity (counts)"))

    return fig, ax, im


_VOIGT_PARAM_KEYS = ["center", "amp", "fwhm_g", "fwhm_l"]


def plot_multi_voigt_overlay(fit_results: dict, fit_windows: dict,
                              mode_names: list = None, ncols: int = 3,
                              xlabel: str = None, ylabel: str = None, x_unit: str = ""):
    """
    Data-vs-fit overlay for several :func:`fitting.fit_multi_voigt` results.

    Draws the fitted sum curve plus each individual Voigt component,
    reconstructed via :func:`fitting.voigt_approx` from that component's
    own ``amp``/``center``/``fwhm_g``/``fwhm_l``, so what is drawn is
    exactly what the fit says rather than a re-derived approximation of
    it. One panel per entry in *fit_results*, arranged in a grid — not a
    single row — once there are more than *ncols* of them.

    Parameters
    ----------
    fit_results : dict of {name: FitResult}
        Results from :func:`fitting.fit_multi_voigt` or a wrapper built on
        it (e.g. :func:`fitting.fit_raman_modes`).
    fit_windows : dict of {name: (x, y)}
        The exact windowed data each fit was run on, for the "data"
        scatter — not re-derived from ``result.residuals + result.y_fit``,
        since a caller with the original array on hand should not be made
        to reconstruct it.
    mode_names : list of str, optional
        Label for each peak index in the legend, e.g. ``["E2g/A1g",
        "2LA(M)", "B2g"]``. A result with more peaks than *mode_names* has
        falls back to ``"peak i"`` for the extra ones rather than raising.
    ncols : int
        Panels per row.
    xlabel, ylabel : str, optional
        Axis labels, set on every panel. Left blank by default: this
        function has no way to know what domain the fit's x-axis is in
        (Raman shift, energy, wavelength, ...) or what the y-axis units
        are.
    x_unit : str, optional
        Appended after each peak's fitted center in the legend, e.g.
        ``" cm$^{-1}$"`` — again left blank rather than assumed.

    Returns
    -------
    fig, axes
        The full grid, including any unused trailing slots (hidden, not
        removed, so the grid shape stays rectangular) — index by
        ``axes[row, col]`` to restyle a panel.
    """
    mode_names = mode_names or []
    names = list(fit_results.keys())
    ncols = min(ncols, len(names))
    nrows = -(-len(names) // ncols)  # ceil division -- a partial last row is fine
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(5.0 * ncols, 4.0 * nrows))
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, names):
        result = fit_results[name]
        x, y = fit_windows[name]
        offset = result.params.get("offset", 0.0)

        ax.plot(x, y, ".", ms=4, alpha=0.4, color="0.4", label="data")
        ax.plot(x, result.y_fit, "-", color="C0", label="fit (sum)")

        n_peaks = sum(1 for k in result.params if k.startswith("center_"))
        for i in range(n_peaks):
            amp, cen, fg, fl = (result.params[f"amp_{i}"], result.params[f"center_{i}"],
                                 result.params[f"fwhm_g_{i}"], result.params[f"fwhm_l_{i}"])
            peak_curve = fitting.voigt_approx(x, amp, cen, fg, fl) + offset
            mode = mode_names[i] if i < len(mode_names) else f"peak {i}"
            ax.plot(x, peak_curve, "--", lw=1.3, label=f"{mode} ({cen:.1f}{x_unit})")

        ax.set_title(name, fontsize=11)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(names):]:
        ax.set_visible(False)  # unused grid slots when len(names) isn't a multiple of ncols

    fig.tight_layout()
    return fig, axes


def plot_fit_param_comparison(fit_results: dict, mode_names: list, param_labels: dict = None):
    """
    Compare :func:`fitting.fit_multi_voigt` parameters across several fits.

    One grid figure — rows are peak indices (from *mode_names*), columns
    are the four Voigt parameter types — rather than one figure per
    (peak, parameter) combination, so the whole comparison fits on screen
    at once. Not every result needs every peak: a sample missing a given
    mode's params (e.g. no B₂g on a monolayer fit) simply leaves a gap at
    that sample's x-position in that panel, rather than raising or
    silently shifting the other samples over.

    Parameters
    ----------
    fit_results : dict of {name: FitResult}
        Results from :func:`fitting.fit_multi_voigt` or a wrapper built on
        it.
    mode_names : list of str
        One label per peak index, e.g. ``["E2g/A1g", "2LA(M)", "B2g"]`` —
        also sets how many peak rows are considered; a row is only drawn
        if at least one result actually has that peak.
    param_labels : dict of {"center"|"amp"|"fwhm_g"|"fwhm_l": str}, optional
        Axis label for each parameter type — pass this to add units, e.g.
        ``{"center": "Center (cm$^{-1}$)"}`` for a Raman shift, ``"Center
        (eV)"`` for a PL peak. Falls back to the bare, unitless key name
        when not given: this function has no way to know what domain the
        fit's x-axis is in.

    Returns
    -------
    fig, axes
        The grid (rows = active peaks, columns = the 4 Voigt parameters)
        — index by ``axes[row, col]`` to restyle a panel.
    """
    param_labels = param_labels or {}
    sample_names = list(fit_results.keys())
    sample_colors = {name: f"C{i}" for i, name in enumerate(sample_names)}
    x = np.arange(len(sample_names))

    active_peaks = [
        peak_i for peak_i in range(len(mode_names))
        if any(f"center_{peak_i}" in fit_results[name].params for name in sample_names)
    ]
    n_rows, n_cols = len(active_peaks), len(_VOIGT_PARAM_KEYS)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                              figsize=(3.2 * n_cols, 2.6 * n_rows))

    for row, peak_i in enumerate(active_peaks):
        for col, param in enumerate(_VOIGT_PARAM_KEYS):
            ax = axes[row, col]
            key = f"{param}_{peak_i}"
            for j, name in enumerate(sample_names):
                result = fit_results[name]
                if key not in result.params:
                    continue
                ax.errorbar(j, result.params[key], yerr=result.errors[key],
                            fmt="o", ms=7, capsize=4, color=sample_colors[name])
            ax.set_xticks(x)
            ax.set_xticklabels(sample_names, rotation=25, ha="right", fontsize=8)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(param_labels.get(param, param), fontsize=10)
            if col == 0:
                ax.set_ylabel(mode_names[peak_i], fontsize=10)

    fig.tight_layout()
    return fig, axes


def _format_frame_title(
    var_array : np.typing.ArrayLike,
    var_label : str,
    units     : str,
    frame     : int,
    fmt       : str,
) -> str:
    """
    Format the per-frame subtitle string for an animated PL map.

    The output format depends on whether *var_label* is supplied:

    * With label : ``"<var_label>: <value> <units>"``
    * Without    : ``"<value> <units>"``

    Trailing whitespace is stripped so an empty *units* string leaves no
    dangling space.

    Parameters
    ----------
    var_array : array-like
        Values of the swept parameter, one per frame.
    var_label : str
        Human-readable label shown before the value. Pass ``""`` to omit.
    units : str
        Unit string appended after the value (e.g. ``"µW"``).
        Accepts LaTeX, e.g. ``r"$\\mu$W"`` or ``r"mV nm$^{-1}$"``.
    frame : int
        Current frame index.
    fmt : str
        Python format spec for the numeric value (e.g. ``".3g"``).

    Returns
    -------
    str
    """
    value      = var_array[frame]
    value_str  = f"{value:{fmt}} {units}".strip()
    return f"{var_label}: {value_str}" if var_label else value_str


def animate_real_space_PL_map(
    scan,
    ax               = None,
    var_array        = None,
    var_label        : str  = "",
    units            : str  = "mV/nm",
    fmt              : str  = ".3g",
    title            : str  = None,
    xlabel           : str  = "x-axis (um)",
    ylabel           : str  = "y-axis (um)",
    laser_annotation : bool = True,
    cmap             : str  = "cork",
) -> tuple:
    """
    Animate a sequence of real-space PL maps from an
    :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace`.

    Parameters
    ----------
    scan : AttoCubePLScanRealSpace
    ax : matplotlib.axes.Axes, optional
    var_array : array-like, optional
        Values of the swept parameter, one per frame (e.g. electric field,
        optical power, gate voltage). When ``None``, no per-frame subtitle
        is shown.
    var_label : str
        Label prepended to the per-frame value, e.g. ``"Power"``.
        Produces ``"Power: 1.23 µW"``.  Pass ``""`` (default) to show
        only the value and units: ``"1.23 µW"``.
    units : str
        Unit string appended to the per-frame value. Default ``"mV/nm"``.
        Accepts LaTeX, e.g. ``r"$\\mu$W"`` or ``r"mV nm$^{-1}$"``.
    fmt : str
        Python format spec for the per-frame numeric value.
        Default ``".3g"`` (compact, handles both small and large numbers).
        Examples: ``".1f"`` for one decimal place, ``".2e"`` for
        explicit scientific notation.
    title : str, optional
        Static heading shown above the axes for the full animation
        (e.g. ``"Device A — power sweep"``).  Uses ``fig.suptitle`` so
        it sits above the per-frame subtitle without collision.
        Omitted when ``None``.
    xlabel, ylabel : str
        Axis labels.
    laser_annotation : bool
        Overlay the laser spot circle if ``scan.laser_ref`` is set.

    Returns
    -------
    fig, anim

    Examples
    --------
    >>> # Electric field sweep
    >>> fig, anim = animate_real_space_PL_map(
    ...     scan,
    ...     var_array = ef_array,
    ...     var_label = "E-field",
    ...     units     = r"mV nm$^{-1}$",
    ...     title     = "Device A — gate sweep",
    ... )

    >>> # Optical power sweep, value only (no label)
    >>> fig, anim = animate_real_space_PL_map(
    ...     scan,
    ...     var_array = power_uW,
    ...     units     = r"$\\mu$W",
    ...     fmt       = ".2f",
    ...     title     = "Power-dependent PL",
    ... )
    """
    fig, ax = plot_real_space_PL_map(scan, ax, idx=0, xlabel=xlabel, ylabel=ylabel, cmap=(cmap))
    im = ax.images[0] if ax.images else ax.imshow(scan.load_frame(0), cmap=get_cmap(cmap))

    # Static overall title (suptitle so it doesn't clash with the per-frame subtitle)
    if title is not None:
        fig.suptitle(title)

    # Per-frame subtitle (ax.set_title, updates every frame)
    frame_title = (
        ax.set_title(_format_frame_title(var_array, var_label, units, 0, fmt))
        if var_array is not None else None
    )

    if laser_annotation and scan.laser_ref is not None:
        _draw_laser_circle(ax, scan.laser_ref, ls="--")

    def update(frame):
        im.set_data(scan.load_frame(frame))
        updated = [im]
        if var_array is not None and frame_title is not None:
            frame_title.set_text(
                _format_frame_title(var_array, var_label, units, frame, fmt)
            )
            updated.append(frame_title)
        return tuple(updated)

    anim = animation.FuncAnimation(
        fig, update, frames=scan.n_frames, blit=True, interval=200,
    )
    return fig, anim


# ---------------------------------------------------------------------------
# Stark shift / dipole length
# ---------------------------------------------------------------------------

def plot_stark_shift(
    dipole_result,
    ax             = None,
    figsize        : tuple = (5, 3.5),
    dpi            : int   = None,
    show_fit       : bool  = True,
    show_errorbars : bool  = True,
    color_data     : str   = "C0",
    color_fit      : str   = "C1",
    ef_range       : tuple = None,
) -> tuple:
    """
    Plot the DC Stark shift (peak energy vs. electric field) and the
    linear fit used to extract the dipole length.

    Parameters
    ----------
    dipole_result : DipoleResult
        Output of :func:`~tmdc_optics_tools.fitting.extract_dipole_length`.
    ax : matplotlib.axes.Axes, optional
    show_fit : bool
        Overlay the best-fit line.
    show_errorbars : bool
        Show per-point center uncertainties as errorbars.
    color_data, color_fit : str
    ef_range : tuple of (F_min, F_max), optional
        Restrict the plotted field range.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    dr   = dipole_result
    mask = dr.converged_mask.copy()
    if ef_range is not None:
        mask &= (dr.ef >= ef_range[0]) & (dr.ef <= ef_range[1])

    ef_plot  = dr.ef[mask]
    E_plot   = dr.peak_energies[mask]
    err_plot = dr.peak_errors[mask]

    if show_errorbars:
        ax.errorbar(
            ef_plot, E_plot, yerr=err_plot,
            fmt="o", color=color_data, markersize=3,
            linewidth=0.8, capsize=2, label="Peak energy",
        )
    else:
        ax.plot(ef_plot, E_plot, "o", color=color_data,
                markersize=3, label="Peak energy")

    if show_fit:
        ef_line = np.linspace(ef_plot.min(), ef_plot.max(), 300)
        label   = (
            f"Linear fit\n"
            f"$d$ = {dr.dipole_length:.3f} ± {dr.dipole_length_err:.3f} nm\n"
            f"$R^2$ = {dr.r_squared:.4f}"
        )
        ax.plot(ef_line, dr.slope * ef_line + dr.intercept,
                "-", color=color_fit, linewidth=1.4, label=label)

    ax.set_xlabel(r"$E_F$ (mV/nm)")
    ax.set_ylabel("Peak energy (eV)")
    ax.legend(frameon=False, fontsize=7)
    return fig, ax


def plot_rise_time_vs_distance(
    distances,
    rise_times,
    rise_time_errs = None,
    labels         = None,
    ax             = None,
    figsize        : tuple = (5, 3.5),
    dpi            : int   = None,
    **errorbar_kwargs,
) -> tuple:
    """
    Plot fitted rise time vs. distance from the excitation spot.

    One point per measurement location, e.g. the ``tau_rise`` of a
    :class:`~tmdc_optics_tools.fitting.SparseLifetimeResult` against that
    spot's distance from the excitation laser.

    Parameters
    ----------
    distances : array-like
        Distance of each spot from the excitation laser (µm).
    rise_times : array-like
        Fitted rise time for each spot (ns).
    rise_time_errs : array-like, optional
        Per-spot rise time uncertainty, drawn as error bars.
    labels : sequence of str, optional
        Per-point legend label (e.g. spot name).
    ax : matplotlib.axes.Axes, optional
    **errorbar_kwargs
        Forwarded to every ``ax.errorbar`` call, e.g. ``fmt``, ``capsize``,
        ``color``.

    Returns
    -------
    fig, ax, artists
        *artists* is the list of ``ErrorbarContainer`` objects, one per
        point, in *distances* order — restyle or re-label individual points
        through these rather than re-plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    distances  = np.asarray(distances, dtype=float)
    rise_times = np.asarray(rise_times, dtype=float)
    errs   = np.zeros_like(distances) if rise_time_errs is None else np.asarray(rise_time_errs, dtype=float)
    labels = [None] * len(distances) if labels is None else list(labels)

    errorbar_kwargs.setdefault("fmt", "o")
    errorbar_kwargs.setdefault("capsize", 3)

    artists = [
        ax.errorbar(d, t, yerr=e, label=lbl, **errorbar_kwargs)
        for d, t, e, lbl in zip(distances, rise_times, errs, labels)
    ]

    ax.set_xlabel("Distance from excitation spot (µm)")
    ax.set_ylabel("Rise time (ns)")
    if any(labels):
        ax.legend(frameon=False, fontsize=8)
    return fig, ax, artists


# ---------------------------------------------------------------------------
# Composable multi-panel animation
# ---------------------------------------------------------------------------
#
# The building blocks below decouple *what* each panel draws from *how* the
# figure is assembled and animated.  An :class:`AnimationPanel` knows how to
# draw frame 0 and how to mutate its artists for a given frame; the engine
# :func:`animate_panels` lays out ``1 x N`` subplots and drives them in lock
# step.  "Any combination of panels" is then simply "pass whichever panels you
# want, in any order" — a 2-panel figure is a 3-panel figure minus one entry.


class AnimationPanel:
    """
    Base class for one panel of a multi-panel animation.

    Subclasses implement the two halves of an animated panel:

    * :meth:`init_artists` — draw frame 0 onto a given axes and stash the
      dynamic artists.  It receives the engine-resolved ``n_frames`` so the
      panel can truncate its data and fix its axes limits once (preventing the
      autoscale "jump" you would otherwise get as frames advance).
    * :meth:`update` — mutate the stored artists for ``frame`` and return the
      ones that changed, so the engine can blit efficiently.

    The :attr:`n_frames` property reports the panel's *native* number of frames;
    the engine takes the minimum across all panels (unless overridden) so panels
    of differing length stay in sync.
    """

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    def init_artists(self, ax, n_frames: int) -> None:
        raise NotImplementedError

    def update(self, frame: int) -> tuple:
        raise NotImplementedError

    def frame_label(self, frame: int):
        """
        Per-frame label string contributed to the shared suptitle.

        Return a non-empty string (e.g. ``"Power: 1.23 uW"``) to have it
        included in the figure suptitle, or ``None`` to contribute nothing.
        The default returns ``None`` so panels without a swept variable are
        silent.
        """
        return None


class ImageSequencePanel(AnimationPanel):
    """
    A panel that animates a sequence of real-space images.

    Wraps an
    :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace` (or any object
    exposing ``n_frames`` and ``load_frame(idx)``).  Frame 0 is drawn with
    :func:`plot_real_space_PL_map`; subsequent frames swap the image data.  If
    the scan carries a ``laser_ref`` and *laser_annotation* is ``True``, the
    1/e² laser-spot circle is overlaid (static, so it sits in the blit
    background and is redrawn for free).

    Parameters
    ----------
    scan : AttoCubePLScanRealSpace
    title : str
        Per-panel heading.
    cmap : str
        Colormap name passed to :func:`get_cmap` via :func:`plot_real_space_PL_map`.
    laser_annotation : bool
        Overlay the laser-spot circle when ``scan.laser_ref`` is set.
    laser_color : str
        Edge colour of the laser circle.
    laser_linewidth : float
        Line width of the laser circle.
    laser_linestyle : str
        Line style of the laser circle.  Defaults to a solid line (``"-"``),
        which survives GIF palette quantization far better than a dashed one.
    laser_halo : bool
        Draw a contrasting outline (halo) behind the circle so it stays
        visible over bright hot spots, where a thin coloured line would
        otherwise be quantized away in the 256-colour GIF palette.
    laser_halo_color : str
        Colour of the halo stroke.
    xlabel, ylabel : str
        Axis labels forwarded to :func:`plot_real_space_PL_map`.
    """

    def __init__(
        self,
        scan,
        title            : str   = "",
        cmap             : str   = "vik",
        laser_annotation : bool  = True,
        laser_color      : str   = "red",
        laser_linewidth  : float = 1.5,
        laser_linestyle  : str   = "-",
        laser_halo       : bool  = True,
        laser_halo_color : str   = "white",
        xlabel           : str   = "x-axis (pixels)",
        ylabel           : str   = "y-axis (pixels)",
    ):
        self.scan             = scan
        self.title            = title
        self.cmap             = cmap
        self.laser_annotation = laser_annotation
        self.laser_color      = laser_color
        self.laser_linewidth  = laser_linewidth
        self.laser_linestyle  = laser_linestyle
        self.laser_halo       = laser_halo
        self.laser_halo_color = laser_halo_color
        self.xlabel           = xlabel
        self.ylabel           = ylabel
        self._im              = None

    @property
    def n_frames(self) -> int:
        return self.scan.n_frames

    def init_artists(self, ax, n_frames: int) -> None:
        plot_real_space_PL_map(
            self.scan, ax=ax, idx=0, cmap=self.cmap,
            xlabel=self.xlabel, ylabel=self.ylabel,
        )
        ax.set_title(self.title)
        self._im = ax.images[0]

        if self.laser_annotation and getattr(self.scan, "laser_ref", None) is not None:
            lr = self.scan.laser_ref
            circle = patches.Circle(
                (lr.center_x, lr.center_y), radius=lr.radius,
                edgecolor=self.laser_color, facecolor="none",
                linewidth=self.laser_linewidth, linestyle=self.laser_linestyle,
                zorder=3,
            )
            if self.laser_halo:
                # Draw a thicker contrasting stroke behind the coloured line so
                # it stays legible over bright hot spots after GIF quantization.
                circle.set_path_effects([
                    path_effects.withStroke(
                        linewidth=self.laser_linewidth + 2.0,
                        foreground=self.laser_halo_color,
                    ),
                ])
            ax.add_patch(circle)

    def update(self, frame: int) -> tuple:
        self._im.set_data(self.scan.load_frame(frame))
        return (self._im,)


class SpectrumLinePanel(AnimationPanel):
    """
    A panel that animates one PL spectrum per frame.

    Wraps an :class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep` (or any object
    exposing ``energy``/``wavelength`` plus ``best_energy_spectra``/``spectra``
    of shape ``(n_pixels, n_sweeps)``).  The x-axis is fixed; each frame swaps
    the y-values and updates a per-panel subtitle showing the swept value.

    Both axes limits are fixed once over the *truncated* extent
    (``[:, :n_frames]``) so the trace does not rescale or jump between frames.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    x_axis : {"energy", "wavelength"}
    sweep_attr : str
        Name of the per-sweep array used for the subtitle value
        (e.g. ``"scanner_y"``, ``"ef"``, ``"v_top"``).
    sweep_label : str
        Human-readable label shown before the value.  Defaults to *sweep_attr*.
    sweep_unit : str
        Unit string appended to the value (e.g. ``"V"``).
    title_fmt : str
        Format string with ``{label}``, ``{value}`` and ``{unit}`` fields.
    color : str, optional
        Line colour.  Matplotlib default when ``None``.
    ylabel : str, optional
        Y-axis label.  Derived from the scan's measurement type when ``None``,
        so a reflectance sweep is not labelled as PL.  A string is used
        **verbatim**, so include the unit.
    """

    def __init__(
        self,
        scan,
        x_axis           : str  = "energy",
        sweep_attr       : str  = "scanner_y",
        sweep_label      : str  = None,
        sweep_unit       : str  = "V",
        title_fmt        : str  = "{label} = {value:.3g} {unit}",
        show_sweep_title : bool = True,
        color            : str  = None,
        ylabel           : str  = None,
    ):
        self.scan             = scan
        self.x_axis           = x_axis
        self.sweep_attr       = sweep_attr
        self.sweep_label      = sweep_label if sweep_label is not None else sweep_attr
        self.sweep_unit       = sweep_unit
        self.title_fmt        = title_fmt
        self.show_sweep_title = show_sweep_title
        self.color            = color
        self.ylabel           = ylabel
        self._line            = None
        self._title           = None
        self._y               = None
        self._sweep_vals      = None

    @property
    def n_frames(self) -> int:
        return self.scan.n_sweeps

    def init_artists(self, ax, n_frames: int) -> None:
        x, xlabel = _resolve_x_axis(self.scan, self.x_axis)
        y_full = (self.scan.best_energy_spectra if self.x_axis == "energy"
                  else self.scan.spectra)
        self._y          = np.asarray(y_full[:, :n_frames], dtype=float)
        self._sweep_vals = np.asarray(getattr(self.scan, self.sweep_attr))[:n_frames]

        # Fix both axes over the truncated extent so the trace doesn't jump.
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(self._y.min(), self._y.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.ylabel if self.ylabel is not None
                      else _signal_label(self.scan))

        (self._line,) = ax.plot(x, self._y[:, 0], color=self.color)
        # show_sweep_title=True keeps the swept value in ax.set_title (useful
        # when this panel is used standalone).  Set to False when animate_panels
        # is already showing it in the suptitle to avoid duplication.
        if self.show_sweep_title:
            self._title = ax.set_title(self._frame_title(0))
        else:
            self._title = None

    def _frame_title(self, frame: int) -> str:
        return self.title_fmt.format(
            label=self.sweep_label,
            value=self._sweep_vals[frame],
            unit=self.sweep_unit,
        )

    def frame_label(self, frame: int):
        """Contribute the swept value to the shared suptitle."""
        if self._sweep_vals is None:
            return None
        return self._frame_title(frame)

    def update(self, frame: int) -> tuple:
        self._line.set_ydata(self._y[:, frame])
        updated = [self._line]
        if self._title is not None:
            self._title.set_text(self._frame_title(frame))
            updated.append(self._title)
        return tuple(updated)


class NormalizedSpectrumPanel(AnimationPanel):
    """
    One spectrum per frame, each normalized to its own range, coloured by
    the global peak-intensity scale.

    :class:`SpectrumLinePanel` fixes one shared y-axis across every frame,
    which is the right choice when comparing absolute intensity is the
    point. Here it is not: a weak frame would otherwise be flattened to a
    sliver next to a bright one. Each spectrum is instead normalized to its
    own [0, 1] range (:func:`~tmdc_optics_tools.processing.normalise_minmax`)
    so every frame fills the same vertical extent — and the intensity that
    normalizing throws away is put back as the line's *colour*, via a
    colormap spanning the peak intensity's *global* range across every
    frame, not each frame's own max, which would erase the same information
    the height normalization already erased.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep (or any object exposing ``energy``/
        ``wavelength`` plus ``best_energy_spectra``/``spectra`` of shape
        ``(n_pixels, n_sweeps)``)
    x_axis : {"energy", "wavelength"}
    secondary_x_axis : bool
        Add the other of energy/wavelength as a secondary top axis, via
        :func:`~tmdc_optics_tools.processing.energy_to_wavelength` /
        :func:`~tmdc_optics_tools.processing.wavelength_to_energy`.
    cmap : str or None
        Colormap name passed to :func:`get_cmap`, mapping each frame's peak
        intensity to a line colour, with a colorbar showing that scale.
        ``None`` turns this encoding off entirely: no colorbar, and the line
        keeps whatever colour :func:`set_style`/rcParams (or a caller
        restyling the line returned by :meth:`update`) gives it, rather than
        this panel overriding it every frame. This is a *different* setting
        from a plain colour: ``cmap`` says a data channel (peak intensity) is
        drawn through colour, ``None`` says it isn't — not a colour name.
    smooth_window, smooth_poly : int or None
        Forwarded to :func:`~tmdc_optics_tools.processing.smooth_savgol`,
        run once per spectrum before either the colour metric or the
        normalized curve is computed, so both reflect the same smoothed
        data rather than two disagreeing versions of it. ``smooth_window=None``
        skips smoothing.
    colorbar_label : str
        Ignored when ``cmap=None``. Labelled as the *peak* of each frame,
        not the trace's own intensity scale: the colour (and this colorbar)
        is normalized over each frame's own maximum, not over the full
        y-axis range the trace itself spans, so the two are different
        quantities even though they share a unit.
    sweep_attrs : str or list of str, optional
        Per-sweep attribute name(s) shown in the frame label — e.g.
        ``"scanner_y"``, or ``["scanner_x", "scanner_y"]`` for a 2-D
        position scan. ``None`` (default) shows no label.
    sweep_units : str or list of str
        Matching unit(s) for *sweep_attrs*.

    Attributes
    ----------
    colorbar : matplotlib.colorbar.Colorbar or None
        Set by :meth:`init_artists`; ``None`` until then, and always
        ``None`` when ``cmap=None``. Restyle through this handle (e.g.
        ``panel.colorbar.set_label(...)``) rather than adding a parameter
        for it.

    Examples
    --------
    >>> panels = [
    ...     ImageSequencePanel(wl_scan, title="White light"),
    ...     ImageSequencePanel(pl_scan, title="Real-space PL"),
    ...     NormalizedSpectrumPanel(
    ...         spectra_scan, secondary_x_axis=True,
    ...         sweep_attrs=["scanner_x", "scanner_y"],
    ...     ),
    ... ]
    >>> fig, anim = animate_panels(panels)
    """

    def __init__(
        self,
        scan,
        x_axis           : str   = "energy",
        secondary_x_axis : bool  = False,
        cmap             : str   = "inferno",
        smooth_window          = 11,
        smooth_poly      : int  = 3,
        colorbar_label   : str  = "Peak intensity (counts)",
        sweep_attrs             = None,
        sweep_units             = "V",
    ):
        self.scan             = scan
        self.x_axis           = x_axis
        self.secondary_x_axis = secondary_x_axis
        self.cmap             = get_cmap(cmap) if cmap is not None else None
        self.smooth_window    = smooth_window
        self.smooth_poly      = smooth_poly
        self.colorbar_label   = colorbar_label

        attrs = ([sweep_attrs] if isinstance(sweep_attrs, str)
                 else list(sweep_attrs) if sweep_attrs else [])
        units = ([sweep_units] * len(attrs) if isinstance(sweep_units, str)
                 else list(sweep_units))
        self.sweep_attrs = attrs
        self.sweep_units = units

        self.colorbar     = None
        self._line        = None
        self._norm        = None
        self._normalized  = None
        self._peaks       = None
        self._sweep_vals  = None

    @property
    def n_frames(self) -> int:
        return self.scan.n_sweeps

    def init_artists(self, ax, n_frames: int) -> None:
        x, xlabel = _resolve_x_axis(self.scan, self.x_axis)
        y_full = (self.scan.best_energy_spectra if self.x_axis == "energy"
                  else self.scan.spectra)
        raw = np.asarray(y_full[:, :n_frames], dtype=float)

        raw = processing.maybe_smooth(raw, self.smooth_window, self.smooth_poly, axis=0)

        self._peaks      = raw.max(axis=0)
        self._normalized = processing.normalise_minmax(raw, axis=0)
        self._sweep_vals = [np.asarray(getattr(self.scan, attr))[:n_frames]
                             for attr in self.sweep_attrs]

        # A previous init_artists call (re-using this panel instance) would
        # otherwise leave its colorbar in place, stacking a second one onto
        # the figure alongside the new one.
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        if self.cmap is not None:
            self._norm = Normalize(vmin=self._peaks.min(), vmax=self._peaks.max())
            sm = ScalarMappable(cmap=self.cmap, norm=self._norm)
            sm.set_array([])
            self.colorbar = ax.figure.colorbar(sm, ax=ax, pad=0.02, label=self.colorbar_label)
        else:
            self._norm = None

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_signal_label(self.scan, normalized=True))

        if self.secondary_x_axis:
            other = "wavelength" if self.x_axis == "energy" else "energy"
            _, other_label = _resolve_x_axis(self.scan, other)
            convert_fn = (processing.energy_to_wavelength if self.x_axis == "energy"
                          else processing.wavelength_to_energy)

            def convert(v):
                # secondary_xaxis's own tick/limit setup probes values outside
                # the real data range, including 0, before syncing to the
                # parent axes -- not a property of real energy/wavelength data.
                with np.errstate(divide="ignore", invalid="ignore"):
                    return convert_fn(v)

            secax = ax.secondary_xaxis("top", functions=(convert, convert))
            secax.set_xlabel(other_label)

        if self.cmap is not None:
            (self._line,) = ax.plot(x, self._normalized[:, 0],
                                     color=self.cmap(self._norm(self._peaks[0])))
        else:
            (self._line,) = ax.plot(x, self._normalized[:, 0])

    def update(self, frame: int) -> tuple:
        self._line.set_ydata(self._normalized[:, frame])
        if self.cmap is not None:
            self._line.set_color(self.cmap(self._norm(self._peaks[frame])))
        return (self._line,)

    def frame_label(self, frame: int):
        if not self.sweep_attrs:
            return None
        parts = [
            f"{attr} = {vals[frame]:.3g} {unit}".strip()
            for attr, vals, unit in zip(self.sweep_attrs, self._sweep_vals, self.sweep_units)
        ]
        return ", ".join(parts)


class TrimmedImageSequence:
    """
    A view over an existing image sequence, addressing a sub-range or
    permutation of its frames.

    Useful when a sequence needs trimming (e.g. one extra frame at the end
    relative to a paired spectral sweep) or reordering, without loading
    anything twice: :meth:`load_frame` is forwarded to *base_scan* by index.

    Parameters
    ----------
    base_scan : object exposing ``load_frame(idx)``
        E.g. :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace`.
    frame_indices : iterable of int
        Which of *base_scan*'s frames to expose, and in what order — a
        contiguous range trims, an arbitrary permutation reorders.

    Examples
    --------
    >>> wl = TrimmedImageSequence(wl_scan, range(scan.n_sweeps))  # doctest: +SKIP
    """

    def __init__(self, base_scan, frame_indices):
        self._base     = base_scan
        self._indices  = list(frame_indices)
        self.laser_ref = getattr(base_scan, "laser_ref", None)

    @property
    def n_frames(self) -> int:
        return len(self._indices)

    def load_frame(self, idx: int):
        return self._base.load_frame(self._indices[idx])


class GridImageSequence:
    """
    A pre-built ``(height, width, n_frames)`` stack, presented as an
    :class:`ImageSequencePanel`-compatible sequence.

    For a stack already reshaped or reordered outside any loader — e.g.
    :meth:`~tmdc_optics_tools.loaders._AttoCubeSweep.as_image_grid` followed
    by :func:`~tmdc_optics_tools.processing.reorder_grid` — rather than one
    that still has a ``load_frame`` of its own to forward to, which is what
    :class:`TrimmedImageSequence` is for.

    Parameters
    ----------
    frames : np.ndarray, shape (height, width, n_frames)
    laser_ref : AttoCubeLaserReferenceImage, optional
    """

    def __init__(self, frames, laser_ref=None):
        self._frames   = frames
        self.laser_ref = laser_ref

    @property
    def n_frames(self) -> int:
        return self._frames.shape[-1]

    def load_frame(self, idx: int):
        return self._frames[:, :, idx]


class GridSweep:
    """
    A spectral sweep's frames, reordered outside any loader, presented as a
    :class:`NormalizedSpectrumPanel`-compatible scan.

    For per-frame arrays already reshaped or reordered — e.g. via
    :meth:`~tmdc_optics_tools.loaders._AttoCubeSweep.as_grid` followed by
    :func:`~tmdc_optics_tools.processing.reorder_grid` — since the original
    sweep object's own arrays are in the wrong order for that reordering to
    apply to directly.

    Parameters
    ----------
    base_scan : object exposing ``energy``/``wavelength``
        Source of the axes, which do not depend on frame order and so are
        taken as-is.
    best_energy_spectra : np.ndarray, shape (n_pixels, n_frames), optional
        Needed for a :class:`NormalizedSpectrumPanel` built with the default
        ``x_axis="energy"``.
    spectra : np.ndarray, shape (n_pixels, n_frames), optional
        Needed only for ``x_axis="wavelength"``.
    **sweep_attrs
        Per-frame arrays, already reordered to match *best_energy_spectra*,
        exposed under their given names — e.g. ``scanner_x=...`` for a
        :class:`NormalizedSpectrumPanel`'s ``sweep_attrs=["scanner_x"]``.

    Raises
    ------
    ValueError
        If neither *best_energy_spectra* nor *spectra* is given — there
        would then be no way to know ``n_sweeps``.
    """

    def __init__(self, base_scan, best_energy_spectra=None, spectra=None, **sweep_attrs):
        self.energy    = base_scan.energy
        self.wavelength = base_scan.wavelength

        sized = best_energy_spectra if best_energy_spectra is not None else spectra
        if sized is None:
            raise ValueError(
                "GridSweep needs best_energy_spectra, spectra, or both, to "
                "know n_sweeps."
            )
        self.n_sweeps = sized.shape[-1]

        if best_energy_spectra is not None:
            self.best_energy_spectra = best_energy_spectra
        if spectra is not None:
            self.spectra = spectra
        for name, value in sweep_attrs.items():
            setattr(self, name, value)


# Map output file extensions to the Matplotlib animation writer that handles
# them.  GIF (Pillow) is the default; the video formats go through FFmpeg,
# which is full-colour and avoids the 256-colour palette quantization that can
# wash out thin overlays like the laser circle.
_WRITER_BY_EXT = {
    ".gif" : "pillow",
    ".mp4" : "ffmpeg",
    ".m4v" : "ffmpeg",
    ".mov" : "ffmpeg",
    ".webm": "ffmpeg",
}


def _writer_for_path(path: str) -> str:
    """Infer the animation writer from a save path's extension (default GIF)."""
    return _WRITER_BY_EXT.get(Path(path).suffix.lower(), "pillow")


def animate_panels(
    panels,
    n_frames           : int   = None,
    panel_width        : float = 5.0,
    panel_height       : float = 4.0,
    figsize            : tuple = None,
    interval_ms        : int   = 250,
    show_frame_count   : bool  = True,
    frame_count_fmt    : str   = "Frame {frame}/{n_frames}",
    suptitle_sep       : str   = "  |  ",
    constrained_layout : bool  = True,
    save               : str   = None,
    fps                : float = None,
    writer             : str   = None,
) -> tuple:
    """
    Assemble and animate a row of :class:`AnimationPanel` objects.

    Lays out ``1 x len(panels)`` subplots, initialises each panel, and drives
    them in lock step with a single :class:`~matplotlib.animation.FuncAnimation`.
    Because the panel list is the only thing that determines the layout, any
    subset/combination/order of panels works with no special-casing.

    The shared ``suptitle`` is built each frame by concatenating up to three
    parts with *suptitle_sep*:

    * The frame counter ``"Frame n/N"`` (when *show_frame_count* is ``True``).
    * Per-panel swept-variable strings returned by each panel's
      :meth:`~AnimationPanel.frame_label` hook (e.g. ``"Power: 1.23 uW"``).

    Any part that is empty or ``None`` is silently omitted so the separator
    never appears at the start or end of the string.

    Parameters
    ----------
    panels : sequence of AnimationPanel
        One panel per subplot, left to right.
    n_frames : int, optional
        Number of frames to animate.  Defaults to the minimum native
        ``n_frames`` across all panels.
    panel_width, panel_height : float
        Per-panel figure size in inches (used when *figsize* is ``None``).
    figsize : tuple, optional
        Overrides the computed ``(panel_width * n, panel_height)``.
    interval_ms : int
        Delay between frames in milliseconds.
    show_frame_count : bool
        When ``True`` (default), prepend ``"Frame n/N"`` to the suptitle.
        Set to ``False`` to show only the swept-variable labels.
    frame_count_fmt : str
        Format string for the frame counter.  Available fields:
        ``{frame}`` (0-based current frame) and ``{n_frames}`` (total).
        Default ``"Frame {frame}/{n_frames}"``.
    suptitle_sep : str
        Separator inserted between suptitle segments.
        Default ``"  |  "``.
    constrained_layout : bool
    save : str, optional
        If given, save the animation to this path.  The output format is
        chosen from the file extension: ``.gif`` (default) uses the Pillow
        writer; ``.mp4`` / ``.m4v`` / ``.mov`` / ``.webm`` use FFmpeg
        (full-colour, no 256-colour quantization — best when a thin overlay
        such as the laser circle must stay crisp).
    fps : float, optional
        Frames per second for saving.  Defaults to ``1000 / interval_ms``.
    writer : str, optional
        Matplotlib animation writer.  When ``None`` (default) it is inferred
        from the *save* extension (GIF by default).  Pass an explicit writer
        name (e.g. ``"pillow"``, ``"ffmpeg"``) to override.

    Returns
    -------
    fig, anim

    Examples
    --------
    >>> panels = [
    ...     ImageSequencePanel(white_light_map, title="White light", cmap="gray"),
    ...     ImageSequencePanel(real_space_PL_map, title="Real-space PL", cmap="lipari"),
    ...     SpectrumLinePanel(spectra_linescan, x_axis="energy"),
    ... ]
    >>> # suptitle shows e.g. "Frame 3/78  |  Power: 1.23 uW"
    >>> fig, anim = animate_panels(panels, save="three_panel_scan.gif")
    """
    panels = list(panels)
    n = len(panels)
    if n == 0:
        raise ValueError("animate_panels requires at least one panel.")

    if n_frames is None:
        n_frames = min(p.n_frames for p in panels)
    if figsize is None:
        figsize = (panel_width * n, panel_height)

    fig, axes = plt.subplots(
        1, n, figsize=figsize,
        constrained_layout=constrained_layout, squeeze=False,
    )
    axes = axes[0]

    for panel, ax in zip(panels, axes):
        panel.init_artists(ax, n_frames)

    # _build_suptitle and _has_suptitle must be evaluated AFTER init_artists
    # has run on every panel.  DiffusionCloudPanel (and any other panel that
    # defers heavy work to init_artists) calls _resolve_var() there, which is
    # what populates self._var_array.  Calling frame_label(0) before
    # init_artists would always return None, suppressing the suptitle entirely.

    def _build_suptitle(frame: int) -> str:
        parts = []
        if show_frame_count:
            parts.append(frame_count_fmt.format(frame=frame, n_frames=n_frames))
        for panel in panels:
            lbl = panel.frame_label(frame)
            if lbl:
                parts.append(lbl)
        return suptitle_sep.join(parts)

    _has_suptitle = show_frame_count or any(
        panel.frame_label(0) is not None for panel in panels
    )

    # fig.suptitle() is a Figure-level artist.  With blit=True, matplotlib
    # only redraws Axes-level artists, so the suptitle text updates correctly
    # in memory but is never repainted on screen — it appears frozen on the
    # frame-0 string for the entire animation.
    #
    # Fix: place the shared title as a centred text artist on the top axes
    # (the leftmost one when there are several panels).  It is an Axes artist
    # so blit picks it up, yet with transform=fig.transFigure it sits at the
    # same visual position as a suptitle would.
    if _has_suptitle:
        title_ax = axes[len(axes) // 2]   # centre panel (or only panel)

        # Check whether any panel has set a non-empty axes title.
        # If so, we need to stack the suptitle above the axes title rather
        # than overlapping it.  We do this by:
        #   - moving the suptitle text higher (y=1.12 instead of 1.04), and
        #   - nudging each panel's axes title downward (pad=-4) so there is
        #     clear vertical separation between the two lines.
        any_panel_title = any(
            ax.get_title() for ax in axes
        )
        if any_panel_title:
            suptitle_y = 1.12
            for ax in axes:
                if ax.get_title():
                    ax.set_title(ax.get_title(), pad=-4)
        else:
            suptitle_y = 1.04

        suptitle = title_ax.text(
            0.5, suptitle_y,
            _build_suptitle(0),
            transform      = title_ax.transAxes,
            ha             = "center",
            va             = "bottom",
            fontsize       = plt.rcParams.get("figure.titlesize", "large"),
            fontweight     = plt.rcParams.get("figure.titleweight", "normal"),
        )
    else:
        suptitle = None

    def update(frame):
        artists = []
        for panel in panels:
            artists.extend(panel.update(frame))
        if suptitle is not None:
            suptitle.set_text(_build_suptitle(frame))
            artists.append(suptitle)
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=True, interval=interval_ms,
    )

    if save is not None:
        if fps is None:
            fps = 1000.0 / interval_ms
        if writer is None:
            writer = _writer_for_path(save)
        anim.save(save, writer=writer, fps=fps)

    return fig, anim


def animate_wl_pl_spectra(
    wl               = None,
    pl               = None,
    spectra          = None,
    laser_ref        = None,
    x_axis           : str = "energy",
    wl_cmap          : str = "gray",
    pl_cmap          : str = "lipari",
    wl_title         : str = "White light",
    pl_title         : str = "Real-space PL",
    sweep_attr       : str = "scanner_y",
    sweep_unit       : str = "V",
    laser_ref_kwargs : dict = None,
    laser_style      : dict = None,
    save             : str  = None,
    **engine_kwargs,
) -> tuple:
    """
    Convenience wrapper: build a white-light / real-space-PL / spectrum
    animation straight from file paths (or pre-built loaders).

    Each of the three panels is optional — omit one (leave it ``None``) and the
    figure simply drops to two (or one) panels.  This is how every combination
    is supported without a separate function per layout.

    Parameters
    ----------
    wl, pl : (dir, prefix) tuple or AttoCubePLScanRealSpace or None
        White-light and real-space-PL image sequences.  A ``(dir, prefix)``
        tuple is loaded into an
        :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace`; an existing
        scan object is used as-is.
    spectra : str or AttoCubeSpectralSweep or None
        Spectrum line-scan.  A path is loaded into an
        :class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep` with
        ``spectra_type="PL"``; pass a pre-built sweep for any other measurement
        type or to declare a ``sweep=``.
    laser_ref : str or AttoCubeLaserReferenceImage or None
        Shared laser-spot reference for the image panels.  A path is loaded
        into an :class:`~tmdc_optics_tools.loaders.AttoCubeLaserReferenceImage`
        (with *laser_ref_kwargs*).
    x_axis : {"energy", "wavelength"}
        Spectrum panel x-axis.
    wl_cmap, pl_cmap : str
        Colormaps for the two image panels.
    wl_title, pl_title : str
        Titles for the two image panels.
    sweep_attr, sweep_unit : str
        Per-sweep attribute and unit shown in the spectrum subtitle.
    laser_ref_kwargs : dict, optional
        Extra keyword arguments for
        :class:`~tmdc_optics_tools.loaders.AttoCubeLaserReferenceImage` when
        *laser_ref* is a path (e.g. ``{"expected_radius_px": 10}``).
    laser_style : dict, optional
        Laser-circle styling forwarded to both image
        :class:`ImageSequencePanel` panels, e.g.
        ``{"laser_color": "red", "laser_linewidth": 1.5, "laser_linestyle": "-",
        "laser_halo": True, "laser_halo_color": "white", "laser_annotation": True}``.
    save : str, optional
        Output path for the animation.  Format is chosen from the extension
        (``.gif`` by default; ``.mp4`` etc. via FFmpeg) — see
        :func:`animate_panels`.
    **engine_kwargs
        Forwarded to :func:`animate_panels` (e.g. ``interval_ms``,
        ``suptitle_fmt``, ``n_frames``, ``writer``).

    Returns
    -------
    fig, anim

    Examples
    --------
    >>> # All three panels
    >>> fig, anim = animate_wl_pl_spectra(
    ...     wl=("./wl/", "wl_"), pl=("./PL/", "PL_"),
    ...     spectra="./PL/PL_..iter_0.csv",
    ...     laser_ref="laser_ref.csv", save="three_panel_scan.gif",
    ... )

    >>> # PL map + spectra only
    >>> fig, anim = animate_wl_pl_spectra(
    ...     pl=("./PL/", "PL_"), spectra="./PL/PL_..iter_0.csv",
    ...     laser_ref="laser_ref.csv",
    ... )
    """
    from .loaders import (
        AttoCubePLScanRealSpace,
        AttoCubeSpectralSweep,
        AttoCubeLaserReferenceImage,
    )

    # Resolve the shared laser reference (path -> loader, object -> as-is).
    if isinstance(laser_ref, (str, Path)):
        laser_ref = AttoCubeLaserReferenceImage(
            str(laser_ref), **(laser_ref_kwargs or {})
        )

    def _image_scan(spec):
        if spec is None:
            return None
        if isinstance(spec, AttoCubePLScanRealSpace):
            return spec
        directory, prefix = spec
        return AttoCubePLScanRealSpace(
            path=str(directory), prefix=prefix, laser_ref=laser_ref,
        )

    def _spectrum_scan(spec):
        # isinstance on the base class also accepts the deprecated
        # AttoCubePLVabScan, which is a subclass of it.
        if spec is None or isinstance(spec, AttoCubeSpectralSweep):
            return spec
        # A bare path: this function animates PL, so declare it rather than
        # letting the loader guess.  Pass a pre-built sweep for anything else.
        return AttoCubeSpectralSweep(path=str(spec), spectra_type="PL")

    laser_style = laser_style or {}

    panels = []
    wl_scan = _image_scan(wl)
    if wl_scan is not None:
        panels.append(ImageSequencePanel(
            wl_scan, title=wl_title, cmap=wl_cmap, **laser_style))

    pl_scan = _image_scan(pl)
    if pl_scan is not None:
        panels.append(ImageSequencePanel(
            pl_scan, title=pl_title, cmap=pl_cmap, **laser_style))

    spec_scan = _spectrum_scan(spectra)
    if spec_scan is not None:
        panels.append(SpectrumLinePanel(
            spec_scan, x_axis=x_axis,
            sweep_attr=sweep_attr, sweep_unit=sweep_unit,
        ))

    if not panels:
        raise ValueError(
            "animate_wl_pl_spectra needs at least one of wl, pl, or spectra."
        )

    return animate_panels(panels, save=save, **engine_kwargs)


def trim_to_sweep_count(image_scan, n_sweeps: int, auto_trim: bool = True):
    """
    Drop trailing frames beyond *n_sweeps*, warning when it happens.

    The AttoCube acquisition can leave one extra frame (e.g. white light) at
    the end of an image sequence relative to a paired spectral sweep — a
    known export quirk, not a corrupted sequence (see CLAUDE.md's AttoCube
    export record). :meth:`~.AttoCubeSpectralSweep.as_image_grid` and
    :func:`animate_wl_pl_spectra_grid` both need an exact frame-count match,
    so this is where that quirk gets handled, once, rather than at every call
    site that runs into it.

    Parameters
    ----------
    image_scan : object exposing ``n_frames`` and ``load_frame(idx)``
        E.g. :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace`.
    n_sweeps : int
        The frame count *image_scan* is expected to match.
    auto_trim : bool
        If ``image_scan`` has more frames than *n_sweeps*, wrap it in a
        :class:`TrimmedImageSequence` keeping only the first *n_sweeps*, with
        a ``UserWarning`` naming how many frames were dropped. ``False``
        returns *image_scan* unchanged, so a caller that requires an exact
        match (e.g. :meth:`~.AttoCubeSpectralSweep.as_image_grid`) raises its
        own, more specific error instead.

    Returns
    -------
    object
        *image_scan* unchanged if its frame count already matches *n_sweeps*
        or *auto_trim* is ``False``; otherwise a :class:`TrimmedImageSequence`
        over it.
    """
    if image_scan.n_frames == n_sweeps:
        return image_scan
    if auto_trim and image_scan.n_frames > n_sweeps:
        warnings.warn(
            f"{type(image_scan).__name__} has {image_scan.n_frames} frames vs "
            f"{n_sweeps} sweep points -- dropping {image_scan.n_frames - n_sweeps} "
            f"frame(s) from the end.",
            UserWarning, stacklevel=2,
        )
        return TrimmedImageSequence(image_scan, range(n_sweeps))
    return image_scan  # let as_image_grid raise its own, more specific error


def animate_wl_pl_spectra_grid(
    scan,
    wl               = None,
    pl               = None,
    inner_axis       : str  = "fast",
    reverse_fast     : bool = False,
    reverse_slow     : bool = False,
    x_axis           : str  = "energy",
    wl_cmap          : str  = "gray",
    pl_cmap          : str  = "lipari",
    wl_title         : str  = "White light",
    pl_title         : str  = "Real-space PL",
    sweep_attrs             = ("scanner_x", "scanner_y"),
    sweep_units             = "V",
    secondary_x_axis : bool = True,
    laser_style      : dict = None,
    auto_trim        : bool = True,
    save             : str  = None,
    **engine_kwargs,
) -> tuple:
    """
    Three-panel animation over a declared 2-D nest, played in a chosen order.

    The nested-sweep sibling of :func:`animate_wl_pl_spectra`: that
    function's :class:`SpectrumLinePanel` plays a flat sweep in file order
    and needs no reordering, while a 2-D raster needs *inner_axis* /
    *reverse_fast* / *reverse_slow* to say what traversal order to play it
    in. Its spectrum panel is a :class:`NormalizedSpectrumPanel` (each frame
    normalized to its own range, colour-coded by peak intensity) rather than
    a :class:`SpectrumLinePanel` (one shared y-axis), since a position
    raster's brightness commonly spans more than one order of magnitude,
    which a shared axis would flatten most frames to a sliver.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
        Must carry a declared nest (``fast_sweep=`` / ``slow_sweep=`` at load
        time) — raises the same way :meth:`~.AttoCubeSpectralSweep.as_grid`
        does otherwise.
    wl, pl : AttoCubePLScanRealSpace, optional
        White-light and real-space-PL image sequences, one frame per *scan*
        sweep point. Omit one (leave it ``None``) and the figure drops to two
        (or one) panels, the same convention as :func:`animate_wl_pl_spectra`.
        Each must already share *scan*'s ``laser_ref``, if any — read off
        the sequence itself, not a separate parameter here.
    inner_axis : {"fast", "slow"}
        Which nest axis varies fastest during playback. ``"fast"`` (default)
        reproduces the order the data was written in.
    reverse_fast, reverse_slow : bool
        Traverse that axis in decreasing order instead of increasing; the two
        combine independently of each other and of *inner_axis*.
    x_axis : {"energy", "wavelength"}
        Spectrum panel x-axis.
    wl_cmap, pl_cmap, wl_title, pl_title : str
        As :func:`animate_wl_pl_spectra`.
    sweep_attrs, sweep_units : str or list of str
        Forwarded to :class:`NormalizedSpectrumPanel`'s own ``sweep_attrs`` /
        ``sweep_units``, for the per-frame position label. Default to both
        nest axes.
    secondary_x_axis : bool
        As :class:`NormalizedSpectrumPanel`.
    laser_style : dict, optional
        Forwarded to both image :class:`ImageSequencePanel`\\ s, as
        :func:`animate_wl_pl_spectra`'s own *laser_style*.
    auto_trim : bool
        An image sequence with more frames than ``scan.n_sweeps`` is trimmed
        to the first ``scan.n_sweeps`` (a known AttoCube acquisition quirk —
        see CLAUDE.md's AttoCube export record), with a ``UserWarning``
        naming how many frames were dropped. ``False`` raises instead, via
        :meth:`~.AttoCubeSpectralSweep.as_image_grid`'s own error.
    save : str, optional
        As :func:`animate_wl_pl_spectra`.
    **engine_kwargs
        Forwarded to :func:`animate_panels` (e.g. ``interval_ms``, ``writer``).

    Returns
    -------
    fig, anim

    See Also
    --------
    animate_wl_pl_spectra : the flat-sweep counterpart.
    """
    laser_style = laser_style or {}
    # NormalizedSpectrumPanel itself accepts sweep_units as a single string or a
    # per-attribute list -- only sweep_attrs needs expanding here, to index
    # axis_grids below.
    sweep_attrs = [sweep_attrs] if isinstance(sweep_attrs, str) else list(sweep_attrs)
    order_kwargs = dict(inner_axis=inner_axis, reverse_fast=reverse_fast, reverse_slow=reverse_slow)

    spectra_key  = "best_energy_spectra" if x_axis == "energy" else "spectra"
    spectra_grid = scan.as_grid(getattr(scan, spectra_key))
    axis_grids   = {attr: scan.as_grid(np.asarray(getattr(scan, attr))) for attr in sweep_attrs}

    panels = []
    for image_scan, title, cmap in ((wl, wl_title, wl_cmap), (pl, pl_title, pl_cmap)):
        if image_scan is None:
            continue
        image_scan = trim_to_sweep_count(image_scan, scan.n_sweeps, auto_trim)
        image_grid = scan.as_image_grid(image_scan)
        panels.append(ImageSequencePanel(
            GridImageSequence(processing.reorder_grid(image_grid, **order_kwargs),
                               laser_ref=getattr(image_scan, "laser_ref", None)),
            title=title, cmap=cmap, **laser_style,
        ))

    panels.append(NormalizedSpectrumPanel(
        GridSweep(
            scan,
            **{spectra_key: processing.reorder_grid(spectra_grid, **order_kwargs)},
            **{attr: processing.reorder_grid(grid, **order_kwargs)
               for attr, grid in axis_grids.items()},
        ),
        x_axis=x_axis, secondary_x_axis=secondary_x_axis,
        sweep_attrs=sweep_attrs, sweep_units=sweep_units,
    ))

    return animate_panels(panels, save=save, **engine_kwargs)


# ---------------------------------------------------------------------------
# Diffusion cloud — shared helpers
# ---------------------------------------------------------------------------

def _draw_laser_circle(
    ax,
    laser_ref,
    color      : str   = "red",
    lw         : float = 1.5,
    ls         : str   = "-",
    halo       : bool  = True,
    halo_color : str   = "white",
) -> patches.Circle:
    """
    Draw the 1/e² laser-spot boundary on *ax* and return the Circle artist.

    Mirrors the implementation used in :class:`ImageSequencePanel` so both
    static single-frame plots and animations get identical laser annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    laser_ref : AttoCubeLaserReferenceImage
        Must expose ``center_x``, ``center_y``, and ``radius`` attributes.
    color : str
        Edge colour of the circle.
    lw : float
        Line width.
    ls : str
        Line style (e.g. ``"-"`` or ``"--"``).
    halo : bool
        Draw a contrasting halo stroke behind the circle so it stays
        visible over bright pixels after GIF palette quantization.
    halo_color : str
        Colour of the halo stroke.

    Returns
    -------
    patches.Circle
    """
    circle = patches.Circle(
        (laser_ref.center_x, laser_ref.center_y),
        radius    = laser_ref.radius,
        edgecolor = color,
        facecolor = "none",
        linewidth = lw,
        linestyle = ls,
        label     = f"Laser 1/e² ({laser_ref.radius:.1f} px)",
        zorder    = 4,
    )
    if halo:
        circle.set_path_effects([
            path_effects.withStroke(linewidth=lw + 2.0, foreground=halo_color),
        ])
    ax.add_patch(circle)
    return circle


# ---------------------------------------------------------------------------
# Diffusion cloud — single image
# ---------------------------------------------------------------------------

def plot_diffusion_cloud(
    image,
    result             = None,
    ax                 = None,
    figsize            : tuple = (4, 4),
    dpi                : int   = None,
    cmap               : str   = "inferno",
    contour_color      : str   = "green",
    contour_lw         : float = 0.9,
    contour_ls         : str   = "--",
    centroid_color     : str   = "white",
    centroid_marker    : str   = "+",
    centroid_ms        : float = 30,
    colorbar           : bool  = True,
    colorbar_label     : str   = "Intensity (counts)",
    xlabel             : str   = "x (px)",
    ylabel             : str   = "y (px)",
    show_roi           : bool  = False,
    show_bg_region     : bool  = False,
    roi                : tuple = None,
    bg_region          : tuple = None,
    bg_stat            : str   = "median",
    roi_color          : str   = "lime",
    bg_region_color    : str   = "orange",
    # laser spot
    laser_ref                  = None,
    laser_annotation   : bool  = True,
    laser_color        : str   = "red",
    laser_linewidth    : float = 1.5,
    laser_linestyle    : str   = "-",
    laser_halo         : bool  = True,
    laser_halo_color   : str   = "white",
    # analyse_diffusion_cloud kwargs (used when result is None)
    threshold                  = "1/e",
    smooth_sigma       : float = 1.0,
    keep_largest       : bool  = True,
    pixel_scale        : float = None,
    origin             : str   = "corner",
) -> tuple:
    """
    Plot a single real-space PL image with the diffusion cloud boundary and
    centroid overlaid.

    You can either supply a pre-computed :class:`~tmdc_optics_tools.diffusion.DiffusionResult`
    via *result*, or let the function run the analysis internally (using the
    keyword arguments that mirror :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_cloud`).

    Parameters
    ----------
    image : np.ndarray, str, pathlib.Path, or _AttoCubeImage
        2-D PL image, forwarded as-is to
        :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_cloud` when
        *result* is ``None`` (same accepted types — see that function).
        Ignored when *result* is supplied: display then uses
        :attr:`~tmdc_optics_tools.diffusion.DiffusionResult.image` from
        *result* itself, so the two never disagree about what was analysed.
    result : DiffusionResult, optional
        Pre-computed result from :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_cloud`.
        When ``None`` the analysis is run here with the remaining kwargs.
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    cmap : str
        Colormap for the image.
    contour_color, contour_lw, contour_ls : str / float / str
        Style of the boundary contour.
    centroid_color, centroid_marker, centroid_ms : str / str / float
        Style of the centroid marker.
    colorbar : bool
    colorbar_label : str
    xlabel, ylabel : str
    laser_ref : AttoCubeLaserReferenceImage or None
        Laser-spot reference.  When supplied (and *laser_annotation* is
        ``True``), the 1/e² spot boundary circle is drawn on the axes.
        Can also be passed implicitly via ``image.laser_ref`` (i.e. from
        :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace` which
        stores it); an explicit *laser_ref* argument takes priority.
    laser_annotation : bool
        Draw the laser circle when *laser_ref* is available.  Default ``True``.
    laser_color : str
        Edge colour of the laser circle.  Default ``"red"``.
    laser_linewidth : float
        Line width of the laser circle.  Default ``1.5``.
    laser_linestyle : str
        Line style of the laser circle.  Default ``"-"``.
    laser_halo : bool
        Draw a contrasting halo stroke behind the circle so it stays
        visible over bright hot spots.  Default ``True``.
    laser_halo_color : str
        Colour of the halo stroke.  Default ``"white"``.
    threshold, smooth_sigma, keep_largest, pixel_scale, origin
        Forwarded to :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_cloud`
        when *result* is ``None``.

    Returns
    -------
    fig, ax, result : (Figure, Axes, DiffusionResult)
        The DiffusionResult is always returned so you can inspect the
        centroid and area without a separate call.
    """
    from . import diffusion as _diffusion

    if result is None:
        # Pass the image object/path/array straight through -- _load_image
        # already knows to use an _AttoCubeImage's *raw* array so that the
        # bg_region below is the only subtraction that happens. Handing it
        # an already-corrected `.img` here would subtract twice.
        result = _diffusion.analyse_diffusion_cloud(
            image,
            threshold    = threshold,
            smooth_sigma = smooth_sigma,
            keep_largest = keep_largest,
            pixel_scale  = pixel_scale,
            origin       = origin,
            roi          = roi,
            bg_region    = bg_region,
            bg_stat      = bg_stat
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    im = ax.imshow(result.image, cmap=get_cmap(cmap), origin="upper")

    for contour in result.contours:
        ax.plot(contour[:, 1], contour[:, 0], color=contour_color,
                linewidth=contour_lw, linestyle=contour_ls)

    cx, cy = result.x_pixel, result.y_pixel
    ax.scatter(cx, cy, s=centroid_ms, c=centroid_color, marker=centroid_marker,
            linewidths=0.8, zorder=5, label=f"({cx:.1f}, {cy:.1f}) px")
    
    if show_roi:
        processing._draw_region_box(ax, roi if roi is not None else result.roi,
                        roi_color, label="ROI")
    if show_bg_region:
        processing._draw_region_box(ax, bg_region if bg_region is not None else getattr(image, "bg_region", None),
                        bg_region_color, label="bg region")

    # Laser circle — explicit arg takes priority, then image.laser_ref.
    _lr = laser_ref if laser_ref is not None else getattr(image, "laser_ref", None)
    if laser_annotation and _lr is not None:
        _draw_laser_circle(
            ax, _lr,
            color    = laser_color,
            lw       = laser_linewidth,
            ls       = laser_linestyle,
            halo     = laser_halo,
            halo_color = laser_halo_color,
        )

    ax.legend(fontsize=5, loc="upper right", frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(colorbar_label)

    return fig, ax, result


# ---------------------------------------------------------------------------
# Diffusion cloud — centroid trajectory plot
# ---------------------------------------------------------------------------

def plot_centroid_trajectory(
    seq_result,
    ax           = None,
    figsize      : tuple = (5, 3.5),
    dpi          : int   = None,
    coord        : str   = "both",
    use_real     : bool  = False,
    color_x      : str   = "C0",
    color_y      : str   = "C1",
    marker       : str   = "o",
    markersize   : float = 3,
    xlabel       : str   = None,
    ylabel       : str   = None,
) -> tuple:
    """
    Plot the centroid position as a function of an external variable.

    Parameters
    ----------
    seq_result : DiffusionSequenceResult
        Output of :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_sequence`.
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    coord : {``"x"``, ``"y"``, ``"both"``}
        Which centroid coordinate(s) to plot.
    use_real : bool
        If ``True`` and real-space coordinates are available, plot those
        instead of pixel coordinates.
    color_x, color_y : str
        Line/marker colours for x and y coordinates.
    marker, markersize : str / float
        Marker style.
    xlabel : str, optional
        X-axis label. Defaults to ``"<var_label> (<var_units>)"`` or
        ``"Frame"`` when no *var_array* is set.
    ylabel : str, optional
        Y-axis label.

    Returns
    -------
    fig, ax
    """
    sr = seq_result

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    x_ax = (sr.var_array if sr.var_array is not None
            else np.arange(sr.n_frames))

    if use_real and sr.x_real is not None:
        cx = sr.x_real
        cy = sr.y_real
        coord_unit = " (real)"
    else:
        cx = sr.x_pixel
        cy = sr.y_pixel
        coord_unit = " (px)"

    if coord in ("x", "both"):
        ax.plot(x_ax, cx, color=color_x, marker=marker,
                markersize=markersize, label="x" + coord_unit)
    if coord in ("y", "both"):
        ax.plot(x_ax, cy, color=color_y, marker=marker,
                markersize=markersize, label="y" + coord_unit)

    if xlabel is None:
        if sr.var_array is not None and sr.var_label:
            xlabel = f"{sr.var_label} ({sr.var_units})"
        elif sr.var_array is not None:
            xlabel = sr.var_units or "External variable"
        else:
            xlabel = "Frame"
    if ylabel is None:
        ylabel = "Centroid" + coord_unit

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7)

    return fig, ax


# ---------------------------------------------------------------------------
# DiffusionCloudPanel — drop-in AnimationPanel for animate_panels()
# ---------------------------------------------------------------------------

class DiffusionCloudPanel(AnimationPanel):
    """
    An :class:`AnimationPanel` that overlays the exciton diffusion cloud
    boundary and centroid on each frame of an image sequence animation.

    Plug this into :func:`animate_panels` alongside
    :class:`ImageSequencePanel` or :class:`SpectrumLinePanel` to build a
    composite animation that shows the boundary evolving frame-by-frame.

    Parameters
    ----------
    scan : AttoCubePLScanRealSpace or list of np.ndarray
        Image sequence.  Any object exposing ``load_frame(idx)`` and
        ``n_frames`` (e.g. :class:`~tmdc_optics_tools.loaders.AttoCubePLScanRealSpace`),
        or a plain list of 2-D arrays.
    seq_result : DiffusionSequenceResult, optional
        Pre-computed sequence result.  When ``None`` the analysis is run
        lazily on ``init_artists`` (requires that *scan* be indexable at
        that point).
    title : str
        Panel heading.
    cmap : str
        Colormap for the image.
    contour_color, contour_lw, contour_ls : str / float / str
        Style of the per-frame boundary contour.
    centroid_color, centroid_marker, centroid_ms : str / str / float
        Style of the per-frame centroid marker.
    xlabel, ylabel : str
    var_array : array-like, optional
        Values of the swept parameter, one per frame (e.g. optical power,
        gate voltage).  When supplied, the panel's ``ax.set_title`` is
        updated every frame to show the current value.  If *seq_result* is
        provided and already carries a ``var_array``, that is used as the
        default — an explicit *var_array* here overrides it.
    var_label : str
        Human-readable label prepended to the per-frame value,
        e.g. ``\"Power\"`` → ``\"Power: 1.23 µW\"``.
        Defaults to *seq_result.var_label* when available.
    var_units : str
        Unit string appended to the value (e.g. ``\"µW\"``).
        Defaults to *seq_result.var_units* when available.
    var_fmt : str
        Python format spec for the numeric value (e.g. ``\".3g\"``).
    threshold, smooth_sigma, keep_largest, pixel_scale, origin
        Forwarded to :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_sequence`
        when *seq_result* is ``None``.

    Examples
    --------
    >>> seq = analyse_diffusion_sequence(scan, pixel_scale=0.065)
    >>> panels = [
    ...     ImageSequencePanel(wl_scan, title="White light"),
    ...     DiffusionCloudPanel(pl_scan, seq_result=seq, title="Exciton cloud"),
    ... ]
    >>> fig, anim = animate_panels(panels, save="diffusion_sweep.gif")
    """

    def __init__(
        self,
        scan,
        seq_result        = None,
        title             : str   = "",
        cmap              : str   = "inferno",
        contour_color     : str   = "cyan",
        contour_lw        : float = 0.9,
        contour_ls        : str   = "--",
        centroid_color    : str   = "white",
        centroid_marker   : str   = "+",
        centroid_ms       : float = 30,
        xlabel            : str   = "x (px)",
        ylabel            : str   = "y (px)",
        # laser spot annotation
        laser_ref                 = None,
        laser_annotation  : bool  = True,
        laser_color       : str   = "red",
        laser_linewidth   : float = 1.5,
        laser_linestyle   : str   = "-",
        laser_halo        : bool  = True,
        laser_halo_color  : str   = "white",
        # swept-variable display — overrides seq_result fields when not None
        var_array                 = None,
        var_label         : str   = None,
        var_units         : str   = None,
        var_fmt           : str   = ".3g",
        # analyse kwargs (used when seq_result is None)
        threshold                 = "otsu",
        smooth_sigma      : float = 1.0,
        keep_largest      : bool  = True,
        pixel_scale       : float = None,
        origin            : str   = "corner",
        bg_region         : tuple = None,
        bg_stat           : str   = "median",
        roi               : tuple = None,
        show_roi          : bool  = False,
        show_bg_region    : bool  = False,
        roi_color         : str   = "lime",
        bg_region_color   : str   = "orange",
    ):
        self.scan           = scan
        self._seq_result    = seq_result
        self.title          = title
        self.cmap           = cmap
        self.contour_color  = contour_color
        self.contour_lw     = contour_lw
        self.contour_ls     = contour_ls
        self.centroid_color = centroid_color
        self.centroid_marker= centroid_marker
        self.centroid_ms    = centroid_ms
        self.xlabel         = xlabel
        self.ylabel         = ylabel
        # laser spot
        # Priority: explicit laser_ref arg → scan.laser_ref → None
        self._laser_ref_override = laser_ref
        self.laser_annotation    = laser_annotation
        self.laser_color         = laser_color
        self.laser_linewidth     = laser_linewidth
        self.laser_linestyle     = laser_linestyle
        self.laser_halo          = laser_halo
        self.laser_halo_color    = laser_halo_color
        self._laser_circle       = None   # artist stored for blit
        # swept-variable display (None means "inherit from seq_result later")
        self._var_array_override = np.asarray(var_array) if var_array is not None else None
        self._var_label_override = var_label
        self._var_units_override = var_units
        self._var_fmt        = var_fmt
        self._var_array      = None   # resolved in init_artists
        self._var_label      = None
        self._var_units      = None
        # analysis kwargs stored for lazy computation
        self._threshold     = threshold
        self._smooth_sigma  = smooth_sigma
        self._keep_largest  = keep_largest
        self._pixel_scale   = pixel_scale
        self._origin        = origin
        self._bg_region     = bg_region
        self._bg_stat       = bg_stat
        self._roi           = roi
        self.show_roi       = show_roi
        self.show_bg_region = show_bg_region
        self.roi_color      = roi_color
        self.bg_region_color= bg_region_color

        # artists (set in init_artists)
        self._im            = None
        self._contour_lines = []
        self._centroid_pt   = None

    @property
    def n_frames(self) -> int:
        if hasattr(self.scan, "n_frames"):
            return self.scan.n_frames
        return len(self.scan)

    def _get_seq_result(self):
        """Run analysis lazily if not pre-supplied."""
        if self._seq_result is None:
            from . import diffusion as _diffusion
            self._seq_result = _diffusion.analyse_diffusion_sequence(
                self.scan,
                threshold    = self._threshold,
                smooth_sigma = self._smooth_sigma,
                keep_largest = self._keep_largest,
                pixel_scale  = self._pixel_scale,
                origin       = self._origin,
                bg_region    = self._bg_region,
                bg_stat      = self._bg_stat,
                roi          = self._roi,
            )
        return self._seq_result

    def _resolve_var(self, seq, n_frames: int) -> None:
        """
        Resolve the swept-variable array and labels.

        Priority (highest first):
        1. Explicit ``var_array`` / ``var_label`` / ``var_units`` passed to
           ``__init__``.
        2. ``seq_result.var_array`` / ``.var_label`` / ``.var_units`` — the
           values that were forwarded from ``analyse_diffusion_sequence``.
        3. ``None`` — no per-frame subtitle is shown.
        """
        arr = self._var_array_override
        if arr is None and seq.var_array is not None:
            arr = seq.var_array
        if arr is not None:
            self._var_array = np.asarray(arr)[:n_frames]
        else:
            self._var_array = None

        self._var_label = (
            self._var_label_override
            if self._var_label_override is not None
            else seq.var_label
        )
        self._var_units = (
            self._var_units_override
            if self._var_units_override is not None
            else seq.var_units
        )

    def _make_frame_title(self, frame: int) -> str:
        """Format the per-frame subtitle string."""
        value_str = f"{self._var_array[frame]:{self._var_fmt}} {self._var_units}".strip()
        return f"{self._var_label}: {value_str}" if self._var_label else value_str

    def frame_label(self, frame: int):
        """Contribute the swept value to the shared suptitle."""
        if self._var_array is None:
            return None
        return self._make_frame_title(frame)

    def init_artists(self, ax, n_frames: int) -> None:
        seq = self._get_seq_result()
        self._resolve_var(seq, n_frames)

        # Draw frame 0
        frame0 = (self.scan.load_frame(0)
                  if hasattr(self.scan, "load_frame") else self.scan[0])
        self._im = ax.imshow(
            np.asarray(frame0, float),
            cmap=get_cmap(self.cmap), origin="upper",
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        # Static panel title only — swept value is handled by frame_label
        # and composed into the figure suptitle by animate_panels.
        ax.set_title(self.title)

        if self.show_roi:
            processing._draw_region_box(
                ax,
                self._roi if self._roi is not None else getattr(seq, "roi", None),
                self.roi_color, label="ROI",
            )
        if self.show_bg_region:
            processing._draw_region_box(
                ax,
                self._bg_region if self._bg_region is not None else getattr(seq, "bg_region", None),
                self.bg_region_color, label="bg region",
            )

        r0 = seq.frames[0]
        # Contour lines for frame 0
        self._contour_lines = []
        for contour in r0.contours:
            (line,) = ax.plot(
                contour[:, 1], contour[:, 0],
                color=self.contour_color,
                linewidth=self.contour_lw,
                linestyle=self.contour_ls,
            )
            self._contour_lines.append(line)

        # Centroid marker
        self._centroid_pt = ax.scatter(
            r0.x_pixel, r0.y_pixel,
            s=self.centroid_ms, c=self.centroid_color,
            marker=self.centroid_marker, linewidths=0.8, zorder=5,
        )

        # Laser circle — static overlay; drawn once in init, never updated.
        # Priority: explicit arg → scan.laser_ref → None.
        _lr = (
            self._laser_ref_override
            if self._laser_ref_override is not None
            else getattr(self.scan, "laser_ref", None)
        )
        self._laser_circle = None
        if self.laser_annotation and _lr is not None:
            self._laser_circle = _draw_laser_circle(
                ax, _lr,
                color      = self.laser_color,
                lw         = self.laser_linewidth,
                ls         = self.laser_linestyle,
                halo       = self.laser_halo,
                halo_color = self.laser_halo_color,
            )

    def update(self, frame: int) -> tuple:
        seq = self._get_seq_result()
        img = (self.scan.load_frame(frame)
               if hasattr(self.scan, "load_frame") else self.scan[frame])
        self._im.set_data(np.asarray(img, float))

        r = seq.frames[frame]

        # Update contours — replace lines (variable length per frame)
        for line in self._contour_lines:
            line.remove()
        ax = self._im.axes
        self._contour_lines = []
        for contour in r.contours:
            (line,) = ax.plot(
                contour[:, 1], contour[:, 0],
                color=self.contour_color,
                linewidth=self.contour_lw,
                linestyle=self.contour_ls,
            )
            self._contour_lines.append(line)

        # Update centroid
        self._centroid_pt.set_offsets([[r.x_pixel, r.y_pixel]])

        updated = [self._im, self._centroid_pt, *self._contour_lines]
        # Laser circle is static but must be included so blit redraws it.
        if self._laser_circle is not None:
            updated.append(self._laser_circle)
        return tuple(updated)

# ---------------------------------------------------------------------------
# Power-series spectrum plot
# ---------------------------------------------------------------------------

def plot_power_series(
    scan,
    ax               = None,
    figsize          : tuple  = (6, 4),
    dpi              : int    = None,
    # --- x-axis ---
    x_axis           : str    = "energy",
    x_range          : tuple  = None,
    twin_axis        : bool   = False,
    # --- spectra source ---
    spectra_source   : str    = "best",
    # --- background subtraction (post-load, in addition to loader bg) ---
    bg_region        : tuple  = None,
    # --- sweep selection and stacking ---
    sweep_step       : int    = 1,
    spectrum_offset  : float  = 0.0,
    # --- colour mapping ---
    cmap             : str    = "viridis",
    power_scale      : str    = "linear",
    power_range      : tuple  = None,
    # --- line style ---
    lw               : float  = 1.0,
    alpha            : float  = 1.0,
    alpha_by_power   : bool   = False,
    alpha_min        : float  = 0.2,
    # --- colorbar ---
    colorbar         : bool   = True,
    cb_label         : str    = "Power (µW)",
    cb_labelpad      : float  = 12.0,
    # --- peak marker ---
    peak_marker      : bool   = False,
    peak_marker_color: str    = "red",
    peak_marker_lw   : float  = 1.0,
    peak_marker_ls   : str    = "--",
    # --- axes labels ---
    ylabel           : str    = None,
) -> tuple:
    """
    Plot a power-series of PL spectra with each line coloured by optical power.

    Each sweep in *scan* is drawn as a line whose colour is taken from *cmap*
    mapped linearly (or logarithmically) onto the ``scan.power`` array.  A
    colorbar indicates the optical power scale.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
        Must expose ``power`` (µW), ``energy``/``wavelength``, and the chosen
        *spectra_source* attribute.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure is created when ``None``.
    figsize : tuple
        Figure size in inches (used when *ax* is ``None``).
    dpi : int, optional
        Figure DPI (used when *ax* is ``None``).

    x-axis
    ------
    x_axis : {"energy", "wavelength"}
        Primary x-axis.  Default ``"energy"``.
    x_range : tuple of (x_min, x_max), optional
        Crop to this range before plotting (same units as *x_axis*).
    twin_axis : bool
        When ``True``, add a secondary x-axis on top showing the other unit
        (energy → wavelength or wavelength → energy).  Default ``False``.

    Spectra
    -------
    spectra_source : str
        Which array to use.  One of:

        * ``"best"``  — :attr:`best_energy_spectra` (bg-corrected if
          configured, otherwise raw energy spectra).  **Default.**
        * ``"raw"``   — :attr:`spectra` (wavelength space, raw counts).
        * ``"energy"``— :attr:`energy_spectra` (Jacobian applied if
          configured; no background subtraction).
        * ``"energy_bg"`` — :attr:`energy_spectra_bg` (background-
          subtracted, Jacobian applied if configured).  Requires
          ``bg_region`` at load time.
        * ``"energy_pre_jacobian"`` — :attr:`energy_spectra_pre_jacobian`
          (always without Jacobian correction).

    bg_region : tuple of (x_min, x_max), optional
        Additional background region subtracted *after* loading (same units
        as *x_axis*).  Applied on top of any background already baked into
        *spectra_source*.  ``None`` (default) skips this step.
    sweep_step : int
        Plot every *sweep_step*-th sweep, starting from the first.  ``1``
        (default) plots all of them, ``2`` every other one, and so on.  Must
        be a positive integer.  Thinning the lines does not change the
        colorbar, which always spans the whole scan's power range.
    spectrum_offset : float
        Stack the plotted spectra by adding a cumulative vertical shift, in
        the units of the plotted array: the first drawn spectrum is shifted by
        ``0``, the second by ``spectrum_offset``, the *j*-th by
        ``j * spectrum_offset``.  Negative values stack downward.  ``0.0``
        (default) draws them overlapping on a shared baseline.

        The shift counts drawn spectra, not sweep indices, so it closes the
        gaps a *sweep_step* leaves rather than stacking blank space.  It is
        absolute, so a value that separates one scan will not suit another
        whose counts differ by orders of magnitude — read a spectrum's peak
        height off an unstacked plot first.  The y tick *values* are no longer
        absolute signal once a shift is applied, which *ylabel* reflects; the
        spacing between them still is.  Hide them with
        ``ax.set_yticks([])`` if the numbers distract.

    Colour mapping
    --------------
    cmap : str
        Matplotlib colormap name.  Default ``"viridis"``.
    power_scale : {"linear", "log"}
        Colormap normalisation.  Default ``"linear"``.
    power_range : tuple of (p_min, p_max), optional
        Clip the colormap to this power range (µW).  Defaults to
        ``(scan.power.min(), scan.power.max())``.

    Line style
    ----------
    lw : float
        Line width.
    alpha : float
        Global line opacity (0–1).  Ignored when *alpha_by_power* is ``True``.
    alpha_by_power : bool
        Scale each line's alpha linearly from *alpha_min* (lowest power) to
        1.0 (highest power).  Overrides *alpha*.
    alpha_min : float
        Minimum alpha used when *alpha_by_power* is ``True``.

    Colorbar
    --------
    colorbar : bool
        Show a colorbar.  Default ``True``.
    cb_label : str
        Colorbar axis label.  Default ``"Power (µW)"``.
    cb_labelpad : float
        Padding between colorbar tick labels and the axis label.

    Peak marker
    -----------
    peak_marker : bool
        Overlay a vertical dashed line at the peak of each spectrum.
        Default ``False``.
    peak_marker_color : str
    peak_marker_lw : float
    peak_marker_ls : str

    Axes
    ----
    ylabel : str, optional
        Y-axis label.  ``None`` (default) takes it from the scan's
        spectroscopy type, so a reflectance scan is not labelled as PL; a
        contrast *spectra_source* is labelled as the ratio it is, and the axis
        is marked offset and arbitrary when *spectrum_offset* is non-zero.  A
        string is used **verbatim**, so include the unit.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Primary axes.
    cb : matplotlib.colorbar.Colorbar or None
        Colorbar object, or ``None`` when *colorbar* is ``False``.
    lines : list of matplotlib.lines.Line2D
        One Line2D per *drawn* sweep, in sweep order — so ``lines[j]`` is the
        spectrum taken at ``scan.power[::sweep_step][j]``.  Their y data
        includes any *spectrum_offset*.

    Raises
    ------
    ValueError
        If *sweep_step* is not a positive integer.
    """
    from .constants import HC_EV_NM  # local import to avoid circular at module level

    if not isinstance(sweep_step, (int, np.integer)) or sweep_step < 1:
        raise ValueError(
            f"sweep_step must be a positive integer, got {sweep_step!r}.  "
            f"Use 1 to plot every sweep, 2 for every other one, and so on."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    # --- x-axis array and label -------------------------------------------
    x, xlabel = _resolve_x_axis(scan, x_axis)

    # --- spectra array (n_pixels, n_sweeps) --------------------------------
    data = _resolve_spectra(scan, spectra_source, x_axis)

    # --- optional post-load background subtraction -------------------------
    if bg_region is not None:
        data = processing.subtract_background(data, bg_region=bg_region, x=x, axis=0)

    # --- optional spectral crop -------------------------------------------
    if x_range is not None:
        mask   = (x >= x_range[0]) & (x <= x_range[1])
        x      = x[mask]
        data   = data[mask, :]

    # --- colour norm ----------------------------------------------------------
    power   = np.asarray(scan.power, dtype=float)
    p_min, p_max = power_range if power_range is not None else (power.min(), power.max())

    if power_scale == "log":
        norm = LogNorm(vmin=max(p_min, 1e-12), vmax=p_max)
    else:
        norm = Normalize(vmin=p_min, vmax=p_max)

    sm      = ScalarMappable(norm=norm, cmap=get_cmap(cmap))
    sm.set_array([])   # required for standalone colorbars

    # --- per-line alpha if requested ---------------------------------------
    power_norm_linear = (power - p_min) / max(p_max - p_min, 1e-12)

    # --- draw lines --------------------------------------------------------
    # Two counters, and they differ once sweep_step > 1: i indexes the scan, so
    # colour and alpha keep tracking each line's own power, while j counts drawn
    # lines, so the offsets stack contiguously instead of leaving gaps where a
    # skipped sweep would have been.
    lines = []
    for j, i in enumerate(range(0, len(power), sweep_step)):
        colour = sm.to_rgba(power[i])
        a = float(alpha_min + (1.0 - alpha_min) * power_norm_linear[i]) \
            if alpha_by_power else float(alpha)
        y = data[:, i] + j * spectrum_offset
        (line,) = ax.plot(x, y, color=colour, lw=lw, alpha=a)
        lines.append(line)

        if peak_marker:
            # argmax is invariant to the additive offset, so the un-shifted
            # spectrum would give the same position.
            x_peak = x[np.argmax(y)]
            ax.axvline(
                x_peak,
                color=peak_marker_color,
                lw=peak_marker_lw,
                ls=peak_marker_ls,
                alpha=a,
            )

    # --- axes formatting --------------------------------------------------
    if ylabel is None:
        # The source decides the quantity -- a contrast is ΔR/R₀, not counts.
        name, unit = _signal_name_unit(scan, spectra_source)
        if spectrum_offset:
            # Stacking destroys the absolute scale, so the unit goes.  A
            # dimensionless signal has none to drop and is only marked as
            # shifted.
            unit = "a.u., offset" if unit else "offset"
        ylabel = f"{name} ({unit})" if unit else name

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # --- twin axis --------------------------------------------------------
    ax_twin = None
    if twin_axis:
        ax_twin = ax.twiny()
        if x_axis == "energy":
            # top axis in nm; tick positions derived from bottom energy ticks
            e_ticks = ax.get_xticks()
            # keep only ticks in range to avoid division by zero / overflow
            e_ticks = e_ticks[(e_ticks > 0) & (e_ticks >= x.min()) & (e_ticks <= x.max())]
            wl_ticks = HC_EV_NM / e_ticks
            ax_twin.set_xlim(ax.get_xlim())
            ax_twin.set_xticks(e_ticks)
            ax_twin.set_xticklabels([f"{w:.0f}" for w in wl_ticks])
            ax_twin.set_xlabel("Wavelength (nm)")
        else:
            wl_ticks = ax.get_xticks()
            wl_ticks = wl_ticks[(wl_ticks > 0) & (wl_ticks >= x.min()) & (wl_ticks <= x.max())]
            e_ticks  = HC_EV_NM / wl_ticks
            ax_twin.set_xlim(ax.get_xlim())
            ax_twin.set_xticks(wl_ticks)
            ax_twin.set_xticklabels([f"{e:.3f}" for e in e_ticks])
            ax_twin.set_xlabel("Energy (eV)")

    # --- colorbar ---------------------------------------------------------
    cb = None
    if colorbar:
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label(cb_label, labelpad=cb_labelpad)

    return fig, ax, cb, lines