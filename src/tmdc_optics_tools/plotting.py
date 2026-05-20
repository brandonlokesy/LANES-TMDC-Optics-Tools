# tmdc_optics_tools/plotting.py
"""
Plotting helpers for TMD spectroscopy.

Provides a consistent Matplotlib style and convenience functions for
the most common plot types encountered in gate-dependent PL experiments.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from skimage.exposure import rescale_intensity

from . import processing

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


# ---------------------------------------------------------------------------
# 2-D map plots
# ---------------------------------------------------------------------------

def plot_pl_map_Vab_scan(
    scan,
    ax             = None,
    figsize        : tuple = (6, 4),
    dpi            : int   = None,
    x_axis         : str   = "energy",
    cmap           : str   = "vik",
    median_kernel  : int   = 3,
    clim           : tuple = None,
    colorbar       : bool  = True,
    colorbar_label : str   = "PL intensity (counts)",
    rescale_img    : bool  = True,
) -> tuple:
    """
    Plot a gate-dependent PL map from an
    :class:`~tmdc_optics_tools.loaders.AttoCubePLVabScan`.

    Background subtraction and Jacobian correction are configured at
    load time on the scan object (via ``bg_region_nm``, ``bg_region_eV``,
    and ``apply_jacobian``).  This function always uses
    :attr:`~tmdc_optics_tools.loaders.AttoCubePLVabScan.best_energy_spectra`,
    which automatically returns the background-corrected array when one
    is available, and falls back to the uncorrected array otherwise.

    Parameters
    ----------
    scan : AttoCubePLVabScan
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
    colorbar_label : str
    rescale_img : bool
        Rescale intensity to [0, 1] before plotting.

    Returns
    -------
    fig, ax, mesh
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    x, xlabel = _resolve_x_axis(scan, x_axis)
    y, ylabel  = scan.gate_axis, scan.gate_axis_label

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
        cb.set_label(colorbar_label)

    return fig, ax, mesh


# ---------------------------------------------------------------------------
# Spectrum plots
# ---------------------------------------------------------------------------

def plot_spectrum(
    scan,
    sweep_index : int,
    ax          = None,
    figsize     : tuple = (5, 3),
    dpi         : int   = None,
    x_axis      : str  = "energy",
    normalize   : bool = False,
    label       : str  = None,
    **line_kwargs,
) -> tuple:
    """
    Plot a single PL spectrum from a scan.

    Parameters
    ----------
    scan : AttoCubePLScan
    sweep_index : int
        Index of the sweep point to plot.
    ax : matplotlib.axes.Axes, optional
    x_axis : {"energy", "wavelength"}
    normalize : bool
        Normalise spectrum to its peak value.
    label : str, optional
        Legend label. Defaults to the gate voltage / field value.
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
    if normalize:
        y = y / y.max()

    if label is None:
        label = (
            f"$E_F$ = {scan.ef[sweep_index]:.1f} mV/nm"
            if scan.ef is not None
            else f"$V_{{top}}$ = {scan.v_top[sweep_index]:.2f} V"
        )

    line, = ax.plot(x, y, label=label, **line_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PL intensity (norm.)" if normalize else "PL intensity (counts)")

    return fig, ax, line


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
    scan : AttoCubePLScan
    ax : matplotlib.axes.Axes, optional
        Must be a standard (non-twin) axes.
    ef_axis : bool
        Use displacement field on the x-axis if available.
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

    if ef_axis and scan.ef is not None:
        x, xlabel = scan.ef, r"$E_F$ (mV/nm)"
    else:
        x, xlabel = scan.v_top, r"$V_\mathrm{top}$ (V)"

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
    xlabel : str = "x-axis (um)",
    ylabel : str = "y-axis (um)",
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
        lr = scan.laser_ref
        ax.add_patch(patches.Circle(
            (lr.center_x, lr.center_y), radius=lr.radius,
            edgecolor="red", facecolor="none",
            linewidth=1, linestyle="--",
            label=f"Laser Spot (1/e² Radius: {lr.radius:.1f} px)",
        ))

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