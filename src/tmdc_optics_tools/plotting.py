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

from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from skimage.exposure import rescale_intensity

from . import processing
from . import diffusion as _diffusion
from .constants import HC_EV_NM, _x_axis_name_unit
# The spectra- and frame-source registries name attributes on the loader classes,
# so they live with them; the resolvers are imported here because this is where
# callers of ``spectra_source=`` and ``frame_source=`` are.
from .loaders import _SPECTRA_SOURCES, _resolve_spectra, _resolve_frame

# Optional colormap packages (pip install "tmdc_optics_tools[colormaps]").
# Imported for their side effect alone: each registers its colormaps into
# Matplotlib's registry under a prefix — "cmc.vik", "cmo.thermal" — which is how
# get_cmap reaches them.  Nothing below refers to either package by name, so the
# import is the whole point and must not be tidied away as unused.
try:
    import cmcrameri            # noqa: F401  — registers the "cmc.*" names
except ImportError:
    pass

try:
    import cmocean              # noqa: F401  — registers the "cmo.*" names
except ImportError:
    pass

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

#: Anything accepted wherever this module takes a ``cmap``: a colormap name, a
#: Matplotlib colormap, or a sequence of colours.  Resolved by :func:`get_cmap`.
ColormapLike = Union[str, mcolors.Colormap, list, tuple, np.ndarray]


def get_cmap(cmap: ColormapLike = "magma") -> mcolors.Colormap:
    """
    Resolve a colormap specification to a Matplotlib colormap.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or sequence of colours
        A **name** is resolved through Matplotlib's colormap registry, so any
        Matplotlib name works (``"magma"``, ``"gray"``, …).  cmcrameri and
        cmocean register into that same registry under a **prefix**, so their
        names carry it: ``"cmc.vik"``, ``"cmo.thermal"``.  Bare ``"vik"`` and
        ``"thermal"`` are not registered by either package and will not
        resolve.

        A **colormap** is returned unchanged, so anything producing one can be
        passed directly — ``seaborn.color_palette(..., as_cmap=True)``,
        ``cmocean.cm.thermal``, ``cmcrameri.cm.vik``, a hand-built
        :class:`~matplotlib.colors.LinearSegmentedColormap`.

        A **sequence of colours** — hex strings, named colours, RGB(A) tuples,
        or an ``(n, 3)`` / ``(n, 4)`` array, as returned by
        ``seaborn.color_palette(...)`` without ``as_cmap=True`` — is wrapped in
        a :class:`~matplotlib.colors.ListedColormap`.

    Returns
    -------
    matplotlib.colors.Colormap

    Raises
    ------
    TypeError
        If *cmap* is neither a name, a colormap, nor a sequence of colours.
    ValueError
        If *cmap* names an unknown colormap, or is an empty sequence.

    Notes
    -----
    A sequence of *n* colours becomes *n* discrete bands, not a continuous
    ramp — the colours are used as given rather than interpolated between.  For
    a smooth colormap from a seaborn palette, ask the palette for one with
    ``as_cmap=True``.

    A prefixed name only resolves once the package that owns it has been
    imported, since registration is an import side effect.  Importing this
    module imports cmcrameri and cmocean when they are installed, so the
    prefixed names are available without importing them yourself.  Passing the
    colormap object depends on no such ordering.

    Examples
    --------
    >>> get_cmap("magma")                                    # doctest: +SKIP
    >>> get_cmap("cmo.thermal")                              # doctest: +SKIP
    >>> get_cmap(cmocean.cm.thermal)                         # doctest: +SKIP
    >>> get_cmap(sns.color_palette("rocket", as_cmap=True))  # doctest: +SKIP
    >>> get_cmap(sns.color_palette("rocket", 8))           # 8 bands  # doctest: +SKIP
    >>> get_cmap(["#1b9e77", "#d95f02", "#7570b3"])        # 3 bands  # doctest: +SKIP
    """
    if isinstance(cmap, mcolors.Colormap):
        return cmap

    if isinstance(cmap, str):
        # One registry, so there is no precedence rule to get wrong.  Looking
        # bare third-party names up here instead would silently shadow
        # Matplotlib: cmocean and Matplotlib both define "gray", and cmcrameri
        # and Matplotlib both define "berlin", "managua" and "vanimo".
        return plt.get_cmap(cmap)

    # Everything else is read as a sequence of colours.  to_rgba_array is the
    # validator as well as the converter: it takes hex strings, named colours,
    # RGB(A) tuples and (n, 3)/(n, 4) arrays alike, and rejects anything that is
    # not one of those.
    try:
        colours = mcolors.to_rgba_array(cmap)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"cmap must be a colormap name, a matplotlib Colormap, or a sequence "
            f"of colours.  Could not read {type(cmap).__name__} as a sequence of "
            f"colours: {exc}"
        ) from exc

    if len(colours) == 0:
        raise ValueError("cmap is an empty sequence of colours.")

    return mcolors.ListedColormap(colours)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_x_axis(scan, x_axis: str) -> tuple:
    """
    Return ``(x_array, xlabel_string)`` for a scan.

    Centralises the repeated ``"energy"`` / ``"wavelength"`` branching so
    every plotting function can call this instead of duplicating the logic.
    """
    name, unit = _x_axis_name_unit(x_axis)
    values     = scan.energy if x_axis == "energy" else scan.wavelength
    return values, f"{name} ({unit})"


#: The conjugate of each spectral x-axis: what a top axis shows, and its label.
_CONJUGATE_AXIS = {"energy": "wavelength", "wavelength": "energy"}


def _conjugate_x_axis(ax, x_axis: str):
    """
    Add a top x-axis showing the other spectral unit, and return it.

    Energy and wavelength are reciprocal through ``HC_EV_NM``, and ``HC_EV_NM / x``
    is its own inverse, so one function serves both directions of the transform.

    Built with ``secondary_xaxis`` rather than ``twiny`` plus relabelled ticks.
    Matplotlib then chooses the ticks in the *displayed* unit, so the nm labels come
    out at round wavelengths instead of wherever the eV ticks happened to fall, and
    the axis follows any later change to the primary limits instead of freezing at
    the ticks that existed when it was built.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Its x-axis must already be in *x_axis* units.
    x_axis : {"energy", "wavelength"}
        The **primary** axis' quantity.  The secondary axis shows the other one.

    Returns
    -------
    matplotlib.axis.SecondaryAxis
    """
    name, unit = _x_axis_name_unit(_CONJUGATE_AXIS[x_axis])

    def _convert(values):
        # Matplotlib evaluates the transform across the whole axis, including 0,
        # where the reciprocal is undefined; without this the first draw emits a
        # divide-by-zero RuntimeWarning that a -W error run would fail on.
        with np.errstate(divide="ignore", invalid="ignore"):
            return HC_EV_NM / np.asarray(values, dtype=float)

    secondary = ax.secondary_xaxis("top", functions=(_convert, _convert))
    secondary.set_xlabel(f"{name} ({unit})")
    return secondary


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
    if source == "contrast":
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
    cmap           : ColormapLike = "magma",
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

    Corrections are configured at load time on the scan object (via
    ``bg_region_nm``, ``bg_region_eV``, ``apply_jacobian`` and ``cosmic_rays``).
    This function plots the most-corrected array the scan has for the chosen axis
    — :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.best_energy_spectra`
    or :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.best_spectra` — so
    a declared background or cosmic-ray repair reaches the map on either axis.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    x_axis : {"energy", "wavelength"}
    cmap : str, Colormap, or sequence of colours
        Passed to :func:`get_cmap`.
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

    # Copied because the filters below build on it; the resolver itself does not.
    data = _resolve_spectra(scan, "best", x_axis).copy()

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

def _coordinate_text(label: str, value: float, unit: str) -> str:
    """
    ``"Top gate (V) = 1.5"`` — one coordinate named, with its unit when known.

    The same composition ``sweep_axis_label`` uses, so every legend this module
    writes reads the same whichever axis the point was addressed on.
    """
    named = f"{label} ({unit})" if unit else label
    return f"{named} = {float(value):.4g}"


# The coordinate each position keyword pairs with, for messages that offer the
# other spelling.
_COORDINATE_FOR_POSITION = {
    "index":      "value=",
    "index_fast": "fast=",
    "index_slow": "slow=",
}


def _select_sweep_point(scan, value, axis, index, fast, slow,
                        index_fast, index_slow, what: str,
                        passthrough: dict = None) -> int:
    """
    Resolve one point request into a single integer sweep position.

    Mirrors the loader's two accessors: *value* / *fast* / *slow* are
    coordinates and *index* / *index_fast* / *index_slow* are integer positions,
    so the two spellings never share a keyword and a request cannot be half of
    each.  The lookup itself is the scan's, so an ambiguous coordinate is refused
    and a distant one warns exactly as they do for ``get_spectrum_at``.

    *passthrough* is the caller's unmatched keyword dict, named in the no-point
    error.  A selector spelled wrongly is absorbed there rather than rejected, so
    without this the caller is told they named no point while looking at the one
    they thought they had named.
    """
    # None is the unspecified default, not a quantity to look up. Without this it
    # would reach the scan, which reads an undeclared axis as the flat index — so
    # a coordinate would be silently searched against 0, 1, 2, … instead.
    if axis is None:
        axis = "sweep"

    named_by_value = [n for n, v in (("value", value), ("fast", fast),
                                     ("slow", slow)) if v is not None]
    named_by_index = [(n, v) for n, v in (("index", index),
                                          ("index_fast", index_fast),
                                          ("index_slow", index_slow))
                      if v is not None]
    index_names = [n for n, _ in named_by_index]

    if named_by_value and named_by_index:
        raise ValueError(
            f"{what}: name the point by value ({', '.join(named_by_value)}) or "
            f"by position ({', '.join(index_names)}), not both."
        )
    if not named_by_value and not named_by_index:
        # Every selector is keyword-only, so a misspelt or renamed one lands in
        # the style passthrough instead of raising. Name what arrived: the usual
        # cause of "no point" is a point named under a keyword that is not one.
        stray = ""
        if passthrough:
            listed = ", ".join(f"{k}={v!r}" for k, v in passthrough.items())
            stray = (f" Received {listed}, which names no point — forwarded to "
                     f"ax.plot as a line property.")
        raise ValueError(
            f"{what} needs a point: value= for a coordinate on the sweep axis, "
            f"or index= for a position. For a declared nest give fast= and "
            f"slow= (coordinates) or index_fast= and index_slow= (positions)."
            f"{stray}"
        )
    if named_by_index and axis != "sweep":
        raise ValueError(
            f"{what}: axis={axis!r} names the quantity a *coordinate* is read "
            f"against, so it does not apply to {', '.join(index_names)}. "
            f"Give the point as a value, or drop axis=."
        )

    for name, given in named_by_index:
        # A position must be exact. The scan would take int(1.9) and plot point 1
        # without comment, and a fractional position is far more likely to be a
        # coordinate that reached the wrong keyword.
        if not isinstance(given, (int, np.integer)):
            raise TypeError(
                f"{what}: {name}={given!r} selects by position, which needs an "
                f"integer. {given!r} looks like a coordinate — pass it as "
                f"{_COORDINATE_FOR_POSITION[name]} to look it up by value, or "
                f"round it if you did mean a position."
            )

    # The scan's own refusals for these two are written for its accessors, where
    # fast=/slow= are whichever spelling that method takes. Here they are always
    # coordinates, so its advice would name the wrong keyword — and following it
    # succeeds, selecting a different point in silence. Refuse first, in this
    # function's vocabulary.
    if index is not None and scan.is_nested:
        raise ValueError(
            f"{what}: this sweep is a declared nest ({scan.nesting}), so a "
            f"single position does not locate a point. Name both axes with "
            f"index_fast= and index_slow=, or address it by coordinate with "
            f"fast= and slow=."
        )
    if (index_fast is not None or index_slow is not None) and not scan.is_nested:
        raise ValueError(
            f"{what}: index_fast= and index_slow= need a declared nest, and this "
            f"sweep is flat ({scan.n_sweeps} points). Use index= for a position "
            f"on the sweep axis, or declare the nest with fast_sweep= and "
            f"slow_sweep= at load time."
        )

    if named_by_value:
        selector = scan._sweep_selector(value, axis=axis, fast=fast, slow=slow,
                                        by_value=True, what=what)
    else:
        selector = scan._sweep_selector(index, fast=index_fast, slow=index_slow,
                                        by_value=False, what=what)

    # A slice means one nest axis was left free, so the accessor would hand back
    # (n_pixels, n) — a line per point along it, which this function has no
    # return contract for.  Naming both axes pins the single spectrum it draws.
    if not isinstance(selector, (int, np.integer)):
        raise ValueError(
            f"{what}: leaving one nest axis free selects every point along the "
            f"other, which is more than one spectrum. Name both axes to pin "
            f"one, or take the block from scan.get_spectrum_at() and plot its "
            f"columns."
        )
    return int(selector)


def _sweep_point_label(scan, idx: int, axis: str) -> str:
    """
    Legend text naming the coordinate a point was addressed on.

    A point on a nest is named by both its coordinates and one reached through
    *axis* by that quantity, so the legend states what was asked for.  On a nest
    the declared sweep axis is normally the flat index, which would say nothing.
    """
    if scan.is_nested:
        nest = scan.nesting
        # Points run n_fast inside n_slow, so the flat position divides back onto
        # the grid — the same arithmetic the selector used to build it.
        i_slow, i_fast = divmod(idx, nest.n_fast)
        return (f"{_coordinate_text(nest.fast_label, nest.fast_axis[i_fast], nest.fast_unit)}, "
                f"{_coordinate_text(nest.slow_label, nest.slow_axis[i_slow], nest.slow_unit)}")

    if axis != "sweep":
        values, lbl, unit = scan._lookup_axis(axis)
        return _coordinate_text(lbl, values[idx], unit)

    # Fall back to whatever the scan says it swept rather than to a gate
    # voltage: a gate role needs a declared wiring, and the sweep axis is
    # already the scan's own answer to "what varied", labelled and in its own
    # units.  A field needs a geometry and two declared gates, so check the
    # device first — reading scan.ef without them raises.
    if scan.is_dual_gated and scan.ef is not None:
        return f"$E_F$ = {scan.ef[idx]:.1f} mV/nm"
    return _coordinate_text(scan.sweep_label, scan.sweep_axis[idx],
                            scan.sweep_unit)


def plot_spectrum(
    scan,
    *,
    value      : float = None,
    axis       : str   = "sweep",
    index      : int   = None,
    fast       : float = None,
    slow       : float = None,
    index_fast : int   = None,
    index_slow : int   = None,
    ax         = None,
    figsize    : tuple = (5, 3),
    dpi        : int   = None,
    x_axis     : str   = "energy",
    normalize  : bool  = False,
    label      : str   = None,
    ylabel     : str   = None,
    **line_kwargs,
) -> tuple:
    """
    Plot one spectrum from a sweep, chosen by coordinate or by position.

    The point is named the way the measurement was:
    ``plot_spectrum(scan, value=2.5)`` takes the sweep point nearest 2.5 in the
    sweep axis's own units.  Integer positions remain available through *index*.

    Every selector is keyword-only, so a call always states which kind it means.
    A bare number could be either, and on a sweep whose coordinates span the same
    range as its positions — a power sweep in µW, say — neither the value nor a
    warning would reveal which was taken.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    value : float, optional
        Coordinate on the sweep axis, in that axis's units.  For a flat sweep;
        a nest is addressed with *fast* and *slow*.
    axis : str
        Which quantity *value* is read against, spelled as ``sweep=`` spells it:
        a registry key such as ``"top_voltage"`` or a raw row label such as
        ``"V_A"``.  The default searches the declared sweep axis.  Use it when a
        sweep is declared in one coordinate and you want a point in another — a
        field sweep driven by both gates at a fixed ratio can be addressed by
        ``axis="top_voltage"``.  Applies to coordinates only; ``None`` means the
        default.
    index : int, optional
        Position on the sweep axis.  Negative counts from the end.  For a flat
        sweep; a nest is addressed with *index_fast* and *index_slow*.
    fast, slow : float, optional
        Coordinates on the nest axes.  Give both — one spectrum is drawn, so
        leaving an axis free is refused.
    index_fast, index_slow : int, optional
        Positions on the nest axes, the integer spelling of *fast* / *slow*.
        A position must be a whole number; a fractional one is refused rather
        than truncated, since it is more likely a coordinate that reached the
        wrong keyword.
    ax : matplotlib.axes.Axes, optional
    x_axis : {"energy", "wavelength"}
    normalize : bool
        Normalise spectrum to its peak value.
    label : str, optional
        Legend label.  Defaults to the coordinate the point was addressed on,
        with its unit; both coordinates for a nest.
    ylabel : str, optional
        Y-axis label.  Derived from the scan's measurement type when ``None``,
        so a reflectance sweep is not labelled as PL.  A string is used
        **verbatim**, so include the unit.
    **line_kwargs
        Passed directly to ``ax.plot``.  A keyword that is not a selector lands
        here, so the no-point error names whatever arrived.

    Returns
    -------
    fig, ax, line

    Raises
    ------
    ValueError
        If the point is named both ways at once, or not at all; if a position
        spelling does not match the sweep's shape; if a coordinate names more
        than one sweep point, since drawing one of them would drop the rest
        without saying so; or if one nest axis is left free, which selects more
        than one spectrum.
    TypeError
        If a position is not a whole number, or if the point is given
        positionally rather than as ``value=`` or ``index=``.

    Warns
    -----
    UserWarning
        When a requested coordinate is further than half a step from any real
        point.  A nearest-value lookup cannot fail, so it returns a real
        spectrum from somewhere else; the warning names what was used.

    See Also
    --------
    tmdc_optics_tools.loaders.AttoCubeSpectralSweep.get_spectrum_at :
        the same selection, returning the array instead of drawing it.
    plot_single_spectrum : plot a spectrum that is not part of a sweep.

    Examples
    --------
    >>> plot_spectrum(scan, value=2.5)                     # doctest: +SKIP
    >>> plot_spectrum(scan, value=15.0, axis="top_voltage")  # doctest: +SKIP
    >>> plot_spectrum(scan, index=-1)                      # doctest: +SKIP
    >>> plot_spectrum(scan, fast=2.5, slow=100.0)          # doctest: +SKIP
    """
    # Resolved before the label is built as well as before the lookup: an
    # unresolved None reads as the flat index there too, and would name the
    # legend after an axis that was never searched.
    if axis is None:
        axis = "sweep"

    sweep_index = _select_sweep_point(
        scan, value, axis, index, fast, slow, index_fast, index_slow,
        what="plot_spectrum()", passthrough=line_kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    x, xlabel = _resolve_x_axis(scan, x_axis)
    y         = _resolve_spectra(scan, "best", x_axis)[:, sweep_index]
    if normalize:
        y = y / y.max()

    if label is None:
        label = _sweep_point_label(scan, sweep_index, axis)

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
        ``best_energy_spectra`` attributes. The most-corrected array available is
        used automatically, on whichever axis is plotted.
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    x_axis : {"wavelength", "energy"}
        Axis to plot against. Default ``"wavelength"`` (the native axis).
    normalize : bool
        Normalise the spectrum to its peak value.
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
    y = _resolve_spectra(spectrum, "best", x_axis)
    if normalize:
        y = y / y.max()

    line, = ax.plot(x, y, label=label, **line_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None
                  else _signal_label(spectrum, normalized=normalize))
    if label:
        ax.legend(frameon=False)

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
    color_power : str  = "C2",
) -> tuple:
    """
    Plot electrode currents and excitation power vs. electric field (or gate
    voltage) to check for dielectric breakdown.

    One trace per electrode the scan declared and for which a current was
    recorded, labelled by role: a dual-gated device gives the two gate leakage
    currents, a contacted single-gated one gives its gate leakage and the
    transport current into the TMDC.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    ax : matplotlib.axes.Axes, optional
        Must be a standard (non-twin) axes.
    ef_axis : bool
        Use displacement field on the x-axis when the scan can supply one — it
        needs both a :class:`DeviceGeometry` and a declared channel-to-gate
        mapping.  Otherwise the scan's declared sweep axis is used.
    color_power : str
        Matplotlib colour for the power trace.

    Returns
    -------
    fig, ax_left, ax_right, lines
        *lines* holds the current traces in role order, for restyling.

    Raises
    ------
    ValueError
        If the scan declared no electrode mapping, or none of its electrodes has
        a recorded current.  Which electrode a current row belongs to is
        per-session wiring; pass ``gates=`` at load time.
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

    # A declared electrode need not have a recorded current: it may be grounded
    # with no row, or driven by something that is not a source-meter channel.  The
    # scan already encodes those rules, so ask it and skip what it refuses rather
    # than restating the conditions here.
    lines = []
    for attr, label in (("i_top",     r"$I_\mathrm{top}$"),
                        ("i_bot",     r"$I_\mathrm{bot}$"),
                        ("i_channel", r"$I_\mathrm{ch}$")):
        try:
            current = getattr(scan, attr)
        except ValueError:
            continue
        line, = ax_left.plot(x, current, label=label)
        lines.append(line)

    if not lines:
        raise ValueError(
            f"plot_current has nothing to plot for '{scan.path}': no declared "
            f"electrode has a recorded current. Which acquisition channel reached "
            f"which electrode is per-session wiring, so pass it at load time — "
            f"gates={{'top': '<row>', 'bottom': '<row>'}}. A gate declared on a "
            f"row that is not a source-meter channel, or an electrode declared "
            f"grounded with no row, records no current either."
        )

    ax_left.axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel("Current (nA)")

    ax_right = ax_left.twinx()
    ax_right.spines["right"].set_visible(True)
    l_power, = ax_right.plot(x, scan.power, color=color_power, linestyle="--",
                             label="Power")
    ax_right.set_ylabel("Power (µW)")

    ax_left.legend(handles=lines + [l_power], loc="best", frameon=False)
    fig.tight_layout()
    return fig, ax_left, ax_right, lines


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
    cmap   : ColormapLike = "magma",
    frame_source : str = "best",
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
    cmap : str, Colormap, or sequence of colours
        Passed to :func:`get_cmap`.
    frame_source : {``"best"``, ``"raw"``, ``"bg"``}
        Which version of the frame to draw.  ``"best"`` is background-corrected
        when the scan was loaded with a *bg_region* and raw otherwise; ``"raw"``
        is always the file's counts; ``"bg"`` requires a *bg_region* and raises
        without one.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    ax.imshow(_resolve_frame(scan, idx, frame_source), cmap=get_cmap(cmap))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_image(
    image,
    ax             = None,
    figsize        : tuple = (6, 5),
    dpi            : int   = None,
    cmap           : ColormapLike = "magma",
    colorbar       : bool  = True,
    colorbar_label : str   = None,
    rescale_img    : bool  = False,
    clim           : tuple = None,
    xlabel         : str   = "x (px)",
    ylabel         : str   = "y (px)",
    show_axes      : bool  = True,
) -> tuple:
    """
    Plot a single 2-D image with a colormap and an optional colorbar.

    Parameters
    ----------
    image : np.ndarray or object with ``.img``
        A 2-D array, or any object exposing a 2-D ``img`` attribute
        (e.g. :class:`~tmdc_optics_tools.loaders.SingleImage`,
        :class:`~tmdc_optics_tools.loaders.AttoCubeSampleImage`).
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    cmap : str, Colormap, or sequence of colours
        Passed to :func:`get_cmap`.
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

    Returns
    -------
    fig, ax, im
    """
    img = image.img if hasattr(image, "img") else np.asarray(image)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    if rescale_img:
        img = rescale_intensity(img, in_range="image", out_range=(0, 1))

    vmin, vmax = clim if clim is not None else (None, None)
    im = ax.imshow(img, cmap=get_cmap(cmap), vmin=vmin, vmax=vmax)

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
    cmap             : ColormapLike = "magma",
    frame_source     : str  = "best",
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
    cmap : str, Colormap, or sequence of colours
        Passed to :func:`get_cmap`.
    frame_source : {``"best"``, ``"raw"``, ``"bg"``}
        Which version of each frame to draw.  ``"best"`` is background-corrected
        when the scan was loaded with a *bg_region* and raw otherwise, so
        animating the same scan twice as ``"raw"`` and ``"bg"`` shows what the
        subtraction removed.

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
    fig, ax = plot_real_space_PL_map(scan, ax, idx=0, xlabel=xlabel, ylabel=ylabel,
                                     cmap=(cmap), frame_source=frame_source)
    im = ax.images[0] if ax.images else ax.imshow(
        _resolve_frame(scan, 0, frame_source), cmap=get_cmap(cmap)
    )

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
        im.set_data(_resolve_frame(scan, frame, frame_source))
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

    * :meth:`init_artists` — draw the animation's first frame onto a given axes
      and stash the dynamic artists.  It receives *frames*, the sequence of
      frame indices the animation will show, so the panel can fix its axes
      limits over exactly those frames (preventing the autoscale "jump" you
      would otherwise get as frames advance).  The first frame to draw is
      ``frames[0]``, which is not necessarily ``0``.
    * :meth:`update` — mutate the stored artists for ``frame`` and return the
      ones that changed.  The engine redraws in full rather than blitting, so the
      returned artists are not what makes the animation work; they are what lets
      a caller drive the panels itself, and what documents which artists a panel
      owns.

    **Frame indices are always the panel's own.** ``frames`` may be any subset,
    in any order — a window, a stride, a single frame — but every value in it
    indexes the panel's data directly, and :meth:`update` is handed those same
    values.  A panel therefore never tracks where a window started, and never
    offsets an index; it keeps its full arrays and reads them at ``frame``.
    Getting that wrong is silent — the animation plays real frames in a
    plausible order while the shared title names different ones — which is why
    the engine passes native indices rather than positions within the window.

    The :attr:`n_frames` property reports the panel's number of frames.  Every
    panel in one figure must report the same count; the engine refuses a
    mismatch rather than quietly animating the shortest.
    """

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    def init_artists(self, ax, frames) -> None:
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
    cmap : str, Colormap, or sequence of colours
        Passed to :func:`get_cmap` via :func:`plot_real_space_PL_map`.
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
    frame_source : {``"best"``, ``"raw"``, ``"bg"``}
        Which version of each frame to draw.  ``"best"`` is background-corrected
        when the scan was loaded with a *bg_region* and raw otherwise.
    """

    def __init__(
        self,
        scan,
        title            : str   = "",
        cmap             : ColormapLike = "magma",
        laser_annotation : bool  = True,
        laser_color      : str   = "red",
        laser_linewidth  : float = 1.5,
        laser_linestyle  : str   = "-",
        laser_halo       : bool  = True,
        laser_halo_color : str   = "white",
        xlabel           : str   = "x-axis (pixels)",
        ylabel           : str   = "y-axis (pixels)",
        frame_source     : str   = "best",
    ):
        self.scan             = scan
        self.title            = title
        self.cmap             = cmap
        self.frame_source     = frame_source
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

    def init_artists(self, ax, frames) -> None:
        plot_real_space_PL_map(
            self.scan, ax=ax, idx=frames[0], cmap=self.cmap,
            xlabel=self.xlabel, ylabel=self.ylabel,
            frame_source=self.frame_source,
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
        self._im.set_data(_resolve_frame(self.scan, frame, self.frame_source))
        return (self._im,)


class SpectrumLinePanel(AnimationPanel):
    """
    A panel that animates one PL spectrum per frame.

    Wraps an :class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep` (or any object
    exposing ``energy``/``wavelength`` plus ``best_energy_spectra``/``best_spectra``
    of shape ``(n_pixels, n_sweeps)``).  The x-axis is fixed; each frame swaps
    the y-values and updates a per-panel subtitle showing the swept value.

    Both axes limits are fixed once over the frames being animated, so the trace
    does not rescale or jump between frames.

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
        Line colour.  Matplotlib default when ``None``.  Cannot be combined with
        *cmap*, which sets the colour itself every frame.
    cmap : ColormapLike, optional
        Encode each frame's **peak** value as the trace's colour, and draw a
        colour bar for it.  ``None`` (default) leaves the line one colour, adds
        no colour bar, and takes no axes space — an encoding is a claim about the
        data, so it is asked for rather than assumed.
        The scale spans the peaks of the **whole scan**, not of the frames being
        animated, so the colour means "this frame's brightness" and two clips of
        one scan can be compared.  Values are read through the same corrected
        arrays the trace is drawn from.
    ylabel : str, optional
        Y-axis label.  Derived from the scan's measurement type when ``None``,
        so a reflectance sweep is not labelled as PL.  A string is used
        **verbatim**, so include the unit.
    colorbar_label : str, optional
        Colour-bar label.  Derived as ``"Peak <signal>"`` when ``None``; a string
        is used **verbatim**.  It is deliberately not the y-axis label: the y-axis
        spans the full data range while the bar spans the range of per-frame
        peaks, so one label on both would put the same words on two scales that
        disagree.
    twin_axis : bool
        Add a top x-axis in the other spectral unit — wavelength above an energy
        axis, energy above a wavelength one.  Default ``False``.

    Attributes
    ----------
    line : matplotlib.lines.Line2D
        The animated trace.  ``None`` until :meth:`init_artists` has run.
    mappable : matplotlib.cm.ScalarMappable
        Carries the colour scale; ``None`` unless *cmap* was given.  Restyle the
        encoding through this — e.g. ``panel.mappable.set_clim(...)`` before
        rendering — rather than through more constructor arguments.
    colorbar : matplotlib.colorbar.Colorbar
        ``None`` unless *cmap* was given.
    ax_twin : matplotlib.axis.SecondaryAxis
        The conjugate top axis; ``None`` unless *twin_axis* was set.
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
        cmap             : ColormapLike = None,
        ylabel           : str  = None,
        colorbar_label   : str  = None,
        twin_axis        : bool = False,
    ):
        if cmap is not None and color is not None:
            raise ValueError(
                "pass either color= or cmap=, not both: cmap sets the line colour "
                "from each frame's peak, so color= would be overwritten on the "
                "first frame and silently ignored thereafter."
            )

        self.scan             = scan
        self.x_axis           = x_axis
        self.sweep_attr       = sweep_attr
        self.sweep_label      = sweep_label if sweep_label is not None else sweep_attr
        self.sweep_unit       = sweep_unit
        self.title_fmt        = title_fmt
        self.show_sweep_title = show_sweep_title
        self.color            = color
        self.cmap             = cmap
        self.ylabel           = ylabel
        self.colorbar_label   = colorbar_label
        self.twin_axis        = twin_axis
        self.line             = None
        self.mappable         = None
        self.colorbar         = None
        self.ax_twin          = None
        self._title           = None
        self._y               = None
        self._sweep_vals      = None

    @property
    def n_frames(self) -> int:
        return self.scan.n_sweeps

    def init_artists(self, ax, frames) -> None:
        x, xlabel = _resolve_x_axis(self.scan, self.x_axis)
        # Both arrays stay full length and are read at the frame's own index, so
        # the panel never has to know where the animated selection began.
        self._y          = np.asarray(_resolve_spectra(self.scan, "best", self.x_axis),
                                      dtype=float)
        self._sweep_vals = np.asarray(getattr(self.scan, self.sweep_attr))

        # (n_pixels, n_shown): only the columns actually animated, so the y-limits
        # frame the traces on screen rather than the whole scan's dynamic range.
        shown = self._y[:, list(frames)]

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(shown.min(), shown.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.ylabel if self.ylabel is not None
                      else _signal_label(self.scan))

        # After set_xlim, so the secondary axis inherits the finished limits.
        if self.twin_axis:
            self.ax_twin = _conjugate_x_axis(ax, self.x_axis)

        color = self.color
        if self.cmap is not None:
            # Max down the pixel axis of every column, so the scale spans the whole
            # scan's peaks rather than the animated frames'.  A window-dependent
            # scale would give one frame different colours in different clips.
            peaks = self._y.max(axis=0)
            self.mappable = ScalarMappable(
                cmap=get_cmap(self.cmap),
                norm=Normalize(vmin=peaks.min(), vmax=peaks.max()),
            )
            self.mappable.set_array([])

            # Re-initialising a panel onto the same figure would otherwise add a
            # second bar beside the first, shrinking the axes again each time.
            if self.colorbar is not None:
                self.colorbar.remove()
            self.colorbar = ax.figure.colorbar(self.mappable, ax=ax, pad=0.02)
            self.colorbar.set_label(self._colorbar_label())
            color = self.mappable.to_rgba(peaks[frames[0]])

        (self.line,) = ax.plot(x, self._y[:, frames[0]], color=color)
        # show_sweep_title=True keeps the swept value in ax.set_title (useful
        # when this panel is used standalone).  Set to False when animate_panels
        # is already showing it in the suptitle to avoid duplication.
        if self.show_sweep_title:
            self._title = ax.set_title(self._frame_title(frames[0]))
        else:
            self._title = None

    def _colorbar_label(self) -> str:
        """
        Label the colour bar, deriving ``"Peak <signal>"`` when none was given.

        Composed from the name and unit rather than by prefixing
        :func:`_signal_label`'s output, so the unit stays inside the brackets:
        "Peak PL intensity (counts)", not "Peak PL intensity (counts)" built by
        string surgery that would break on a signal with no unit.
        """
        if self.colorbar_label is not None:
            return self.colorbar_label
        name, unit = _signal_name_unit(self.scan)
        return f"Peak {name} ({unit})" if unit else f"Peak {name}"

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
        y = self._y[:, frame]
        self.line.set_ydata(y)
        if self.mappable is not None:
            self.line.set_color(self.mappable.to_rgba(y.max()))
        updated = [self.line]
        if self._title is not None:
            self._title.set_text(self._frame_title(frame))
            updated.append(self._title)
        return tuple(updated)


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


def _panel_frame_count(panels) -> int:
    """
    The frame count every panel agrees on, or a ``ValueError`` naming the disagreement.

    Refusing is the point.  Taking the minimum instead animates the shortest panel's
    worth of frames and says nothing, so a figure built from a scan and an image
    sequence that do not correspond renders happily and looks right.
    """
    counts = [p.n_frames for p in panels]
    if len(set(counts)) == 1:
        return counts[0]

    listing = ", ".join(
        f"{type(p).__name__} has {c}" for p, c in zip(panels, counts)
    )
    raise ValueError(
        f"the panels disagree on how many frames they have: {listing}. Every panel "
        f"in one figure must cover the same measurements, so there is no safe way to "
        f"pick. The AttoCube exports one extra white-light frame by default, which is "
        f"the usual cause of an off-by-one; drop the trailing frame(s) before "
        f"animating. If the panels really do cover different measurements, animate "
        f"them separately."
    )


def _resolve_frames(frames, n_frames: int):
    """
    Validate a caller's frame selection against the panels' frame count.

    Returns a list of native frame indices.  Every refusal here is a case that would
    otherwise fail deep inside a writer, halfway through a render, with an
    ``IndexError`` naming neither the panel nor the offending index.
    """
    try:
        selected = list(frames)
    except TypeError as exc:
        raise TypeError(
            f"frames must be a sequence of frame indices, e.g. range(10, 20); "
            f"got {type(frames).__name__}."
        ) from exc

    if not selected:
        raise ValueError("frames is empty; an animation needs at least one frame.")

    for value in selected:
        # bool is an int subclass, and True would silently mean frame 1.
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"frames must contain whole frame indices; got {value!r}. To select "
                f"by a physical coordinate rather than by index, resolve it against "
                f"the scan first."
            )
        if value < 0:
            raise ValueError(
                f"frames contains {value}, but frame indices are counted from 0. "
                f"For the last few frames write range(n - 10, n), not negatives — "
                f"a negative mixed with positives would silently reorder the "
                f"animation."
            )
        if value >= n_frames:
            raise ValueError(
                f"frames contains {value}, but the panels have {n_frames} frames "
                f"(0 to {n_frames - 1})."
            )

    return [int(v) for v in selected]


def frame_window(scan, start=None, end=None, *, axis=None) -> range:
    """
    Frame indices between two coordinates on a scan, for :func:`animate_panels`.

    ``animate_panels`` takes frame *indices*, which is the only thing every panel
    understands — an image sequence has no coordinates at all.  This turns a pair of
    physical coordinates into such a selection, naming the scan they belong to at the
    call site, so the assumption that one scan's coordinates describe every panel in
    the figure is written down rather than inferred.

    Both endpoints are **inclusive**: a caller who names two points is asking to see
    both of them.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
        The scan whose coordinates *start* and *end* refer to.
    start, end : float, optional
        Coordinates on *axis*.  Each resolves to its nearest sweep point, with the
        scan's own policy: an ambiguous coordinate is refused and a distant one
        warns, exactly as for ``get_spectrum_at``.  ``None`` means the first / last
        frame.
    axis : str, optional
        Which coordinate *start* and *end* are on — ``"piezo_y"``,
        ``"top_voltage"``, a raw row label, or anything else ``sweep=`` accepts.
        Defaults to the scan's declared sweep axis.

    Returns
    -------
    range
        Frame indices, ascending, for ``animate_panels(panels, frames=…)``.

    Raises
    ------
    ValueError
        If *end* resolves before *start*.  Reversed endpoints are far more often a
        typo than a request to play backwards, and playing backwards is still one
        slice away — see below — so the reversal is made explicit rather than
        guessed at.  Silently swapping them would make the typo invisible.

    Examples
    --------
    >>> window = frame_window(sweep, 3.2, 4.8, axis="piezo_y")   # doctest: +SKIP
    >>> fig, anim = animate_panels(panels, frames=window)        # doctest: +SKIP

    Backwards, and every third frame — ``frames=`` takes any sequence, so slicing the
    window is all either needs:

    >>> animate_panels(panels, frames=window[::-1])              # doctest: +SKIP
    >>> animate_panels(panels, frames=window[::3])               # doctest: +SKIP
    """
    n_frames = scan.n_sweeps

    def _endpoint(value, default, what):
        if value is None:
            return default
        return _select_sweep_point(
            scan, value=value, axis=axis, index=None, fast=None, slow=None,
            index_fast=None, index_slow=None, what=what,
        )

    first = _endpoint(start, 0,            "frame_window(start=…)")
    last  = _endpoint(end,   n_frames - 1, "frame_window(end=…)")

    if last < first:
        raise ValueError(
            f"frame_window does not do reverse playback: start must not come after "
            f"end. start={start!r} resolves to frame {first}, end={end!r} to frame "
            f"{last}. Swap them if they are the wrong way round. For reverse "
            f"playback, reverse the window itself — "
            f"frame_window(scan, {end!r}, {start!r})[::-1] — which says so, where a "
            f"reversed pair would just read as a typo."
        )

    return range(first, last + 1)


def animate_panels(
    panels,
    *,
    frames                     = None,
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
        One panel per subplot, left to right.  All panels must report the same
        ``n_frames``; a mismatch is refused rather than silently truncated.
    frames : sequence of int, optional
        Which frames to animate, in order.  Defaults to every frame.  Any
        sequence of indices works, so one parameter covers a window
        (``range(500, 520)``), a stride (``range(0, 2091, 10)``), a single
        frame (``[7]``) or an arbitrary order — which is what makes a
        thousand-frame scan quick to render and to embed.  Keyword-only,
        because the parameter that used to sit in this position took a *count*
        rather than indices and a stale positional call would otherwise be
        read as a different window.
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
        ``{frame}`` (the frame's own index in the scan) and ``{n_frames}``
        (the panels' total), plus ``{position}`` (0-based place within the
        animated selection) and ``{n_shown}`` (how many frames are shown).
        Default ``"Frame {frame}/{n_frames}"``, so a windowed animation
        captions its frames with the indices they have in the scan —
        ``"Frame 203/2091"``, not ``"Frame 3/40"`` — and a still lifted from
        one can be traced back to a file.  Use
        ``"Frame {position}/{n_shown}"`` for the other reading.
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
    ...     ImageSequencePanel(real_space_PL_map, title="Real-space PL", cmap="magma"),
    ...     SpectrumLinePanel(spectra_linescan, x_axis="energy"),
    ... ]
    >>> # suptitle shows e.g. "Frame 3/78  |  Power: 1.23 uW"
    >>> fig, anim = animate_panels(panels, save="three_panel_scan.gif")

    Animate one interval of a long scan, then every tenth frame of it:

    >>> fig, anim = animate_panels(panels, frames=range(500, 520))     # doctest: +SKIP
    >>> fig, anim = animate_panels(panels, frames=range(0, 2091, 10))  # doctest: +SKIP
    """
    panels = list(panels)
    n = len(panels)
    if n == 0:
        raise ValueError("animate_panels requires at least one panel.")

    # With no selection the panels must agree, because picking for them is what
    # hides a mismatch.  With an explicit selection there is nothing to guess: the
    # caller named the frames, so they only have to be valid for every panel.
    counts = [p.n_frames for p in panels]
    if frames is None:
        n_frames = _panel_frame_count(panels)
        selected = list(range(n_frames))
    else:
        n_frames = min(counts)
        selected = _resolve_frames(frames, n_frames)
    n_shown = len(selected)
    # Where each frame sits in the animation, for {position} in the counter.  A
    # frame may legitimately appear twice (a caller can repeat one to hold on it);
    # the first occurrence is the one the counter names.
    position_of = {}
    for i, f in enumerate(selected):
        position_of.setdefault(f, i)

    if figsize is None:
        figsize = (panel_width * n, panel_height)

    fig, axes = plt.subplots(
        1, n, figsize=figsize,
        constrained_layout=constrained_layout, squeeze=False,
    )
    axes = axes[0]

    for panel, ax in zip(panels, axes):
        panel.init_artists(ax, selected)

    # _build_suptitle and _has_suptitle must be evaluated AFTER init_artists
    # has run on every panel.  DiffusionCloudPanel (and any other panel that
    # defers heavy work to init_artists) calls _resolve_var() there, which is
    # what populates self._var_array.  Calling frame_label(0) before
    # init_artists would always return None, suppressing the suptitle entirely.

    def _build_suptitle(frame: int) -> str:
        parts = []
        if show_frame_count:
            parts.append(frame_count_fmt.format(
                frame=frame, n_frames=n_frames,
                position=position_of.get(frame, 0), n_shown=n_shown,
            ))
        for panel in panels:
            lbl = panel.frame_label(frame)
            if lbl:
                parts.append(lbl)
        return suptitle_sep.join(parts)

    _has_suptitle = show_frame_count or any(
        panel.frame_label(selected[0]) is not None for panel in panels
    )

    # A real fig.suptitle, because it is the only shared title the layout engine
    # reserves vertical space for.  Anything placed in a panel's own coordinates
    # instead — the obvious way to get a title blit can repaint — collides as soon
    # as a panel gains furniture on top: constrained_layout does not grow the
    # figure to fit a secondary axis, it *shrinks the panel*, so a position given
    # as a fraction of the panel slides down while the panel's own title stays at
    # the top.  Measured 20 px of overlap once a panel draws a secondary x-axis,
    # against 8.3 px of clearance here for every panel count and figure size tried.
    suptitle = fig.suptitle(_build_suptitle(selected[0])) if _has_suptitle else None

    def update(frame):
        artists = []
        for panel in panels:
            artists.extend(panel.update(frame))
        if suptitle is not None:
            suptitle.set_text(_build_suptitle(frame))
            artists.append(suptitle)
        return tuple(artists)

    # blit=False because the shared title is a Figure artist and blitting only
    # repaints Axes ones, so with blit=True the title freezes on its frame-0 text.
    # Verified across all three output paths: frozen in the notebook slider
    # (to_jshtml) and in MP4, and updating only in GIF — a Pillow-writer accident,
    # not a guarantee.  Nothing is given up: both save paths draw full frames
    # regardless (measured slightly *faster* without blit), and the notebook slider
    # steps through frames rendered in advance, which blitting cannot speed up.
    # Only live playback in a desktop window or %matplotlib widget redraws more.
    # frames=<sequence> rather than a count, so FuncAnimation hands update() the
    # frame's own index.  Every panel then reads its data at that index directly,
    # with no window offset to carry and get wrong.
    anim = animation.FuncAnimation(
        fig, update, frames=selected, blit=False, interval=interval_ms,
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
    wl_cmap          : ColormapLike = "gray",
    pl_cmap          : ColormapLike = "magma",
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
    wl_cmap, pl_cmap : str, Colormap, or sequence of colours
        Colormaps for the two image panels, passed to :func:`get_cmap`.
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
        ``frame_count_fmt``, ``suptitle_sep``, ``frames``, ``writer``).
        Pass ``frames=range(a, b)`` to animate only part of a long scan.

    Notes
    -----
    The AttoCube exports **one more white-light frame than PL frames** by default;
    the extra one has no PL frame to pair with.  When *wl* is exactly one frame
    longer than the shortest other panel, that trailing frame is dropped and a
    warning names the counts.  Any other disagreement is left to
    :func:`animate_panels` to refuse, because only this function knows which of
    its arguments is the white light, and so only here is the off-by-one
    identifiable rather than guessed at.

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

    # The AttoCube's trailing white-light frame.  This is the only place the
    # off-by-one can be *identified* rather than guessed at, because only here is
    # one panel known to be the white light — animate_panels sees a row of
    # ImageSequencePanels and rightly refuses to pick among them.  Exactly one
    # extra frame is the documented export quirk; anything else falls through to
    # the engine's refusal.
    if wl_scan is not None and len(panels) > 1 and "frames" not in engine_kwargs:
        wl_count    = panels[0].n_frames          # wl is appended first
        other_count = min(p.n_frames for p in panels[1:])
        if wl_count == other_count + 1:
            warnings.warn(
                f"takes {other_count} images out of a possible {wl_count}: the "
                f"AttoCube exports one more white-light frame than PL frames by "
                f"default, and the last one has no PL frame to pair with. Pass "
                f"frames= explicitly to override.",
                UserWarning, stacklevel=2,
            )
            engine_kwargs["frames"] = range(other_count)

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
    cmap               : ColormapLike = "inferno",
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
    image : np.ndarray or object with ``.img``
        Raw 2-D PL image.
    result : DiffusionResult, optional
        Pre-computed result from :func:`~tmdc_optics_tools.diffusion.analyse_diffusion_cloud`.
        When ``None`` the analysis is run here with the remaining kwargs.
    ax : matplotlib.axes.Axes, optional
        Creates a new figure if ``None``.
    cmap : str, Colormap, or sequence of colours
        Colormap for the image, passed to :func:`get_cmap`.
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

    img = image.img if hasattr(image, "img") else np.asarray(image, float)

    if result is None:
        result = _diffusion.analyse_diffusion_cloud(
            img,
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

    im = ax.imshow(img, cmap=get_cmap(cmap), origin="upper")

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
    cmap : str, Colormap, or sequence of colours
        Colormap for the image, passed to :func:`get_cmap`.
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
        cmap              : ColormapLike = "inferno",
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

    def _resolve_var(self, seq) -> None:
        """
        Resolve the swept-variable array and labels.

        Priority (highest first):
        1. Explicit ``var_array`` / ``var_label`` / ``var_units`` passed to
           ``__init__``.
        2. ``seq_result.var_array`` / ``.var_label`` / ``.var_units`` — the
           values that were forwarded from ``analyse_diffusion_sequence``.
        3. ``None`` — no per-frame subtitle is shown.

        The array is kept at full length and read at each frame's own index, so a
        window shows the values belonging to the frames it displays.
        """
        arr = self._var_array_override
        if arr is None and seq.var_array is not None:
            arr = seq.var_array
        if arr is not None:
            self._var_array = np.asarray(arr)
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

    def init_artists(self, ax, frames) -> None:
        seq = self._get_seq_result()
        self._resolve_var(seq)

        first = frames[0]
        frame0 = (self.scan.load_frame(first)
                  if hasattr(self.scan, "load_frame") else self.scan[first])
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

        r0 = seq.frames[first]
        # Contour lines for the first frame shown
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

# Lazy imports for colour-norm helpers (avoid polluting the module namespace
# with rarely-used names while keeping the import cost near zero).
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.cm import ScalarMappable


def plot_spectral_series(
    scan,
    ax               = None,
    figsize          : tuple  = (6, 4),
    dpi              : int    = None,
    # --- nest pinning (nested scans only) ---
    fast             : float  = None,
    index_fast       : int    = None,
    slow             : float  = None,
    index_slow       : int    = None,
    # --- x-axis ---
    x_axis           : str    = "energy",
    x_range          : tuple  = None,
    twin_axis        : bool   = False,
    # --- spectra source ---
    spectra_source   : str    = "best",
    # --- background subtraction (post-load, in addition to loader bg) ---
    bg_region        : tuple  = None,
    # --- series selection and stacking ---
    series_axis      : str    = "sweep",
    series_range     : tuple  = None,
    sweep_step       : int    = 1,
    spectrum_offset  : float  = 0.0,
    # --- colour mapping ---
    cmap             : ColormapLike = "magma",
    color_scale      : str    = "linear",
    color_range      : tuple  = None,
    # --- line style ---
    lw               : float  = 1.0,
    alpha            : float  = 1.0,
    alpha_by_series  : bool   = False,
    alpha_min        : float  = 0.2,
    # --- colorbar ---
    colorbar         : bool   = True,
    cb_label         : str    = None,
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
    Plot a series of spectra, one line per sweep point, coloured by coordinate.

    Every point of the series is drawn as a line whose colour is taken from
    *cmap* mapped onto the series coordinate, with a colorbar naming that
    coordinate.  A flat sweep is itself the series.  A declared nest is pinned
    on one of its two axes, and the axis left free becomes the series.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
        Must expose ``sweep_axis``, ``energy`` / ``wavelength``, and the chosen
        *spectra_source* attribute.  A nested scan must have been loaded with
        ``fast_sweep=`` and ``slow_sweep=``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure is created when ``None``.
    figsize : tuple
        Figure size, used only when *ax* is ``None``.
    dpi : int, optional
        Figure resolution, used only when *ax* is ``None``.

    Nest pinning
    ------------
    fast, slow : float, optional
        Coordinate at which to hold that nest axis.  The other axis becomes the
        series.  Nested scans only.
    index_fast, index_slow : int, optional
        The same, by integer position rather than by coordinate.

        Name exactly one of these four on a nested scan and none of them on a
        flat one.  Holding the fast axis returns one spectrum per slow point and
        vice versa, so the series always runs along the axis *not* named.

    Axes
    ----
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
        Which correction state to plot; *x_axis* decides the axis it is served on.

        * ``"best"``  — the most-corrected state the scan holds.  **Default.**
        * ``"raw"``   — the file's own counts.
        * ``"cr"``    — cosmic-ray repaired.  Requires ``cosmic_rays`` at load time.
        * ``"bg"``    — background-subtracted.  Requires ``bg_region_nm`` /
          ``bg_region_eV`` or ``bg_spectrum`` at load time.
        * ``"contrast"`` — ΔR/R₀ against the reference.  Requires ``reference``.
        * ``"pre_jacobian"`` — the raw counts on the energy axis with the Jacobian
          left off, whatever ``apply_jacobian`` says.  Energy axis only.

    bg_region : tuple of (x_min, x_max), optional
        Additional background region subtracted *after* loading (same units
        as *x_axis*).  Applied on top of any background already baked into
        *spectra_source*.  ``None`` (default) skips this step.

    Series
    ------
    series_axis : str
        Which quantity the series is read against — the coordinate that colours
        the lines, labels the colorbar, and *series_range* is measured in.
        ``"sweep"`` (default) is the scan's declared sweep axis.  Anything else
        is spelled as ``sweep=`` spells it: a registry key such as
        ``"top_voltage"`` or ``"carrier_density"``, or a raw row label such as
        ``"V_A"``.  Use it when a scan is declared in one coordinate and the
        figure is wanted in another — a displacement-field sweep driven by both
        gates, read in top-gate volts.

        Flat sweeps only.  On a nest the free axis carries its own coordinate,
        label and unit, and the colours follow it.
    series_range : tuple of (low, high), optional
        Draw only the points whose series coordinate falls within these bounds,
        in the units of *series_axis*.  Inclusive at both ends.  ``None`` for
        either end leaves it unbounded, ``None`` for the whole argument (default)
        draws every point.

        The endpoints are bounds on the coordinate, not a direction of travel, so
        a descending sweep still takes them low-to-high; reversed endpoints
        raise.  A non-finite coordinate never falls inside a bound, so a derived
        axis with gaps drops those points.
    sweep_step : int
        Plot every *sweep_step*-th point of what *series_range* kept, starting
        from the first.  ``1`` (default) plots all of them, ``2`` every other
        one, and so on.  Must be a positive integer.
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
    cmap : str, Colormap, or sequence of colours
        Passed to :func:`get_cmap`.  Default ``"viridis"``.
    color_scale : {"linear", "log"}
        Colormap normalisation.  Default ``"linear"``.  A log scale needs a
        positive coordinate; on one that crosses zero the lower limit is clamped
        and the colours no longer report the value.
    color_range : tuple of (low, high), optional
        Clip the colormap to this range, in the units of *series_axis*.
        Defaults to the span of the points actually drawn, so *series_range*
        shrinks it and the drawn lines always use the whole colormap.  Set it
        explicitly to hold one scale across several figures.

    Line style
    ----------
    lw : float
        Line width.
    alpha : float
        Global line opacity (0–1).  Ignored when *alpha_by_series* is ``True``.
    alpha_by_series : bool
        Scale each line's alpha linearly from *alpha_min* at the low end of the
        colour range to 1.0 at the high end.  Overrides *alpha*.
    alpha_min : float
        Minimum alpha used when *alpha_by_series* is ``True``.

    Colorbar
    --------
    colorbar : bool
        Show a colorbar.  Default ``True``.
    cb_label : str, optional
        Colorbar label.  ``None`` (default) takes it from *series_axis*, with
        its unit.  A string is used **verbatim**, so include the unit.
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

    Signal axis
    -----------
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
        One Line2D per *drawn* point, in series order — so ``lines[j]`` is the
        spectrum taken at the *j*-th coordinate surviving *series_range* and
        *sweep_step*.  Their y data includes any *spectrum_offset*.

    Raises
    ------
    ValueError
        If *sweep_step* is not a positive integer; if the nest pinning does not
        match the scan (none named on a nest, or any named on a flat sweep); if
        *series_axis* is not ``"sweep"`` on a nested scan, or names no quantity
        this scan holds; or if *series_range* runs backwards or selects no point.

    See Also
    --------
    tmdc_optics_tools.loaders.AttoCubeSpectralSweep.get_spectrum_at :
        one spectrum at a coordinate, rather than the series.
    tmdc_optics_tools.loaders.AttoCubeSpectralSweep.as_grid :
        a nested sweep reshaped onto its grid, rather than pinned to a line.

    Examples
    --------
    >>> fig, ax, cb, lines = plot_spectral_series(power_scan)   # doctest: +SKIP

    A displacement-field sweep, read in top-gate volts instead:

    >>> plot_spectral_series(field_scan, series_axis="top_voltage")  # doctest: +SKIP

    A raster, holding the slow axis and drawing the fast line, over part of it:

    >>> plot_spectral_series(raster, slow=2.0, series_range=(10.0, 20.0))  # doctest: +SKIP
    """
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

    # --- which spectra become the series -----------------------------------
    # A nest is pinned on one axis and the free axis becomes the series; a flat
    # sweep is already the series.  Either way `data` ends up (n_pixels, n) and
    # `series` is the matching (n,) coordinate that colours the lines.
    pinned = [name for name, arg in (("fast", fast), ("index_fast", index_fast),
                                     ("slow", slow), ("index_slow", index_slow))
              if arg is not None]

    if scan.is_nested:
        if len(pinned) != 1:
            raise ValueError(
                f"plot_spectral_series(): this sweep is a declared nest "
                f"({scan.nesting}), so name exactly one axis to hold: fast=, "
                f"index_fast=, slow= or index_slow=. The axis left free becomes "
                f"the series. Got {', '.join(pinned) if pinned else 'none'}. "
                f"Naming two pins every axis and leaves a single spectrum, "
                f"which is get_spectrum_at(); for the whole grid, use "
                f"scan.as_grid()."
            )
        if series_axis != "sweep":
            raise ValueError(
                f"plot_spectral_series(): series_axis={series_axis!r} reads the "
                f"series against another quantity, which applies to a flat "
                f"sweep. This one is a declared nest ({scan.nesting}), where the "
                f"free axis already carries its own coordinate, label and unit, "
                f"so the colours follow it. To colour by a different quantity, "
                f"declare the nest in it with fast_sweep= / slow_sweep= at load "
                f"time."
            )

        held_fast = fast is not None or index_fast is not None

        if index_fast is not None:
            data = scan.get_spectrum_by_index(
                fast=index_fast, source=spectra_source, x_axis=x_axis)
        elif fast is not None:
            data = scan.get_spectrum_at(
                fast=fast, source=spectra_source, x_axis=x_axis)
        elif index_slow is not None:
            data = scan.get_spectrum_by_index(
                slow=index_slow, source=spectra_source, x_axis=x_axis)
        else:
            data = scan.get_spectrum_at(
                slow=slow, source=spectra_source, x_axis=x_axis)

        # Holding the fast axis strides across the grid and returns one spectrum
        # per *slow* point, so the free axis — the other one — is what varies
        # along the columns of `data`, and therefore what the colours mean.
        nest         = scan.nesting
        series       = np.asarray(
            nest.slow_axis if held_fast else nest.fast_axis, dtype=float)
        series_label = nest.slow_axis_label if held_fast else nest.fast_axis_label
    else:
        if pinned:
            raise ValueError(
                f"plot_spectral_series(): {', '.join(pinned)} needs a declared "
                f"nest, and this sweep is flat ({scan.n_sweeps} points). A flat "
                f"sweep is already the series, so pass none of fast=, "
                f"index_fast=, slow=, index_slow=. If these points are a grid, "
                f"declare it with fast_sweep= and slow_sweep= at load time."
            )
        data = _resolve_spectra(scan, spectra_source, x_axis)
        # param= so a bad name reports itself as series_axis=, not as the axis=
        # of the loader accessors the resolver is shared with.
        values, label, unit = scan._lookup_axis(series_axis, param="series_axis")
        series       = np.asarray(values, dtype=float)
        series_label = f"{label} ({unit})" if unit else label

    # --- subset: which points of the series are drawn -----------------------
    # Bounds are read on `series`, the same coordinate that colours the lines,
    # so the selection and the colorbar can never disagree about units.  A
    # nested scan needs no special case: `series` is already the free axis.
    if series_range is not None:
        low_given, high_given = series_range
        low  = -np.inf if low_given  is None else float(low_given)
        high =  np.inf if high_given is None else float(high_given)

        if low > high:
            raise ValueError(
                f"plot_spectral_series(): series_range={series_range!r} runs "
                f"backwards. Give it low-to-high in {series_label}. The "
                f"endpoints are bounds on the coordinate, not a direction of "
                f"travel, so a descending sweep still takes (low, high). Pass "
                f"None for either end to leave it unbounded."
            )

        keep = (series >= low) & (series <= high)   # inclusive, as x_range is
        if not keep.any():
            raise ValueError(
                f"plot_spectral_series(): series_range={series_range!r} selects "
                f"no sweep point. {series_label} runs {series.min():.4g} to "
                f"{series.max():.4g} across {series.size} points."
            )

        # Boolean indexing copies, unlike the selection above, which returned a
        # view: the kept columns are an arbitrary subset, so no slice expresses
        # them.  The copy is the plotted subset, not the whole scan.
        data   = data[:, keep]
        series = series[keep]

    # --- optional post-load background subtraction -------------------------
    if bg_region is not None:
        data = processing.subtract_background(data, bg_region=bg_region, x=x, axis=0)

    # --- optional spectral crop -------------------------------------------
    if x_range is not None:
        mask   = (x >= x_range[0]) & (x <= x_range[1])
        x      = x[mask]
        data   = data[mask, :]

    # --- colour norm --------------------------------------------------------
    # Defaults to the span of what survived series_range, so the drawn lines use
    # the whole colormap.  sweep_step does not shrink it further: thinning
    # samples the kept span evenly and its endpoints stay in the figure.
    c_min, c_max = color_range if color_range is not None \
        else (series.min(), series.max())

    if color_scale == "log":
        norm = LogNorm(vmin=max(c_min, 1e-12), vmax=c_max)
    else:
        norm = Normalize(vmin=c_min, vmax=c_max)

    sm      = ScalarMappable(norm=norm, cmap=get_cmap(cmap))
    sm.set_array([])   # required for standalone colorbars

    # --- per-line alpha if requested ---------------------------------------
    series_norm_linear = (series - c_min) / max(c_max - c_min, 1e-12)

    # --- draw lines --------------------------------------------------------
    # Two counters, and they differ once sweep_step > 1: i indexes the series, so
    # colour and alpha keep tracking each line's own coordinate, while j counts
    # drawn lines, so the offsets stack contiguously instead of leaving gaps
    # where a skipped point would have been.
    lines = []
    for j, i in enumerate(range(0, len(series), sweep_step)):
        colour = sm.to_rgba(series[i])
        a = float(alpha_min + (1.0 - alpha_min) * series_norm_linear[i]) \
            if alpha_by_series else float(alpha)
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
        cb.set_label(series_label if cb_label is None else cb_label,
                     labelpad=cb_labelpad)

    return fig, ax, cb, lines