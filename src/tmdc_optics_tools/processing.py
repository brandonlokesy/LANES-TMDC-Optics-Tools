# tmdc_optics_tools/processing.py
"""
Spectral processing and normalisation routines.

All functions operate on plain NumPy arrays and are independent of any
particular loader class, so they can be used standalone or piped together.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import matplotlib.patches as patches

from .constants import HC_EV_NM


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_peak(spectra: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalise each spectrum to its maximum value.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps) or (n_pixels,)
    axis : int
        Axis along which the spectra run. Default 0 (pixels along rows).

    Returns
    -------
    np.ndarray
        Normalised spectra. Sweeps with zero max are left as-is.
    """
    spectra = np.asarray(spectra, float)
    peak = spectra.max(axis=axis, keepdims=True)
    peak[peak == 0] = 1.0
    return spectra / peak


def normalise_area(
    spectra : np.ndarray,
    x       : np.ndarray = None,
    axis    : int = 0,
) -> np.ndarray:
    """
    Normalise each spectrum to its integrated area.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps) or (n_pixels,)
    x : np.ndarray, shape (n_pixels,), optional
        x-axis values. Used for trapezoidal integration if provided;
        otherwise a rectangular sum is used.
    axis : int
        Pixel axis.

    Returns
    -------
    np.ndarray
    """
    spectra = np.asarray(spectra, float)
    area    = np.trapz(spectra, x=x, axis=axis) if x is not None else spectra.sum(axis=axis)
    area    = np.where(area == 0, 1.0, area)
    return spectra / np.expand_dims(area, axis)


def subtract_background(
    spectra   : np.ndarray,
    bg_region : tuple,
    x         : np.ndarray,
    axis      : int = 0,
) -> np.ndarray:
    """
    Subtract a constant background estimated from a spectral region.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps)
    bg_region : tuple of (x_min, x_max)
        Spectral range used to estimate the background.
    x : np.ndarray, shape (n_pixels,)
        x-axis values (energy or wavelength).
    axis : int
        Pixel axis.

    Returns
    -------
    np.ndarray
        Background-subtracted spectra.
    """
    mask = (x >= bg_region[0]) & (x <= bg_region[1])
    if not mask.any():
        raise ValueError(f"No pixels found in bg_region {bg_region}.")
    bg = np.take(spectra, np.where(mask)[0], axis=axis).mean(axis=axis, keepdims=True)
    return spectra - bg


def subtract_spectrum(
    spectra    : np.ndarray,
    background : np.ndarray,
    axis       : int = 0,
) -> np.ndarray:
    """
    Subtract a separately measured background *spectrum*.

    The counterpart of :func:`subtract_background`, which removes a scalar
    estimated from a window of the same spectrum.  Here the background is a
    measured spectrum of its own — a dark frame, a stray-light or substrate
    reference — so it removes wavelength-dependent structure that a flat offset
    cannot.  Applies equally to PL and reflectance.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps) or (n_pixels,)
        Sample spectra.
    background : np.ndarray, shape (n_pixels,) or matching *spectra*
        Measured background, on the **same** x-axis as *spectra*.  Alignment is
        the caller's responsibility: nothing here can check it, since only the
        arrays are passed.
    axis : int
        Pixel axis of *spectra*.  Default 0.

    Returns
    -------
    np.ndarray
        ``spectra - background``, of the same shape as *spectra*.

    Raises
    ------
    ValueError
        If *background* is 1-D and its length does not match *spectra* along
        *axis*, or if it is neither 1-D nor the same shape as *spectra*.
    """
    spectra    = np.asarray(spectra, float)
    background = np.asarray(background, float)

    if background.shape == spectra.shape:
        return spectra - background
    if background.ndim != 1:
        raise ValueError(
            f"background must be 1-D or the same shape as spectra "
            f"{spectra.shape}, got {background.shape}."
        )
    if background.size != spectra.shape[axis]:
        raise ValueError(
            f"background has {background.size} points but spectra has "
            f"{spectra.shape[axis]} along axis {axis}. The two must share an "
            f"x-axis."
        )
    # Reshape (n_pixels,) so it broadcasts along the pixel axis only: one measured
    # background, subtracted from every sweep.
    shape = [1] * spectra.ndim
    shape[axis] = background.size
    return spectra - background.reshape(shape)


# ---------------------------------------------------------------------------
# Contrast against a reference spectrum
# ---------------------------------------------------------------------------

_CONTRAST_MODES = {
    "contrast": "(S - R) / R",
    "ratio":    "S / R",
}


def spectral_contrast(
    spectra       : np.ndarray,
    reference     : np.ndarray,
    mode          : str = "contrast",
    min_reference : float = None,
    axis          : int = 0,
) -> tuple:
    """
    Divide spectra by a reference spectrum, as reflectance contrast or a ratio.

    For reflectance the sample and a bare-substrate reference give

    .. math :: \\frac{\\Delta R}{R_0} = \\frac{S - R}{R}
               \\qquad\\text{or}\\qquad \\frac{R}{R_0} = \\frac{S}{R}

    Both are standard read-outs, which is why both modes exist and no others: an
    absorbance mode would be an untested promise until absorption data has a
    loader.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps) or (n_pixels,)
        Sample spectra, background-subtracted if that is wanted — do it first,
        because a pedestal in either array biases a ratio non-linearly.
    reference : np.ndarray, shape (n_pixels,) or matching *spectra*
        Reference spectrum on the **same** x-axis, likewise background-subtracted.
    mode : {"contrast", "ratio"}
        ``"contrast"`` gives ``(S - R) / R``; ``"ratio"`` gives ``S / R``.
    min_reference : float, optional
        Reference pixels at or below this value are treated as unusable and come
        back as ``NaN``.  ``None`` (default) guards only non-positive pixels: a
        zero-count reference pixel would otherwise divide to ``inf`` and a
        negative one would silently flip the sign of the contrast.
    axis : int
        Pixel axis of *spectra*.  Default 0.

    Returns
    -------
    result : np.ndarray
        Contrast or ratio, same shape as *spectra*, ``NaN`` at guarded pixels.
    guarded : np.ndarray of bool, shape (n_pixels,)
        Which reference pixels were guarded.  Returned rather than only warned
        about so the caller can mask, crop or interpolate over them knowingly.

    Warns
    -----
    UserWarning
        When any pixel is guarded, naming how many.

    Notes
    -----
    **The Jacobian cancels in a ratio.** ``(S·λ²/hc) / (R·λ²/hc) = S / R``, so a
    contrast spectrum must not be Jacobian-corrected — the factor divides out
    exactly, and applying it to the numerator alone would be an error.

    **Sample and reference must share an exposure.** For a reference scaled by
    *k*, ``(S − kR)/(kR)`` is not a rescaling of the contrast but a biased
    version of it, so a reference taken at a different integration time or
    excitation power gives a wrong answer that no later normalisation repairs.

    Examples
    --------
    >>> rc, guarded = spectral_contrast(sample, substrate)
    >>> guarded.sum()
    0
    """
    if mode not in _CONTRAST_MODES:
        raise ValueError(
            f"mode={mode!r} is not recognised. Choose from "
            + ", ".join(f"{k!r} ({v})" for k, v in _CONTRAST_MODES.items())
            + "."
        )

    spectra   = np.asarray(spectra, float)
    reference = np.asarray(reference, float)

    if reference.ndim == 1:
        if reference.size != spectra.shape[axis]:
            raise ValueError(
                f"reference has {reference.size} points but spectra has "
                f"{spectra.shape[axis]} along axis {axis}. The two must share an "
                f"x-axis."
            )
        shape = [1] * spectra.ndim
        shape[axis] = reference.size
        ref = reference.reshape(shape)
        guarded_1d = reference
    elif reference.shape == spectra.shape:
        ref = reference
        guarded_1d = None
    else:
        raise ValueError(
            f"reference must be 1-D or the same shape as spectra "
            f"{spectra.shape}, got {reference.shape}."
        )

    floor   = 0.0 if min_reference is None else float(min_reference)
    unusable = ref <= floor
    if unusable.any():
        n = int(np.count_nonzero(unusable if guarded_1d is None
                                 else np.asarray(guarded_1d) <= floor))
        warnings.warn(
            f"{n} reference pixel(s) at or below {floor:g} were excluded from the "
            f"{mode}; the result is NaN there. A reference cannot be divided by "
            f"zero, and a negative one would invert the contrast. The returned "
            f"mask says which pixels.",
            UserWarning, stacklevel=2,
        )

    # Divide only where the reference is usable; NaN elsewhere rather than inf, so
    # the gap propagates visibly instead of dominating a colour scale or a fit.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(unusable, np.nan,
                          (spectra - ref) / ref if mode == "contrast"
                          else spectra / ref)

    guarded = (np.asarray(guarded_1d) <= floor if guarded_1d is not None
               else np.any(unusable, axis=1 - axis if spectra.ndim > 1 else 0))
    return result, guarded


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_median(spectra: np.ndarray, kernel: int = 3) -> np.ndarray:
    """
    Median filter applied to a spectrum (1-D) or a PL map (2-D).

    Parameters
    ----------
    spectra : np.ndarray
    kernel : int
        Filter kernel size.

    Returns
    -------
    np.ndarray
    """
    return median_filter(spectra, size=kernel, mode="mirror")


def smooth_savgol(
    spectra    : np.ndarray,
    window     : int = 11,
    poly_order : int = 3,
    axis       : int = 0,
) -> np.ndarray:
    """
    Savitzky-Golay smoothing along the pixel axis.

    Parameters
    ----------
    spectra : np.ndarray
    window : int
        Window length (must be odd).
    poly_order : int
        Polynomial order for the filter.
    axis : int
        Pixel axis.

    Returns
    -------
    np.ndarray
    """
    return savgol_filter(spectra, window_length=window, polyorder=poly_order, axis=axis)


# ---------------------------------------------------------------------------
# Spectral operations
# ---------------------------------------------------------------------------

def crop(
    spectra : np.ndarray,
    x       : np.ndarray,
    x_range : tuple,
    axis    : int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop spectra and x-axis to a given range.

    Parameters
    ----------
    spectra : np.ndarray
    x : np.ndarray
    x_range : tuple of (x_min, x_max)
    axis : int

    Returns
    -------
    x_cropped : np.ndarray
    spectra_cropped : np.ndarray
    """
    mask = (x >= x_range[0]) & (x <= x_range[1])
    idx  = np.where(mask)[0]
    return x[idx], np.take(spectra, idx, axis=axis)


def wavelength_to_energy(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    Convert wavelength in nm to photon energy in eV.

    Parameters
    ----------
    wavelength_nm : array-like

    Returns
    -------
    np.ndarray
        Energy in eV.
    """
    return HC_EV_NM / np.asarray(wavelength_nm, float)


def energy_to_wavelength(energy_eV: np.ndarray) -> np.ndarray:
    """
    Convert photon energy in eV to wavelength in nm.

    Parameters
    ----------
    energy_eV : array-like

    Returns
    -------
    np.ndarray
        Wavelength in nm.
    """
    return HC_EV_NM / np.asarray(energy_eV, float)


def jacobian_correction_wvl2E(
    spectra       : np.ndarray,
    wavelength_nm : np.ndarray,
    axis          : int = 0,
) -> np.ndarray:
    """
    Apply the Jacobian correction when converting PL from wavelength to energy.

    When replotting on an energy axis, the spectral density must be
    multiplied by dλ/dE = λ²/(hc) to conserve integrated intensity.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra as a function of wavelength.
    wavelength_nm : np.ndarray, shape (n_pixels,)
        Corresponding wavelength axis in nm.
    axis : int
        Pixel axis of *spectra*.

    Returns
    -------
    np.ndarray
        Corrected spectra.
    """
    jacobian = wavelength_nm**2 / HC_EV_NM
    shape = [1] * spectra.ndim
    shape[axis] = len(jacobian)
    return spectra * jacobian.reshape(shape)

def _draw_region_box(ax, region, color, label=None, lw=1.2, ls="-"):
    if region is None:
        return None
    row_slice, col_slice = region
    x0, y0 = (col_slice.start or 0) - 0.5, (row_slice.start or 0) - 0.5
    width  = col_slice.stop - (col_slice.start or 0)
    height = row_slice.stop - (row_slice.start or 0)
    rect = patches.Rectangle((x0, y0), width, height, edgecolor=color,
                              facecolor="none", linewidth=lw, linestyle=ls,
                              label=label, zorder=4)
    ax.add_patch(rect)
    return rect

def _apply_bg_region(img: np.ndarray, region, stat: str = "median") -> np.ndarray:
    stat_fn = np.median if stat == "median" else np.mean
    return img - stat_fn(img[region])

def _fill_flagged(working: np.ndarray, cr_mask: np.ndarray, median_window: int) -> np.ndarray:
    """
    One replacement pass: overwrite flagged pixels with the local median.

    Operates **in place** on ``working`` (which must already be a copy) and
    returns it.  The median is taken from ``working`` rather than the raw
    spectrum so that a multi-pixel spike is not still sitting inside its own
    median window on later passes.
    """
    if cr_mask.any():
        local_med          = median_filter(working, size=median_window)
        working[cr_mask]   = local_med[cr_mask]
    return working


def _detect_cosmic_rays_1d(
        spectrum        : np.ndarray,
        sigma_threshold : float,
        median_window   : int,
        max_iter        : int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterative Laplacian sigma-clip on a single spectrum.

    Returns
    -------
    cr_mask : ndarray[bool]
        True at pixels flagged as cosmic rays.
    working : ndarray
        Partially-cleaned spectrum carried across iterations.  Flags raised on
        the final iteration are not yet filled in it; it exists so the caller
        can take replacement medians from uncontaminated data.
    """
    n       = spectrum.size
    cr_mask = np.zeros(n, dtype=bool)

    # Partially-cleaned copy, carried across iterations.  Never modifies the
    # original spectrum.
    working = spectrum.copy()

    for _ in range(max_iter):

        _fill_flagged(working, cr_mask, median_window)

        # Compute the laplacian on the good pixels
        laplacian = np.zeros(n)
        laplacian[1:-1]  = working[:-2] - 2.0 * working[1:-1] + working[2:]

        # Robust noise estimate from unflagged pixels.
        # MAD → σ conversion factor for a Gaussian: 1/0.6745.

        # Returns laplacian without cosmic ray pixels
        lap_good  = laplacian[~cr_mask]
        if lap_good.size == 0:            # everything flagged — nothing left to estimate from
            break
        # Noise estimate on the laplacian without cosmic rays
        mad       = np.median(np.abs(lap_good - np.median(lap_good)))
        sigma_lap = mad / 0.6745 # MAD ≈ 0.6745σ

        if sigma_lap == 0:
            break

        # Flag where Laplacian is a large negative outlier.
        # When the Laplacian is more than the threshold, we flag a cosmic ray
        new_flags   = laplacian < -sigma_threshold * sigma_lap
        # Newly identified cosmic ray pixels
        newly_found = new_flags & ~cr_mask
        # Combine old cosmic ray pixels with newly found cosmic ray pixels
        cr_mask |= new_flags

        if not newly_found.any():
            break

    return cr_mask, working


def _cross_sweep_veto(
        spectra         : np.ndarray,
        cr_mask         : np.ndarray,
        sigma_threshold : float,
        window          : int,
    ) -> np.ndarray:
    """
    Drop detections that repeat at the same pixel in neighbouring sweeps.

    A cosmic ray cannot recur: a second particle would have to strike the same
    detector pixel during the next exposure.  A narrow *spectral* feature (Raman
    line, laser leakage past the filter edge, sharp emitter line) is present in
    every sweep, and is indistinguishable from a CR to a 3-point Laplacian.
    Comparing each detection against its own sweep neighbourhood separates them.

    The comparison is strictly along the sweep axis — the median footprint is
    ``(1, window)``, so no information is ever mixed between detector pixels.

    This can only ever *remove* flags.  Where the spectrum changes fast with the
    sweep parameter (a charging transition), the local scatter inflates and more
    detections are vetoed — i.e. the failure mode is a missed cosmic ray, never
    an overwritten real feature.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps)
        Raw counts, pixel-major.
    cr_mask : np.ndarray[bool], shape (n_pixels, n_sweeps)
        Detections from the per-spectrum Laplacian pass.
    sigma_threshold : float
        A detection survives only if it stands this many sigma above the local
        sweep median, sigma taken from the same neighbourhood.
    window : int
        Number of neighbouring sweeps (forced odd) used for the median.

    Returns
    -------
    np.ndarray[bool]
        The subset of ``cr_mask`` confirmed as non-repeating.
    """
    if window % 2 == 0:
        window += 1

    footprint = (1, window)                        # sweeps only, never pixels
    local_med = median_filter(spectra, size=footprint)
    local_mad = median_filter(np.abs(spectra - local_med), size=footprint)
    sigma_cs  = local_mad / 0.6745

    # A pixel whose sweep trace happens to be unusually quiet should still need a
    # real excess, so floor sigma at the typical value across the detector.
    positive  = sigma_cs[sigma_cs > 0]
    floor     = np.median(positive) if positive.size else 0.0
    sigma_eff = np.maximum(sigma_cs, floor)

    confirmed = (spectra - local_med) > sigma_threshold * sigma_eff
    return cr_mask & confirmed


# Fraction of sweeps above which a repeatedly-flagged pixel stops being credible
# as a cosmic ray.  Module-level so it can be tuned without a signature change;
# the warning itself is suppressible through the standard `warnings` filters.
PERSISTENT_FLAG_FRACTION = 0.8


def _warn_persistent_flags(cr_mask: np.ndarray, fraction: float = PERSISTENT_FLAG_FRACTION) -> None:
    """
    Warn when the per-spectrum pass keeps flagging the same pixel every sweep.

    A cosmic ray cannot recur at one pixel, so a detection that repeats across
    the sweep is a hot pixel or a real narrow spectral feature — and without the
    veto both have just been median-replaced in every sweep, silently.  This is
    the only signal the caller gets that data was destroyed rather than cleaned.
    """
    n_sweeps   = cr_mask.shape[1]
    counts     = cr_mask.sum(axis=1)
    persistent = np.flatnonzero(counts > fraction * n_sweeps)
    if persistent.size == 0:
        return

    shown = ", ".join(f"{p} ({counts[p]}/{n_sweeps})" for p in persistent[:3])
    if persistent.size > 3:
        shown += ", ..."

    # ASCII only: this is runtime output, and group terminals are cp1252.
    warnings.warn(
        f"{persistent.size} pixel(s) flagged in more than {fraction:.0%} of sweeps: "
        f"{shown}. A cosmic ray cannot recur at one pixel, so these are hot pixels "
        f"or real narrow spectral features (Raman, laser leakage, sharp emitter "
        f"lines) - and they have been median-replaced in every sweep. Pass "
        f"cross_sweep_veto=True to keep them.",
        stacklevel=3,
    )


def remove_cosmic_rays(
        spectra : np.ndarray,
        sigma_threshold : float = 5.0,
        median_window : int = 7,
        max_iter: int = 3,
        cross_sweep_veto : bool = False,
        cross_sweep_window : int = 5,
        axis : int = 0,

    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosmic Ray Removal from PL spectra

    Standard method: Iterative sigma-clipping on the Laplacian (second derivative).
    Detection always runs **one spectrum at a time**, independently of the sweep
    parameter: a cosmic ray is a single-exposure event, so the dispersion axis is
    the only axis along which it is anomalous, and the MAD noise estimate has to
    be per-exposure because PL intensity can move by an order of magnitude across
    a gate sweep.

    Scientific basis:
    - Cosmic rays produce sharp, narrow spikes (1–3 pixels wide).
    - The discrete Laplacian  L[i] = flux[i-1] - 2·flux[i] + flux[i+1]  is near
        zero for a smooth spectrum but strongly NEGATIVE at a CR spike centre, because
        the centre pixel is far above its neighbours.
    - Flagging pixels where L < -N·σ_L identifies CR centres (σ_L estimated robustly
        via the median absolute deviation, MAD).
    - Iteration is essential for multi-pixel CRs:
        • Pass 1 detects the *edges* (their neighbours are normal → large |L|).
        • Detected pixels are replaced with local medians, then the Laplacian is
            recomputed. Now interior "flat-top" pixels show a large negative L because
            their neighbours have been restored.
        • Repeat until no new pixels are found.
    - This approach is the 1-D analogue of LA Cosmic (van Dokkum 2001, PASP 113, 1420).

    Parameters
    ----------
    spectra : np.ndarray, shape (n_pixels, n_sweeps) or (n_pixels,)
        Raw spectra in counts.  A 2-D array is treated as independent spectra,
        one per column, each detected and cleaned on its own.
    sigma_threshold : float
        Detection threshold in MAD-based sigma units.
        Typical values: 4–7 (lower = more aggressive).
    median_window : int
        Width (pixels, forced odd) of the median filter used for both noise
        estimation and pixel replacement. If median_window is even, the value is incremented by 1
        to ensure an odd window.
    max_iter : int
        Maximum number of sigma-clipping iterations.  Convergence is usually
        reached in 2–4 passes.
    cross_sweep_veto : bool
        If True, discard detections that recur at the same pixel in neighbouring
        sweeps — those are spectral features or hot pixels, not cosmic rays.
        Requires 2-D input with at least 3 sweeps, and assumes the sweep axis is
        an ordered physical parameter along which the spectrum evolves smoothly.
        Default False: each spectrum is cleaned in complete isolation, and any
        pixel flagged in more than ``PERSISTENT_FLAG_FRACTION`` of the sweeps
        raises a ``UserWarning`` instead (see Notes).
    cross_sweep_window : int
        Number of neighbouring sweeps (forced odd) used by the veto.  Keep it
        small; a window that straddles a charging transition compares spectra
        that are not physically comparable.
    axis : int
        Pixel axis of a 2-D input. Default 0 (pixels along rows, sweeps along
        columns), matching the rest of this module.

    Returns
    -------
    cleaned : ndarray
        Flux with cosmic ray pixels replaced by local median values.
        Same shape as the input; the input array is never modified.
    cr_mask : ndarray[bool]
        True at pixels identified as cosmic rays.  Same shape as the input.

    Notes
    -----
    The default is deliberately the conservative one: no assumption is made
    about the sweep axis, and results do not depend on whether spectra are
    passed one at a time or as a block.  Its risk is that a real narrow feature
    is replaced in every sweep and the output simply looks clean, so the
    per-spectrum path warns whenever a pixel is flagged in more than
    ``PERSISTENT_FLAG_FRACTION`` of the sweeps (2-D input, ≥3 sweeps).  Silence
    the warning through ``warnings.filterwarnings`` if it is expected.

    The warning fires only with the veto off; with it on those detections are
    dropped rather than replaced.  Note that the veto keeps a hot pixel in
    ``cleaned`` — correctly, it is not a cosmic ray — but it cannot tell a hot
    pixel from a real spectral feature, so it does not label one.  To find
    detector defects, inspect ``cr_mask.mean(axis=1)`` on the default mask.
    """

    arr = np.asarray(spectra, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"spectra must be 1-D or 2-D, got {arr.ndim}-D with shape {arr.shape}."
        )
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}.")

    if median_window % 2 == 0:
        median_window += 1                     # enforce odd window

    # Work pixel-major, (n_pixels, n_sweeps), and undo at the end.
    flip = arr.ndim == 2 and axis == 1
    work = arr.T if flip else arr

    if work.ndim == 1:
        if cross_sweep_veto:
            raise ValueError(
                "cross_sweep_veto needs 2-D input (n_pixels, n_sweeps): a single "
                "spectrum has no neighbouring sweeps to compare against."
            )
        cr_mask, working = _detect_cosmic_rays_1d(
            work, sigma_threshold, median_window, max_iter
        )
        cleaned = work.copy()
        if cr_mask.any():
            local_median      = median_filter(working, size=median_window)
            cleaned[cr_mask]  = local_median[cr_mask]
        return cleaned, cr_mask

    n_sweeps = work.shape[1]
    if cross_sweep_veto and n_sweeps < 3:
        raise ValueError(
            f"cross_sweep_veto needs at least 3 sweeps to form a median, got {n_sweeps}."
        )

    cr_mask  = np.zeros(work.shape, dtype=bool)
    workings = np.empty_like(work)
    for j in range(n_sweeps):
        cr_mask[:, j], workings[:, j] = _detect_cosmic_rays_1d(
            work[:, j], sigma_threshold, median_window, max_iter
        )

    if cross_sweep_veto:
        cr_mask = _cross_sweep_veto(work, cr_mask, sigma_threshold, cross_sweep_window)
        # Vetoed pixels must go back to their raw values before replacement, so
        # rebuild the fills against the surviving mask.
        workings = work.copy()
        for j in range(n_sweeps):
            _fill_flagged(workings[:, j], cr_mask[:, j], median_window)
    elif n_sweeps >= 3:
        # Nothing here vetoes a repeating detection, so at least say it happened.
        _warn_persistent_flags(cr_mask)

    # Replace flagged pixels with the local median.
    cleaned = work.copy()
    for j in range(n_sweeps):
        col_mask = cr_mask[:, j]
        if col_mask.any():
            local_median              = median_filter(workings[:, j], size=median_window)
            cleaned[col_mask, j]      = local_median[col_mask]

    return (cleaned.T, cr_mask.T) if flip else (cleaned, cr_mask)