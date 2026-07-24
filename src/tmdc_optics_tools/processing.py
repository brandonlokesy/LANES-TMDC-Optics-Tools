# tmdc_optics_tools/processing.py
"""
Spectral processing and normalisation routines.

All functions operate on plain NumPy arrays and are independent of any
particular loader class, so they can be used standalone or piped together.
"""

from __future__ import annotations

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

def remove_cosmic_rays(
        spectra : np.ndarray,
        sigma_threshold : float = 5.0,
        median_window : int = 7,
        max_iter: int = 3

    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosmic Ray Removal from a Single Spectrum
    
    Standard method: Iterative sigma-clipping on the Laplacian (second derivative).
    
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
    spectra : np.ndarray
        Raw 1-D spectra in counts
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

    Returns
    -------
    cleaned : ndarray
        Flux with cosmic ray pixels replaced by local median values.
    cr_mask : ndarray[bool]
        True at pixels identified as cosmic rays.

    """
    
    if median_window % 2 == 0:
        median_window += 1                     # enforce odd window

    spectra      = spectra.astype(float)
    n            = len(spectra)
    cr_mask      = np.zeros(n, dtype=bool)
 
    for iteration in range(max_iter):

        # Never modifies the original spectra
        working = spectra.copy()
        if cr_mask.any():
            # Smoothen the spectra
            local_med = median_filter(spectra, size = median_window)
            # Replace cosmic ray pixels with smoothened values
            working[cr_mask] = local_med[cr_mask]

        # Compute the laplacian on the good pixels
        laplacian = np.zeros(n)
        laplacian[1:-1]  = working[:-2] - 2.0 * working[1:-1] + working[2:]

        # Robust noise estimate from unflagged pixels.
        # MAD → σ conversion factor for a Gaussian: 1/0.6745.

        # Compute non cosmic ray pixels
        good      = ~cr_mask
        # Returns laplacian without cosmic ray pixels
        lap_good  = laplacian[good]
        # Noise estimate on the laplacian without cosmic rays
        mad       = np.median(np.abs(lap_good - np.median(lap_good)))
        sigma_lap = mad / 0.6745 # MAD ≈ 0.6745σ

    
        if sigma_lap == 0:
            break
 
        # Flag where Laplacian is a large negative outlier.
        # When the Laplacian is more than the threshold, we flag a cosmic ray
        new_flags   = laplacian < -sigma_threshold * sigma_lap
        # Newly identified cosmic ray pixels
        newly_found = new_flags & ~cosmic_mask
        # Combine old cosmic ray pixels with newly found cosmic ray pixels
        cosmic_mask |= new_flags

        if not newly_found.any():
            print(f"  Converged after {iteration + 1} iteration(s).")
            break
 
    # Replace flagged pixels with the local median.
    cleaned = spectra.copy()
    if cosmic_mask.any():
        local_median          = median_filter(spectra, size=median_window)
        cleaned[cosmic_mask]  = local_median[cosmic_mask]
 
    return cleaned, cosmic_mask