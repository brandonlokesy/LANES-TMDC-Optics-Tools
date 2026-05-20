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