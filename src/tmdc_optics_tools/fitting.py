# tmdc_optics_tools/fitting.py
"""
Common fitting routines for TMD spectroscopy.

All public fitting functions follow the convention:
    fit_*(x, y, ...) -> FitResult

where FitResult is a lightweight dataclass holding parameters, errors,
the best-fit curve, and goodness-of-fit metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Container for a single spectral fit.

    Attributes
    ----------
    params : dict[str, float]
        Best-fit parameter values keyed by name.
    errors : dict[str, float]
        1-sigma uncertainties (sqrt of diagonal of covariance matrix).
    x_fit : np.ndarray
        x values used for the fit.
    y_fit : np.ndarray
        Best-fit curve evaluated on *x_fit*.
    residuals : np.ndarray
        y - y_fit on *x_fit*.
    r_squared : float
        Coefficient of determination R².
    model : str
        Name of the model function used.
    converged : bool
        Whether ``curve_fit`` converged.
    """
    params    : dict
    errors    : dict
    x_fit     : np.ndarray
    y_fit     : np.ndarray
    residuals : np.ndarray
    r_squared : float
    model     : str
    converged : bool = True

    def __repr__(self) -> str:
        lines = [f"FitResult [{self.model}]  R²={self.r_squared:.4f}"]
        for k, v in self.params.items():
            lines.append(f"  {k:12s} = {v:.5g} ± {self.errors[k]:.2g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Line-shape functions
# ---------------------------------------------------------------------------

def lorentzian(x, amplitude, center, fwhm):
    """Single Lorentzian peak."""
    gamma = fwhm / 2.0
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def gaussian(x, amplitude, center, fwhm):
    """Single Gaussian peak."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)


def voigt_approx(x, amplitude, center, fwhm_g, fwhm_l):
    """
    Pseudo-Voigt approximation (Thompson et al. 1987).
    A weighted sum of Gaussian and Lorentzian with the same FWHM.
    """
    fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
    eta  = (1.36603 * (fwhm_l / fwhm)
            - 0.47719 * (fwhm_l / fwhm)**2
            + 0.11116 * (fwhm_l / fwhm)**3)
    return amplitude * (
        eta * lorentzian(x, 1.0, center, fwhm) +
        (1 - eta) * gaussian(x, 1.0, center, fwhm)
    )


def multi_lorentzian(x, *params):
    """
    Sum of N Lorentzians. ``params`` must have length 3N:
    ``[amp1, cen1, fwhm1, amp2, cen2, fwhm2, ...]``
    """
    if len(params) % 3 != 0:
        raise ValueError("multi_lorentzian requires 3 parameters per peak.")
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        result += lorentzian(x, params[i], params[i + 1], params[i + 2])
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear(x, slope, intercept):
    """Simple linear model for use with curve_fit."""
    return slope * x + intercept


def _r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _make_result(
    model_name  : str,
    model_fn,
    param_names : list,
    popt        : np.ndarray,
    pcov,
    x           : np.ndarray,
    y           : np.ndarray,
    converged   : bool,
) -> FitResult:
    """Build a FitResult from curve_fit outputs."""
    perr  = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
    y_fit = model_fn(x, *popt)
    return FitResult(
        params    = dict(zip(param_names, popt)),
        errors    = dict(zip(param_names, perr)),
        x_fit     = x,
        y_fit     = y_fit,
        residuals = y - y_fit,
        r_squared = _r_squared(y, y_fit),
        model     = model_name,
        converged = converged,
    )


def _fit_single_peak(
    model_fn,
    model_name : str,
    x          : np.ndarray,
    y          : np.ndarray,
    p0         : tuple,
    bounds     : tuple,
) -> FitResult:
    """
    Shared implementation for single-peak fits (Lorentzian / Gaussian).
    Avoids duplicating the try/except and _make_result call in every fitter.
    """
    try:
        popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=5000)
        converged  = True
    except RuntimeError:
        popt, pcov = np.array(p0, dtype=float), None
        converged  = False
        warnings.warn(f"{model_name} fit did not converge.")

    return _make_result(model_name, model_fn,
                        ["amplitude", "center", "fwhm"],
                        popt, pcov, x, y, converged)


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def fit_lorentzian(
    x      : np.ndarray,
    y      : np.ndarray,
    p0     : tuple = None,
    bounds : tuple = (-np.inf, np.inf),
) -> FitResult:
    """
    Fit a single Lorentzian peak.

    Parameters
    ----------
    x, y : array-like
        Spectral data.
    p0 : tuple of (amplitude, center, fwhm), optional
        Initial guess. Auto-estimated if ``None``.
    bounds : tuple
        Passed to ``scipy.optimize.curve_fit``.

    Returns
    -------
    FitResult
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if p0 is None:
        p0 = (y.max(), x[np.argmax(y)], (x[-1] - x[0]) / 10)
    return _fit_single_peak(lorentzian, "lorentzian", x, y, p0, bounds)


def fit_gaussian(
    x      : np.ndarray,
    y      : np.ndarray,
    p0     : tuple = None,
    bounds : tuple = (-np.inf, np.inf),
) -> FitResult:
    """
    Fit a single Gaussian peak. Same signature as :func:`fit_lorentzian`.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if p0 is None:
        p0 = (y.max(), x[np.argmax(y)], (x[-1] - x[0]) / 10)
    return _fit_single_peak(gaussian, "gaussian", x, y, p0, bounds)


def fit_multi_lorentzian(
    x           : np.ndarray,
    y           : np.ndarray,
    n_peaks     : int  = None,
    p0          : list = None,
    bounds      : tuple = None,
    peak_kwargs : dict = None,
) -> FitResult:
    """
    Fit a sum of N Lorentzian peaks.

    Parameters
    ----------
    x, y : array-like
    n_peaks : int, optional
        Number of peaks. Inferred from ``p0`` length if not given.
    p0 : list of (amp, center, fwhm) per peak, optional
        If ``None``, peaks are found automatically via
        ``scipy.signal.find_peaks``.
    bounds : tuple, optional
        ``([lower, ...], [upper, ...])`` passed to ``curve_fit``.
        Auto-constructed if ``None``.
    peak_kwargs : dict, optional
        Extra kwargs forwarded to ``scipy.signal.find_peaks`` during
        automatic peak detection.

    Returns
    -------
    FitResult
        ``params`` keys are ``amp_0``, ``center_0``, ``fwhm_0``, ``amp_1``, …
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = x[-1] - x[0]

    if p0 is None:
        pk_kw = peak_kwargs or {}
        peaks, _ = find_peaks(y, height=y.max() * 0.1, **pk_kw)
        if n_peaks is not None:
            order = np.argsort(y[peaks])[::-1]
            peaks = peaks[order[:n_peaks]]
        p0 = []
        for pk in peaks:
            p0.extend([y[pk], x[pk], span / 20])

    p0_flat = np.array(p0).ravel()
    n = len(p0_flat) // 3

    if bounds is None:
        lo = [0,      x.min(), 0   ] * n
        hi = [np.inf, x.max(), span] * n
        bounds = (lo, hi)

    param_names = [name for i in range(n)
                   for name in (f"amp_{i}", f"center_{i}", f"fwhm_{i}")]

    try:
        popt, pcov = curve_fit(
            multi_lorentzian, x, y, p0=p0_flat, bounds=bounds, maxfev=10000
        )
        converged = True
    except RuntimeError:
        popt, pcov = p0_flat, None
        converged  = False
        warnings.warn("Multi-Lorentzian fit did not converge.")

    perr  = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
    y_fit = multi_lorentzian(x, *popt)
    return FitResult(
        params    = dict(zip(param_names, popt)),
        errors    = dict(zip(param_names, perr)),
        x_fit     = x,
        y_fit     = y_fit,
        residuals = y - y_fit,
        r_squared = _r_squared(y, y_fit),
        model     = "multi_lorentzian",
        converged = converged,
    )


def fit_scan_peak(
    scan,
    x_axis  : str   = "energy",
    x_range : tuple = None,
    model   : str   = "lorentzian",
    bg_region : tuple = None,
    sweep_mask : np.ndarray = None,
) -> list[FitResult]:
    """
    Fit a single peak in every sweep of an
    :class:`~tmdc_optics_tools.loaders.AttoCubePLScan`.

    Parameters
    ----------
    scan : AttoCubePLScan
    x_axis : {"energy", "wavelength"}
    x_range : tuple of (x_min, x_max), optional
        Restrict the fit to this spectral window. Fits the full range
        if ``None``.
    model : {"lorentzian", "gaussian"}
        Peak shape to fit.
    bg_region : tuple of (x_min, x_max), optional
        Spectral range (nm) for background subtraction

    Returns
    -------
    list of FitResult, length = scan.n_sweeps
    """

    from .processing import subtract_background

    x       = scan.energy     if x_axis == "energy" else scan.wavelength
    spectra = scan.energy_spectra if x_axis == "energy" else scan.spectra
    fit_fn  = fit_lorentzian if model == "lorentzian" else fit_gaussian

    if bg_region is not None:
        spectra = subtract_background(spectra, bg_region=bg_region, x=x, axis=0)

    if x_range is not None:
        px_mask = (x >= x_range[0]) & (x <= x_range[1])
        x       = x[px_mask]
    else:
        px_mask = np.ones(len(x), dtype=bool)

    if sweep_mask is None:
        sweep_mask = np.ones(scan.n_sweeps, dtype=bool)

    # Fit only selected sweeps; insert a non-converged placeholder for the rest
    results = []
    for i in range(scan.n_sweeps):
        if sweep_mask[i]:
            results.append(fit_fn(x, spectra[px_mask, i].astype(float)))
        else:
            # Placeholder so indices stay aligned with scan.ef
            results.append(FitResult(
                params    = {"amplitude": np.nan, "center": np.nan, "fwhm": np.nan},
                errors    = {"amplitude": np.nan, "center": np.nan, "fwhm": np.nan},
                x_fit     = x,
                y_fit     = np.full_like(x, np.nan),
                residuals = np.full_like(x, np.nan),
                r_squared = np.nan,
                model     = model,
                converged = False,
            ))
    return results


# ---------------------------------------------------------------------------
# Dipole length extraction
# ---------------------------------------------------------------------------

@dataclass
class DipoleResult:
    """
    Result of a dipole length extraction from a gate-dependent PL scan.

    The dipole length is extracted from the linear Stark shift:

        ΔE = −d · F   →   d = −dE/dF · (1/e)

    Because energies are already in eV (divided by e), the (1/e) factor
    drops out and the dipole length in nm is:

        d [nm] = |dE [eV] / dF [V/nm]|
               = |slope [eV/(mV/nm)]| × 1000

    Attributes
    ----------
    ef : np.ndarray
        Electric field values for all sweep points (mV/nm).
    peak_energies : np.ndarray
        Fitted peak center energies at each field point (eV).
    peak_errors : np.ndarray
        1-sigma uncertainty on each peak center (eV).
        ``NaN`` where the lineshape fit did not converge.
    slope : float
        Linear slope dE/dF in eV/(mV/nm).
    slope_err : float
        1-sigma uncertainty on the slope (eV/(mV/nm)).
    intercept : float
        Linear intercept E₀ in eV (energy at zero field).
    intercept_err : float
        1-sigma uncertainty on the intercept (eV).
    dipole_length : float
        |slope| × 1000, in nm.
    dipole_length_err : float
        Propagated 1-sigma uncertainty on the dipole length (nm).
    dipole_length_angstrom : float
        Dipole length in Ångström (dipole_length × 10).
    r_squared : float
        R² of the linear fit.
    peak_model : str
        Lineshape model used to extract peak centers (e.g. ``"lorentzian"``).
    converged_mask : np.ndarray of bool
        True for sweep points where the lineshape fit converged.
    """
    ef                    : np.ndarray
    peak_energies         : np.ndarray
    peak_errors           : np.ndarray
    slope                 : float
    slope_err             : float
    intercept             : float
    intercept_err         : float
    dipole_length         : float
    dipole_length_err     : float
    dipole_length_angstrom: float
    r_squared             : float
    peak_model            : str
    converged_mask        : np.ndarray

    def __repr__(self) -> str:
        return (
            f"DipoleResult\n"
            f"  Dipole length : {self.dipole_length:.4f} ± {self.dipole_length_err:.4f} nm"
            f"  ({self.dipole_length_angstrom:.2f} Å)\n"
            f"  Slope dE/dF   : {self.slope:.4e} ± {self.slope_err:.2e} eV/(mV/nm)\n"
            f"  Intercept E₀  : {self.intercept:.4f} ± {self.intercept_err:.4f} eV\n"
            f"  R²            : {self.r_squared:.4f}\n"
            f"  Peak model    : {self.peak_model}\n"
            f"  Sweep points  : {self.converged_mask.sum()} / {len(self.converged_mask)} converged"
        )


def extract_dipole_length(
    scan,
    x_range  : tuple = None,
    model    : str   = "lorentzian",
    ef_range : tuple = None,
    bg_region : tuple = None,
    Efield_range : tuple = None,
) -> DipoleResult:
    """
    Extract the excitonic dipole length from the DC Stark shift in a
    gate-dependent PL scan.

    The procedure is:

    1. Fit a lineshape to the PL spectrum at every sweep point on the
       **energy** axis (restricted to *x_range* if supplied). This gives
       a fitted peak center energy E(F) at each field F, together with a
       per-point uncertainty σ from the covariance matrix.
    2. Optionally restrict the field range used for the linear fit to
       *ef_range* (e.g. to exclude the non-linear high-field regime).
    3. Perform a weighted linear fit E(F) = slope · F + intercept, using
       per-point weights 1/σ². Non-converged sweep points are excluded.
    4. Derive the dipole length and propagate uncertainties.

    .. note::
        ``fit_scan_peak`` is used rather than ``extract_spectra_peak``
        because:

        * The fit center is insensitive to single noisy/hot pixels.
        * The fit operates on the energy axis, avoiding the
          wavelength→energy Jacobian bias that affects a raw ``argmax``.
        * Per-point uncertainties are propagated into the final error.

    Parameters
    ----------
    scan : AttoCubePLScan
        Must have ``ef`` set (requires a
        :class:`~tmdc_optics_tools.loaders.DeviceGeometry`).
    x_range : tuple of (E_min, E_max) in eV, optional
        Spectral window for the lineshape fit. Strongly recommended to
        zoom in on the exciton of interest.
    model : {"lorentzian", "gaussian"}
        Lineshape model. Lorentzian is the physical choice for
        homogeneously broadened excitons.
    ef_range : tuple of (F_min, F_max) in mV/nm, optional
        Restrict the linear fit to this field range.
    bg_region : tuple of (x_min, x_max) in nm, optional
        Spectral range for background subtraction before fitting. 

    Returns
    -------
    DipoleResult

    Raises
    ------
    ValueError
        If ``scan.ef`` is ``None`` or fewer than 2 usable sweep points remain.

    Examples
    --------
    >>> result = extract_dipole_length(scan, x_range=(1.30, 1.42))
    >>> print(result)
    """

    if scan.ef is None:
        raise ValueError(
            "scan.ef is None — supply a DeviceGeometry when loading the scan."
        )

    # Efield_range restricts which sweeps are fitted at all
    active_range = Efield_range if Efield_range is not None else ef_range
    if active_range is not None:
        sweep_mask = (scan.ef >= active_range[0]) & (scan.ef <= active_range[1])
    else:
        sweep_mask = None

    # --- Step 1: lineshape fit at selected sweep points only ---
    fit_results   = fit_scan_peak(
        scan, x_axis="energy", x_range=x_range, model=model,
        bg_region=bg_region, sweep_mask=sweep_mask,
    )
    peak_energies = np.array([r.params["center"] for r in fit_results])
    peak_errors   = np.array([r.errors["center"]  for r in fit_results])
    converged     = np.array([r.converged          for r in fit_results])

    peak_errors = np.where(np.isfinite(peak_errors) & (peak_errors > 0),
                           peak_errors, np.inf)

    # --- Step 2: mask for linear fit (converged + within Efield_range) ---
    ef   = scan.ef.copy()
    mask = converged.copy()
    if active_range is not None:
        mask &= (ef >= active_range[0]) & (ef <= active_range[1])

    if mask.sum() < 2:
        raise ValueError(
            f"Only {mask.sum()} usable sweep point(s) after applying Efield_range "
            f"and removing non-converged fits. Need at least 2."
        )

    ef_fit  = ef[mask]
    E_fit   = peak_energies[mask]
    sig_fit = peak_errors[mask]

    # --- Step 3: weighted linear fit ---
    try:
        popt, pcov = curve_fit(
            _linear, ef_fit, E_fit,
            sigma=sig_fit, absolute_sigma=True,
        )
        slope, intercept         = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        warnings.warn(
            "Weighted linear fit failed; falling back to unweighted polyfit."
        )
        slope, intercept = np.polyfit(ef_fit, E_fit, 1)
        slope_err = intercept_err = np.nan

    # --- Step 4: derived quantities ---
    r_squared         = _r_squared(E_fit, slope * ef_fit + intercept)
    dipole_length     = abs(slope) * 1000.0
    dipole_length_err = abs(slope_err) * 1000.0 if np.isfinite(slope_err) else np.nan

    return DipoleResult(
        ef                     = ef,
        peak_energies          = peak_energies,
        peak_errors            = np.where(np.isinf(peak_errors), np.nan, peak_errors),
        slope                  = slope,
        slope_err              = slope_err,
        intercept              = intercept,
        intercept_err          = intercept_err,
        dipole_length          = dipole_length,
        dipole_length_err      = dipole_length_err,
        dipole_length_angstrom = dipole_length * 10.0,
        r_squared              = r_squared,
        peak_model             = model,
        converged_mask         = converged,
    )