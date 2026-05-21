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
    x_axis     : str        = "energy",
    x_range    : tuple      = None,
    model      : str        = "lorentzian",
    sweep_mask : np.ndarray = None,
) -> list[FitResult]:
    """
    Fit a single peak in every sweep of an
    :class:`~tmdc_optics_tools.loaders.AttoCubePLVabScan`.

    Background subtraction and Jacobian correction are configured at load
    time on the scan object (via ``bg_region_nm`` / ``bg_region_eV`` and
    ``apply_jacobian``).  This function always uses
    :attr:`~tmdc_optics_tools.loaders.AttoCubePLVabScan.best_energy_spectra`
    for the energy axis, which automatically returns the background-corrected
    array when one is available.

    Parameters
    ----------
    scan : AttoCubePLVabScan
    x_axis : {"energy", "wavelength"}
    x_range : tuple of (x_min, x_max), optional
        Restrict the fit to this spectral window. Units match *x_axis*.
        Fits the full range if ``None``.
    model : {"lorentzian", "gaussian"}
        Peak shape to fit.
    sweep_mask : np.ndarray of bool, optional
        Boolean mask of length ``scan.n_sweeps``. Only sweeps where the
        mask is ``True`` are fitted; the rest receive a non-converged
        placeholder so that result indices stay aligned with ``scan.ef``.
        Fits all sweeps when ``None``.

    Returns
    -------
    list of FitResult, length = scan.n_sweeps
    """
    x       = scan.energy     if x_axis == "energy" else scan.wavelength
    spectra = scan.best_energy_spectra if x_axis == "energy" else scan.spectra
    fit_fn  = fit_lorentzian if model == "lorentzian" else fit_gaussian

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
    method : str
        Linear-fit method used: ``"wls"``, ``"minmax"``, or ``"bootstrap"``.
    n_bootstrap : int or None
        Number of bootstrap iterations. ``None`` unless ``method="bootstrap"``.
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
    method                : str = "wls"
    n_bootstrap           : int = None   # only meaningful for method="bootstrap"

    def __repr__(self) -> str:
        method_str = self.method
        if self.method == "bootstrap" and self.n_bootstrap is not None:
            method_str += f" (n={self.n_bootstrap})"
        return (
            f"DipoleResult\n"
            f"  Dipole length : {self.dipole_length:.4f} ± {self.dipole_length_err:.4f} nm"
            f"  ({self.dipole_length_angstrom:.2f} Å)\n"
            f"  Slope dE/dF   : {self.slope:.4e} ± {self.slope_err:.2e} eV/(mV/nm)\n"
            f"  Intercept E₀  : {self.intercept:.4f} ± {self.intercept_err:.4f} eV\n"
            f"  R²            : {self.r_squared:.4f}\n"
            f"  Peak model    : {self.peak_model}\n"
            f"  Method        : {method_str}\n"
            f"  Sweep points  : {self.converged_mask.sum()} / {len(self.converged_mask)} converged"
        )


def _prepare_dipole_data(
    scan,
    x_range      : tuple,
    model        : str,
    active_range : tuple,
) -> tuple:
    """
    Shared setup for all dipole extraction methods.

    Runs the per-sweep lineshape fits and constructs the masked arrays
    (ef_fit, E_fit, sig_fit) ready for a linear fit.

    Parameters
    ----------
    scan : AttoCubePLVabScan
    x_range : tuple or None
    model : str
    active_range : tuple or None
        Combined ef_range / Efield_range already resolved by the caller.

    Returns
    -------
    ef : np.ndarray
        Full electric field array (all sweeps).
    peak_energies : np.ndarray
        Fitted peak centers (all sweeps, NaN where not converged).
    peak_errors : np.ndarray
        1-sigma uncertainties on peak centers (NaN where not converged or
        where the covariance was unusable).
    converged : np.ndarray of bool
    ef_fit, E_fit, sig_fit : np.ndarray
        Masked arrays for the linear fit (converged + within active_range).
        sig_fit contains NaN where the covariance was unusable; each
        linear fitter handles these internally.
    """
    sweep_mask = None
    if active_range is not None:
        sweep_mask = (scan.ef >= active_range[0]) & (scan.ef <= active_range[1])

    fit_results   = fit_scan_peak(
        scan, x_axis="energy", x_range=x_range, model=model,
        sweep_mask=sweep_mask,
    )
    peak_energies = np.array([r.params["center"] for r in fit_results])
    peak_errors   = np.array([r.errors["center"]  for r in fit_results])
    converged     = np.array([r.converged          for r in fit_results])

    # Mark bad/zero errors as inf so they act as zero-weight points
    peak_errors = np.where(
        np.isfinite(peak_errors) & (peak_errors > 0), peak_errors, np.inf
    )

    ef   = scan.ef.copy()
    mask = converged.copy()
    if active_range is not None:
        mask &= (ef >= active_range[0]) & (ef <= active_range[1])

    if mask.sum() < 2:
        raise ValueError(
            f"Only {mask.sum()} usable sweep point(s) after applying field range "
            f"and removing non-converged fits. Need at least 2."
        )

    # Restore inf → NaN for the returned full arrays (clean display)
    peak_errors_out = np.where(np.isinf(peak_errors), np.nan, peak_errors)
    # sig_fit passed to fitters: NaN where inf (each fitter handles it)
    sig_fit = np.where(np.isinf(peak_errors[mask]), np.nan, peak_errors[mask])

    return ef, peak_energies, peak_errors_out, converged, ef[mask], peak_energies[mask], sig_fit


def _dipole_wls(
    ef_fit  : np.ndarray,
    E_fit   : np.ndarray,
    sig_fit : np.ndarray,
) -> tuple:
    """
    Weighted least squares (WLS / χ² minimisation) linear fit.

    Each point is weighted by 1/σ², and ``absolute_sigma=True`` ensures
    the covariance matrix has correct physical units so slope_err is a
    genuine 1-sigma uncertainty in eV/(mV/nm).

    Points with NaN sigma are given a very large sigma (effectively zero
    weight) so they don't influence the fit but don't cause it to fail.
    Falls back to unweighted polyfit if curve_fit fails.

    Returns
    -------
    slope, slope_err, intercept, intercept_err
    """
    sig_safe = np.where(np.isfinite(sig_fit), sig_fit, 1e10)
    try:
        popt, pcov = curve_fit(
            _linear, ef_fit, E_fit,
            sigma=sig_safe, absolute_sigma=True,
        )
        slope, intercept         = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        warnings.warn("WLS fit failed; falling back to unweighted polyfit.")
        slope, intercept         = np.polyfit(ef_fit, E_fit, 1)
        slope_err = intercept_err = np.nan
    return slope, slope_err, intercept, intercept_err


def _dipole_minmax(
    ef_fit  : np.ndarray,
    E_fit   : np.ndarray,
    sig_fit : np.ndarray,
) -> tuple:
    """
    Min/max slope method (extremal fit).

    Finds the steepest and shallowest lines still consistent with the
    data by solving a linear program with per-point slack variables s_i ≥ 0:

        minimise / maximise   slope
        subject to            slope·F_i + intercept ≤ E_i + σ_i + s_i
                              slope·F_i + intercept ≥ E_i - σ_i - s_i
                              s_i ≥ 0  for all i

    The slack terms penalise constraint violations so the LP is always
    feasible. A large penalty (1e6) on Σ s_i discourages slack from being
    used except where unavoidable (e.g. a point whose noise exceeds σ_i).

    Points with NaN σ are skipped — they impose no constraint.

    The best-fit slope and intercept come from an unweighted polyfit of
    the valid points.

    Returns
    -------
    slope, slope_err, intercept, intercept_err

    Notes
    -----
    slope_err     = (slope_max - slope_min) / 2
    intercept_err = (intercept_max - intercept_min) / 2
    """
    from scipy.optimize import linprog

    finite = np.isfinite(sig_fit)
    ef_v, E_v, sig_v = ef_fit[finite], E_fit[finite], sig_fit[finite]
    n = len(ef_v)

    if n < 2:
        warnings.warn("minmax: fewer than 2 finite-error points; returning NaN errors.")
        slope, intercept = np.polyfit(ef_fit, E_fit, 1)
        return slope, np.nan, intercept, np.nan

    # Variables: x = [slope, intercept, s_0, ..., s_{n-1}]
    # Constraints (2n inequalities):
    #   upper:  slope·F_i + intercept - s_i ≤  E_i + σ_i
    #   lower: -slope·F_i - intercept - s_i ≤ -(E_i - σ_i)
    A_ub = np.vstack([
        np.hstack([np.column_stack([ ef_v,  np.ones(n)]), -np.eye(n)]),
        np.hstack([np.column_stack([-ef_v, -np.ones(n)]), -np.eye(n)]),
    ])
    b_ub    = np.concatenate([E_v + sig_v, -(E_v - sig_v)])
    bounds  = [(None, None), (None, None)] + [(0, None)] * n

    slack_penalty = 1e6
    slopes, intercepts = [], []
    for sign in (+1, -1):   # maximise then minimise slope
        c     = np.zeros(2 + n)
        c[0]  = sign
        c[2:] = slack_penalty
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            slopes.append(res.x[0])
            intercepts.append(res.x[1])
        else:
            warnings.warn(f"minmax linprog did not converge (sign={sign:+d}).")
            slopes.append(np.nan)
            intercepts.append(np.nan)

    slope_max, slope_min         = max(slopes), min(slopes)
    intercept_max, intercept_min = max(intercepts), min(intercepts)

    slope, intercept = np.polyfit(ef_v, E_v, 1)
    slope_err        = (slope_max - slope_min) / 2.0
    intercept_err    = (intercept_max - intercept_min) / 2.0

    return slope, slope_err, intercept, intercept_err


def _dipole_bootstrap(
    ef_fit      : np.ndarray,
    E_fit       : np.ndarray,
    sig_fit     : np.ndarray,
    n_bootstrap : int = 2000,
    rng         : np.random.Generator = None,
) -> tuple:
    """
    Bootstrap resampling of the linear slope.

    For each iteration, each energy point is perturbed by a Gaussian
    draw scaled by its 1-sigma uncertainty:

        E'_i = E_i + ε_i,   ε_i ~ N(0, σ_i)

    A weighted least squares line is then fitted to the perturbed dataset.
    The slope uncertainty is the standard deviation of the resulting slope
    distribution. The best-fit slope and intercept come from a single WLS
    fit to the unperturbed data.

    Points with NaN σ receive zero perturbation (their uncertainty is
    unknown, so they are kept fixed) and very large sigma in the WLS
    weight (effectively zero weight).

    Parameters
    ----------
    ef_fit, E_fit, sig_fit : np.ndarray
    n_bootstrap : int
        Number of resampling iterations. Default 2000.
    rng : np.random.Generator, optional
        For reproducibility: pass ``np.random.default_rng(seed)``.

    Returns
    -------
    slope, slope_err, intercept, intercept_err
    """
    if rng is None:
        rng = np.random.default_rng()

    slope, _, intercept, _ = _dipole_wls(ef_fit, E_fit, sig_fit)

    sig_perturb = np.where(np.isfinite(sig_fit), sig_fit, 0.0)
    sig_wls     = np.where(np.isfinite(sig_fit), sig_fit, 1e10)

    boot_slopes     = np.empty(n_bootstrap)
    boot_intercepts = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        E_perturbed = E_fit + rng.normal(0.0, sig_perturb)
        try:
            popt, _ = curve_fit(
                _linear, ef_fit, E_perturbed,
                sigma=sig_wls, absolute_sigma=True,
            )
            boot_slopes[i], boot_intercepts[i] = popt
        except (RuntimeError, ValueError):
            boot_slopes[i]     = np.nan
            boot_intercepts[i] = np.nan

    slope_err     = np.nanstd(boot_slopes)
    intercept_err = np.nanstd(boot_intercepts)

    return slope, slope_err, intercept, intercept_err


def extract_dipole_length(
    scan,
    x_range      : tuple = None,
    model        : str   = "lorentzian",
    ef_range     : tuple = None,
    Efield_range : tuple = None,
    method       : str   = "wls",
    n_bootstrap  : int   = 2000,
    rng          : np.random.Generator = None,
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
    3. Fit a line E(F) = slope · F + intercept using the chosen *method*.
    4. Derive the dipole length and propagate uncertainties.

    .. note::
        Background subtraction is configured at load time on the scan
        object via ``bg_region_nm`` or ``bg_region_eV``.
        :func:`fit_scan_peak` automatically uses
        :attr:`~tmdc_optics_tools.loaders.AttoCubePLVabScan.best_energy_spectra`,
        which returns the background-corrected array when available.

    Parameters
    ----------
    scan : AttoCubePLVabScan
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
    Efield_range : tuple of (F_min, F_max) in mV/nm, optional
        Alias for *ef_range*. Takes precedence if both are supplied.
    method : {"wls", "minmax", "bootstrap"}
        Linear-fit method used to extract the slope and its uncertainty:

        ``"wls"``
            Weighted least squares (χ² minimisation). Each point is
            weighted by 1/σ². The slope uncertainty comes from the
            covariance matrix with ``absolute_sigma=True``. Statistically
            optimal when the σ_i are accurate and errors are Gaussian.

        ``"minmax"``
            Extremal fit (min/max slope). Finds the steepest and
            shallowest lines still consistent with all error bars via a
            linear program. The uncertainty is half the range between
            these extremes. Conservative and visually intuitive — the two
            extreme lines can be overlaid directly on the Stark-shift plot.

        ``"bootstrap"``
            Bootstrap resampling. Perturbs each E_i by a Gaussian draw
            scaled by σ_i, refits the slope n_bootstrap times, and
            reports the standard deviation of the slope distribution.
            Makes no assumptions beyond Gaussian per-point errors.

    n_bootstrap : int
        Number of bootstrap iterations. Only used when
        ``method="bootstrap"``. Default 2000.
    rng : np.random.Generator, optional
        Random number generator for reproducibility when using bootstrap,
        e.g. ``np.random.default_rng(42)``.

    Returns
    -------
    DipoleResult

    Raises
    ------
    ValueError
        If ``scan.ef`` is ``None``, fewer than 2 usable sweep points remain,
        or *method* is not recognised.

    Examples
    --------
    >>> # Default: weighted least squares
    >>> result = extract_dipole_length(scan, x_range=(1.30, 1.42))

    >>> # Min/max slope (conservative, visually intuitive)
    >>> result = extract_dipole_length(scan, x_range=(1.30, 1.42), method="minmax")

    >>> # Bootstrap with fixed seed for reproducibility
    >>> result = extract_dipole_length(
    ...     scan, x_range=(1.30, 1.42),
    ...     method="bootstrap", n_bootstrap=5000,
    ...     rng=np.random.default_rng(42),
    ... )
    >>> print(result)
    """
    _METHODS = ("wls", "minmax", "bootstrap")
    if method not in _METHODS:
        raise ValueError(
            f"method='{method}' is not recognised. Choose from {_METHODS}."
        )

    if scan.ef is None:
        raise ValueError(
            "scan.ef is None — supply a DeviceGeometry when loading the scan."
        )

    active_range = Efield_range if Efield_range is not None else ef_range

    # --- Shared setup: lineshape fits + masking ---
    ef, peak_energies, peak_errors, converged, ef_fit, E_fit, sig_fit = (
        _prepare_dipole_data(scan, x_range, model, active_range)
    )

    # --- Linear fit: dispatch to chosen method ---
    if method == "wls":
        slope, slope_err, intercept, intercept_err = _dipole_wls(
            ef_fit, E_fit, sig_fit
        )
    elif method == "minmax":
        slope, slope_err, intercept, intercept_err = _dipole_minmax(
            ef_fit, E_fit, sig_fit
        )
    else:  # bootstrap
        slope, slope_err, intercept, intercept_err = _dipole_bootstrap(
            ef_fit, E_fit, sig_fit, n_bootstrap=n_bootstrap, rng=rng,
        )

    # --- Derived quantities ---
    r_squared         = _r_squared(E_fit, slope * ef_fit + intercept)
    dipole_length     = abs(slope) * 1000.0
    dipole_length_err = abs(slope_err) * 1000.0 if np.isfinite(slope_err) else np.nan

    return DipoleResult(
        ef                     = ef,
        peak_energies          = peak_energies,
        peak_errors            = peak_errors,
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
        method                 = method,
        n_bootstrap            = n_bootstrap if method == "bootstrap" else None,
    )