# -*- coding: utf-8 -*-
"""
Post-processing for the RPA I-V curve (Retarding Potential Analyzer).

Provides:
    - derivative()        : smoothed dI/dV (Savitzky-Golay)
    - weighted_mean_std()  : mean / standard deviation of the energy distribution
    - fit_gaussian()       : Gaussian fit of the energy distribution

Author: Camille Da Silva
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def gaussian(x, amplitude, mean, sigma, offset):
    """Gaussian model with offset."""
    return offset + amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _odd(n: int) -> int:
    """Force an integer to be odd (rounded up)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def derivative(voltages, currents, window_length: int = 9, polyorder: int = 2):
    """
    Compute dI/dV with a Savitzky-Golay filter (smoothing + derivative in one
    pass), which is much more robust to noise than a simple np.gradient on
    noisy experimental data.

    Parameters
    ----------
    voltages, currents : array-like, same length, voltages assumed to be
        regularly sampled (which is the case for a linear sweep).
    window_length : int
        Smoothing window width (will be forced to an odd number).
    polyorder : int
        Local polynomial order (must be < window_length).

    Returns
    -------
    np.ndarray
        dI/dV, same length as the input.
    """
    voltages = np.asarray(voltages, dtype=float)
    currents = np.asarray(currents, dtype=float)
    n = len(voltages)

    if n < 5:
        # Not enough points for a reliable Savitzky-Golay filter
        return np.gradient(currents, voltages)

    wl = _odd(min(window_length, n - 1 if n % 2 == 0 else n))
    wl = max(wl, _odd(polyorder + 2))  # wl must be > polyorder
    wl = min(wl, n if n % 2 == 1 else n - 1)

    dv = np.mean(np.diff(voltages))

    try:
        return savgol_filter(currents, window_length=wl, polyorder=polyorder,
                              deriv=1, delta=abs(dv))
    except Exception:
        return np.gradient(currents, voltages)


def weighted_mean_std(voltages, distribution):
    """
    Mean and standard deviation of the ion energy distribution, weighted by
    |distribution| (weights are always positive, regardless of the sign of
    dI/dV according to the Keithley convention).

    Returns
    -------
    (mean, std) : tuple of floats, (nan, nan) if the distribution is zero.
    """
    voltages = np.asarray(voltages, dtype=float)
    weights = np.abs(np.asarray(distribution, dtype=float))
    total = np.sum(weights)

    if total <= 0:
        return float('nan'), float('nan')

    mean = np.sum(voltages * weights) / total
    variance = np.sum(weights * (voltages - mean) ** 2) / total
    return mean, np.sqrt(variance)


def fit_gaussian(voltages, distribution):
    """
    Fit a Gaussian to the energy distribution (typically dI/dV or -dI/dV,
    depending on the convention used).

    Returns
    -------
    (params, fitted_curve)
        params = (amplitude, mean, sigma, offset), or (None, None) if the fit
        fails (for example if the distribution is too noisy or flat).
    """
    voltages = np.asarray(voltages, dtype=float)
    distribution = np.asarray(distribution, dtype=float)

    mean0, sigma0 = weighted_mean_std(voltages, distribution)
    if np.isnan(mean0):
        return None, None
    if sigma0 <= 0 or np.isnan(sigma0):
        sigma0 = (voltages.max() - voltages.min()) / 6

    amplitude0 = np.max(distribution) - np.min(distribution)
    offset0 = np.median(distribution)
    p0 = [amplitude0, mean0, sigma0, offset0]

    try:
        popt, _ = curve_fit(gaussian, voltages, distribution, p0=p0, maxfev=5000)
    except Exception:
        return None, None

    fitted = gaussian(voltages, *popt)
    return popt, fitted