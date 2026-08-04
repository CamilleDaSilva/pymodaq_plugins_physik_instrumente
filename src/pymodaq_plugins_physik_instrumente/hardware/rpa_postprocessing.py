# -*- coding: utf-8 -*-
"""
Post-traitement de la courbe I-V du RPA (Retarding Potential Analyzer).

Fournit :
    - derivative()        : dI/dV lissée (Savitzky-Golay)
    - weighted_mean_std()  : moyenne / écart-type de la distribution en énergie
    - fit_gaussian()       : ajustement gaussien de la distribution en énergie

Author: Camille Da Silva
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def gaussian(x, amplitude, mean, sigma, offset):
    """Modèle gaussien avec offset."""
    return offset + amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _odd(n: int) -> int:
    """Force un entier à être impair (arrondi vers le haut)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def derivative(voltages, currents, window_length: int = 9, polyorder: int = 2):
    """
    Calcule dI/dV avec un filtre de Savitzky-Golay (lissage + dérivée en un
    seul passage), ce qui est nettement plus robuste au bruit qu'un simple
    np.gradient sur des données expérimentales bruitées.

    Parameters
    ----------
    voltages, currents : array-like, même longueur, voltages supposé
        régulièrement échantillonné (c'est le cas pour un balayage linéaire).
    window_length : int
        Largeur de la fenêtre de lissage (sera forcée à un nombre impair).
    polyorder : int
        Ordre du polynôme local (doit être < window_length).

    Returns
    -------
    np.ndarray
        dI/dV, même longueur que l'entrée.
    """
    voltages = np.asarray(voltages, dtype=float)
    currents = np.asarray(currents, dtype=float)
    n = len(voltages)

    if n < 5:
        # Pas assez de points pour un Savitzky-Golay fiable
        return np.gradient(currents, voltages)

    wl = _odd(min(window_length, n - 1 if n % 2 == 0 else n))
    wl = max(wl, _odd(polyorder + 2))  # wl doit être > polyorder
    wl = min(wl, n if n % 2 == 1 else n - 1)

    dv = np.mean(np.diff(voltages))

    try:
        return savgol_filter(currents, window_length=wl, polyorder=polyorder,
                              deriv=1, delta=abs(dv))
    except Exception:
        return np.gradient(currents, voltages)


def weighted_mean_std(voltages, distribution):
    """
    Moyenne et écart-type de la distribution en énergie des ions, pondérés
    par |distribution| (poids toujours positif, peu importe le signe de
    dI/dV selon la convention du Keithley).

    Returns
    -------
    (mean, std) : tuple de float, (nan, nan) si la distribution est nulle.
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
    Ajuste une gaussienne sur la distribution en énergie (typiquement
    dI/dV ou -dI/dV selon la convention retenue).

    Returns
    -------
    (params, fitted_curve)
        params = (amplitude, mean, sigma, offset), ou (None, None) en cas
        d'échec du fit (ex : distribution trop bruitée / plate).
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