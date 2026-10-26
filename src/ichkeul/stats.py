"""Small statistics helpers: Mann-Kendall / Sen's slope (series and per pixel),
Olofsson et al. (2014) error-adjusted area estimation."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def mann_kendall(y: np.ndarray, x: np.ndarray | None = None, alpha: float = 0.05) -> dict:
    """Mann-Kendall test with tie correction and Sen's slope with its CI (Gilbert 1987)."""
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    ok = np.isfinite(y)
    y, x = y[ok], x[ok]
    n = len(y)
    i, j = np.triu_indices(n, 1)
    s = np.sign(y[j] - y[i]).sum()
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - (counts * (counts - 1) * (2 * counts + 5)).sum()) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / math.sqrt(var)
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    slopes = np.sort((y[j] - y[i]) / (x[j] - x[i]))
    c = sps.norm.ppf(1 - alpha / 2) * math.sqrt(var)
    m1, m2 = int(round((len(slopes) - c) / 2)), int(round((len(slopes) + c) / 2))
    lo = slopes[max(m1 - 1, 0)]
    hi = slopes[min(m2, len(slopes) - 1)]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    tau = s / (n * (n - 1) / 2)
    return {"n": n, "S": float(s), "tau": float(tau), "z": float(z), "p": float(p), "sen_slope": slope,
            "slope_ci95": [float(lo), float(hi)], "intercept": intercept,
            "trend": "increasing" if (p < alpha and slope > 0) else "decreasing" if (p < alpha and slope < 0)
            else "no significant trend"}


def mk_pixels(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Mann-Kendall z-score and Sen slope over axis 0 (no tie correction)."""
    n = stack.shape[0]
    i, j = np.triu_indices(n, 1)
    d = stack[j] - stack[i]
    s = np.sign(d).sum(0)
    var = n * (n - 1) * (2 * n + 5) / 18
    z = np.where(s > 0, (s - 1) / np.sqrt(var), np.where(s < 0, (s + 1) / np.sqrt(var), 0.0))
    slope = np.nanmedian(d / (j - i)[:, None, None], axis=0)
    return z.astype(np.float32), slope.astype(np.float32)


def olofsson(cm_counts: np.ndarray, mapped_area: np.ndarray, conf: float = 0.95) -> dict:
    """Error-adjusted areas from a stratified sample (rows = map class, cols = reference).

    cm_counts[i, j]: number of sample units mapped as i and labelled j.
    mapped_area[i]: mapped area of class i (any unit). Olofsson et al. 2014, eqs. 9-10.
    """
    n_i = cm_counts.sum(1, keepdims=True).astype(float)
    W = mapped_area / mapped_area.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_i > 0, W[:, None] * cm_counts / n_i, 0.0)
        p_j = p.sum(0)
        A = mapped_area.sum() * p_j
        se = np.sqrt(np.nansum(np.where(n_i > 1, (W[:, None] ** 2) * (cm_counts / n_i) * (1 - cm_counts / n_i)
                                        / (n_i - 1), 0.0), axis=0))
        users = np.diag(p) / p.sum(1)
        producers = np.diag(p) / p_j
    zc = sps.norm.ppf(1 - (1 - conf) / 2)
    return {"area": A.tolist(), "ci": (zc * se * mapped_area.sum()).tolist(), "overall_accuracy": float(np.trace(p)),
            "users_accuracy": np.nan_to_num(users).tolist(), "producers_accuracy": np.nan_to_num(producers).tolist()}
