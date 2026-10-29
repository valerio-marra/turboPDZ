"""Shared point-estimate, correlation, and retained-fraction metrics."""

from __future__ import annotations
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def _vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Metrics require a non-empty, finite, one-dimensional cohort")
    return array


def residual(z_pred: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    """Compute (z_pred-z_spec)/(1+z_spec), with no implicit row filtering."""
    pred, truth = _vector(z_pred), _vector(z_spec)
    if pred.shape != truth.shape or (truth <= -1).any():
        raise ValueError("Predictions/labels must be aligned and z_spec greater than -1")
    return (pred - truth) / (1 + truth)


def sigma_nmad(dz: np.ndarray) -> float:
    """Return 1.48 times the median absolute deviation about the median."""
    values = _vector(dz)
    return float(1.48 * np.median(np.abs(values - np.median(values))))


def eta_outliers(dz: np.ndarray, threshold: float = 0.15) -> float:
    """Return the fractional outlier rate, not percentage points."""
    if threshold <= 0:
        raise ValueError("Outlier threshold must be positive")
    return float(np.mean(np.abs(_vector(dz)) > threshold))


def photoz_metrics(z_pred: np.ndarray, z_spec: np.ndarray, threshold: float = 0.15) -> dict[str, float]:
    """Return redshift MAE, median bias, NMAD, and fractional outliers."""
    dz = residual(z_pred, z_spec)
    return {"mae": float(np.mean(np.abs(np.asarray(z_pred) - z_spec))),
            "bias": float(np.median(dz)), "absolute_bias": float(abs(np.median(dz))),
            "scatter": sigma_nmad(dz), "outliers": eta_outliers(dz, threshold)}


def error_correlations(abs_dz: np.ndarray, proxy: np.ndarray) -> dict[str, float]:
    """Correlate an error-scale proxy with an absolute residual on common rows."""
    target, scale = _vector(abs_dz), _vector(proxy)
    if target.shape != scale.shape:
        raise ValueError("Residual and proxy must be row aligned")
    if np.ptp(target) == 0 or np.ptp(scale) == 0:
        return {"spearman": np.nan, "pearson": np.nan}
    return {"spearman": float(spearmanr(target, scale).statistic),
            "pearson": float(pearsonr(target, scale).statistic)}


def retained_metrics(dz: np.ndarray, selector: np.ndarray, *, higher_is_better: bool,
                     fractions: np.ndarray | None = None, threshold: float = 0.15) -> pd.DataFrame:
    """Measure one residual target sorted by a declared selector direction."""
    values, score = _vector(dz), _vector(selector)
    if values.shape != score.shape:
        raise ValueError("Residual and selector must be row aligned")
    f = np.linspace(0.05, 1.0, 100) if fractions is None else _vector(fractions)
    if (f <= 0).any() or (f > 1).any() or (np.diff(f) <= 0).any():
        raise ValueError("Retained fractions must increase within (0,1]")
    rows = []
    for fraction in f:
        boundary = np.quantile(score, 1-fraction if higher_is_better else fraction)
        keep = score >= boundary if higher_is_better else score <= boundary
        selected = values[keep]
        count = int(keep.sum())
        rows.append({"retained_fraction": count / len(values), "requested_fraction": float(fraction), "n_retained": count,
                     "scatter": sigma_nmad(selected), "outliers": eta_outliers(selected, threshold)})
    return pd.DataFrame(rows).sort_values("retained_fraction", kind="stable").reset_index(drop=True)


def filter_auc(dz: np.ndarray, selector: np.ndarray, *, higher_is_better: bool,
               fractions: np.ndarray | None = None, threshold: float = 0.15) -> dict[str, float]:
    """Integrate retained-fraction metrics; smaller AUC is better."""
    curves = retained_metrics(dz, selector, higher_is_better=higher_is_better,
                             fractions=fractions, threshold=threshold)
    return {name: float(np.trapz(curves[name], curves["retained_fraction"]))
            for name in ("scatter", "outliers")}


def common_finite_mask(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build one shared mask before comparison, never one per estimator."""
    arrays = [np.asarray(value) for value in columns.values()]
    if not arrays or any(x.ndim != 1 or len(x) != len(arrays[0]) for x in arrays):
        raise ValueError("Columns must be non-empty and row aligned")
    return np.logical_and.reduce([np.isfinite(x) for x in arrays])
