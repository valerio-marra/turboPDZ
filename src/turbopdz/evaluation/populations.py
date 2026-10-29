"""Exploratory PCA population/branch checks, independent of network fitting."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from ..data import quadrature_weights
from .diagnostics import numeric_observable
from .metrics import residual


def _edges(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all() or (np.diff(result) <= 0).any():
        raise ValueError("Population bin edges must be finite and increasing")
    return result


def _wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return np.nan, np.nan
    z, p = 1.95996398454, k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def pca_population_tables(frame: pd.DataFrame, pdz: np.ndarray, z_grid: np.ndarray,
                           scores: np.ndarray, *, representation: str,
                           redshift_edges: np.ndarray,
                           magnitude_edges: np.ndarray | None = None,
                           tail_fraction: float = 0.2, min_common: int = 15,
                           branches: Mapping | None = None,
                           mass_thresholds: tuple[float, ...] = (0.05, 0.1, 0.2),
                           investigation_window: tuple[float, float] | None = None) -> dict[str, pd.DataFrame]:
    """Compare PCA tails, population mix and optional competing PDZ solutions.

    Parameters
    ----------
    frame : pandas.DataFrame
        Row-aligned unique IDs, z_spec, z_ml and optional i_cmodel_mag.
    pdz, z_grid, scores : numpy.ndarray
        The same rows' native PDZs and fitted component scores.
    representation : {'mass', 'density'}
        Declared native convention; branch integrals honor quadrature weights.
    redshift_edges, magnitude_edges : numpy.ndarray, optional
        Prespecified population cells. No bins are relaxed to obtain overlap.
    tail_fraction : float
        Inclusive lower/upper quantiles. Ties may increase tail membership.
    min_common : int
        Minimum members in both low/other groups of each overlap cell.
    branches : mapping, optional
        Explicit true_low/true_high and pdz_low/pdz_high windows, plus
        predicted_low_max/predicted_high_min. No physical defaults are assumed.
    mass_thresholds : tuple of float
        Sensitivity thresholds for probability in each of the two PDZ windows.
    investigation_window : tuple of float, optional
        Additional redshift interval to inspect, without claiming completeness.

    Returns
    -------
    dict of pandas.DataFrame
        Objects, tail membership, bin error contributions, overlap controls and
        optional branch/ambiguity/sensitivity tables. Wilson intervals describe
        wrong-branch counts, not model-retraining uncertainty or causal effects.

    Notes
    -----
    Two-peak checks use one-native-bin Gaussian smoothing only on uniform grids;
    nonuniform grids retain branch integrals but report peak checks as missing.
    """
    table = frame.reset_index(drop=True).copy()
    if branches is not None and not isinstance(branches, Mapping):
        raise ValueError("branches must be a mapping; use Papermill -y for nested parameters, not -p JSON text")
    scores, pdz, grid = np.asarray(scores), np.asarray(pdz), np.asarray(z_grid)
    if (scores.shape != (len(table),) or pdz.shape != (len(table), len(grid))
            or not np.isfinite(scores).all() or table.object_id.duplicated().any()
            or not np.isfinite(table[["z_spec", "z_ml"]]).all(axis=None)):
        raise ValueError("PCA population inputs must align with unique IDs and finite scores")
    if not 0 < tail_fraction < 0.5 or min_common < 1:
        raise ValueError("Require 0 < tail_fraction < 0.5 and positive min_common")
    if representation not in {"mass", "density"}:
        raise ValueError("PDZ representation must be mass or density")
    if not np.isfinite(pdz).all() or (pdz < 0).any():
        raise ValueError("PDZ values must be finite and non-negative")
    weights = quadrature_weights(grid)
    masses = pdz if representation == "mass" else pdz * weights
    normalizer = masses.sum(axis=1)
    if (normalizer <= 0).any():
        raise ValueError("PDZ normalization must be positive")
    masses = masses / normalizer[:, None]
    redshift_edges = _edges(redshift_edges)
    low, high = np.quantile(scores, [tail_fraction, 1 - tail_fraction])
    low_mask, high_mask = scores <= low, scores >= high
    if (low_mask & high_mask).any():
        raise ValueError("Tied PCA boundaries overlap; choose a varying component")
    table["score"] = scores
    table["tail"] = np.select([low_mask, high_mask], ["low", "high"], default="middle")
    table["abs_error"] = np.abs(table.z_ml - table.z_spec)
    table["abs_dz"] = np.abs(residual(table.z_ml.to_numpy(), table.z_spec.to_numpy()))
    table["outlier"] = table.abs_dz > 0.15
    masks = {"low": low_mask, "high": high_mask, "other": ~low_mask}
    rows = []
    for name, mask in masks.items():
        total_error, total_outliers = table.loc[mask, "abs_error"].sum(), table.loc[mask, "outlier"].sum()
        for i, (lo, hi) in enumerate(zip(redshift_edges[:-1], redshift_edges[1:])):
            selected = mask & (table.z_spec >= lo) & ((table.z_spec <= hi) if i == len(redshift_edges) - 2 else (table.z_spec < hi))
            subset = table.loc[selected]
            rows.append(dict(group=name, bin=i, lower=lo, upper=hi, center=(lo + hi) / 2,
                             n_objects=len(subset), n_outliers=int(subset.outlier.sum()),
                             fraction_of_group=len(subset) / int(mask.sum()), mae=subset.abs_error.mean(),
                             outlier_rate=subset.outlier.mean(),
                             share_of_absolute_error=subset.abs_error.sum() / total_error if total_error else np.nan,
                             share_of_outliers=subset.outlier.sum() / total_outliers if total_outliers else np.nan))
    result = {"objects": table, "redshift_breakdown": pd.DataFrame(rows)}
    if magnitude_edges is not None and "i_cmodel_mag" in table:
        magnitude_edges = _edges(magnitude_edges)
        magnitude = numeric_observable(table, "i_cmodel_mag")
        cells = []
        # pd.cut includes the upper boundary through the final-bin extension.
        z_edges = redshift_edges.copy()
        m_edges = magnitude_edges.copy()
        z_edges[-1], m_edges[-1] = np.nextafter(z_edges[-1], np.inf), np.nextafter(m_edges[-1], np.inf)
        z_bin = pd.cut(table.z_spec, z_edges, right=False, labels=False)
        m_bin = pd.cut(magnitude, m_edges, right=False, labels=False)
        for i in range(len(redshift_edges) - 1):
            for j in range(len(magnitude_edges) - 1):
                selected = (z_bin == i) & (m_bin == j)
                a, b = table.loc[selected & low_mask], table.loc[selected & ~low_mask]
                if min(len(a), len(b)) < min_common:
                    continue
                cells.append(dict(z_bin=i, mag_bin=j, n_low=len(a), n_other=len(b),
                                  weight=min(len(a), len(b)), mae_low=a.abs_error.mean(), mae_other=b.abs_error.mean(),
                                  outliers_low=a.outlier.mean(), outliers_other=b.outlier.mean()))
        overlap = pd.DataFrame(cells, columns=["z_bin", "mag_bin", "n_low", "n_other", "weight",
                                               "mae_low", "mae_other", "outliers_low", "outliers_other"])
        result["common_redshift_magnitude_cells"] = overlap
        summary = dict(n_common_cells=len(overlap), low_tail_coverage=0.0,
                       standardized_mae_low=np.nan, standardized_mae_other=np.nan,
                       mae_difference_low_minus_other=np.nan, outlier_difference_low_minus_other=np.nan)
        if len(overlap):
            weight = overlap.weight
            summary.update(low_tail_coverage=overlap.n_low.sum() / int(low_mask.sum()),
                           standardized_mae_low=np.average(overlap.mae_low, weights=weight),
                           standardized_mae_other=np.average(overlap.mae_other, weights=weight),
                           mae_difference_low_minus_other=np.average(overlap.mae_low - overlap.mae_other, weights=weight),
                           outlier_difference_low_minus_other=np.average(overlap.outliers_low - overlap.outliers_other, weights=weight))
        result["overlap_standardized"] = pd.DataFrame([summary])
    if branches is None:
        return result
    windows = {name: _edges(branches[name]) for name in ("true_low", "true_high", "pdz_low", "pdz_high")}
    if any(len(window) != 2 for window in windows.values()):
        raise ValueError("Each branch window requires exactly two bounds")
    pred_low, pred_high = branches["predicted_low_max"], branches["predicted_high_min"]
    if (not np.isfinite([pred_low, pred_high]).all() or pred_low >= pred_high
            or windows["pdz_low"][1] >= windows["pdz_high"][0]
            or windows["true_low"][1] > windows["true_high"][0]):
        raise ValueError("Branch regions must be ordered and non-overlapping")
    branch_rows = []
    for name, mask in masks.items():
        for label, window, wrong in (("low", windows["true_low"], table.z_ml >= pred_high),
                                     ("high", windows["true_high"], table.z_ml < pred_low)):
            selected = mask & (table.z_spec >= window[0]) & (table.z_spec < window[1])
            n, k = int(selected.sum()), int((selected & wrong).sum())
            lo, hi = _wilson(k, n)
            branch_rows.append(dict(group=name, true_branch=label, n_objects=n, n_wrong_branch=k,
                                    wrong_branch_rate=k / n if n else np.nan, wilson95_low=lo, wilson95_high=hi,
                                    mae=table.loc[selected, "abs_error"].mean(), outliers=table.loc[selected, "outlier"].mean()))
    result["branch_confusion"] = pd.DataFrame(branch_rows)
    low_bins = (grid >= windows["pdz_low"][0]) & (grid < windows["pdz_low"][1])
    high_bins = (grid >= windows["pdz_high"][0]) & (grid < windows["pdz_high"][1])
    gap = (grid >= windows["pdz_low"][1]) & (grid < windows["pdz_high"][0])
    if not low_bins.any() or not high_bins.any() or not gap.any():
        raise ValueError("Native grid must sample both PDZ branches and their gap")
    table["p_low"], table["p_high"] = masses[:, low_bins].sum(axis=1), masses[:, high_bins].sum(axis=1)
    table["dual_branch_mass"] = (table.p_low >= 0.1) & (table.p_high >= 0.1)
    if np.allclose(np.diff(grid), np.diff(grid)[0]):
        density = gaussian_filter1d(masses / weights, sigma=1, axis=1, mode="nearest")
        table["two_branch_peaks"] = table.dual_branch_mass & (density[:, gap].min(axis=1) <
            0.75 * np.minimum(density[:, low_bins].max(axis=1), density[:, high_bins].max(axis=1)))
    else:
        table["two_branch_peaks"] = np.nan
    ranges = {"all": np.ones(len(table), dtype=bool)}
    for name in ("true_low", "true_high"):
        ranges[name] = (table.z_spec >= windows[name][0]) & (table.z_spec < windows[name][1])
    if investigation_window is not None:
        window = _edges(investigation_window)
        if len(window) != 2:
            raise ValueError("Investigation window needs exactly two bounds")
        ranges["investigation_window"] = (table.z_spec >= window[0]) & (table.z_spec < window[1])
    ambiguity, sensitivity = [], []
    for name, mask in masks.items():
        for label, region in ranges.items():
            subset = table.loc[mask & region]
            ambiguity.append(dict(group=name, redshift_range=label, n_objects=len(subset),
                                  dual_mass_fraction=subset.dual_branch_mass.mean(), two_peak_fraction=subset.two_branch_peaks.mean(),
                                  mean_p_low=subset.p_low.mean(), mean_p_high=subset.p_high.mean(),
                                  mae=subset.abs_error.mean(), outliers=subset.outlier.mean()))
            for threshold in mass_thresholds:
                if not 0 < threshold < 0.5:
                    raise ValueError("Branch mass thresholds must lie in (0, 0.5)")
                dual = (subset.p_low >= threshold) & (subset.p_high >= threshold)
                sensitivity.append(dict(group=name, redshift_range=label, threshold=threshold,
                                        n_objects=len(subset), dual_mass_fraction=dual.mean()))
    result["individual_pdz_ambiguity"] = pd.DataFrame(ambiguity)
    result["dual_mass_sensitivity"] = pd.DataFrame(sensitivity)
    return result
