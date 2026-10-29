"""Single-catalog diagnostic tables, with explicit cohorts and finite masks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .metrics import photoz_metrics, error_correlations, sigma_nmad

if TYPE_CHECKING:
    from ..preprocessing import Preprocessor

_EMPTY_POINT = dict.fromkeys(("mae", "bias", "absolute_bias", "scatter", "outliers"), np.nan)
_EMPTY_SCALE = dict.fromkeys(("spearman", "pearson", "median_abs_error", "median_scale",
                             "median_abs_scaled", "frac_abs_scaled_le_1", "frac_abs_scaled_gt_3"), np.nan)


def numeric_observable(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Read a numeric column, honoring HSC null flags and finite sentinels."""
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    flag = f"{name}_isnull"
    if flag in frame:
        values[frame[flag].fillna(True).astype(bool).to_numpy()] = np.nan
    return values


def align_predictions(frame: pd.DataFrame, observables: pd.DataFrame,
                      object_id: np.ndarray) -> pd.DataFrame:
    """Attach observables by unique string ID, never by row position."""
    result = frame.copy()
    result["object_id"] = result.object_id.astype(str)
    obs = observables.reset_index(drop=True).copy()
    obs["object_id"] = np.asarray(object_id).astype(str)
    if result.object_id.duplicated().any() or obs.object_id.duplicated().any():
        raise ValueError("Diagnostic inputs require unique object IDs")
    if not result.object_id.isin(obs.object_id).all():
        raise ValueError("Prediction IDs are not present in the dataset")
    columns = [name for name in obs if name not in result or name == "object_id"]
    result = result.merge(obs[columns], on="object_id", how="left", validate="one_to_one", sort=False)
    for name in observables:
        if name != "object_id" and pd.api.types.is_numeric_dtype(observables[name]) and not name.endswith("_isnull"):
            result[name] = numeric_observable(result, name)
    return result


def point_summary(frame: pd.DataFrame, *, cohort: str) -> pd.DataFrame:
    """Summarize both point estimates on one common finite-object mask."""
    mask = np.isfinite(frame[["z_spec", "z_ml", "z_best"]]).all(axis=1)
    rows = []
    for subset, selected in (("all", mask), ("z_spec<2", mask & (frame.z_spec < 2))):
        for estimator in ("z_ml", "z_best"):
            row = {"cohort": cohort, "subset": subset, "estimator": estimator,
                   "n_objects": int(selected.sum()), **_EMPTY_POINT}
            if selected.any():
                row.update(photoz_metrics(frame.loc[selected, estimator].to_numpy(), frame.loc[selected, "z_spec"].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def point_metrics_by_feature(frame: pd.DataFrame, column: str, *, cohort: str,
                             n_bins: int = 10, edges: np.ndarray | None = None) -> pd.DataFrame:
    """Return bin and cumulative metrics with common estimator masks/counts.

    Bins are left-closed/right-open except for the inclusive final edge.
    Cumulative rows retain objects up to each upper edge. Empty bins are recorded
    with count zero and NaN metrics, not silently discarded or assigned zeros.
    """
    values = numeric_observable(frame, column)
    valid = np.isfinite(values) & np.isfinite(frame[["z_spec", "z_ml", "z_best"]]).all(axis=1).to_numpy()
    if edges is None:
        if not valid.any() or n_bins < 1:
            raise ValueError("Binning requires finite objects and positive n_bins")
        lo, hi = values[valid].min(), values[valid].max()
        edges = np.linspace(lo, hi if hi > lo else lo + 1, n_bins + 1)
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 2 or np.isnan(edges).any() or (np.diff(edges) <= 0).any():
        raise ValueError("Bin edges must increase")
    rows = []
    for kind in ("bin", "cumulative"):
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            below = values <= high if i == len(edges) - 2 else values < high
            selected = valid & below & ((values >= low) if kind == "bin" else (values >= edges[0]))
            for estimator in ("z_ml", "z_best"):
                row = dict(cohort=cohort, feature=column, kind=kind, bin=i,
                           lower=low, upper=high, center=(low + high) / 2,
                           estimator=estimator, n_objects=int(selected.sum()), **_EMPTY_POINT)
                if selected.any():
                    row.update(photoz_metrics(frame.loc[selected, estimator].to_numpy(), frame.loc[selected, "z_spec"].to_numpy()))
                rows.append(row)
    return pd.DataFrame(rows)


def error_scale_table(frame: pd.DataFrame, *, edges=(0, 0.5, 0.8, np.inf),
                       cohort: str = "final_test") -> pd.DataFrame:
    """Compare native/common residuals, error scales and shape diagnostics.

    All target/proxy cells share one finite positive-scale mask. The fraction
    |Delta z|/scale <= 1 is descriptive; empirical median-error scales are not
    Gaussian posterior sigmas and this is not a promised 68-percent coverage.
    """
    proxies = [name for name in ("sigma_ml", "s_best") if name in frame]
    required = ["z_spec", "dz_ml", "dz_best", *proxies]
    common = np.isfinite(frame[required]).all(axis=1).to_numpy()
    for name in proxies:
        common &= frame[name].to_numpy() > 0
    bins = [("all", -np.inf, np.inf)] + [(f"bin_{i}", lo, hi) for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    rows = []
    for label, lo, hi in bins:
        selected = common & (frame.z_spec.to_numpy() >= lo) & (frame.z_spec.to_numpy() < hi)
        for target in ("dz_ml", "dz_best"):
            for proxy in proxies:
                row = dict(cohort=cohort, redshift_bin=label, lower=lo, upper=hi,
                           target=target, proxy=proxy, n_objects=int(selected.sum()), **_EMPTY_SCALE)
                if selected.any():
                    error = np.abs(frame.loc[selected, target].to_numpy())
                    scale = frame.loc[selected, proxy].to_numpy()
                    ratio = error / scale
                    row.update(error_correlations(error, scale), median_abs_error=np.median(error),
                               median_scale=np.median(scale), median_abs_scaled=np.median(ratio),
                               frac_abs_scaled_le_1=np.mean(ratio <= 1), frac_abs_scaled_gt_3=np.mean(ratio > 3))
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_profile(raw_scale: np.ndarray, dz: np.ndarray, evaluated_scale: np.ndarray,
                         edges: np.ndarray, *, cohort: str, version: str) -> pd.DataFrame:
    """Evaluate fixed calibration-derived bins without fitting on test data."""
    raw, target, scale = map(lambda x: np.asarray(x, dtype=float), (raw_scale, dz, evaluated_scale))
    if raw.ndim != 1 or raw.shape != target.shape or raw.shape != scale.shape:
        raise ValueError("Calibration profile vectors must align")
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.isnan(edges).any() or (np.diff(edges) <= 0).any():
        raise ValueError("Calibration profile edges must increase")
    common = np.isfinite(raw) & np.isfinite(target) & np.isfinite(scale) & (raw > 0) & (scale > 0)
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = common & (raw >= low) & (raw < high)
        row = dict(cohort=cohort, version=version, bin=i, lower=low, upper=high,
                   n_objects=int(selected.sum()), **dict.fromkeys(
                       ("raw_mid", "median_scale", "median_abs_error", "m_bin", "mad1_bin",
                        "median_abs_scaled", "smad_element_bin", "frac_abs_scaled_le_1"), np.nan))
        if selected.any():
            ratio = np.abs(target[selected]) / scale[selected]
            m_bin = np.median(np.abs(target[selected])) / np.median(scale[selected])
            row.update(raw_mid=np.median(raw[selected]), median_scale=np.median(scale[selected]),
                       median_abs_error=np.median(np.abs(target[selected])),
                       m_bin=m_bin, mad1_bin=abs(m_bin - 1), median_abs_scaled=np.median(ratio),
                       smad_element_bin=sigma_nmad(ratio), frac_abs_scaled_le_1=np.mean(ratio <= 1))
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_metrics(predict: Callable[[np.ndarray], np.ndarray], inputs: np.ndarray,
                         groups: Mapping[str, list[int]],
                         metrics: Mapping[str, Callable[[np.ndarray], float]], *,
                         repeats: int = 5, seed: int = 50) -> pd.DataFrame:
    """Evaluate several losses on the same block permutations/predictions.

    Positive increases indicate dependence of fitted predictions on a block.
    The standard deviations measure permutation variability, not bootstrap CIs.
    """
    x = np.asarray(inputs)
    if x.ndim != 2 or not len(x) or repeats < 1 or not groups or not metrics:
        raise ValueError("Require non-empty inputs/groups/metrics and positive repeats")
    if any(not cols or min(cols) < 0 or max(cols) >= x.shape[1] for cols in groups.values()):
        raise ValueError("Permutation columns outside model inputs")
    baseline_pred = predict(x)
    baseline = {name: float(metric(baseline_pred)) for name, metric in metrics.items()}
    rng = np.random.default_rng(seed)
    values = {(group, metric): [] for group in groups for metric in metrics}
    for _ in range(repeats):
        order = rng.permutation(len(x))
        for group, columns in groups.items():
            changed = x.copy()
            changed[:, columns] = x[order][:, columns]
            pred = predict(changed)
            for name, metric in metrics.items():
                values[group, name].append(float(metric(pred)) - baseline[name])
    return pd.DataFrame([dict(group=group, metric=name, baseline=baseline[name],
                              mean_increase=np.mean(draws), std_increase=np.std(draws),
                              n_columns=len(groups[group]), repeats=repeats)
                         for (group, name), draws in values.items()]).sort_values(
        ["metric", "mean_increase"], ascending=[True, False]).reset_index(drop=True)


def pc_loading_table(preprocessor: Preprocessor, grid: np.ndarray,
                      components: list[int]) -> pd.DataFrame:
    """Back-project PCA modes through the fitted PDZ scaler, retaining signs."""
    rows = []
    for component in components:
        vector = preprocessor.pca.components_[component]
        multiplier = np.sqrt(preprocessor.pca.explained_variance_[component]) if preprocessor.pca.whiten else 1
        raw = vector * preprocessor.sc_pdf.scale_ * multiplier
        for z, loading, physical in zip(grid, vector, raw):
            rows.append(dict(pc=f"PC{component + 1}", z=z, standardized_loading=loading,
                             pdz_loading=physical, explained_variance=preprocessor.pca.explained_variance_ratio_[component]))
    return pd.DataFrame(rows)
