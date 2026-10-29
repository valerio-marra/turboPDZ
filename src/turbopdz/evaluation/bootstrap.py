"""Generic paired percentile bootstrap of fixed predictions."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapResult:
    """Summaries and raw (candidate, reference, improvement) draws."""

    summary: pd.DataFrame
    draws: dict[str, np.ndarray]
    metadata: dict


def paired_bootstrap(columns: Mapping[str, np.ndarray],
                     comparisons: Mapping[str, Callable[[Mapping[str, np.ndarray]], tuple[float, float]]],
                     *, higher_is_better: Mapping[str, bool] | None = None,
                     n_boot: int = 2000, confidence_level: float = 0.95, seed: int = 50,
                     groups: np.ndarray | None = None) -> BootstrapResult:
    """Resample the same object indices for every model/metric comparison.

    Parameters
    ----------
    comparisons
        Functions returning ``(candidate_metric, reference_metric)`` on one
        prefiltered cohort. Positive improvements always favor the candidate.
    higher_is_better
        Metric directions; unspecified metrics default to smaller-is-better.
    groups
        Optional row-aligned labels for resampling entire groups instead of rows.

    Notes
    -----
    No models are refitted. Intervals are conditional on the fitted models and
    the observed population; they do not measure training variability.
    """
    arrays = {name: np.asarray(value) for name, value in columns.items()}
    if not arrays or not comparisons:
        raise ValueError("Provide columns and at least one comparison")
    n = len(next(iter(arrays.values())))
    if n < 2 or any(a.ndim != 1 or len(a) != n or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Bootstrap requires finite aligned vectors on one cohort of >=2 rows")
    if n_boot < 2 or not 0 < confidence_level < 1:
        raise ValueError("Require n_boot>=2 and confidence_level within (0,1)")
    direction = higher_is_better or {}
    if set(direction) - set(comparisons):
        raise ValueError("Metric directions must name existing comparisons")
    blocks = None
    if groups is not None:
        group_array = np.asarray(groups)
        if group_array.shape != (n,) or pd.isna(group_array).any():
            raise ValueError("groups must be non-missing and row aligned")
        unique = pd.unique(group_array)
        if len(unique) < 2:
            raise ValueError("Cluster bootstrap requires at least two groups")
        blocks = [np.flatnonzero(group_array == label) for label in unique]
    observed = {name: fn(arrays) for name, fn in comparisons.items()}
    draws = {name: np.full((n_boot, 3), np.nan) for name in comparisons}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n, n) if blocks is None else np.concatenate(
            [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        sample = {name: values[idx] for name, values in arrays.items()}
        for name, fn in comparisons.items():
            candidate, reference = fn(sample)
            improvement = candidate - reference if direction.get(name, False) else reference - candidate
            draws[name][b] = candidate, reference, improvement
    alpha = (1 - confidence_level) / 2
    rows = []
    for name, (candidate, reference) in observed.items():
        valid = np.isfinite(draws[name]).all(axis=1)
        if not valid.any() or not np.isfinite([candidate, reference]).all():
            raise ValueError(f"Metric {name!r} is undefined on the cohort or every resample")
        low, high = np.quantile(draws[name][valid, 2], [alpha, 1 - alpha])
        candidate_ci = np.quantile(draws[name][valid, 0], [alpha, 1 - alpha])
        reference_ci = np.quantile(draws[name][valid, 1], [alpha, 1 - alpha])
        sign = 1 if direction.get(name, False) else -1
        rows.append({"metric": name, "candidate": candidate, "reference": reference,
                     "improvement": sign * (candidate - reference), "ci_low": low, "ci_high": high,
                     "candidate_ci_low": candidate_ci[0], "candidate_ci_high": candidate_ci[1],
                     "reference_ci_low": reference_ci[0], "reference_ci_high": reference_ci[1],
                     "status": "candidate" if low > 0 else "reference" if high < 0 else "unresolved",
                     "n_objects": n, "n_valid_boot": int(valid.sum())})
    return BootstrapResult(pd.DataFrame(rows), draws, {
        "method": "paired cluster percentile bootstrap" if blocks else "paired object percentile bootstrap",
        "seed": seed, "n_boot": n_boot, "confidence_level": confidence_level,
        "conditional_on_fitted_models": True,
    })
