"""Per-object PDZ descriptors and explicit, independently selectable features."""

from __future__ import annotations
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._pmf import pmf_stats, _local_maxima
from .data import PDZDataset, quadrature_weights
from .objectives import resolve_callable

DESCRIPTOR_GROUPS = {
    "location": ["mean", "median", "mode"],
    "shape": ["std", "skewness", "kurtosis"],
    "peaks": ["main_peak_height", "peak_height_ratio", "peak_influence",
              "distance_to_mode", "has_second_peak"],
}


def pdz_descriptors(dataset: PDZDataset) -> pd.DataFrame:
    """Compute descriptors with mass-weighted moments and native-height peaks.

    Missing second-peak descriptors are zero, with an explicit indicator. A
    degenerate distribution's undefined kurtosis is zero. Density peaks use
    density heights, not quadrature-weighted masses on a non-uniform grid.
    """
    rows = []
    for raw_heights in dataset.pdz:
        heights = raw_heights
        masses = heights if dataset.representation == "mass" else heights * quadrature_weights(dataset.z_grid)
        values = pmf_stats(dataset.z_grid, masses)
        if dataset.representation == "density":
            integral = float(masses.sum())
            heights = heights / integral
            interval = np.diff(dataset.z_grid)
            cumulative = np.r_[0.0, np.cumsum(0.5 * (heights[:-1] + heights[1:]) * interval)]
            segment = min(int(np.searchsorted(cumulative[1:], 0.5, side="left")), len(interval)-1)
            area = max(0.0, 0.5 - cumulative[segment])
            slope = (heights[segment+1] - heights[segment]) / interval[segment]
            denominator = heights[segment] + np.sqrt(max(0.0, heights[segment]**2 + 2*slope*area))
            offset = 2*area/denominator if denominator > 0 else 0.0
            values["median"] = float(dataset.z_grid[segment] + np.clip(offset, 0, interval[segment]))
            mode_idx = int(np.argmax(heights))
            values["mode"] = float(dataset.z_grid[mode_idx])
            peaks = _local_maxima(heights)
            peaks = np.unique(np.r_[peaks, mode_idx])
            peaks = peaks[np.argsort(heights[peaks])[::-1]]
            values["main_peak_height"] = float(heights[peaks[0]])
            values["second_peak_pos"] = float(dataset.z_grid[peaks[1]]) if len(peaks) > 1 else None
            values["peak_height_ratio"] = float(heights[peaks[1]] / heights[peaks[0]]) if len(peaks) > 1 else None
        has_peak = values["second_peak_pos"] is not None
        distance = values["second_peak_pos"] - values["mode"] if has_peak else 0.0
        ratio = values["peak_height_ratio"] if has_peak else 0.0
        values.update(has_second_peak=float(has_peak), distance_to_mode=distance,
                      peak_height_ratio=ratio, peak_influence=distance * ratio)
        rows.append({name: 0.0 if values[name] is None else float(values[name])
                     for names in DESCRIPTOR_GROUPS.values() for name in names})
    return pd.DataFrame(rows, columns=[name for group in DESCRIPTOR_GROUPS.values() for name in group])


def feature_table(dataset: PDZDataset, config: Mapping) -> pd.DataFrame:
    """Select descriptor groups/columns, optional observables, and stateless hooks.

    Hooks receive one object cohort and must not fit population-dependent
    transformations or access labels. Learned preprocessing belongs in a fold.
    """
    groups = config.get("descriptor_groups", ["location", "shape", "peaks"])
    if set(groups) - set(DESCRIPTOR_GROUPS):
        raise ValueError("Unknown descriptor group")
    descriptor_names = config.get("descriptor_columns", [name for group in groups for name in DESCRIPTOR_GROUPS[group]])
    names = list(descriptor_names) + list(config.get("extra_columns", []))
    if config.get("descriptor_source", "compute") == "observables":
        available = dataset.observables.copy()
    else:
        available = pdz_descriptors(dataset)
        for name in config.get("extra_columns", []):
            if name not in dataset.observables:
                raise KeyError(f"Missing required observable: {name}")
            available[name] = dataset.observables[name].to_numpy()
    selected = available.loc[:, names].copy()
    if config.get("descriptor_missing") == "zero":
        selected.loc[:, descriptor_names] = selected.loc[:, descriptor_names].fillna(0.0)
    for hook in config.get("hooks", []):
        extra = resolve_callable(hook["callable"])(pdz=dataset.pdz, z_grid=dataset.z_grid,
                                                  observables=dataset.observables.copy(),
                                                  **hook.get("kwargs", {}))
        if not isinstance(extra, pd.DataFrame) or len(extra) != len(dataset.pdz):
            raise ValueError("Feature hook must return a row-aligned DataFrame")
        selected = pd.concat([selected.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    excluded = config.get("exclude", [])
    if set(excluded) - set(selected.columns):
        raise ValueError("Excluded features must exist in the selected feature table")
    selected = selected.drop(columns=excluded)
    if selected.columns.has_duplicates:
        raise ValueError("Feature names must be unique across descriptors/observables/hooks")
    selected = selected.astype(np.float64)
    for name in selected:
        null_name = f"{name}_isnull"
        if null_name in dataset.observables:
            mask = dataset.observables[null_name].fillna(True).astype(bool).to_numpy()
            selected.loc[mask, name] = np.nan
    return selected
