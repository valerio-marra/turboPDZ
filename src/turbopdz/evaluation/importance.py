"""Grouped permutation diagnostics in the fitted input representation."""

from __future__ import annotations
from collections.abc import Callable, Mapping
import numpy as np
import pandas as pd

from ..features import DESCRIPTOR_GROUPS


def input_feature_groups(feature_names: tuple[str, ...], n_components: int) -> dict[str, list[int]]:
    """Map saved tabular/PCA input columns into interpretable feature blocks."""
    groups = {}
    assigned = set()
    for group, names in DESCRIPTOR_GROUPS.items():
        indices = [i for i, name in enumerate(feature_names) if name in names]
        if indices:
            groups[group] = indices
            assigned.update(indices)
    for i, name in enumerate(feature_names):
        if i not in assigned:
            groups[name] = [i]
    if n_components:
        groups["PDZ-PCA"] = list(range(len(feature_names), len(feature_names) + n_components))
    return groups


def grouped_permutation_importance(predict: Callable, inputs: np.ndarray, target: np.ndarray,
                                   groups: Mapping[str, list[int]], score: Callable,
                                   *, repeats: int = 5, seed: int = 50) -> pd.DataFrame:
    """Measure increased loss after jointly permuting a declared feature block.

    The same permutation is used across groups within a repetition. ``score``
    is smaller-is-better and can be the configured selection objective. These
    repetitions are permutation variability, not paired-bootstrap intervals.
    """
    x = np.asarray(inputs)
    if x.ndim != 2 or len(x) != len(target) or repeats < 1:
        raise ValueError("Require aligned matrix/target and positive repeats")
    if any(not indices or min(indices) < 0 or max(indices) >= x.shape[1] for indices in groups.values()):
        raise ValueError("Permutation groups must reference existing input columns")
    baseline = float(score(predict(x), target))
    rng = np.random.default_rng(seed)
    changes = {name: [] for name in groups}
    for _ in range(repeats):
        permutation = rng.permutation(len(x))
        for name, indices in groups.items():
            modified = x.copy()
            modified[:, indices] = x[permutation][:, indices]
            changes[name].append(float(score(predict(modified), target)) - baseline)
    result = pd.DataFrame([{"group": name, "mean_increase": np.mean(values),
                            "std_increase": np.std(values), "baseline": baseline,
                            "repeats": repeats} for name, values in changes.items()])
    return result.sort_values("mean_increase", ascending=False).reset_index(drop=True)
