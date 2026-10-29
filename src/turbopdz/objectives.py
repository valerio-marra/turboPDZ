"""Validation-selection objectives, separate from network training loss."""

from __future__ import annotations
from collections.abc import Callable, Mapping
import importlib

import numpy as np

from .evaluation.metrics import photoz_metrics


def resolve_callable(path: str) -> Callable:
    """Import an explicit ``module:function`` from trusted user extension code."""
    module, separator, name = path.partition(":")
    if not separator:
        raise ValueError("Custom hooks must use 'importable_module:function'")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise TypeError(f"{path} is not callable")
    return function


def objective_score(z_pred: np.ndarray, z_spec: np.ndarray, config: Mapping,
                    references: Mapping[str, float]) -> float:
    """Evaluate a preset or custom function on held-out validation predictions."""
    metrics = photoz_metrics(z_pred, z_spec, config.get("outlier_threshold", 0.15))
    name = config.get("name", "normalized_composite")
    if name == "mae":
        score = metrics["mae"]
    elif name == "normalized_composite":
        weights = config.get("weights", {"scatter": 0.5, "outliers": 0.5, "absolute_bias": 0.0})
        if set(weights) - {"scatter", "outliers", "absolute_bias"}:
            raise ValueError("Unsupported composite metric")
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Objective weights must be finite, non-negative, and not all zero")
        score = 0.0
        for metric, weight in weights.items():
            if weight == 0:
                continue
            scale = config.get("bias_tolerance") if metric == "absolute_bias" else references.get(metric)
            if scale is None or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"A positive normalization scale is required for {metric}")
            score += weight * metrics[metric] / scale
    elif name == "custom":
        score = resolve_callable(config["callable"])(z_pred=z_pred, z_spec=z_spec,
                                                    metrics=metrics, references=references,
                                                    **config.get("kwargs", {}))
    else:
        raise ValueError(f"Unknown selection objective: {name}")
    if not np.isfinite(score):
        raise ValueError("Selection objective must return a finite scalar")
    return float(score)
