"""Multiplicative empirical error-scale calibration and reference ranking."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def log_to_scale(t_pred: np.ndarray, epsilon: float) -> np.ndarray:
    """Invert the log-error target with the paper's positive numerical floor."""
    values = np.asarray(t_pred)
    if not np.isfinite(values).all() or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Require finite log predictions and non-negative epsilon")
    output = np.maximum(np.exp(values) - epsilon, 1e-12)
    if not np.isfinite(output).all():
        raise ValueError("Exponentiation overflow in sigmaML predictions")
    return output


@dataclass
class Calibration:
    """Bin anchors, smoothed factors, and calibration-only reliability CDF."""

    sigma_anchor: np.ndarray
    factors: np.ndarray
    sigma_reference: np.ndarray
    epsilon: float

    @classmethod
    def fit(cls, raw_scale: np.ndarray, abs_residual: np.ndarray, *, epsilon: float,
            n_bins: int = 30, smooth_bw_dex: float | None = 0.2) -> "Calibration":
        """Fit only on a designated calibration cohort, never the final test."""
        scale, target = np.asarray(raw_scale, dtype=float), np.asarray(abs_residual, dtype=float)
        if scale.ndim != 1 or scale.shape != target.shape or not len(scale):
            raise ValueError("Calibration requires aligned non-empty vectors")
        if not np.isfinite(scale).all() or not np.isfinite(target).all() or (scale <= 0).any() or (target < 0).any():
            raise ValueError("Calibration requires finite positive scales and non-negative residuals")
        if n_bins < 1 or smooth_bw_dex is not None and smooth_bw_dex <= 0:
            raise ValueError("Calibration bins/bandwidth must be positive")
        order = np.argsort(scale, kind="stable")
        groups = np.array_split(order, min(n_bins, len(scale)))
        anchors = np.array([np.median(scale[idx]) for idx in groups])
        factors = np.array([max(float(np.median(target[idx]) / np.median(scale[idx])), 1e-12) for idx in groups])
        counts = np.array([len(idx) for idx in groups])
        keep = np.r_[True, np.diff(anchors) > 0]
        anchors, factors, counts = anchors[keep], factors[keep], counts[keep]
        if smooth_bw_dex is not None and len(anchors) > 1:
            log = np.log10(anchors)
            weights = np.exp(-0.5 * ((log[:, None] - log[None, :]) / smooth_bw_dex) ** 2) * counts[None, :]
            factors = (weights @ factors) / weights.sum(axis=1)
        result = cls(anchors, factors, np.empty(0), epsilon)
        result.sigma_reference = np.sort(result.apply(scale))
        result.validate()
        return result

    def validate(self) -> None:
        if self.sigma_anchor.ndim != 1 or not len(self.sigma_anchor) or self.factors.shape != self.sigma_anchor.shape:
            raise ValueError("Invalid calibration anchors/factors")
        for values in (self.sigma_anchor, self.factors, self.sigma_reference):
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Calibration arrays must be finite and positive")
        if not len(self.sigma_reference) or (np.diff(self.sigma_reference) < 0).any() or (np.diff(self.sigma_anchor) <= 0).any():
            raise ValueError("Anchors must increase and the non-empty reference must be sorted")
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("Invalid log-target epsilon")

    def apply(self, raw_scale: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_scale)
        return values * np.interp(values, self.sigma_anchor, self.factors,
                                  left=self.factors[0], right=self.factors[-1])

    def reliability(self, calibrated_scale: np.ndarray) -> np.ndarray:
        """Return 1-F_calibration(sigma), with the original right-rank convention."""
        self.validate()
        return np.clip(1 - np.searchsorted(self.sigma_reference, calibrated_scale, side="right") / len(self.sigma_reference), 0, 1)

    def save(self, path: str | Path) -> None:
        self.validate()
        np.savez_compressed(path, sigma_anchor=self.sigma_anchor, factors=self.factors,
                            sigma_reference=self.sigma_reference, epsilon=np.asarray(self.epsilon))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        with np.load(path, allow_pickle=False) as arrays:
            result = cls(arrays["sigma_anchor"], arrays["factors"], arrays["sigma_reference"], float(arrays["epsilon"]))
        result.validate()
        return result
