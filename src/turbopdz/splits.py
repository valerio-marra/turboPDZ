"""Persistent object-level splits shared across stages and paired experiments."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from .data import PDZDataset


def stratification_bins(labels: np.ndarray, min_count: int = 5, max_bins: int = 20) -> np.ndarray | None:
    """Construct safe quantile bins; return None for an unstratifiable cohort."""
    for q in range(min(max_bins, len(labels) // min_count), 1, -1):
        bins = np.asarray(pd.qcut(labels, q, labels=False, duplicates="drop"))
        counts = pd.Series(bins).value_counts()
        if len(counts) > 1 and counts.min() >= min_count and np.isfinite(bins).all():
            return bins
    return None


@dataclass
class SplitPlan:
    """Development, calibration, final test, and per-development OOF membership."""

    development: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    fold_id: np.ndarray
    object_id: np.ndarray
    dataset_fingerprint: str
    seed: int

    @classmethod
    def create(cls, dataset: PDZDataset, *, development: float = 0.85,
               calibration: float = 0.075, test: float = 0.075,
               oof_folds: int = 5, seed: int = 50) -> "SplitPlan":
        """Split labels once, then assign fixed CV/OOF folds inside development."""
        if dataset.z_spec is None:
            raise ValueError("Training splits require spectroscopic labels")
        fractions = np.asarray([development, calibration, test])
        if not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
            raise ValueError("Positive development/calibration/test fractions must sum to one")
        holdout_fraction = calibration + test
        holdout_count = int(np.ceil(len(dataset.pdz) * holdout_fraction))
        bins = stratification_bins(dataset.z_spec, max(oof_folds, 2),
                                   min(holdout_count, len(dataset.pdz)-holdout_count))
        dev, holdout = train_test_split(np.arange(len(dataset.pdz)), test_size=holdout_fraction,
                                        random_state=seed, stratify=bins)
        final_count = int(np.ceil(len(holdout) * test / holdout_fraction))
        hold_bins = stratification_bins(dataset.z_spec[holdout], 2, min(final_count, len(holdout)-final_count))
        calib, final = train_test_split(holdout, test_size=test/(calibration+test),
                                        random_state=seed, stratify=hold_bins)
        fold_id = np.full(len(dataset.pdz), -1, dtype=int)
        dev_bins = stratification_bins(dataset.z_spec[dev], oof_folds)
        splitter = KFold(oof_folds, shuffle=True, random_state=seed) if dev_bins is None else StratifiedKFold(oof_folds, shuffle=True, random_state=seed)
        for fold, (_, valid) in enumerate(splitter.split(dev, dev_bins)):
            fold_id[dev[valid]] = fold
        result = cls(dev, calib, final, fold_id, dataset.object_id.copy(), dataset.fingerprint(), seed)
        result.validate(dataset)
        return result

    def validate(self, dataset: PDZDataset) -> None:
        """Reject changed data/order, overlapping rows, and missing fold assignments."""
        if not np.array_equal(self.object_id, dataset.object_id) or self.dataset_fingerprint != dataset.fingerprint():
            raise ValueError("Cannot reuse splits with changed data or object order")
        all_indices = np.concatenate([self.development, self.calibration, self.test])
        if not np.array_equal(np.sort(all_indices), np.arange(len(dataset.pdz))):
            raise ValueError("Splits must partition every object exactly once")
        if self.fold_id.shape != (len(dataset.pdz),) or (self.fold_id[self.development] < 0).any():
            raise ValueError("Missing development OOF memberships")
        if (self.fold_id[np.r_[self.calibration, self.test]] != -1).any():
            raise ValueError("Calibration/test objects must not belong to OOF folds")
        if not np.array_equal(np.unique(self.fold_id[self.development]), np.arange(self.n_folds)):
            raise ValueError("OOF fold labels must be contiguous")

    @property
    def n_folds(self) -> int:
        return int(self.fold_id.max()) + 1

    def folds(self):
        """Yield global train/held-out indices restricted to development rows."""
        for fold in range(self.n_folds):
            yield self.development[self.fold_id[self.development] != fold], self.development[self.fold_id[self.development] == fold]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, development=self.development, calibration=self.calibration,
                            test=self.test, fold_id=self.fold_id, object_id=self.object_id,
                            dataset_fingerprint=np.asarray(self.dataset_fingerprint), seed=np.asarray(self.seed))

    @classmethod
    def load(cls, path: str | Path, dataset: PDZDataset) -> "SplitPlan":
        with np.load(path, allow_pickle=False) as arrays:
            result = cls(arrays["development"], arrays["calibration"], arrays["test"],
                         arrays["fold_id"], arrays["object_id"], str(arrays["dataset_fingerprint"]), int(arrays["seed"]))
        result.validate(dataset)
        return result
