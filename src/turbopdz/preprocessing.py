"""Fold-fitted scaling/PCA, including the float32 underflow guard."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    """Fitted input schema, imputation, scalers, and optional PDZ PCA."""

    feature_names: tuple[str, ...]
    feature_fill: np.ndarray
    sc_feats: StandardScaler | None
    sc_pdf: StandardScaler | None
    pca: PCA | None
    guarded_bins: tuple[int, ...] = ()
    missing_policy: str = "raise"

    @classmethod
    def fit(cls, pdz: np.ndarray, features: pd.DataFrame, *, variance: float | int = 0.99,
            use_pca: bool = True, missing: str = "raise", seed: int = 50) -> "Preprocessor":
        """Fit only on supplied training rows; all output inputs use float32."""
        feats = features.to_numpy(dtype=np.float64)
        if missing not in {"raise", "median"}:
            raise ValueError("missing must be 'raise' or 'median'")
        if missing == "raise" and not np.isfinite(feats).all():
            raise ValueError("Selected features contain missing values; choose an explicit policy")
        fill = np.nanmedian(np.where(np.isfinite(feats), feats, np.nan), axis=0) if missing == "median" else np.zeros(feats.shape[1])
        if not np.isfinite(fill).all():
            raise ValueError("Cannot impute a feature missing in every training row")
        feats = np.where(np.isfinite(feats), feats, fill)
        sc_feats = StandardScaler().fit(feats.astype(np.float32)) if feats.shape[1] else None
        sc_pdf = pca = None
        guarded = ()
        if use_pca:
            pdf = np.asarray(pdz, dtype=np.float32)
            sc_pdf = StandardScaler().fit(pdf)
            zero = (sc_pdf.scale_.astype(np.float32) == 0) | ~np.isfinite(sc_pdf.scale_)
            guarded = tuple(int(i) for i in np.flatnonzero(zero))
            sc_pdf.scale_[zero] = 1.0
            scaled = sc_pdf.transform(pdf)
            if not np.isfinite(scaled).all():
                raise ValueError("Non-finite scaled PDZ values after float32 guard")
            if isinstance(variance, float) and variance == 1.0:
                variance = None
            pca = PCA(n_components=variance, svd_solver="full", random_state=seed).fit(scaled)
        if sc_feats is None and pca is None:
            raise ValueError("At least one feature or the PDZ-PCA block is required")
        return cls(tuple(features.columns), fill, sc_feats, sc_pdf, pca, guarded, missing)

    def transform(self, pdz: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Apply the exact saved feature order and fitted transformations."""
        if tuple(features.columns) != self.feature_names:
            raise ValueError("Feature order/schema differs from the trained model")
        feats = features.to_numpy(dtype=np.float64)
        if self.missing_policy == "raise" and not np.isfinite(feats).all():
            raise ValueError("Selected features contain missing values; model was fitted without imputation")
        feats = np.where(np.isfinite(feats), feats, self.feature_fill).astype(np.float32)
        parts = []
        if self.sc_feats is not None:
            parts.append(self.sc_feats.transform(feats))
        if self.pca is not None:
            parts.append(self.pca.transform(self.sc_pdf.transform(np.asarray(pdz, dtype=np.float32))))
        result = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite model inputs after preprocessing")
        return result

    def save(self, path: str | Path) -> None:
        """Save scaler/PCA arrays without executable pickle state."""
        arrays = {"feature_names": np.asarray(self.feature_names, dtype=str),
                  "feature_fill": self.feature_fill, "guarded_bins": np.asarray(self.guarded_bins, dtype=int),
                  "missing_policy": np.asarray(self.missing_policy)}
        for name, scaler in (("feats", self.sc_feats), ("pdf", self.sc_pdf)):
            if scaler is not None:
                for attr in ("mean_", "scale_", "var_"):
                    arrays[f"{name}_{attr}"] = getattr(scaler, attr)
        if self.pca is not None:
            for attr in ("components_", "mean_", "explained_variance_", "explained_variance_ratio_", "singular_values_"):
                arrays[f"pca_{attr}"] = getattr(self.pca, attr)
            arrays["pca_whiten"] = np.asarray(self.pca.whiten)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Reconstruct fitted sklearn transforms from portable numeric arrays."""
        with np.load(path, allow_pickle=False) as arrays:
            scalers = []
            for name in ("feats", "pdf"):
                scaler = None
                if f"{name}_mean_" in arrays:
                    scaler = StandardScaler()
                    for attr in ("mean_", "scale_", "var_"):
                        setattr(scaler, attr, arrays[f"{name}_{attr}"])
                    scaler.n_features_in_ = len(scaler.mean_)
                    scaler.n_samples_seen_ = 1
                scalers.append(scaler)
            pca = None
            if "pca_components_" in arrays:
                pca = PCA(whiten=bool(arrays["pca_whiten"]))
                for attr in ("components_", "mean_", "explained_variance_", "explained_variance_ratio_", "singular_values_"):
                    setattr(pca, attr, arrays[f"pca_{attr}"])
                pca.n_components_, pca.n_features_in_ = pca.components_.shape
            return cls(tuple(arrays["feature_names"].tolist()), arrays["feature_fill"],
                       *scalers, pca, tuple(arrays["guarded_bins"].tolist()), str(arrays["missing_policy"]))
