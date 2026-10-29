"""Chunked inference from hash-verified zML/sigmaML bundles."""

from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .artifacts import load_manifest, file_sha256
from .data import PDZDataset
from .features import feature_table
from .preprocessing import Preprocessor
from .calibration import Calibration, log_to_scale


class TurboPDZ:
    """Loaded pipeline; no Optuna database or training catalogs are required."""

    @classmethod
    def load(cls, directory: str | Path, *, verify: bool = True) -> "TurboPDZ":
        """Load a combined bundle containing ``zml/`` and optional ``sigmaml/``."""
        from .training.common import tensorflow
        root = Path(directory)
        result = cls()
        if (root / "zml").is_dir() and (root / "manifest.json").is_file():
            load_manifest(root, verify=verify)
        zroot = root / "zml" if (root / "zml").is_dir() else root
        result.zml_info = load_manifest(zroot, verify=verify)
        result.zml_model = tensorflow().keras.models.load_model(zroot / "model.keras", compile=False)
        result.zml_pp = Preprocessor.load(zroot / "preprocessor.npz")
        with np.load(zroot / "grid.npz", allow_pickle=False) as arrays:
            result.z_grid = arrays["z_grid"].copy()
        result.sigma_model = result.sigma_info = result.sigma_pp = result.calibration = None
        sroot = root / "sigmaml"
        if sroot.is_dir():
            result.sigma_info = load_manifest(sroot, verify=verify)
            if result.sigma_info["parent_zml_manifest_hash"] != file_sha256(zroot / "manifest.json"):
                raise ValueError("sigmaML bundle references a different zML model")
            with np.load(sroot / "grid.npz", allow_pickle=False) as arrays:
                if not np.array_equal(arrays["z_grid"], result.z_grid):
                    raise ValueError("zML and sigmaML use different redshift grids")
            result.sigma_model = tensorflow().keras.models.load_model(sroot / "model.keras", compile=False)
            result.sigma_pp = Preprocessor.load(sroot / "preprocessor.npz")
            result.calibration = Calibration.load(sroot / "calibration.npz")
        result.applicability = None
        if (root / "applicability.npz").is_file():
            from .applicability import ApplicabilityReference
            result.applicability = ApplicabilityReference.load(root / "applicability.npz")
        return result

    def predict(self, pdz: np.ndarray, z_grid: np.ndarray, observables: pd.DataFrame | None = None,
                *, object_id: np.ndarray | None = None, representation: str | None = None,
                batch_size: int = 4096, precomputed_descriptors: pd.DataFrame | None = None,
                applicability: bool = False) -> pd.DataFrame:
        """Return aligned z_ml, sigma_ml, and r_ml; optionally attach support flags.

        Precomputed descriptors are an explicit compatibility shortcut and
        must have been computed with the bundle's documented descriptor policy.
        Missing selected observables raise; no catalog rows are silently dropped.
        """
        from .training.common import predict_model
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        grid = np.asarray(z_grid, dtype=float)
        if grid.shape != self.z_grid.shape or not np.allclose(grid, self.z_grid, rtol=1e-7, atol=1e-8):
            raise ValueError("Native redshift grid differs from training; no implicit regridding")
        representation = representation or self.zml_info["representation"]
        if representation != self.zml_info["representation"]:
            raise ValueError("PDZ mass/density convention differs from training")
        n = len(pdz)
        frame = pd.DataFrame(index=np.arange(n)) if observables is None else observables.reset_index(drop=True)
        ids = np.arange(n) if object_id is None else np.asarray(object_id)
        if len(frame) != n or ids.shape != (n,) or pd.Index(ids).has_duplicates:
            raise ValueError("Observables and unique IDs must match PDZ rows")
        if precomputed_descriptors is not None and len(precomputed_descriptors) != n:
            raise ValueError("Precomputed descriptors must be row aligned")
        rows = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            obs = frame.iloc[start:stop].reset_index(drop=True).copy()
            if precomputed_descriptors is not None:
                for name in precomputed_descriptors:
                    obs[name] = precomputed_descriptors[name].iloc[start:stop].to_numpy()
            data = PDZDataset(pdz[start:stop], grid, obs, ids[start:stop], representation=representation)
            zfeatures = dict(self.zml_info["features"])
            if precomputed_descriptors is not None:
                zfeatures["descriptor_source"] = "observables"
            features = feature_table(data, zfeatures)
            pred = predict_model(self.zml_model, self.zml_pp.transform(data.pdz, features), batch_size)
            result = pd.DataFrame({"object_id": data.object_id, "z_ml": pred})
            if self.sigma_model is not None:
                sfeatures = dict(self.sigma_info["features"])
                if precomputed_descriptors is not None:
                    sfeatures["descriptor_source"] = "observables"
                features = feature_table(data, sfeatures)
                log_scale = predict_model(self.sigma_model, self.sigma_pp.transform(data.pdz, features), batch_size)
                result["sigma_ml"] = self.calibration.apply(log_to_scale(log_scale, self.calibration.epsilon))
                result["r_ml"] = self.calibration.reliability(result["sigma_ml"].to_numpy())
            if applicability:
                if self.applicability is None:
                    raise ValueError("No applicability reference attached to this bundle")
                from .applicability import compute_applicability_flags
                result = pd.concat([result, compute_applicability_flags(obs, self.applicability)], axis=1)
            rows.append(result)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["object_id", "z_ml", "sigma_ml", "r_ml"])
