"""Validated, row-aligned PDZ datasets with an explicit probability convention."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd


def quadrature_weights(z_grid: np.ndarray) -> np.ndarray:
    """Return trapezoidal weights on a finite, strictly increasing grid."""
    z = np.asarray(z_grid, dtype=np.float64)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all() or (np.diff(z) <= 0).any():
        raise ValueError("z_grid must be finite, strictly increasing, and contain >=2 points")
    delta = np.diff(z)
    return np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]


@dataclass
class PDZDataset:
    """PDZs, observables, IDs, and optional labels on one ordered cohort.

    Parameters
    ----------
    pdz
        Finite non-negative matrix with one row per object.
    z_grid
        Shared native redshift grid; no implicit regridding is performed.
    observables
        Row-aligned catalog columns. Its index is not an object identifier.
    object_id
        Unique stable integer or string identifiers.
    z_spec
        Optional finite labels, always greater than -1.
    representation
        ``mass`` means discrete masses; ``density`` means sampled densities
        integrated with trapezoidal weights. Raw amplitudes are retained for
        PCA; :meth:`masses` normalizes for descriptor calculations only.
    """

    pdz: np.ndarray
    z_grid: np.ndarray
    observables: pd.DataFrame
    object_id: np.ndarray
    z_spec: np.ndarray | None = None
    representation: str = "mass"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pdz = np.asarray(self.pdz, dtype=np.float64)
        self.z_grid = np.asarray(self.z_grid, dtype=np.float64)
        quadrature_weights(self.z_grid)
        if self.representation not in {"mass", "density"}:
            raise ValueError("Declare representation='mass' or 'density'")
        if self.pdz.ndim != 2 or self.pdz.shape[1] != len(self.z_grid):
            raise ValueError("pdz must have shape (n_objects, len(z_grid))")
        if not np.isfinite(self.pdz).all() or (self.pdz < 0).any():
            raise ValueError("PDZs must be finite/non-negative; filter invalid rows explicitly")
        if (self.pdz.sum(axis=1) <= 0).any():
            raise ValueError("Every PDZ must have positive total probability")
        self.observables = self.observables.reset_index(drop=True).copy()
        if self.observables.columns.has_duplicates:
            raise ValueError("Observable names must be unique")
        if not all(isinstance(name, str) for name in self.observables.columns):
            raise ValueError("Observable names must be strings")
        ids = np.asarray(self.object_id)
        if ids.ndim != 1 or len(ids) != len(self.pdz) or len(self.observables) != len(ids):
            raise ValueError("PDZs, observables, and IDs must be row aligned")
        if pd.isna(ids).any() or pd.Index(ids).has_duplicates:
            raise ValueError("object_id must contain unique, non-missing identifiers")
        if ids.dtype.kind not in "iuUSO" or ids.dtype.kind == "O" and not all(isinstance(x, (str, int, np.integer)) and not isinstance(x, bool) for x in ids):
            raise ValueError("Identifiers must be integers or strings, never floating point")
        self.object_id = ids.astype(str) if ids.dtype.kind == "O" else ids
        if pd.Index(self.object_id).has_duplicates:
            raise ValueError("Identifier normalization produced duplicate IDs")
        if self.z_spec is not None:
            self.z_spec = np.asarray(self.z_spec, dtype=np.float64)
            if self.z_spec.shape != (len(ids),) or not np.isfinite(self.z_spec).all():
                raise ValueError("z_spec must be a finite row-aligned vector")
            if (self.z_spec <= -1).any():
                raise ValueError("z_spec must be greater than -1")

    def masses(self) -> np.ndarray:
        """Return normalized masses, accounting for density quadrature."""
        weights = quadrature_weights(self.z_grid) if self.representation == "density" else 1.0
        values = self.pdz * weights
        return values / values.sum(axis=1, keepdims=True)

    def subset(self, indices: np.ndarray) -> "PDZDataset":
        """Select the same rows in every component, preserving their identifiers."""
        idx = np.asarray(indices)
        return PDZDataset(self.pdz[idx], self.z_grid, self.observables.iloc[idx],
                          self.object_id[idx], None if self.z_spec is None else self.z_spec[idx],
                          self.representation, self.metadata.copy())

    def fingerprint(self) -> str:
        """Hash content and row order, not machine-specific filenames."""
        digest = hashlib.sha256()
        for array in (self.pdz, self.z_grid, self.object_id, self.z_spec):
            if array is not None:
                values = np.ascontiguousarray(array)
                digest.update(str((values.shape, values.dtype.str)).encode())
                digest.update(values.tobytes())
        digest.update(pd.util.hash_pandas_object(self.observables, index=False).values.tobytes())
        digest.update(json.dumps(list(self.observables.columns)).encode())
        digest.update(self.representation.encode())
        return digest.hexdigest()

    def save(self, directory: str | Path) -> Path:
        """Save numeric/string arrays without pickle or floating-point text loss."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        arrays = dict(pdz=self.pdz, z_grid=self.z_grid, object_id=self.object_id)
        if self.z_spec is not None:
            arrays["z_spec"] = self.z_spec
        for i, column in enumerate(self.observables.columns):
            values = self.observables[column].to_numpy()
            if values.dtype.kind == "O":
                if not all(isinstance(x, str) for x in values):
                    raise ValueError("Object-valued observables must contain strings, not arbitrary Python objects")
                values = values.astype(str)
            arrays[f"observable_{i}"] = values
        np.savez_compressed(root / "arrays.npz", **arrays)
        (root / "dataset.json").write_text(json.dumps({
            "schema_version": 1, "representation": self.representation,
            "observable_columns": list(self.observables.columns),
            "fingerprint": self.fingerprint(), "metadata": self.metadata,
        }, indent=2), encoding="utf-8")
        return root

    @classmethod
    def load(cls, directory: str | Path) -> "PDZDataset":
        """Load arrays and verify content/order against their recorded fingerprint."""
        root = Path(directory)
        info = json.loads((root / "dataset.json").read_text())
        if info["schema_version"] != 1:
            raise ValueError("Unsupported dataset schema")
        with np.load(root / "arrays.npz", allow_pickle=False) as data:
            frame = pd.DataFrame({name: data[f"observable_{i}"]
                                  for i, name in enumerate(info["observable_columns"])},
                                 index=np.arange(len(data["pdz"])))
            result = cls(data["pdz"], data["z_grid"], frame, data["object_id"],
                         data["z_spec"] if "z_spec" in data else None,
                         info["representation"], info["metadata"])
        if result.fingerprint() != info["fingerprint"]:
            raise ValueError("Dataset content/order differs from the saved fingerprint")
        return result
