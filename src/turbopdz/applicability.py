"""Training-support flags for the turboPDZ value-added catalog.

The routines in this module are independent of the zML and sigmaML models.
They summarize the observed ``grizy`` color distribution of a spectroscopic
reference sample and later apply that summary to a photometric catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REFERENCE_VERSION = "1.0"
DEFAULT_MAG_COLUMNS = (
    "g_cmodel_mag",
    "r_cmodel_mag",
    "i_cmodel_mag",
    "z_cmodel_mag",
    "y_cmodel_mag",
)
DEFAULT_COLOR_NAMES = ("g-r", "r-i", "i-z", "z-y")
FLAG_COLUMNS = (
    "turbopdz_in_mag_domain",
    "turbopdz_specz_support_percentile",
    "turbopdz_in_specz_support_99",
    "turbopdz_applicability_flag",
)


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return selected columns as a two-dimensional float64 array.

    When a FITS-derived ``<column>_isnull`` field is present, rows marked by
    that field are converted to ``NaN`` even if the stored numerical sentinel
    itself is finite (often zero in the HSC tables).
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing required photometry columns: {missing}")
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2:
        raise ValueError("Photometry must be a two-dimensional table")
    for position, name in enumerate(columns):
        null_column = f"{name}_isnull"
        if null_column in frame.columns:
            null_values = frame[null_column].to_numpy(copy=False)
            marked_null = pd.isna(null_values) | np.asarray(
                null_values, dtype=bool
            )
            values[marked_null, position] = np.nan
    return values


def adjacent_colors(
    frame: pd.DataFrame,
    mag_columns: Sequence[str] = DEFAULT_MAG_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct adjacent colors and identify rows with finite photometry.

    Parameters
    ----------
    frame
        Table containing two or more ordered magnitude columns.
    mag_columns
        Magnitudes ordered from blue to red. The default produces
        ``g-r``, ``r-i``, ``i-z``, and ``z-y``.

    Returns
    -------
    colors, valid
        The ``(n_rows, n_bands-1)`` color matrix and a finite-photometry mask.
        Invalid rows remain present in ``colors`` and are marked by ``valid``.
    """
    if len(mag_columns) < 2 or len(set(mag_columns)) != len(mag_columns):
        raise ValueError("At least two distinct ordered magnitude columns are required")
    magnitudes = _numeric_matrix(frame, mag_columns)
    colors = magnitudes[:, :-1] - magnitudes[:, 1:]
    valid = np.isfinite(magnitudes).all(axis=1) & np.isfinite(colors).all(axis=1)
    return colors, valid


def _query_neighbor_distance(
    tree: cKDTree,
    values: np.ndarray,
    *,
    k: int,
    chunk_size: int,
    workers: int,
) -> np.ndarray:
    """Query the distance to neighbor ``k`` without a large temporary array."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    result = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        try:
            distance, _ = tree.query(values[start:stop], k=k, workers=workers)
        except TypeError:  # Compatibility with older SciPy releases.
            distance, _ = tree.query(values[start:stop], k=k)
        result[start:stop] = distance if k == 1 else distance[:, -1]
    return result


@dataclass(frozen=True)
class ApplicabilityReference:
    """Serializable color-space reference fitted to spectroscopic photometry."""

    reference_scaled: np.ndarray
    training_distance_sorted: np.ndarray
    color_center: np.ndarray
    color_scale: np.ndarray
    mag_bounds: tuple[float, float]
    n_neighbors: int
    support_level: float = 0.99
    mag_columns: tuple[str, ...] = DEFAULT_MAG_COLUMNS
    magnitude_column: str = "i_cmodel_mag"
    color_names: tuple[str, ...] = DEFAULT_COLOR_NAMES
    version: str = REFERENCE_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def support_distance_99(self) -> float:
        """Return the kNN-distance boundary containing 99% of the reference."""
        return float(
            np.quantile(
                self.training_distance_sorted,
                self.support_level,
                method="higher",
            )
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-compatible description of the reference."""
        return {
            "version": self.version,
            "n_reference": int(len(self.reference_scaled)),
            "n_neighbors": int(self.n_neighbors),
            "support_level": float(self.support_level),
            "support_distance_99": self.support_distance_99,
            "magnitude_column": self.magnitude_column,
            "mag_bounds": [float(value) for value in self.mag_bounds],
            "mag_columns": list(self.mag_columns),
            "color_names": list(self.color_names),
            "percentile_direction": "0=densest/familiar, 100=sparsest/outlying",
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Save the reference as a portable compressed NumPy artifact."""
        output = Path(path)
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            reference_scaled=np.asarray(self.reference_scaled, dtype=np.float32),
            training_distance_sorted=np.asarray(
                self.training_distance_sorted, dtype=np.float32
            ),
            color_center=np.asarray(self.color_center, dtype=np.float64),
            color_scale=np.asarray(self.color_scale, dtype=np.float64),
            mag_bounds=np.asarray(self.mag_bounds, dtype=np.float64),
            n_neighbors=np.asarray(self.n_neighbors, dtype=np.int64),
            support_level=np.asarray(self.support_level, dtype=np.float64),
            mag_columns=np.asarray(self.mag_columns),
            magnitude_column=np.asarray(self.magnitude_column),
            color_names=np.asarray(self.color_names),
            version=np.asarray(self.version),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True, default=str)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ApplicabilityReference":
        """Load and validate a reference produced by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            reference = cls(
                reference_scaled=np.asarray(data["reference_scaled"], dtype=np.float32),
                training_distance_sorted=np.asarray(
                    data["training_distance_sorted"], dtype=np.float64
                ),
                color_center=np.asarray(data["color_center"], dtype=np.float64),
                color_scale=np.asarray(data["color_scale"], dtype=np.float64),
                mag_bounds=tuple(
                    float(value) for value in np.asarray(data["mag_bounds"]).tolist()
                ),
                n_neighbors=int(data["n_neighbors"]),
                support_level=float(data["support_level"]),
                mag_columns=tuple(str(value) for value in data["mag_columns"].tolist()),
                magnitude_column=str(data["magnitude_column"]),
                color_names=tuple(str(value) for value in data["color_names"].tolist()),
                version=str(data["version"]),
                metadata=json.loads(str(data["metadata_json"])),
            )
        reference.validate()
        return reference

    def validate(self) -> None:
        """Raise ``ValueError`` if the stored arrays or schema are inconsistent."""
        n_colors = len(self.mag_columns) - 1
        if self.version != REFERENCE_VERSION:
            raise ValueError(
                f"Unsupported reference version {self.version!r}; "
                f"expected {REFERENCE_VERSION!r}"
            )
        if self.reference_scaled.ndim != 2 or self.reference_scaled.shape[1] != n_colors:
            raise ValueError("reference_scaled has an invalid shape")
        if len(self.reference_scaled) <= self.n_neighbors:
            raise ValueError("Reference is too small for the configured n_neighbors")
        if self.training_distance_sorted.shape != (len(self.reference_scaled),):
            raise ValueError("training_distance_sorted has an invalid shape")
        if not np.all(np.isfinite(self.reference_scaled)):
            raise ValueError("reference_scaled contains non-finite values")
        if (
            not np.all(np.isfinite(self.training_distance_sorted))
            or np.any(self.training_distance_sorted < 0)
        ):
            raise ValueError("Training distances must be finite and non-negative")
        if np.any(np.diff(self.training_distance_sorted) < 0):
            raise ValueError("training_distance_sorted must be sorted")
        if self.color_center.shape != (n_colors,) or self.color_scale.shape != (n_colors,):
            raise ValueError("Color center/scale dimensions do not match the features")
        if not np.all(np.isfinite(self.color_center)):
            raise ValueError("Color center contains non-finite values")
        if not np.all(np.isfinite(self.color_scale)) or np.any(self.color_scale <= 0):
            raise ValueError("Color scale must contain finite positive values")
        if self.magnitude_column not in self.mag_columns:
            raise ValueError("magnitude_column is absent from mag_columns")
        if not np.isclose(self.support_level, 0.99):
            raise ValueError("The published flag schema requires support_level=0.99")


def fit_applicability_reference(
    training: pd.DataFrame,
    *,
    mag_bounds: tuple[float, float],
    mag_columns: Sequence[str] = DEFAULT_MAG_COLUMNS,
    magnitude_column: str = "i_cmodel_mag",
    n_neighbors: int = 20,
    max_reference_size: int | None = None,
    random_state: int = 50,
    chunk_size: int = 100_000,
    workers: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> ApplicabilityReference:
    """Fit a robust k-nearest-neighbor color-support reference.

    Colors are centered by their median and scaled by their interquartile
    ranges. Local support is represented by the distance to the ``k``-th
    nearest neighbor. The 99th percentile of leave-one-out training distances
    defines the published support boundary.

    Parameters
    ----------
    training
        Spectroscopic reference photometry. Spectroscopic quantities may be
        present but are not used as features.
    mag_bounds
        Inclusive adopted bounds for ``magnitude_column``.
    mag_columns
        Five ordered magnitude columns used to form adjacent colors.
    magnitude_column
        Column used by the independent magnitude-domain flag.
    n_neighbors
        Number of neighboring reference objects used for the local distance.
    max_reference_size
        Optional deterministic subsample size. ``None`` uses every row with
        finite five-band photometry inside ``mag_bounds``.
    random_state
        Seed used only if ``max_reference_size`` triggers subsampling.
    chunk_size, workers
        Controls for the k-d tree queries.
    metadata
        Provenance stored with the reference artifact.

    Returns
    -------
    ApplicabilityReference
        Reference that can be saved and applied without either ML model.
    """
    lower, upper = (float(mag_bounds[0]), float(mag_bounds[1]))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("mag_bounds must contain two finite increasing values")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if magnitude_column not in mag_columns:
        raise ValueError("magnitude_column must be one of mag_columns")

    colors, valid_colors = adjacent_colors(training, mag_columns)
    magnitude = _numeric_matrix(training, [magnitude_column])[:, 0]
    valid = (
        valid_colors
        & np.isfinite(magnitude)
        & (magnitude >= lower)
        & (magnitude <= upper)
    )
    colors = colors[valid]
    n_color_valid = len(colors)
    if n_color_valid <= n_neighbors:
        raise ValueError(
            "Too few rows with finite five-band photometry inside mag_bounds"
        )

    n_input = len(training)
    was_subsampled = False
    if max_reference_size is not None:
        if max_reference_size <= n_neighbors:
            raise ValueError("max_reference_size must exceed n_neighbors")
        if n_color_valid > max_reference_size:
            rng = np.random.default_rng(random_state)
            selected = np.sort(
                rng.choice(n_color_valid, size=max_reference_size, replace=False)
            )
            colors = colors[selected]
            was_subsampled = True

    center = np.median(colors, axis=0)
    q25, q75 = np.percentile(colors, [25.0, 75.0], axis=0)
    scale = q75 - q25
    fallback = np.std(colors, axis=0)
    scale = np.where(scale > 0, scale, fallback)
    scale = np.where(scale > 0, scale, 1.0)
    reference_scaled = ((colors - center) / scale).astype(np.float32)

    tree = cKDTree(reference_scaled, compact_nodes=True, balanced_tree=True)
    # k+1 includes the row itself plus k genuinely neighboring training rows.
    training_distance = _query_neighbor_distance(
        tree,
        reference_scaled,
        k=n_neighbors + 1,
        chunk_size=chunk_size,
        workers=workers,
    )

    provenance = dict(metadata or {})
    provenance.update(
        {
            "n_training_input": int(n_input),
            "n_training_reference_valid": int(n_color_valid),
            "n_training_reference_invalid": int(n_input - n_color_valid),
            "reference_subsampled": bool(was_subsampled),
            "random_state_if_subsampled": int(random_state),
            "method": "robust_scaled_adjacent_colors_knn",
        }
    )
    reference = ApplicabilityReference(
        reference_scaled=reference_scaled,
        training_distance_sorted=np.sort(training_distance),
        color_center=center,
        color_scale=scale,
        mag_bounds=(lower, upper),
        n_neighbors=int(n_neighbors),
        mag_columns=tuple(mag_columns),
        color_names=DEFAULT_COLOR_NAMES if tuple(mag_columns) == DEFAULT_MAG_COLUMNS else tuple(
            f"{left}-{right}" for left, right in zip(mag_columns[:-1], mag_columns[1:])
        ),
        magnitude_column=magnitude_column,
        metadata=provenance,
    )
    reference.validate()
    return reference


def compute_applicability_flags(
    catalog: pd.DataFrame,
    reference: ApplicabilityReference,
    *,
    chunk_size: int = 100_000,
    workers: int = 1,
) -> pd.DataFrame:
    """Compute the four published turboPDZ applicability columns.

    ``turbopdz_specz_support_percentile`` is the empirical percentile of the
    query object's local color-space distance relative to the spectroscopic
    reference. Values near zero are in dense, familiar regions; values near
    100 are increasingly sparse or outlying. Invalid five-band photometry
    receives ``NaN`` and fails both support and combined applicability flags.
    The returned table preserves the input index and contains only flag
    columns, which makes it convenient to attach the results to a FITS table.
    """
    reference.validate()
    colors, valid_colors = adjacent_colors(catalog, reference.mag_columns)
    magnitude = _numeric_matrix(catalog, [reference.magnitude_column])[:, 0]
    lower, upper = reference.mag_bounds
    in_mag_domain = (
        np.isfinite(magnitude) & (magnitude >= lower) & (magnitude <= upper)
    )

    support_percentile = np.full(len(catalog), np.nan, dtype=np.float32)
    in_support = np.zeros(len(catalog), dtype=bool)
    if np.any(valid_colors):
        scaled = (
            (colors[valid_colors] - reference.color_center) / reference.color_scale
        ).astype(np.float32)
        tree = cKDTree(
            reference.reference_scaled,
            compact_nodes=True,
            balanced_tree=True,
        )
        distance = _query_neighbor_distance(
            tree,
            scaled,
            k=reference.n_neighbors,
            chunk_size=chunk_size,
            workers=workers,
        )
        rank = np.searchsorted(
            reference.training_distance_sorted,
            distance,
            side="right",
        )
        percentile = 100.0 * rank / len(reference.training_distance_sorted)
        support_percentile[valid_colors] = percentile.astype(np.float32)
        in_support[valid_colors] = distance <= reference.support_distance_99

    output = pd.DataFrame(index=catalog.index)
    output["turbopdz_in_mag_domain"] = in_mag_domain
    output["turbopdz_specz_support_percentile"] = support_percentile
    output["turbopdz_in_specz_support_99"] = in_support
    output["turbopdz_applicability_flag"] = in_mag_domain & in_support
    return output


def compute_reference_support_diagnostics(
    reference: ApplicabilityReference,
    *,
    chunk_size: int = 100_000,
    workers: int = 1,
) -> pd.DataFrame:
    """Return row-aligned leave-one-out diagnostics for the reference.

    The reference artifact stores sorted training distances for efficient VAC
    percentile evaluation. This function recomputes the unsorted distances so
    they can be associated with individual reference points in diagnostic
    projections. The self-match is excluded by querying neighbor ``k + 1``.
    """
    reference.validate()
    tree = cKDTree(
        reference.reference_scaled,
        compact_nodes=True,
        balanced_tree=True,
    )
    distance = _query_neighbor_distance(
        tree,
        reference.reference_scaled,
        k=reference.n_neighbors + 1,
        chunk_size=chunk_size,
        workers=workers,
    )
    rank = np.searchsorted(
        reference.training_distance_sorted,
        distance,
        side="right",
    )
    percentile = 100.0 * rank / len(reference.training_distance_sorted)
    return pd.DataFrame(
        {
            "knn_distance": distance,
            "support_percentile": percentile.astype(np.float32),
            "in_specz_support_99": distance <= reference.support_distance_99,
        }
    )


def add_applicability_flags(
    catalog: pd.DataFrame,
    reference: ApplicabilityReference,
    *,
    copy: bool = True,
    overwrite: bool = False,
    chunk_size: int = 100_000,
    workers: int = 1,
) -> pd.DataFrame:
    """Append applicability columns to a pandas catalog.

    Use :func:`compute_applicability_flags` when only the four new columns are
    needed, for example before attaching them to an Astropy FITS table.
    """
    existing = [name for name in FLAG_COLUMNS if name in catalog.columns]
    if existing and not overwrite:
        raise ValueError(
            f"Applicability columns already exist: {existing}; "
            "pass overwrite=True to replace them"
        )

    output = catalog.copy() if copy else catalog
    flags = compute_applicability_flags(
        catalog,
        reference,
        chunk_size=chunk_size,
        workers=workers,
    )
    for name in FLAG_COLUMNS:
        output[name] = flags[name]
    return output
