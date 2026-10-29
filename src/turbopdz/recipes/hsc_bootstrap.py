"""Paired-bootstrap utilities for the HSC reference comparisons.

The functions in this module operate only on saved predictions and catalog
columns.  They never refit preprocessing, zML, or sigmaML models.  Bootstrap
draws are paired at object level within each evaluation cohort:

* the 15% development holdout for point-estimate comparisons;
* the independent 7.5% sigmaML final-test set for error-scale and reliability
  comparisons.

All reported improvements use a positive-is-better convention for turboPDZ.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DEFAULT_RETAINED_GRID = np.linspace(0.05, 1.0, 100)
OUTLIER_THRESHOLD = 0.15


def sigma_nmad(dz: np.ndarray) -> float:
    """Return the normalized median absolute deviation of residuals."""

    values = np.asarray(dz, dtype=float)
    median = np.median(values)
    return float(1.48 * np.median(np.abs(values - median)))


def eta_outliers(dz: np.ndarray, threshold: float = OUTLIER_THRESHOLD) -> float:
    """Return the fraction of residuals whose absolute value exceeds threshold."""

    return float(np.mean(np.abs(np.asarray(dz, dtype=float)) > threshold))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _require_files(directory: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing required files {missing}")


def _assert_run_metadata(
    info: Mapping[str, Any],
    *,
    cat: str,
    layer: str,
    seed: int,
) -> None:
    expected = {
        "cat": cat,
        "layer": layer,
        "seed": seed,
        "specz_selection": "homogeneous_true",
    }
    mismatches = {
        key: (info.get(key), value)
        for key, value in expected.items()
        if info.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"Run metadata mismatch: {mismatches}")
    if info.get("specz_flag_homogeneous_required") is False:
        raise AssertionError("Run does not require homogeneous spectroscopy")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        values = np.asarray(array)
        if len(values) != len(mask):
            raise ValueError("All arrays must have the same length")
        mask &= np.isfinite(values)
    return mask


def load_configuration(
    project_root: str | Path,
    spec: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate one zML/sigmaML pipeline-layer configuration."""

    root = Path(project_root).resolve()
    cat = str(spec["cat"])
    layer = str(spec["layer"])
    seed = int(spec.get("seed", 50))
    zml_tag = str(spec["zml_run"])
    sigmaml_tag = str(spec["sigmaml_run"])
    zml_dir = root / "models_optuna" / zml_tag
    sigmaml_dir = root / "models_optuna" / sigmaml_tag

    _require_files(zml_dir, ["run_info.json", "split_idx.npz", "test_outputs.npz"])
    _require_files(
        sigmaml_dir,
        [
            "run_info.json",
            "split_idx.npz",
            "test_outputs.npz",
            "reliability_artifacts.npz",
        ],
    )

    zml_info = _read_json(zml_dir / "run_info.json")
    sigmaml_info = _read_json(sigmaml_dir / "run_info.json")
    _assert_run_metadata(zml_info, cat=cat, layer=layer, seed=seed)
    _assert_run_metadata(sigmaml_info, cat=cat, layer=layer, seed=seed)

    if zml_info.get("run_tag") != zml_tag:
        raise AssertionError(f"zML run_info tag does not match {zml_tag}")
    if sigmaml_info.get("run_tag") != sigmaml_tag:
        raise AssertionError(f"sigmaML run_info tag does not match {sigmaml_tag}")
    if sigmaml_info.get("run_tag_zml") != zml_tag:
        raise AssertionError(
            "sigmaML provenance does not point to the declared zML run: "
            f"{sigmaml_info.get('run_tag_zml')} != {zml_tag}"
        )
    if sigmaml_info.get("protocol") != "zml_oof_85__calibration_7p5__test_7p5":
        raise AssertionError("Unexpected sigmaML split protocol")
    if int(sigmaml_info.get("zml_oof_folds", -1)) != 5:
        raise AssertionError("Expected five zML OOF folds")

    zml_split = _load_npz(zml_dir / "split_idx.npz")
    sigma_split = _load_npz(sigmaml_dir / "split_idx.npz")
    zml_pool = np.asarray(zml_split["pool_idx"], dtype=np.int64)
    zml_holdout = np.asarray(zml_split["test_idx"], dtype=np.int64)
    trainval = np.asarray(sigma_split["trainval_idx"], dtype=np.int64)
    calibration = np.asarray(sigma_split["calib_idx"], dtype=np.int64)
    final_test = np.asarray(sigma_split["test_idx"], dtype=np.int64)

    if not np.array_equal(zml_pool, np.asarray(sigma_split["zml_pool_idx"])):
        raise AssertionError("sigmaML train pool differs from the zML source pool")
    if not np.array_equal(zml_pool, trainval):
        raise AssertionError("sigmaML trainval rows differ from the zML source pool")
    if not np.array_equal(zml_holdout, np.asarray(sigma_split["zml_holdout_idx"])):
        raise AssertionError("sigmaML holdout differs from the zML source holdout")
    if np.intersect1d(trainval, calibration).size:
        raise AssertionError("Training and calibration rows overlap")
    if np.intersect1d(trainval, final_test).size:
        raise AssertionError("Training and final-test rows overlap")
    if np.intersect1d(calibration, final_test).size:
        raise AssertionError("Calibration and final-test rows overlap")
    rebuilt_holdout = np.sort(np.concatenate([calibration, final_test]))
    if not np.array_equal(rebuilt_holdout, np.sort(zml_holdout)):
        raise AssertionError("Calibration and final test do not partition the zML holdout")

    zml_outputs = _load_npz(zml_dir / "test_outputs.npz")
    table_path = root / str(zml_info["data_files"]["table_parquet"])
    table = pd.read_parquet(
        table_path,
        columns=["specz_redshift", "photoz_best"],
    )
    z_spec_point = table["specz_redshift"].to_numpy(dtype=float)[zml_holdout]
    z_best_point = table["photoz_best"].to_numpy(dtype=float)[zml_holdout]
    z_ml_point = np.asarray(zml_outputs["z_pred_test"], dtype=float)
    np.testing.assert_allclose(
        np.asarray(zml_outputs["y_test"], dtype=float),
        z_spec_point,
        rtol=0.0,
        atol=1e-6,
    )
    point_mask = _finite_mask(z_spec_point, z_best_point, z_ml_point)
    if not point_mask.all():
        raise AssertionError("Non-finite values in the point-estimate paired cohort")
    dz_ml_point = (z_ml_point - z_spec_point) / (1.0 + z_spec_point)
    dz_best_point = (z_best_point - z_spec_point) / (1.0 + z_spec_point)
    point = {
        "z_spec": z_spec_point,
        "z_ml": z_ml_point,
        "z_best": z_best_point,
        "dz_ml": dz_ml_point,
        "dz_best": dz_best_point,
    }

    sigma_outputs = _load_npz(sigmaml_dir / "test_outputs.npz")
    reliability_artifacts = _load_npz(sigmaml_dir / "reliability_artifacts.npz")
    zml_table_path = root / str(sigmaml_info["data_files"]["zml_table_parquet"])
    required_columns = [
        "specz_redshift",
        "z_ml",
        "photoz_best",
        "Dz_zml",
        "Dz_best",
        "|Dz_zml|",
        "|Dz_best|",
        "s_best",
        "photoz_conf_best",
        "photoz_risk_best",
    ]
    zml_table = pd.read_parquet(zml_table_path, columns=required_columns)
    final = zml_table.iloc[final_test]
    z_spec_final = final["specz_redshift"].to_numpy(dtype=float)
    z_ml_final = final["z_ml"].to_numpy(dtype=float)
    sigma_ml = np.asarray(sigma_outputs["sigma_ml_test"], dtype=float)
    r_ml = np.asarray(reliability_artifacts["rml_test"], dtype=float)
    np.testing.assert_allclose(
        np.asarray(sigma_outputs["z_spec_test"], dtype=float),
        z_spec_final,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sigma_outputs["zml_pred_test"], dtype=float),
        z_ml_final,
        rtol=1e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sigma_outputs["abs_dz_test"], dtype=float),
        final["|Dz_zml|"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=5e-6,
    )

    reliability = {
        "dz_ml": final["Dz_zml"].to_numpy(dtype=float),
        "dz_best": final["Dz_best"].to_numpy(dtype=float),
        "abs_dz_ml": final["|Dz_zml|"].to_numpy(dtype=float),
        "abs_dz_best": final["|Dz_best|"].to_numpy(dtype=float),
        "sigma_ml": sigma_ml,
        "s_best": final["s_best"].to_numpy(dtype=float),
        "r_ml": r_ml,
        "photoz_conf_best": final["photoz_conf_best"].to_numpy(dtype=float),
        "photoz_risk_best": final["photoz_risk_best"].to_numpy(dtype=float),
    }
    common_mask = _finite_mask(*reliability.values())
    reliability = {key: values[common_mask] for key, values in reliability.items()}
    if len(reliability["dz_ml"]) < 50:
        raise AssertionError("Too few common finite sigmaML final-test rows")

    provenance = {
        "cat": cat,
        "layer": layer,
        "seed": seed,
        "zml_run": zml_tag,
        "sigmaml_run": sigmaml_tag,
        "n_point_test": int(len(point["z_spec"])),
        "n_reliability_test": int(len(reliability["dz_ml"])),
        "n_calibration": int(len(calibration)),
        "n_trainval": int(len(trainval)),
        "zml_data_checksums": zml_info.get("data_checksums", {}),
        "sigmaml_data_checksums": sigmaml_info.get("data_checksums", {}),
    }
    return point, reliability, provenance


def point_metric_pairs(data: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Return (turboPDZ, catalog) values for the primary point metrics."""

    return {
        "mae": (
            float(np.mean(np.abs(data["z_ml"] - data["z_spec"]))),
            float(np.mean(np.abs(data["z_best"] - data["z_spec"]))),
        ),
        "sigma_nmad": (sigma_nmad(data["dz_ml"]), sigma_nmad(data["dz_best"])),
        "eta_0_15": (
            eta_outliers(data["dz_ml"]),
            eta_outliers(data["dz_best"]),
        ),
    }


def _correlation_matrix(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.column_stack(
        [
            data["abs_dz_ml"],
            data["abs_dz_best"],
            data["sigma_ml"],
            data["s_best"],
        ]
    )
    pearson = np.corrcoef(values, rowvar=False)
    ranks = np.column_stack([rankdata(values[:, i]) for i in range(values.shape[1])])
    spearman = np.corrcoef(ranks, rowvar=False)
    return pearson, spearman


def correlation_metric_pairs(
    data: Mapping[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Return native and controlled correlation pairs.

    Column order is |Dz_ml|, |Dz_best|, sigma_ml, and s_best.
    """

    pearson, spearman = _correlation_matrix(data)
    return {
        "native__pearson": (float(pearson[0, 2]), float(pearson[1, 3])),
        "native__spearman": (float(spearman[0, 2]), float(spearman[1, 3])),
        "controlled_z_ml__pearson": (float(pearson[0, 2]), float(pearson[0, 3])),
        "controlled_z_ml__spearman": (float(spearman[0, 2]), float(spearman[0, 3])),
        "controlled_z_best__pearson": (float(pearson[1, 2]), float(pearson[1, 3])),
        "controlled_z_best__spearman": (float(spearman[1, 2]), float(spearman[1, 3])),
    }


def filter_auc_values(
    data: Mapping[str, np.ndarray],
    retained_grid: np.ndarray = DEFAULT_RETAINED_GRID,
    outlier_threshold: float = OUTLIER_THRESHOLD,
    scale: float = 100.0,
) -> dict[tuple[str, str, str], float]:
    """Compute all native inputs needed for retained-fraction AUC comparisons.

    The implementation reproduces ``pdz_metrics.compute_filter_auc`` while
    sorting each selector only once.  Keys are ``(selector, target, metric)``.
    """

    scores = {
        "r_ml": np.asarray(data["r_ml"], dtype=float),
        "photoz_conf_best": np.asarray(data["photoz_conf_best"], dtype=float),
        "photoz_risk_best": np.asarray(data["photoz_risk_best"], dtype=float),
    }
    directions = {
        "r_ml": "high",
        "photoz_conf_best": "high",
        "photoz_risk_best": "low",
    }
    targets = {
        "z_ml": np.asarray(data["dz_ml"], dtype=float),
        "z_best": np.asarray(data["dz_best"], dtype=float),
    }
    n = len(next(iter(scores.values())))
    result: dict[tuple[str, str, str], float] = {}

    for selector, score in scores.items():
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        direction = directions[selector]
        quantiles = (
            1.0 - retained_grid if direction == "high" else retained_grid
        )
        thresholds = np.quantile(sorted_score, quantiles)
        if direction == "high":
            bounds = np.searchsorted(sorted_score, thresholds, side="left")
            counts = n - bounds
        else:
            bounds = np.searchsorted(sorted_score, thresholds, side="right")
            counts = bounds
        fractions = counts.astype(float) / n
        curve_order = np.argsort(fractions, kind="mergesort")

        for target, residual in targets.items():
            sorted_residual = residual[order]
            scatter = np.empty(len(retained_grid), dtype=float)
            outliers = np.empty(len(retained_grid), dtype=float)
            for i, bound in enumerate(bounds):
                selected = (
                    sorted_residual[bound:]
                    if direction == "high"
                    else sorted_residual[:bound]
                )
                scatter[i] = sigma_nmad(selected)
                outliers[i] = eta_outliers(selected, threshold=outlier_threshold)
            result[(selector, target, "sigma_nmad")] = scale * float(
                np.trapz(scatter[curve_order], fractions[curve_order])
            )
            result[(selector, target, "eta_0_15")] = scale * float(
                np.trapz(outliers[curve_order], fractions[curve_order])
            )
    return result


def auc_metric_pairs(
    data: Mapping[str, np.ndarray],
    retained_grid: np.ndarray = DEFAULT_RETAINED_GRID,
) -> dict[str, tuple[float, float]]:
    """Return native and controlled AUC pairs for both catalog selectors."""

    values = filter_auc_values(data, retained_grid=retained_grid)
    pairs: dict[str, tuple[float, float]] = {}
    comparators = ("photoz_conf_best", "photoz_risk_best")
    metrics = ("sigma_nmad", "eta_0_15")
    for comparator in comparators:
        for metric in metrics:
            pairs[f"native__{comparator}__{metric}"] = (
                values[("r_ml", "z_ml", metric)],
                values[(comparator, "z_best", metric)],
            )
            for target in ("z_ml", "z_best"):
                pairs[f"controlled_{target}__{comparator}__{metric}"] = (
                    values[("r_ml", target, metric)],
                    values[(comparator, target, metric)],
                )
    return pairs


def validate_comparison_products(
    project_root: str | Path,
    spec: Mapping[str, Any],
    *,
    point_pairs: Mapping[str, tuple[float, float]],
    correlation_pairs: Mapping[str, tuple[float, float]],
    auc_pairs: Mapping[str, tuple[float, float]],
    atol: float = 1e-4,
) -> dict[str, float]:
    """Verify that saved comparison tables reproduce the selected run outputs.

    The comparison notebooks may rebuild ``r_ml`` after CPU inference, whereas
    this module deliberately uses the immutable saved training artifacts.  The
    resulting threshold-rank changes can move an integrated AUC by a few
    ``1e-5``; the accepted tolerance is recorded through the returned deltas.
    """

    root = Path(project_root).resolve()
    cat = str(spec["cat"])
    layer = str(spec["layer"])
    zml_dir = root / "models_optuna" / str(spec["zml_run"])
    zml_info = _read_json(zml_dir / "run_info.json")
    deltas: dict[str, float] = {}

    def check(label: str, actual: float, expected: float) -> None:
        delta = abs(float(actual) - float(expected))
        deltas[label] = delta
        if not np.isclose(actual, expected, rtol=0.0, atol=atol):
            raise AssertionError(
                f"{cat}/{layer}: stale or inconsistent {label}: "
                f"computed={actual:.12g}, saved={expected:.12g}, |delta|={delta:.3g}"
            )

    zml_metrics = zml_info["test_metrics"]
    point_keys = {
        "mae": "mae",
        "sigma_nmad": "sigma_nmad",
        "eta_0_15": "eta_outliers",
    }
    for metric, saved_key in point_keys.items():
        check(
            f"zML {metric}",
            point_pairs[metric][0],
            float(zml_metrics[saved_key]),
        )

    comparison_dir = root / "comparison_tables" / "sigmaml"
    sigma_bins_path = comparison_dir / f"tab_sigma_bins_{cat}_{layer}.csv"
    filter_auc_path = comparison_dir / f"filter_auc_{cat}_{layer}.csv"
    controlled_corr_path = comparison_dir / f"common_residual_correlations_{cat}_{layer}.csv"
    controlled_auc_path = comparison_dir / f"common_residual_filter_metrics_{cat}_{layer}.csv"
    _require_files(
        comparison_dir,
        [
            sigma_bins_path.name,
            filter_auc_path.name,
            controlled_corr_path.name,
            controlled_auc_path.name,
        ],
    )

    sigma_bins = pd.read_csv(sigma_bins_path, comment="#")
    all_rows = sigma_bins[sigma_bins["z_spec bin"].astype(str).str.contains("All")]
    if len(all_rows) != 1:
        raise AssertionError(f"{sigma_bins_path}: expected one all-redshift row")
    all_row = all_rows.iloc[0]
    check("native Pearson sigma_ml", correlation_pairs["native__pearson"][0], all_row["Pearson r (σ_ml)"])
    check("native Pearson s_best", correlation_pairs["native__pearson"][1], all_row["Pearson r (s_best)"])
    check("native Spearman sigma_ml", correlation_pairs["native__spearman"][0], all_row["Spearman ρ (σ_ml)"])
    check("native Spearman s_best", correlation_pairs["native__spearman"][1], all_row["Spearman ρ (s_best)"])

    native_auc = pd.read_csv(filter_auc_path, comment="#").set_index("filter")
    for comparator in ("photoz_conf_best", "photoz_risk_best"):
        for metric, csv_col in (("sigma_nmad", "area_sigma"), ("eta_0_15", "area_eta")):
            turbo, catalog = auc_pairs[f"native__{comparator}__{metric}"]
            check(f"native AUC {metric} r_ml", turbo, native_auc.loc["r_ml", csv_col])
            check(
                f"native AUC {metric} {comparator}",
                catalog,
                native_auc.loc[comparator, csv_col],
            )

    controlled_corr = pd.read_csv(controlled_corr_path, comment="#")
    for target in ("z_ml", "z_best"):
        block = controlled_corr[controlled_corr["target"] == target].set_index("uncertainty")
        for metric, csv_col in (("pearson", "pearson_r"), ("spearman", "spearman_rho")):
            turbo, catalog = correlation_pairs[f"controlled_{target}__{metric}"]
            check(
                f"controlled {target} {metric} sigma_ml",
                turbo,
                block.loc["sigma_ml", csv_col],
            )
            check(
                f"controlled {target} {metric} s_best",
                catalog,
                block.loc["s_best", csv_col],
            )

    controlled_auc = pd.read_csv(controlled_auc_path, comment="#")
    for target in ("z_ml", "z_best"):
        block = controlled_auc[controlled_auc["target"] == target].set_index("filter")
        for comparator in ("photoz_conf_best", "photoz_risk_best"):
            for metric, csv_col in (("sigma_nmad", "area_sigma"), ("eta_0_15", "area_eta")):
                turbo, catalog = auc_pairs[f"controlled_{target}__{comparator}__{metric}"]
                check(
                    f"controlled {target} AUC {metric} r_ml",
                    turbo,
                    block.loc["r_ml", csv_col],
                )
                check(
                    f"controlled {target} AUC {metric} {comparator}",
                    catalog,
                    block.loc[comparator, csv_col],
                )
    return deltas


def _draw_improvement_pairs(
    pairs: Mapping[str, tuple[float, float]],
    *,
    lower_is_better: bool,
) -> dict[str, float]:
    if lower_is_better:
        return {key: catalog - turbo for key, (turbo, catalog) in pairs.items()}
    return {key: turbo - catalog for key, (turbo, catalog) in pairs.items()}


def bootstrap_point_metrics(
    data: Mapping[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]:
    """Bootstrap primary point-estimate metric improvements."""

    observed_pairs = point_metric_pairs(data)
    draws = {key: np.empty(n_boot, dtype=float) for key in observed_pairs}
    rng = np.random.default_rng(seed)
    n = len(data["z_spec"])
    for b in range(n_boot):
        index = rng.integers(0, n, size=n, dtype=np.int32)
        sampled = {key: values[index] for key, values in data.items()}
        improvement = _draw_improvement_pairs(
            point_metric_pairs(sampled), lower_is_better=True
        )
        for key, value in improvement.items():
            draws[key][b] = value
    return observed_pairs, draws


def bootstrap_reliability_metrics(
    data: Mapping[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
    retained_grid: np.ndarray = DEFAULT_RETAINED_GRID,
    progress_every: int = 0,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, np.ndarray],
    dict[str, tuple[float, float]],
    dict[str, np.ndarray],
]:
    """Bootstrap correlations and retained-fraction AUCs on shared draws."""

    observed_corr = correlation_metric_pairs(data)
    observed_auc = auc_metric_pairs(data, retained_grid=retained_grid)
    corr_draws = {key: np.empty(n_boot, dtype=float) for key in observed_corr}
    auc_draws = {key: np.empty(n_boot, dtype=float) for key in observed_auc}
    rng = np.random.default_rng(seed)
    n = len(data["dz_ml"])

    for b in range(n_boot):
        index = rng.integers(0, n, size=n, dtype=np.int32)
        sampled = {key: values[index] for key, values in data.items()}
        corr_improvement = _draw_improvement_pairs(
            correlation_metric_pairs(sampled), lower_is_better=False
        )
        auc_improvement = _draw_improvement_pairs(
            auc_metric_pairs(sampled, retained_grid=retained_grid),
            lower_is_better=True,
        )
        for key, value in corr_improvement.items():
            corr_draws[key][b] = value
        for key, value in auc_improvement.items():
            auc_draws[key][b] = value
        if progress_every and (b + 1) % progress_every == 0:
            print(f"  reliability bootstrap {b + 1}/{n_boot}", flush=True)
    return observed_corr, corr_draws, observed_auc, auc_draws


def _comparison_metadata(family: str, key: str) -> dict[str, str]:
    if family == "point_estimate":
        return {
            "scope": "native",
            "target": "z_ml_vs_z_best",
            "comparator": "z_best",
            "metric": key,
            "turbo_quantity": "z_ml",
            "catalog_quantity": "z_best",
        }

    parts = key.split("__")
    if family == "correlation":
        scope = parts[0]
        metric = parts[1]
        target = "native_residuals" if scope == "native" else scope.removeprefix("controlled_")
        return {
            "scope": "native" if scope == "native" else "controlled",
            "target": target,
            "comparator": "s_best",
            "metric": metric,
            "turbo_quantity": "sigma_ml",
            "catalog_quantity": "s_best",
        }

    if family == "filter_auc":
        scope, comparator, metric = parts
        target = "native_residuals" if scope == "native" else scope.removeprefix("controlled_")
        return {
            "scope": "native" if scope == "native" else "controlled",
            "target": target,
            "comparator": comparator,
            "metric": metric,
            "turbo_quantity": "r_ml",
            "catalog_quantity": comparator,
        }
    raise ValueError(f"Unknown family {family!r}")


def summarize_pairs(
    *,
    cat: str,
    layer: str,
    cohort_n: int,
    family: str,
    observed_pairs: Mapping[str, tuple[float, float]],
    draws: Mapping[str, np.ndarray],
    ci_level: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize positive-is-better paired improvements and percentile CIs."""

    alpha = 1.0 - ci_level
    rows = []
    lower_is_better = family in {"point_estimate", "filter_auc"}
    for key, (turbo, catalog) in observed_pairs.items():
        improvement = catalog - turbo if lower_is_better else turbo - catalog
        values = np.asarray(draws[key], dtype=float)
        ci_low, ci_high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        if ci_low > 0.0:
            status = "turbo_supported"
        elif ci_high < 0.0:
            status = "catalog_supported"
        else:
            status = "unresolved"
        rows.append(
            {
                "cat": cat,
                "layer": layer,
                "family": family,
                **_comparison_metadata(family, key),
                "N": int(cohort_n),
                "turbo_value": float(turbo),
                "catalog_value": float(catalog),
                "improvement": float(improvement),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_improvement": float(np.mean(values > 0.0)),
                "status": status,
                "ci_level": float(ci_level),
                "n_boot": int(n_boot),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def run_configuration(
    project_root: str | Path,
    spec: Mapping[str, Any],
    *,
    n_boot: int,
    bootstrap_seed: int,
    ci_level: float = 0.95,
    retained_grid: np.ndarray = DEFAULT_RETAINED_GRID,
) -> dict[str, Any]:
    """Validate, bootstrap, and summarize one manifest entry."""

    point, reliability, provenance = load_configuration(project_root, spec)
    config_index = int(spec.get("config_index", 0))
    point_seed = int(bootstrap_seed + 10_000 * config_index + 101)
    reliability_seed = int(bootstrap_seed + 10_000 * config_index + 202)

    point_pairs, point_draws = bootstrap_point_metrics(
        point, n_boot=n_boot, seed=point_seed
    )
    corr_pairs, corr_draws, auc_pairs, auc_draws = bootstrap_reliability_metrics(
        reliability,
        n_boot=n_boot,
        seed=reliability_seed,
        retained_grid=retained_grid,
    )
    validation_deltas = validate_comparison_products(
        project_root,
        spec,
        point_pairs=point_pairs,
        correlation_pairs=corr_pairs,
        auc_pairs=auc_pairs,
    )
    summaries = pd.concat(
        [
            summarize_pairs(
                cat=provenance["cat"],
                layer=provenance["layer"],
                cohort_n=provenance["n_point_test"],
                family="point_estimate",
                observed_pairs=point_pairs,
                draws=point_draws,
                ci_level=ci_level,
                n_boot=n_boot,
                seed=point_seed,
            ),
            summarize_pairs(
                cat=provenance["cat"],
                layer=provenance["layer"],
                cohort_n=provenance["n_reliability_test"],
                family="correlation",
                observed_pairs=corr_pairs,
                draws=corr_draws,
                ci_level=ci_level,
                n_boot=n_boot,
                seed=reliability_seed,
            ),
            summarize_pairs(
                cat=provenance["cat"],
                layer=provenance["layer"],
                cohort_n=provenance["n_reliability_test"],
                family="filter_auc",
                observed_pairs=auc_pairs,
                draws=auc_draws,
                ci_level=ci_level,
                n_boot=n_boot,
                seed=reliability_seed,
            ),
        ],
        ignore_index=True,
    )
    draw_payload = {
        **{f"point__{key}": value for key, value in point_draws.items()},
        **{f"correlation__{key}": value for key, value in corr_draws.items()},
        **{f"filter_auc__{key}": value for key, value in auc_draws.items()},
    }
    return {
        "summary": summaries,
        "draws": draw_payload,
        "provenance": {
            **provenance,
            "comparison_validation_max_abs_delta": float(max(validation_deltas.values())),
        },
    }


def run_manifest(
    project_root: str | Path,
    manifest: Iterable[Mapping[str, Any]],
    *,
    n_boot: int,
    bootstrap_seed: int,
    ci_level: float = 0.95,
    retained_grid: np.ndarray = DEFAULT_RETAINED_GRID,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run all manifest entries, optionally in parallel by configuration."""

    specs = [dict(spec) for spec in manifest]
    if n_jobs <= 1:
        return [
            run_configuration(
                project_root,
                spec,
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed,
                ci_level=ci_level,
                retained_grid=retained_grid,
            )
            for spec in specs
        ]

    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                run_configuration,
                project_root,
                spec,
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed,
                ci_level=ci_level,
                retained_grid=retained_grid,
            ): int(spec.get("config_index", i))
            for i, spec in enumerate(specs)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            provenance = results[index]["provenance"]
            print(
                f"Completed {provenance['cat']} / {provenance['layer']}",
                flush=True,
            )
    return [results[int(spec.get("config_index", i))] for i, spec in enumerate(specs)]


def _format_ci(row: pd.Series, digits: int, scale: float = 1.0) -> str:
    values = scale * np.array([row["improvement"], row["ci_low"], row["ci_high"]])
    body = f"{values[0]:+.{digits}f} [{values[1]:+.{digits}f}, {values[2]:+.{digits}f}]"
    if row["status"] == "turbo_supported":
        return rf"\textcolor{{red}}{{\textbf{{{body}}}}}"
    if row["status"] == "catalog_supported":
        return rf"\textcolor{{blue}}{{\textbf{{{body}}}}}"
    return body


def _lookup(
    summary: pd.DataFrame,
    *,
    cat: str,
    layer: str,
    family: str,
    scope: str,
    target: str,
    metric: str,
    comparator: str,
) -> pd.Series:
    mask = (
        (summary["cat"] == cat)
        & (summary["layer"] == layer)
        & (summary["family"] == family)
        & (summary["scope"] == scope)
        & (summary["target"] == target)
        & (summary["metric"] == metric)
        & (summary["comparator"] == comparator)
    )
    rows = summary.loc[mask]
    if len(rows) != 1:
        raise AssertionError(f"Expected one summary row, found {len(rows)}")
    return rows.iloc[0]


def make_latex_tables(summary: pd.DataFrame) -> dict[str, str]:
    """Create appendix-ready LaTeX tabular environments from bootstrap results."""

    order = [(cat, layer) for cat in ("demp", "dnnz", "mizuki") for layer in ("wide", "dud")]
    labels = {"demp": "DEmP", "dnnz": "DNNz", "mizuki": "Mizuki"}
    layer_labels = {"wide": "Wide", "dud": "DUD"}

    point_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Pipeline & Layer & $I_{\rm MAE}$ & $I_{\sigma}$ & $I_{\eta}$ [pp] \\",
        r"\midrule",
    ]
    point_no_mae_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Pipeline & Layer & $I_{\sigma}$ & $I_{\eta}$ [pp] \\",
        r"\midrule",
    ]
    for cat, layer in order:
        values = {}
        for metric in ("mae", "sigma_nmad", "eta_0_15"):
            row = _lookup(
                summary,
                cat=cat,
                layer=layer,
                family="point_estimate",
                scope="native",
                target="z_ml_vs_z_best",
                metric=metric,
                comparator="z_best",
            )
            values[metric] = _format_ci(
                row,
                digits=4 if metric != "eta_0_15" else 2,
                scale=100.0 if metric == "eta_0_15" else 1.0,
            )
        point_lines.append(
            f"{labels[cat]} & {layer_labels[layer]} & {values['mae']} & "
            f"{values['sigma_nmad']} & {values['eta_0_15']} \\\\"
        )
        point_no_mae_lines.append(
            f"{labels[cat]} & {layer_labels[layer]} & "
            f"{values['sigma_nmad']} & {values['eta_0_15']} \\\\"
        )
    point_lines.extend([r"\bottomrule", r"\end{tabular}"])
    point_no_mae_lines.extend([r"\bottomrule", r"\end{tabular}"])

    corr_lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Pipeline & Layer & Residual target & $I_{\rho}$ & $I_r$ \\",
        r"\midrule",
    ]
    for cat, layer in order:
        for scope, target, target_label in (
            ("native", "native_residuals", "native"),
            ("controlled", "z_ml", r"$|\Delta z_{\rm ml}|$"),
            ("controlled", "z_best", r"$|\Delta z_{\rm best}|$"),
        ):
            rho = _lookup(
                summary,
                cat=cat,
                layer=layer,
                family="correlation",
                scope=scope,
                target=target,
                metric="spearman",
                comparator="s_best",
            )
            pearson = _lookup(
                summary,
                cat=cat,
                layer=layer,
                family="correlation",
                scope=scope,
                target=target,
                metric="pearson",
                comparator="s_best",
            )
            corr_lines.append(
                f"{labels[cat]} & {layer_labels[layer]} & {target_label} & "
                f"{_format_ci(rho, 3)} & {_format_ci(pearson, 3)} \\\\"
            )
        corr_lines.append(r"\addlinespace")
    corr_lines.extend([r"\bottomrule", r"\end{tabular}"])

    comparator_labels = {
        "photoz_conf_best": r"\texttt{conf\_best}",
        "photoz_risk_best": r"\texttt{risk\_best}",
    }

    def make_auc_lines(
        target_specs: tuple[tuple[str, str, str], ...],
    ) -> list[str]:
        lines = [
            r"\begin{tabular}{llllrr}",
            r"\toprule",
            r"Pipeline & Layer & Residual target & Comparator & $I_{{\rm AUC},\sigma}$ & $I_{{\rm AUC},\eta}$ \\",
            r"\midrule",
        ]
        for cat, layer in order:
            for scope, target, target_label in target_specs:
                for comparator in ("photoz_conf_best", "photoz_risk_best"):
                    sigma = _lookup(
                        summary,
                        cat=cat,
                        layer=layer,
                        family="filter_auc",
                        scope=scope,
                        target=target,
                        metric="sigma_nmad",
                        comparator=comparator,
                    )
                    eta = _lookup(
                        summary,
                        cat=cat,
                        layer=layer,
                        family="filter_auc",
                        scope=scope,
                        target=target,
                        metric="eta_0_15",
                        comparator=comparator,
                    )
                    lines.append(
                        f"{labels[cat]} & {layer_labels[layer]} & {target_label} & "
                        f"{comparator_labels[comparator]} & {_format_ci(sigma, 2)} & "
                        f"{_format_ci(eta, 2)} \\\\"
                    )
            lines.append(r"\addlinespace")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return lines

    native_auc_specs = (("native", "native_residuals", "native"),)
    controlled_auc_specs = (
        ("controlled", "z_ml", r"$|\Delta z_{\rm ml}|$"),
        ("controlled", "z_best", r"$|\Delta z_{\rm best}|$"),
    )
    native_auc_lines = make_auc_lines(native_auc_specs)
    controlled_auc_lines = make_auc_lines(controlled_auc_specs)
    combined_auc_lines = make_auc_lines(native_auc_specs + controlled_auc_specs)

    return {
        "bootstrap_point_estimates.tex": "\n".join(point_lines) + "\n",
        "bootstrap_point_estimates_no_mae.tex": "\n".join(point_no_mae_lines) + "\n",
        "bootstrap_correlations.tex": "\n".join(corr_lines) + "\n",
        "bootstrap_native_filter_auc.tex": "\n".join(native_auc_lines) + "\n",
        "bootstrap_controlled_filter_auc.tex": "\n".join(controlled_auc_lines) + "\n",
        "bootstrap_filter_auc.tex": "\n".join(combined_auc_lines) + "\n",
    }


def claim_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Count supported, unresolved, and catalog-supported comparisons."""

    return (
        summary.groupby(["family", "scope", "metric", "status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["family", "scope", "metric", "status"])
        .reset_index(drop=True)
    )
