"""Small stage functions shared by notebooks and the command-line interface."""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import load_config, run_configs_compatible
from .data import PDZDataset
from .splits import SplitPlan
from .artifacts import write_json, seal_bundle, load_manifest
from .inference import TurboPDZ
from .objectives import resolve_callable
from .evaluation.metrics import photoz_metrics, residual, retained_metrics, filter_auc, error_correlations, common_finite_mask
from .evaluation.bootstrap import paired_bootstrap


def prepare_run(config: dict, directory: str | Path) -> Path:
    """Build a portable dataset and persistent split plan without fitting models."""
    root = Path(directory)
    if (root / "dataset").exists():
        if not run_configs_compatible(json.loads((root / "config.json").read_text()), config):
            raise ValueError("Run configuration changed; choose a new run directory")
        load_run(root)
        write_json(root / "config.json", config)
        return root
    settings = config.get("dataset", {})
    adapter = settings.get("adapter", "synthetic")
    if adapter == "synthetic":
        from .synthetic import make_synthetic_dataset
        dataset = make_synthetic_dataset(seed=config["seed"], **settings.get("kwargs", {}))
    elif adapter == "portable":
        dataset = PDZDataset.load(settings["path"])
    elif adapter == "hsc_products":
        from .recipes.hsc import read_products
        dataset = read_products(settings["path"])
    elif adapter == "custom":
        dataset = resolve_callable(settings["callable"])(**settings.get("kwargs", {}))
        if not isinstance(dataset, PDZDataset):
            raise TypeError("Dataset adapter must return PDZDataset")
    else:
        raise ValueError(f"Unknown dataset adapter: {adapter}")
    root.mkdir(parents=True, exist_ok=True)
    dataset.save(root / "dataset")
    if config["splits"].get("split_path"):
        plan = SplitPlan.load(config["splits"]["split_path"], dataset)
    else:
        plan = SplitPlan.create(dataset, seed=config["seed"], **config["splits"])
    plan.save(root / "splits.npz")
    write_json(root / "config.json", config)
    return root


def load_run(directory: str | Path):
    """Load data/config/splits and verify that every component is still aligned."""
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    dataset = PDZDataset.load(root / "dataset")
    splits = SplitPlan.load(root / "splits.npz", dataset)
    return dataset, splits, config


def evaluate_run(directory: str | Path) -> pd.DataFrame:
    """Save final-test comparisons and a separate original point-estimate holdout."""
    root = Path(directory)
    dataset, splits, config = load_run(root)
    pipeline = TurboPDZ.load(root)
    reference = config.get("dataset", {}).get("reference_column", "photoz_best")
    if reference not in dataset.observables:
        raise KeyError(f"Comparison reference column missing: {reference}")
    all_rows = []
    for cohort, idx in (("final_test", splits.test), ("point_holdout", np.r_[splits.calibration, splits.test])):
        data = dataset.subset(idx)
        predictions = pipeline.predict(data.pdz, data.z_grid, data.observables,
                                       object_id=data.object_id, representation=data.representation)
        predictions["z_spec"] = data.z_spec
        predictions["z_best"] = data.observables[reference].to_numpy()
        predictions["dz_ml"] = residual(predictions["z_ml"].to_numpy(), data.z_spec)
        predictions["dz_best"] = residual(predictions["z_best"].to_numpy(), data.z_spec)
        for name in ("s_best", "photoz_conf_best", "photoz_risk_best"):
            if name in data.observables:
                predictions[name] = data.observables[name].to_numpy()
        predictions.to_csv(root / f"{cohort}.csv", index=False)
        for estimate in ("z_ml", "z_best"):
            for metric, value in photoz_metrics(predictions[estimate].to_numpy(), data.z_spec).items():
                all_rows.append({"cohort": cohort, "estimator": estimate, "metric": metric,
                                 "value": value, "n_objects": len(idx)})
    final = pd.read_csv(root / "final_test.csv")
    if "sigma_ml" in final:
        for target in ("dz_ml", "dz_best"):
            for proxy in ("sigma_ml", "s_best"):
                if proxy in final:
                    cols = {target: final[target].to_numpy(), proxy: final[proxy].to_numpy()}
                    mask = common_finite_mask(cols)
                    for metric, value in error_correlations(np.abs(cols[target][mask]), cols[proxy][mask]).items():
                        all_rows.append({"cohort": "final_test", "target": target, "estimator": proxy,
                                         "metric": metric, "value": value, "n_objects": int(mask.sum())})
        selectors = {name: direction for name, direction in
                     (("r_ml", True), ("photoz_conf_best", True), ("photoz_risk_best", False)) if name in final}
        columns = {name: final[name].to_numpy() for name in ["dz_ml", "dz_best", *selectors]}
        mask = common_finite_mask(columns)
        curve_tables = []
        for scope in ("native", "controlled"):
            for target in (("dz_ml",) if scope == "native" else ("dz_ml", "dz_best")):
                for name, higher in selectors.items():
                    residual_name = "dz_best" if scope == "native" and name != "r_ml" else target
                    curve = retained_metrics(columns[residual_name][mask], columns[name][mask], higher_is_better=higher)
                    curve = curve.assign(scope=scope, target=target, selector=name)
                    curve_tables.append(curve)
        pd.concat(curve_tables, ignore_index=True).to_csv(root / "retained_curves.csv", index=False)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(root / "metrics.csv", index=False)
    write_json(root / "evaluation.json", {"point_cohort": "development holdout = calibration + final test",
                                           "reliability_cohort": "independent final test",
                                           "n_point_holdout": len(splits.calibration) + len(splits.test),
                                           "n_final_test": len(splits.test)})
    return metrics


def bootstrap_run(directory: str | Path, *, bootstrap_config: dict | None = None):
    """Compute point, native, and controlled paired improvements on saved cohorts."""
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    settings = config["bootstrap"] if bootstrap_config is None else bootstrap_config
    options = dict(n_boot=settings["resamples"], confidence_level=settings["confidence_level"], seed=config["seed"])
    point = pd.read_csv(root / "point_holdout.csv")
    columns = {name: point[name].to_numpy() for name in ("z_ml", "z_best", "z_spec")}
    columns = {name: values[common_finite_mask(columns)] for name, values in columns.items()}
    comparisons = {metric: (lambda values, metric=metric: (
        photoz_metrics(values["z_ml"], values["z_spec"])[metric],
        photoz_metrics(values["z_best"], values["z_spec"])[metric]))
        for metric in ("mae", "scatter", "outliers", "absolute_bias")}
    result = paired_bootstrap(columns, comparisons, **options)
    summaries = [result.summary.assign(family="point_estimate", cohort="point_holdout")]
    draws = {f"point__{name}": value for name, value in result.draws.items()}
    final = pd.read_csv(root / "final_test.csv")
    directions = {"r_ml": True, "photoz_conf_best": True, "photoz_risk_best": False}
    catalog = [name for name in directions if name != "r_ml" and name in final]
    needed = ["dz_ml", "dz_best"] + [name for name in ("r_ml", "sigma_ml", "s_best", *catalog) if name in final]
    if "r_ml" in final and (catalog or {"sigma_ml", "s_best"}.issubset(final.columns)):
        arrays = {name: final[name].to_numpy() for name in needed}
        mask = common_finite_mask(arrays)
        arrays = {name: values[mask] for name, values in arrays.items()}
        comparisons, higher = {}, {}
        for scope in ("native", "controlled"):
            for target in (("dz_ml",) if scope == "native" else ("dz_ml", "dz_best")):
                for selector in catalog:
                    for metric in ("scatter", "outliers"):
                        key = f"{scope}__{target}__{selector}__auc_{metric}"
                        reference_target = "dz_best" if scope == "native" else target
                        comparisons[key] = (lambda values, target=target, reference_target=reference_target, selector=selector, metric=metric: (
                            filter_auc(values[target], values["r_ml"], higher_is_better=True)[metric],
                            filter_auc(values[reference_target], values[selector], higher_is_better=directions[selector])[metric]))
                if "s_best" in arrays:
                    for metric in ("spearman", "pearson"):
                        key = f"{scope}__{target}__s_best__{metric}"
                        reference_target = "dz_best" if scope == "native" else target
                        comparisons[key] = (lambda values, target=target, reference_target=reference_target, metric=metric: (
                            error_correlations(np.abs(values[target]), values["sigma_ml"])[metric],
                            error_correlations(np.abs(values[reference_target]), values["s_best"])[metric]))
                        higher[key] = True
        result = paired_bootstrap(arrays, comparisons, higher_is_better=higher, **options)
        summaries.append(result.summary.assign(family="reliability", cohort="final_test"))
        draws.update({f"reliability__{name}": value for name, value in result.draws.items()})
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(root / "bootstrap.csv", index=False)
    np.savez_compressed(root / "bootstrap_draws.npz", **draws)
    write_json(root / "bootstrap.json", dict(options, conditional_on_fitted_models=True))
    return summary


def export_run(directory: str | Path, destination: str | Path) -> Path:
    """Copy only deployable artifacts, never labels, object IDs, or predictions."""
    import shutil
    root, target = Path(directory), Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    for track in ("zml", "sigmaml"):
        source = root / track
        if not source.is_dir():
            continue
        info = load_manifest(source)
        stage = target / track
        stage.mkdir()
        for name in ("model.keras", "preprocessor.npz", "grid.npz", "best_params.json", "calibration.npz"):
            if (source / name).is_file():
                shutil.copy2(source / name, stage / name)
        if track == "sigmaml":
            from .artifacts import file_sha256
            info["parent_zml_manifest_hash"] = file_sha256(target / "zml" / "manifest.json")
        for key in ("files", "software", "bundle_version"):
            info.pop(key, None)
        seal_bundle(stage, info)
    if (root / "applicability.npz").is_file():
        shutil.copy2(root / "applicability.npz", target / "applicability.npz")
    seal_bundle(target, {"kind": "deployable_pipeline", "contains_catalog_extracts": False})
    return target


def run_stage(stage: str, config_path: str | Path, directory: str | Path):
    """Run the same stage from a notebook, CLI, or cluster submission."""
    root = Path(directory)
    config = load_config(config_path)
    if stage == "prepare":
        return prepare_run(config, root)
    saved = json.loads((root / "config.json").read_text())
    if not run_configs_compatible(saved, config, allow_path_changes=stage in {"evaluate", "diagnostics", "bootstrap", "applicability"}):
        raise ValueError("Notebook configuration differs from the prepared run")
    if stage == "bootstrap":
        return bootstrap_run(root, bootstrap_config=config["bootstrap"])
    dataset, splits, saved = load_run(root)
    if stage == "train-zml":
        from .training import train_zml
        return train_zml(dataset, splits, config, root / "zml")
    if stage == "oof":
        from .training import generate_oof
        return generate_oof(dataset, splits, config, root / "zml", root / "oof.npz")
    if stage == "train-sigmaml":
        from .training import train_sigmaml
        return train_sigmaml(dataset, splits, config, root / "zml", root / "oof.npz", root / "sigmaml")
    if stage == "evaluate":
        return evaluate_run(root)
    if stage == "diagnostics":
        from .analysis import write_evaluation_diagnostics
        return write_evaluation_diagnostics(root)
    if stage == "applicability":
        from .applicability import fit_applicability_reference
        settings = dict(config.get("applicability", {}))
        if settings.get("reference_path"):
            from .applicability import ApplicabilityReference
            return ApplicabilityReference.load(settings["reference_path"]).save(root / "applicability.npz")
        if "mag_bounds" not in settings:
            raise ValueError("Declare applicability.mag_bounds and required photometry explicitly")
        reference = fit_applicability_reference(dataset.observables.iloc[splits.development], **settings)
        return reference.save(root / "applicability.npz")
    raise ValueError(f"Unknown workflow stage: {stage}")
