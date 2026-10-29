"""Log-error training on OOF targets with held-out scale calibration."""

from __future__ import annotations
from pathlib import Path
import hashlib

import numpy as np

from ..artifacts import load_manifest, file_sha256, seal_bundle, write_json
from ..calibration import Calibration, log_to_scale
from ..features import feature_table
from ..evaluation.metrics import residual
from ..config import validate_config, semantic_fingerprint
from .common import fit_rows, predict_model, selected_parameters, feature_config, training_config
from .hpo import DEFAULT_SEARCH_SPACE, optimize_parameters


def train_sigmaml(dataset, splits, config: dict, zml_directory: str | Path,
                  oof_path: str | Path, directory: str | Path) -> Path:
    """Fit sigmaML on development only and calibration/CDF on calibration only."""
    config = validate_config(config)
    splits.validate(dataset)
    zroot, root = Path(zml_directory), Path(directory)
    training = training_config(config, "sigmaml")
    mode = training["hpo"]["mode"]
    if (root / "manifest.json").exists() or (root.exists() and mode != "resume"):
        raise FileExistsError("sigmaML output exists; choose a new run directory or resume an unfinished search")
    info = load_manifest(zroot)
    if info["dataset_fingerprint"] != dataset.fingerprint():
        raise ValueError("Different dataset from parent zML")
    with np.load(oof_path, allow_pickle=False) as arrays:
        if str(arrays["source_hash"]) != file_sha256(zroot / "manifest.json"):
            raise ValueError("OOF predictions do not belong to the selected zML model")
        if not np.array_equal(arrays["object_id"], dataset.object_id) or not np.array_equal(arrays["fold_id"], splits.fold_id):
            raise ValueError("OOF object order or fold assignments differ")
        zml_prediction = arrays["prediction"].copy()
    abs_dz = np.abs(residual(zml_prediction, dataset.z_spec))
    sigma = config["sigmaml"]
    epsilon = max(float(sigma["epsilon_floor"]), float(np.quantile(abs_dz[splits.development], sigma["epsilon_quantile"])))
    targets = np.log(abs_dz + epsilon)
    features = feature_table(dataset, feature_config(config, "sigmaml"))
    params, epochs = selected_parameters(training)
    split_digest = hashlib.sha256(splits.fold_id.tobytes() + splits.development.tobytes()).hexdigest()
    fingerprint = semantic_fingerprint(
        config, dataset.fingerprint(), split_digest, track="sigmaml",
        parent_zml_hash=file_sha256(zroot / "manifest.json"), oof_hash=file_sha256(oof_path),
    )
    selection = {"mode": mode, "metric": "mae_log_error"}
    if mode in {"search", "resume"}:
        if mode == "resume" and not (root / "study.sqlite3").is_file():
            raise FileNotFoundError("resume requires an existing sigmaML study.sqlite3")
        root.mkdir(parents=True, exist_ok=True)
        params, epochs, selection = optimize_parameters(
            dataset, features, splits, config, training, root,
            track="sigmaml", params=params, targets=targets,
            score=lambda pred, truth: float(np.mean(np.abs(pred - truth))),
            fingerprint=fingerprint, metric_name="mae_log_error",
        )
    model, pp, _ = fit_rows(dataset, features, splits.development, params, config=config,
                            track="sigmaml", targets=targets, epochs=epochs)
    idx = splits.calibration
    raw = log_to_scale(predict_model(model, pp.transform(dataset.pdz[idx], features.iloc[idx])), epsilon)
    calibration = Calibration.fit(raw, abs_dz[idx], epsilon=epsilon, **sigma["calibration"])
    root.mkdir(parents=True, exist_ok=mode in {"search", "resume"})
    model.save(root / "model.keras")
    if hasattr(model, "_turbopdz_history"):
        model._turbopdz_history.to_csv(root / "training_history.csv", index=False)
    pp.save(root / "preprocessor.npz")
    calibration.save(root / "calibration.npz")
    np.savez_compressed(root / "grid.npz", z_grid=dataset.z_grid)
    write_json(root / "best_params.json", {"best_params": params, "final_epochs": epochs})
    write_json(root / "config.json", config)
    seal_bundle(root, {"track": "sigmaml", "dataset_fingerprint": dataset.fingerprint(),
                       "parent_zml_manifest_hash": file_sha256(zroot / "manifest.json"),
                       "representation": dataset.representation, "features": feature_config(config, "sigmaml"),
                       "epsilon": epsilon, "seed": config["seed"],
                       "semantic_fingerprint": fingerprint, "selection": selection,
                       "crossfit_scope": "weights_and_preprocessing_non_nested",
                       "calibration_method": "fixed equal-count bins with log-scale multiplicative smoothing"})
    return root
