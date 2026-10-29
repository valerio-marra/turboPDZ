"""OOF residual generation using frozen zML hyperparameters and epochs."""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np

from ..artifacts import load_manifest, file_sha256
from ..features import feature_table
from ..preprocessing import Preprocessor
from .common import fit_rows, predict_model, tensorflow


def generate_oof(dataset, splits, config: dict, zml_directory: str | Path, output: str | Path) -> Path:
    """Exclude each OOF fold from both preprocessing and fixed-epoch model fitting.

    Hyperparameter selection is not nested inside these folds. The scope is
    recorded explicitly, matching the paper's cross-fitting interpretation.
    """
    splits.validate(dataset)
    root, path = Path(zml_directory), Path(output)
    info = load_manifest(root)
    if info["dataset_fingerprint"] != dataset.fingerprint():
        raise ValueError("zML source was fitted on different data or row order")
    source_hash = file_sha256(root / "manifest.json")
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["source_hash"]) != source_hash or not np.array_equal(cache["fold_id"], splits.fold_id):
                raise ValueError("Cached OOF predictions have incompatible provenance")
        return path
    source_config = json.loads((root / "config.json").read_text())
    features = feature_table(dataset, info["features"])
    selected = json.loads((root / "best_params.json").read_text())
    prediction = np.full(len(dataset.pdz), np.nan)
    for fold, (fit, valid) in enumerate(splits.folds()):
        print(f"zML OOF fold {fold + 1}/{splits.n_folds}", flush=True)
        model, pp, _ = fit_rows(dataset, features, fit, selected["best_params"],
                                config=source_config, epochs=selected["final_epochs"],
                                seed=source_config["seed"] + fold)
        prediction[valid] = predict_model(model, pp.transform(dataset.pdz[valid], features.iloc[valid]))
    pp = Preprocessor.load(root / "preprocessor.npz")
    model = tensorflow().keras.models.load_model(root / "model.keras", compile=False)
    holdout = np.r_[splits.calibration, splits.test]
    prediction[holdout] = predict_model(model, pp.transform(dataset.pdz[holdout], features.iloc[holdout]))
    if not np.isfinite(prediction).all():
        raise ValueError("Missing/non-finite OOF or held-out zML predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=prediction, fold_id=splits.fold_id,
                        source_hash=np.asarray(source_hash), object_id=dataset.object_id,
                        scope=np.asarray("weights_and_preprocessing_non_nested"))
    return path
