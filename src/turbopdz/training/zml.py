"""Point-estimate training and compatible Optuna search/resumption."""

from __future__ import annotations
from pathlib import Path
import hashlib

import numpy as np

from ..artifacts import write_json, seal_bundle
from ..config import validate_config, semantic_fingerprint
from ..features import feature_table
from ..objectives import objective_score
from ..evaluation.metrics import photoz_metrics
from .common import feature_config, selected_parameters, fit_rows, predict_model
from .hpo import DEFAULT_SEARCH_SPACE, optimize_parameters


def train_zml(dataset, splits, config: dict, directory: str | Path) -> Path:
    """Train on development rows only, then save an independently loadable stage."""
    config = validate_config(config)
    splits.validate(dataset)
    root = Path(directory)
    if (root / "manifest.json").exists():
        raise FileExistsError("Completed zML run exists; choose a new run directory")
    root.mkdir(parents=True, exist_ok=True)
    features = feature_table(dataset, feature_config(config, "zml"))
    params, epochs = selected_parameters(config["training"])
    references = {}
    reference_column = config.get("dataset", {}).get("reference_column", "photoz_best")
    if reference_column in dataset.observables:
        baseline = photoz_metrics(dataset.observables[reference_column].to_numpy()[splits.development],
                                 dataset.z_spec[splits.development],
                                 config["selection_objective"].get("outlier_threshold", 0.15))
        references.update({key: max(value, 1e-6) for key, value in baseline.items()})
    references.update(config["selection_objective"].get("scales", {}))
    split_digest = hashlib.sha256(splits.fold_id.tobytes() + splits.development.tobytes()).hexdigest()
    fingerprint = semantic_fingerprint(config, dataset.fingerprint(), split_digest)
    hpo = config["training"]["hpo"]
    if hpo["mode"] in {"search", "resume"}:
        params, epochs, _ = optimize_parameters(
            dataset, features, splits, config, config["training"], root,
            track="zml", params=params, targets=dataset.z_spec,
            score=lambda pred, truth: objective_score(pred, truth, config["selection_objective"], references),
            fingerprint=fingerprint, metric_name=config["selection_objective"]["name"],
        )
    model, pp, _ = fit_rows(dataset, features, splits.development, params,
                            config=config, epochs=epochs)
    model.save(root / "model.keras")
    if hasattr(model, "_turbopdz_history"):
        model._turbopdz_history.to_csv(root / "training_history.csv", index=False)
    pp.save(root / "preprocessor.npz")
    np.savez_compressed(root / "grid.npz", z_grid=dataset.z_grid)
    write_json(root / "best_params.json", {"best_params": params, "final_epochs": epochs})
    write_json(root / "config.json", config)
    seal_bundle(root, {"track": "zml", "dataset_fingerprint": dataset.fingerprint(),
                       "semantic_fingerprint": fingerprint, "representation": dataset.representation,
                       "features": feature_config(config, "zml"), "references": references,
                       "seed": config["seed"], "crossfit_scope": "weights_and_preprocessing_non_nested"})
    return root
