"""One persistent, non-nested Optuna search per model track."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from ..data import PDZDataset
from ..splits import SplitPlan
from .common import fit_rows, predict_model, tensorflow
from ..runtime import import_optuna


DEFAULT_SEARCH_SPACE = {
    "n_layers": {"type": "int", "low": 2, "high": 5},
    "units": {"type": "int", "low": 64, "high": 512, "step": 64},
    "dropout_rate": {"type": "float", "low": 0.05, "high": 0.3},
    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.002, "log": True},
    "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
    "activation": {"type": "categorical", "choices": ["relu", "swish"]},
    "batch_size": {"type": "categorical", "choices": [128, 256, 512]},
    "explained_variance_pdf": {"type": "float", "low": 0.8, "high": 0.995},
}


def suggest_parameters(trial, base_params: dict, search_space: dict) -> dict:
    """Sample only declared parameters, expanding shared units per hidden layer."""
    def suggest(name, spec):
        kind = spec["type"]
        if kind not in {"int", "float", "categorical"}:
            raise ValueError(f"Unsupported search-space type: {kind}")
        kwargs = {key: value for key, value in spec.items() if key != "type"}
        return getattr(trial, f"suggest_{kind}")(name, **kwargs)

    candidate = dict(base_params)
    for name, spec in search_space.items():
        if name != "units":
            candidate[name] = suggest(name, spec)
    if "units" in search_space:
        for layer in range(int(candidate["n_layers"])):
            name = f"n_units_l{layer}"
            candidate[name] = suggest(name, search_space["units"])
    return candidate


def optimize_parameters(
    dataset: PDZDataset,
    features: pd.DataFrame,
    splits: SplitPlan,
    config: dict,
    training: dict,
    directory: str | Path,
    *,
    track: str,
    params: dict,
    targets: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
    fingerprint: str,
    metric_name: str,
) -> tuple[dict, int, dict]:
    """Select parameters/epochs by fixed development-only CV and save the study.

    Parameters
    ----------
    targets
        Row-aligned regression targets. Only development indices enter fitting
        or scoring; sigmaML uses frozen log-OOF-error targets.
    score
        Uncalibrated held-fold selection metric to minimize.
    fingerprint
        Compatibility key covering scientific inputs and implementation.

    Notes
    -----
    ``trials`` is a total attempted-trial budget, not an increment per resume.
    Calibration/CDF fitting and the final full-development refit are separate.
    """
    try:
        optuna = import_optuna()
    except ImportError as exc:
        raise ImportError("Install turbopdz[train] for Optuna optimization") from exc

    root = Path(directory)
    hpo = training["hpo"]
    storage = f"sqlite:///{(root / 'study.sqlite3').resolve()}"
    name = hpo.get("study_name", track)
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    if hpo["mode"] == "resume":
        if not (root / "study.sqlite3").is_file():
            raise FileNotFoundError("resume requires an existing study.sqlite3")
        study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)
        if study.user_attrs.get("semantic_fingerprint") != fingerprint:
            raise ValueError("Incompatible Optuna study: data, features, targets, parent, splits, or settings changed")
        # Keep the saved history, without replaying the initial random sequence.
        # This is reproducible for the same resumed history, not a serialized RNG.
        study.sampler = optuna.samplers.TPESampler(seed=config["seed"] + len(study.trials))
    else:
        study = optuna.create_study(study_name=name, storage=storage,
                                    direction="minimize", sampler=sampler)
        study.set_user_attr("semantic_fingerprint", fingerprint)
        study.set_user_attr("track", track)
        study.set_user_attr("selection_metric", metric_name)
        study.set_user_attr("crossfit_scope", "weights_and_preprocessing_non_nested")

    space = hpo.get("search_space", DEFAULT_SEARCH_SPACE)

    def objective(trial):
        candidate = suggest_parameters(trial, params, space)
        scores, best_epochs = [], []
        for fold, (fit, valid) in enumerate(splits.folds()):
            model = pp = None
            try:
                model, pp, best_epoch = fit_rows(
                    dataset, features, fit, candidate, config=config, track=track,
                    targets=targets, valid_idx=valid, epochs=training["epochs"],
                    seed=config["seed"] + fold,
                )
                predictions = predict_model(model, pp.transform(dataset.pdz[valid], features.iloc[valid]))
                value = float(score(predictions, targets[valid]))
                if not np.isfinite(value):
                    raise optuna.TrialPruned(f"Non-finite selection metric on fold {fold}")
                scores.append(value)
                best_epochs.append(int(best_epoch))
                history = getattr(model, "_turbopdz_history", None)
                if history is not None:
                    trial.set_user_attr(f"history_fold_{fold}", history.to_dict("records"))
                trial.report(float(np.mean(scores)), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except tensorflow().errors.ResourceExhaustedError as exc:
                raise optuna.TrialPruned(f"Out of memory on fold {fold}") from exc
            finally:
                del model, pp
                tensorflow().keras.backend.clear_session()
                gc.collect()
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("fold_best_epochs", best_epochs)
        trial.set_user_attr("final_epochs", max(1, int(np.median(best_epochs))))
        return float(np.mean(scores))

    study.optimize(objective, n_trials=max(0, int(hpo["trials"]) - len(study.trials)),
                   gc_after_trial=True)
    study.trials_dataframe().to_csv(root / "trials.csv", index=False)
    histories = [dict(row, trial=trial.number, fold=fold)
                 for trial in study.trials
                 for fold in range(splits.n_folds)
                 for row in trial.user_attrs.get(f"history_fold_{fold}", [])]
    if histories:
        pd.DataFrame(histories).to_csv(root / "cv_learning_curves.csv", index=False)
    if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        raise RuntimeError("No completed Optuna trials; inspect the study and increase the resume budget")
    best = study.best_trial
    selected = dict(params, **best.params)
    metadata = {"mode": hpo["mode"], "study_name": name, "metric": metric_name,
                "best_trial": best.number, "best_value": float(best.value),
                "n_trials": len(study.trials), "fold_scores": best.user_attrs["fold_scores"],
                "fold_best_epochs": best.user_attrs["fold_best_epochs"]}
    return selected, int(best.user_attrs["final_epochs"]), metadata
