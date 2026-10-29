"""Validated experiment configurations and content-based compatibility keys."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import inspect
import sys

import yaml

DEFAULTS = {
    "schema_version": 1,
    "python_paths": [],
    "seed": 50,
    "features": {"descriptor_groups": ["location", "shape", "peaks"],
                 "extra_columns": [], "pdz_pca": True, "missing": "raise"},
    "selection_objective": {"name": "normalized_composite",
                            "weights": {"scatter": 0.5, "outliers": 0.5, "absolute_bias": 0.0}},
    "splits": {"development": 0.85, "calibration": 0.075, "test": 0.075, "oof_folds": 5},
    "training": {"epochs": 150, "loss": "mae", "verbose": 1, "patience": 18,
                 "reduce_lr_patience": 6, "reduce_lr_factor": 0.5, "prefetch": 1,
                 "params": {"n_layers": 2, "n_units_l0": 64, "n_units_l1": 64,
                            "dropout_rate": 0.1, "learning_rate": 0.001, "weight_decay": 0.00001,
                            "activation": "relu", "batch_size": 128, "explained_variance_pdf": 0.99},
                 "hpo": {"mode": "fixed", "trials": 50}},
    "sigmaml": {"epochs": 100, "loss": "mae", "hpo": {"mode": "fixed", "trials": 50},
                "epsilon_quantile": 0.01, "epsilon_floor": 1e-6,
                "calibration": {"n_bins": 30, "smooth_bw_dex": 0.2}},
    "bootstrap": {"resamples": 2000, "confidence_level": 0.95},
}


def _merge(base: dict, override: Mapping) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, Mapping) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def validate_config(config: Mapping) -> dict:
    """Merge documented defaults and reject invalid high-level configurations."""
    if set(config) - set(DEFAULTS) - {"dataset", "name", "applicability", "reproduction"}:
        raise ValueError(f"Unknown top-level settings: {set(config) - set(DEFAULTS)}")
    result = _merge(DEFAULTS, config)
    if result["schema_version"] != 1 or isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
        raise ValueError("Require schema_version=1 and an integer seed")
    for name in ("training", "sigmaml"):
        hpo = result[name]["hpo"]
        if hpo["mode"] not in {"fixed", "search", "resume", "reuse_parameters"}:
            raise ValueError(f"{name}.hpo.mode must be fixed, search, resume, or reuse_parameters")
        if isinstance(hpo["trials"], bool) or not isinstance(hpo["trials"], int) or hpo["trials"] < 0:
            raise ValueError(f"{name}.hpo.trials must be a non-negative integer")
        if hpo["mode"] == "search" and hpo["trials"] < 1:
            raise ValueError(f"{name}.hpo.trials must be positive for a new search")
        if int(result[name]["epochs"]) < 1:
            raise ValueError("Training epochs must be positive")
    if result["splits"]["oof_folds"] < 2:
        raise ValueError("OOF requires at least two folds")
    fractions = [result["splits"][key] for key in ("development", "calibration", "test")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to one")
    if result["bootstrap"]["resamples"] < 2 or not 0 < result["bootstrap"]["confidence_level"] < 1:
        raise ValueError("Invalid bootstrap settings")
    for track in (None, "zml", "sigmaml"):
        features = result["features"] if track is None else result["features"].get(track, {})
        if "pdz_pca" in features and not isinstance(features["pdz_pca"], bool):
            raise ValueError("pdz_pca must be a Boolean, not a string")
    return result


def load_config(path: str | Path) -> dict:
    """Load YAML; dataset/source paths are relative to the configuration file."""
    file = Path(path).resolve()
    with file.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    result = validate_config(raw)
    def resolve(values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                resolve(value)
            elif key in {"path", "source_run", "params_path", "split_path", "reference_path"} and isinstance(value, str):
                values[key] = str((file.parent / value).resolve())
    resolve(result)
    result["python_paths"] = [str((file.parent / value).resolve()) for value in result["python_paths"]]
    for value in reversed(result["python_paths"]):
        if value not in sys.path:
            sys.path.insert(0, value)
    return result


def semantic_fingerprint(config: Mapping, dataset_fingerprint: str, split_digest: str,
                         *, track: str = "zml", parent_zml_hash: str | None = None,
                         oof_hash: str | None = None) -> str:
    """Hash scientific settings, excluding trial budgets and machine paths."""
    selected = {key: deepcopy(config[key]) for key in ("seed", "features", "selection_objective", "splits")}
    selected["training"] = {key: deepcopy(value) for key, value in config["training"].items() if key not in {"verbose", "hpo", "prefetch"}}
    training = config["training"]
    if track == "sigmaml":
        if parent_zml_hash is None or oof_hash is None:
            raise ValueError("sigmaML fingerprint requires parent and OOF provenance")
        from .training.common import training_config, feature_config
        training = training_config(config, track)
        selected["features"] = feature_config(config, track)
        selected["training"] = {key: deepcopy(value) for key, value in training.items() if key not in {"verbose", "hpo", "prefetch"}}
        selected["selection_objective"] = {"name": "mae_log_error"}
        selected["sigmaml"] = {key: deepcopy(config["sigmaml"][key]) for key in ("epsilon_quantile", "epsilon_floor", "calibration")}
        selected.update(parent_zml_hash=parent_zml_hash, oof_hash=oof_hash, track=track)
    elif track != "zml":
        raise ValueError("Model track must be zml or sigmaml")
    selected["search_space"] = deepcopy(training["hpo"].get("search_space", {}))
    selected["algorithm_version"] = 1
    selected["dataset_semantics"] = {key: config.get("dataset", {}).get(key) for key in ("reference_column", "z_max")}
    from .artifacts import file_sha256, software_versions
    package = Path(__file__).parent
    selected["implementation"] = {name: file_sha256(package / name) for name in (
        "data.py", "features.py", "preprocessing.py", "splits.py", "objectives.py",
        "evaluation/metrics.py", "training/common.py", "training/hpo.py", "training/zml.py")}
    if track == "sigmaml":
        selected["implementation"]["training/sigmaml.py"] = file_sha256(package / "training/sigmaml.py")
    selected["software"] = software_versions()
    source_path = selected["training"].pop("params_path", None) or training["hpo"].get("params_path")
    if source_path:
        selected["training"]["parameter_source"] = json.loads(Path(source_path).read_text())
    hooks = []
    def find_hooks(settings):
        for key, value in settings.items():
            if isinstance(value, dict):
                find_hooks(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        find_hooks(item)
            elif key == "callable":
                from .objectives import resolve_callable
                function = resolve_callable(value)
                hooks.append({"name": value, "source": inspect.getsource(function)})
    find_hooks(selected)
    selected["hook_sources"] = hooks
    selected.update(dataset_fingerprint=dataset_fingerprint, split_digest=split_digest)
    return hashlib.sha256(json.dumps(selected, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_configs_compatible(previous: Mapping, current: Mapping, *, allow_path_changes: bool = False) -> bool:
    """Allow resume budgets/report settings, but not changed scientific inputs."""
    old, new = validate_config(previous), validate_config(current)
    for track in ("training", "sigmaml"):
        old_mode = old[track]["hpo"]["mode"]
        new_mode = new[track]["hpo"]["mode"]
        if old_mode != new_mode and not (old_mode in {"search", "resume"} and new_mode == "resume"):
            return False
    for value in (old, new):
        value.pop("bootstrap", None)
        for track in ("training", "sigmaml"):
            for key in ("verbose", "prefetch"):
                value[track].pop(key, None)
            for key in ("mode", "trials"):
                value[track]["hpo"].pop(key, None)
        if allow_path_changes:
            value.pop("python_paths", None)
            def strip_paths(settings):
                for key in list(settings):
                    if key in {"path", "source_run", "params_path", "split_path", "reference_path"}:
                        settings.pop(key)
                    elif isinstance(settings[key], dict):
                        strip_paths(settings[key])
            strip_paths(value)
    return old == new
