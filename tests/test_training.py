"""Tiny CPU regressions, never long scientific optimization runs."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from turbopdz.config import validate_config
from turbopdz.synthetic import make_synthetic_dataset
from turbopdz.splits import SplitPlan
from turbopdz.training import train_zml, generate_oof, train_sigmaml
from turbopdz.inference import TurboPDZ
from turbopdz.workflow import export_run


@pytest.mark.training
def test_dataset_remains_shuffled_with_aligned_targets():
    pytest.importorskip("tensorflow")
    from turbopdz.training.common import _batches
    x = np.arange(80, dtype=np.float32)[:, None]
    batches = _batches(x, x[:, 0] * 2, None, 16, seed=50, shuffle=True, prefetch=1)
    epochs = []
    for _ in range(2):
        epoch = []
        for features, targets in batches:
            np.testing.assert_array_equal(features.numpy()[:, 0] * 2, targets.numpy())
            epoch.extend(features.numpy()[:, 0])
        epochs.append(epoch)
    np.testing.assert_array_equal(np.sort(epochs[0]), x[:, 0])
    assert epochs[0] != x[:, 0].tolist() and epochs[0] != epochs[1]


@pytest.mark.training
def test_small_end_to_end_fit_and_export(tmp_path):
    pytest.importorskip("tensorflow")
    data = make_synthetic_dataset(80, 24)
    plan = SplitPlan.create(data, oof_folds=2)
    config = validate_config({"training": {"epochs": 2, "verbose": 0,
        "params": {"n_layers": 1, "n_units_l0": 8, "batch_size": 16, "explained_variance_pdf": 0.8}},
        "features": {"extra_columns": []}, "splits": {"oof_folds": 2},
        "selection_objective": {"scales": {"scatter": 0.2, "outliers": 0.1}},
        "sigmaml": {"epochs": 2, "calibration": {"n_bins": 3}},
        "bootstrap": {"resamples": 8}})
    train_zml(data, plan, config, tmp_path / "zml")
    generate_oof(data, plan, config, tmp_path / "zml", tmp_path / "oof.npz")
    train_sigmaml(data, plan, config, tmp_path / "zml", tmp_path / "oof.npz", tmp_path / "sigmaml")
    fitted = TurboPDZ.load(tmp_path)
    assert fitted.zml_info["references"]["scatter"] == 0.2
    assert fitted.zml_info["references"]["outliers"] == 0.1
    result = fitted.predict(data.pdz[plan.test], data.z_grid, data.observables.iloc[plan.test], object_id=data.object_id[plan.test], batch_size=3)
    assert list(result.columns) == ["object_id", "z_ml", "sigma_ml", "r_ml"]
    assert np.isfinite(result.iloc[:, 1:].to_numpy()).all()
    assert result.sigma_ml.gt(0).all()
    assert result.r_ml.between(0, 1).all()
    export_run(tmp_path, tmp_path / "deployable")
    reloaded = TurboPDZ.load(tmp_path / "deployable")
    compare = reloaded.predict(data.pdz[plan.test], data.z_grid, data.observables.iloc[plan.test], object_id=data.object_id[plan.test], batch_size=3)
    np.testing.assert_allclose(result.iloc[:, 1:], compare.iloc[:, 1:], atol=1e-7)
    # Generate every applicable single-catalog report without altering bundles.
    from turbopdz.artifacts import write_json, file_sha256, load_manifest
    from turbopdz.workflow import evaluate_run, bootstrap_run
    from turbopdz.applicability import fit_applicability_reference
    from turbopdz.analysis import (
        write_data_diagnostics, write_training_diagnostics, write_oof_diagnostics,
        write_evaluation_diagnostics, write_bootstrap_diagnostics,
        write_applicability_diagnostics, write_pca_population_diagnostics,
    )
    data.save(tmp_path / "dataset")
    plan.save(tmp_path / "splits.npz")
    write_json(tmp_path / "config.json", config)
    hashes = {track: file_sha256(tmp_path / track / "manifest.json") for track in ("zml", "sigmaml")}
    evaluate_run(tmp_path)
    bootstrap_run(tmp_path)
    fit_applicability_reference(data.observables.iloc[plan.development],
        n_neighbors=3, mag_bounds=[19, 25]).save(tmp_path / "applicability.npz")
    outputs = [write_data_diagnostics(tmp_path, n_examples=2),
               write_training_diagnostics(tmp_path, "zml"),
               write_training_diagnostics(tmp_path, "sigmaml"),
               write_oof_diagnostics(tmp_path),
               write_evaluation_diagnostics(tmp_path, repeats=1, top_pcs=2, n_examples=2, n_bins=3),
               write_bootstrap_diagnostics(tmp_path),
               write_applicability_diagnostics(tmp_path),
               write_pca_population_diagnostics(tmp_path, magnitude_edges=np.arange(19, 26),
                   min_common=1, min_rate_count=1, branches={"true_low": [0, 0.5], "true_high": [1.25, 1.75],
                   "pdz_low": [0, 0.6], "pdz_high": [1.2, 1.8],
                   "predicted_low_max": 0.6, "predicted_high_min": 1.2})]
    assert sum((output.kind == "plot").sum() for output in outputs) >= 30
    for track in hashes:
        assert file_sha256(tmp_path / track / "manifest.json") == hashes[track]
        load_manifest(tmp_path / track, verify=True)
    for path in (tmp_path / "plots").rglob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF-")
    for path in (tmp_path / "tables").rglob("*.csv"):
        assert len(pd.read_csv(path).columns) > 0
    with pytest.raises(ValueError, match="grid"):
        reloaded.predict(data.pdz[plan.test], data.z_grid + 0.1, data.observables.iloc[plan.test])


@pytest.mark.training
def test_optuna_search_and_compatible_resume(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("optuna")
    data = make_synthetic_dataset(64, 20)
    plan = SplitPlan.create(data, oof_folds=2)
    config = validate_config({"training": {"epochs": 1, "verbose": 0,
        "hpo": {"mode": "search", "trials": 1, "search_space": {
            "n_layers": {"type": "categorical", "choices": [1]},
            "units": {"type": "categorical", "choices": [8]}}}},
        "selection_objective": {"name": "mae"}, "splits": {"oof_folds": 2}})
    train_zml(data, plan, config, tmp_path / "first")
    import shutil
    (tmp_path / "resume").mkdir()
    shutil.copy2(tmp_path / "first" / "study.sqlite3", tmp_path / "resume" / "study.sqlite3")
    resumed = deepcopy(config)
    resumed["training"]["hpo"]["mode"] = "resume"
    resumed["features"]["extra_columns"] = ["i_cmodel_mag"]
    with pytest.raises(ValueError, match="Incompatible"):
        train_zml(data, plan, resumed, tmp_path / "resume")
    resumed["features"]["extra_columns"] = []
    train_zml(data, plan, resumed, tmp_path / "resume")


@pytest.mark.training
def test_sigmaml_search_resume_and_development_only_refit(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    optuna = pytest.importorskip("optuna")
    from turbopdz.training import hpo, sigmaml
    from turbopdz.artifacts import file_sha256, load_manifest

    data = make_synthetic_dataset(64, 20)
    plan = SplitPlan.create(data, oof_folds=2)
    config = validate_config({
        "training": {"epochs": 2, "verbose": 0,
                     "params": {"n_layers": 1, "n_units_l0": 8,
                                "batch_size": 16, "explained_variance_pdf": 0.8}},
        "splits": {"oof_folds": 2},
        "sigmaml": {"epochs": 1, "calibration": {"n_bins": 3},
                    "hpo": {"mode": "search", "trials": 2, "search_space": {
                        "n_layers": {"type": "categorical", "choices": [1]},
                        "units": {"type": "categorical", "choices": [8, 16]}}}},
    })
    zroot, oof = tmp_path / "zml", tmp_path / "oof.npz"
    train_zml(data, plan, config, zroot)
    generate_oof(data, plan, config, zroot, oof)
    original_oof_hash = file_sha256(oof)

    cv_calls, final_calls = [], []
    original_cv_fit, original_final_fit = hpo.fit_rows, sigmaml.fit_rows

    def record_cv(*args, **kwargs):
        cv_calls.append((args[2].copy(), kwargs["valid_idx"].copy(), kwargs["epochs"]))
        assert kwargs["track"] == "sigmaml"
        return original_cv_fit(*args, **kwargs)

    def record_final(*args, **kwargs):
        final_calls.append((args[2].copy(), kwargs["epochs"]))
        return original_final_fit(*args, **kwargs)

    monkeypatch.setattr(hpo, "fit_rows", record_cv)
    monkeypatch.setattr(sigmaml, "fit_rows", record_final)
    first = tmp_path / "sigmaml"
    train_sigmaml(data, plan, config, zroot, oof, first)
    assert len(cv_calls) == 4
    excluded = np.r_[plan.calibration, plan.test]
    for fit, valid, epochs in cv_calls:
        assert epochs == 1
        assert not np.intersect1d(fit, valid).size
        assert not np.intersect1d(np.r_[fit, valid], excluded).size
        np.testing.assert_array_equal(np.sort(np.r_[fit, valid]), np.sort(plan.development))
    np.testing.assert_array_equal(final_calls[0][0], plan.development)
    assert final_calls[0][1] == 1
    info = load_manifest(first)
    assert info["selection"]["metric"] == "mae_log_error"
    assert info["selection"]["n_trials"] == 2
    assert file_sha256(oof) == original_oof_hash
    study = optuna.load_study(study_name="sigmaml", storage=f"sqlite:///{first / 'study.sqlite3'}")
    assert len(study.trials) == 2
    assert study.best_value == pytest.approx(np.mean(study.best_trial.user_attrs["fold_scores"]))
    assert study.best_trial.user_attrs["final_epochs"] == 1

    import shutil
    resumed_root = tmp_path / "resumed"
    resumed_root.mkdir()
    shutil.copy2(first / "study.sqlite3", resumed_root / "study.sqlite3")
    resumed = deepcopy(config)
    resumed["sigmaml"]["hpo"]["mode"] = "resume"
    changed = deepcopy(resumed)
    changed["features"]["sigmaml"] = {"extra_columns": ["i_cmodel_mag"]}
    with pytest.raises(ValueError, match="Incompatible"):
        train_sigmaml(data, plan, changed, zroot, oof, resumed_root)
    changed = deepcopy(resumed)
    changed["sigmaml"]["epsilon_quantile"] = 0.02
    with pytest.raises(ValueError, match="Incompatible"):
        train_sigmaml(data, plan, changed, zroot, oof, resumed_root)
    assert len(cv_calls) == 4
    train_sigmaml(data, plan, resumed, zroot, oof, resumed_root)
    assert len(cv_calls) == 4  # completed budget is reused, not optimized again
    with pytest.raises(FileExistsError, match="output exists"):
        train_sigmaml(data, plan, resumed, zroot, oof, resumed_root)

    extended_root = tmp_path / "extended"
    extended_root.mkdir()
    shutil.copy2(first / "study.sqlite3", extended_root / "study.sqlite3")
    resumed["sigmaml"]["hpo"]["trials"] = 3
    train_sigmaml(data, plan, resumed, zroot, oof, extended_root)
    assert len(cv_calls) == 6  # exactly one additional trial, with two folds
    assert load_manifest(extended_root)["selection"]["n_trials"] == 3
    deployed = tmp_path / "deployable"
    export_run(tmp_path, deployed)
    result = TurboPDZ.load(deployed).predict(
        data.pdz[plan.test], data.z_grid, data.observables.iloc[plan.test],
        object_id=data.object_id[plan.test],
    )
    assert np.isfinite(result.iloc[:, 1:].to_numpy()).all()
    assert not (deployed / "sigmaml/study.sqlite3").exists()
