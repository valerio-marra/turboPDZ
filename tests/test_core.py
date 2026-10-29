"""Regression checks for data contracts, preprocessing, splits, and metrics."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from turbopdz import PDZDataset
from turbopdz.synthetic import make_synthetic_dataset
from turbopdz.features import feature_table
from turbopdz.preprocessing import Preprocessor
from turbopdz.splits import SplitPlan
from turbopdz.config import validate_config, semantic_fingerprint
from turbopdz.objectives import objective_score
from turbopdz.calibration import Calibration
from turbopdz.evaluation.metrics import residual, photoz_metrics, retained_metrics
from turbopdz.evaluation.bootstrap import paired_bootstrap
from turbopdz.artifacts import seal_bundle, load_manifest


def test_import_does_not_load_training_or_plotting_backends():
    output = subprocess.check_output([sys.executable, "-c", "import turbopdz,sys; print(any(x in sys.modules for x in ['tensorflow','optuna','matplotlib.pyplot']))"], text=True)
    assert output.strip() == "False"


def test_dataset_roundtrip_and_alignment(tmp_path):
    data = make_synthetic_dataset(80)
    data.save(tmp_path / "data")
    loaded = PDZDataset.load(tmp_path / "data")
    assert loaded.fingerprint() == data.fingerprint()
    assert np.array_equal(loaded.object_id, data.object_id)
    bad = data.object_id.copy()
    bad[1] = bad[0]
    with pytest.raises(ValueError, match="unique"):
        PDZDataset(data.pdz, data.z_grid, data.observables, bad)


def test_density_on_nonuniform_grid_and_invalid_pdz():
    grid = np.array([0, 1, 3.])
    data = PDZDataset(np.ones((1, 3)), grid, pd.DataFrame(index=[0]), np.array([123]), representation="density")
    np.testing.assert_allclose(data.masses(), [[1/6, 1/2, 1/3]])
    assert feature_table(data, {"descriptor_groups": ["location"]}).iloc[0]["median"] == 1.5
    with pytest.raises(ValueError, match="positive total"):
        PDZDataset(np.zeros((1, 3)), grid, pd.DataFrame(index=[0]), np.array([123]))
    with pytest.raises(ValueError, match="increasing"):
        PDZDataset(np.ones((1, 3)), grid[::-1], pd.DataFrame(index=[0]), np.array([123]))


def test_features_no_i_and_custom_observables():
    data = make_synthetic_dataset(80)
    only_pdz = feature_table(data, {"extra_columns": []})
    with_i = feature_table(data, {"extra_columns": ["i_cmodel_mag"]})
    assert "i_cmodel_mag" not in only_pdz
    assert with_i.shape[1] == only_pdz.shape[1] + 1
    data.observables["g-r"] = data.observables.g_cmodel_mag - data.observables.r_cmodel_mag
    assert "g-r" in feature_table(data, {"extra_columns": ["g-r"]})


def test_preprocessing_roundtrip_schema_and_missing(tmp_path):
    data = make_synthetic_dataset(80)
    features = feature_table(data, {"extra_columns": ["i_cmodel_mag"]})
    pp = Preprocessor.fit(data.pdz[:60], features.iloc[:60], variance=0.9)
    pp.save(tmp_path / "pp.npz")
    loaded = Preprocessor.load(tmp_path / "pp.npz")
    np.testing.assert_allclose(loaded.transform(data.pdz, features), pp.transform(data.pdz, features), atol=1e-6)
    with pytest.raises(ValueError, match="schema"):
        loaded.transform(data.pdz, features[features.columns[::-1]])
    missing = features.copy()
    missing.loc[0, "i_cmodel_mag"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        loaded.transform(data.pdz, missing)
    imputed = Preprocessor.fit(data.pdz[:60], missing.iloc[:60], missing="median")
    assert np.isfinite(imputed.transform(data.pdz, missing)).all()


def test_float32_scaler_underflow_guard():
    data = make_synthetic_dataset(80)
    pdf = data.pdz.copy()
    pdf[:, -1] = 0
    pdf[0, -1] = np.nextafter(np.float32(0), np.float32(1))
    features = feature_table(data, {})
    pp = Preprocessor.fit(pdf, features)
    assert np.isfinite(pp.transform(pdf, features)).all()
    assert pp.sc_pdf.scale_[-1] == 1


def test_split_disjointness_and_reuse(tmp_path):
    data = make_synthetic_dataset(120)
    plan = SplitPlan.create(data, oof_folds=3)
    for fit, valid in plan.folds():
        assert not np.intersect1d(fit, valid).size
        assert not np.intersect1d(fit, np.r_[plan.calibration, plan.test]).size
    plan.save(tmp_path / "splits.npz")
    assert np.array_equal(SplitPlan.load(tmp_path / "splits.npz", data).fold_id, plan.fold_id)
    changed = data.subset(np.arange(120)[::-1])
    with pytest.raises(ValueError, match="changed"):
        SplitPlan.load(tmp_path / "splits.npz", changed)


def test_bias_is_optional_and_has_explicit_scale():
    truth = np.array([0.2, 0.5, 1., 2.])
    pred = truth + 0.05
    base = {"name": "normalized_composite", "weights": {"scatter": 0.5, "outliers": 0.5}}
    refs = {"scatter": 0.03, "outliers": 0.1}
    original = objective_score(pred, truth, base, refs)
    with_bias = {"name": "normalized_composite", "weights": {"scatter": 0.5, "outliers": 0.5, "absolute_bias": 0.2}, "bias_tolerance": 0.01}
    assert objective_score(pred, truth, with_bias, refs) > original
    with_bias.pop("bias_tolerance")
    with pytest.raises(ValueError, match="positive normalization"):
        objective_score(pred, truth, with_bias, refs)


def test_fingerprints_change_with_scientific_choices():
    config = validate_config({})
    baseline = semantic_fingerprint(config, "dataset", "split")
    config["features"]["extra_columns"] = ["i_cmodel_mag"]
    assert semantic_fingerprint(config, "dataset", "split") != baseline
    with pytest.raises(ValueError, match="Boolean"):
        validate_config({"features": {"pdz_pca": "False"}})


def test_paired_bootstrap_identical_models_and_shared_draws():
    x = np.linspace(0, 1, 60)
    results = paired_bootstrap({"x": x}, {"same": lambda d: (d["x"].mean(), d["x"].mean()),
                                          "double": lambda d: (2*d["x"].mean(), 2*d["x"].mean())}, n_boot=40)
    assert (results.draws["same"][:, 2] == 0).all()
    np.testing.assert_allclose(results.draws["double"][:, 0], 2*results.draws["same"][:, 0])
    assert results.summary.iloc[0].ci_low == results.summary.iloc[0].ci_high == 0
    assert results.summary.iloc[0].status == "unresolved"


def test_calibration_and_ranking_roundtrip(tmp_path):
    x = np.linspace(0.01, 0.3, 80)
    fitted = Calibration.fit(x, 2*x, epsilon=1e-5, n_bins=8)
    np.testing.assert_allclose(fitted.apply(x), 2*x)
    assert (np.diff(fitted.reliability(fitted.apply(x))) <= 0).all()
    fitted.save(tmp_path / "calibration.npz")
    np.testing.assert_array_equal(Calibration.load(tmp_path / "calibration.npz").reliability(fitted.apply(x)), fitted.reliability(fitted.apply(x)))


def test_empirical_cdf_keeps_ties_and_exposes_discontinuous_ranks():
    reference = np.array([0.01, 0.02, 0.02, 0.02, 0.03])
    calibration = Calibration(np.array([0.01, 0.03]), np.ones(2), reference, 1e-5)
    query = np.array([np.nextafter(0.02, 0), 0.02])
    np.testing.assert_allclose(calibration.reliability(query), [0.8, 0.2])


def test_resume_and_bootstrap_changes_do_not_change_model_inputs():
    from turbopdz.config import run_configs_compatible
    previous = validate_config({"training": {"hpo": {"mode": "search"}}})
    current = validate_config({"training": {"hpo": {"mode": "resume", "trials": 200}},
                               "bootstrap": {"resamples": 1000}})
    assert run_configs_compatible(previous, current)
    current["features"]["extra_columns"] = ["i_cmodel_mag"]
    assert not run_configs_compatible(previous, current)


def test_relocated_evaluation_keeps_scientific_inputs_and_normalizes_defaults():
    from copy import deepcopy
    from turbopdz.config import run_configs_compatible
    previous = validate_config({"dataset": {"adapter": "portable", "path": "/old/data"},
                                "training": {"params_path": "/old/params.json"}})
    current = deepcopy(previous)
    previous.pop("python_paths")
    assert run_configs_compatible(previous, current)
    current["dataset"]["path"] = "/new/data"
    current["training"]["params_path"] = "/new/params.json"
    assert not run_configs_compatible(previous, current)
    assert run_configs_compatible(previous, current, allow_path_changes=True)
    current["selection_objective"]["weights"]["absolute_bias"] = 0.2
    assert not run_configs_compatible(previous, current, allow_path_changes=True)


def test_sigmaml_fingerprint_checks_targets_parent_and_independent_training(tmp_path):
    from copy import deepcopy
    config = validate_config({"sigmaml": {"hpo": {"mode": "search", "trials": 2}}})
    def fingerprint(value, parent="parent", oof="oof"):
        return semantic_fingerprint(value, "data", "split", track="sigmaml",
                                    parent_zml_hash=parent, oof_hash=oof)
    original = fingerprint(config)
    resumed = deepcopy(config)
    resumed["sigmaml"]["hpo"].update(mode="resume", trials=20)
    assert fingerprint(resumed) == original
    assert fingerprint(config, parent="different") != original
    assert fingerprint(config, oof="different") != original
    for changes in ({"epsilon_floor": 0.001}, {"epochs": 5}, {"loss": "mse"},
                    {"params": {"n_layers": 3}}, {"sample_weight_column": "weight"}):
        changed = deepcopy(config)
        changed["sigmaml"].update(changes)
        assert fingerprint(changed) != original
    source = tmp_path / "params.json"
    source.write_text(json.dumps({"best_params": {"n_layers": 1}, "final_epochs": 3}))
    config["sigmaml"]["hpo"]["params_path"] = str(source)
    sourced = fingerprint(config)
    source.write_text(json.dumps({"best_params": {"n_layers": 2}, "final_epochs": 3}))
    assert fingerprint(config) != sourced


def test_sigmaml_resume_configuration_and_training_overrides():
    from turbopdz.config import run_configs_compatible
    from turbopdz.training.common import training_config
    previous = validate_config({"training": {"hpo": {"mode": "search"},
                                            "params_path": "/point-only/params.json"},
                                "sigmaml": {"epochs": 7, "patience": 3,
                                            "params": {"n_units_l0": 24},
                                            "hpo": {"mode": "search", "trials": 2}}})
    settings = training_config(previous, "sigmaml")
    assert "params_path" not in settings
    assert settings["epochs"] == 7 and settings["patience"] == 3
    assert settings["params"]["n_layers"] == 2
    assert settings["params"]["n_units_l0"] == 24
    current = validate_config(previous)
    current["sigmaml"]["hpo"].update(mode="resume", trials=20)
    assert run_configs_compatible(previous, current)
    current["sigmaml"]["loss"] = "mse"
    assert not run_configs_compatible(previous, current)


@pytest.mark.parametrize("settings", [{"mode": "unsupported"},
                                      {"mode": "search", "trials": 0},
                                      {"mode": "search", "trials": 1.5}])
def test_invalid_sigmaml_search_configuration(settings):
    with pytest.raises(ValueError, match="sigmaml.hpo"):
        validate_config({"sigmaml": {"hpo": settings}})


def test_common_residual_curves_coincide_at_full_retention():
    dz = np.linspace(-0.3, 0.2, 50)
    first = retained_metrics(dz, np.arange(50), higher_is_better=True)
    second = retained_metrics(dz, np.arange(50)[::-1], higher_is_better=True)
    assert first.iloc[-1].scatter == second.iloc[-1].scatter
    assert first.iloc[-1].outliers == second.iloc[-1].outliers


def test_artifact_hashes_reject_changed_content(tmp_path):
    (tmp_path / "data.txt").write_text("original")
    seal_bundle(tmp_path, {"track": "test"})
    assert load_manifest(tmp_path)["track"] == "test"
    (tmp_path / "data.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(tmp_path)


def test_grouped_importance_follows_saved_feature_names():
    from turbopdz.evaluation.importance import input_feature_groups, grouped_permutation_importance
    groups = input_feature_groups(("mean", "i_cmodel_mag"), 2)
    assert groups == {"location": [0], "i_cmodel_mag": [1], "PDZ-PCA": [2, 3]}
    x = np.arange(80, dtype=float).reshape(20, 4)
    target = x[:, 0]
    result = grouped_permutation_importance(lambda values: values[:, 0], x, target,
                                            groups, lambda pred, truth: np.mean(abs(pred-truth)), repeats=3)
    assert result.iloc[0].group == "location"


def test_bootstrap_error_correlations_without_catalog_selectors(tmp_path):
    from turbopdz.workflow import bootstrap_run
    truth = np.linspace(0.1, 2, 50)
    candidate = truth + np.linspace(-0.08, 0.13, 50)
    reference = truth + np.linspace(0.12, -0.1, 50)
    point = pd.DataFrame({"z_ml": candidate, "z_best": reference, "z_spec": truth})
    point.to_csv(tmp_path / "point_holdout.csv", index=False)
    final = pd.DataFrame({"dz_ml": residual(candidate, truth),
                          "dz_best": residual(reference, truth),
                          "sigma_ml": np.linspace(0.01, 0.1, 50),
                          "s_best": np.linspace(0.02, 0.12, 50),
                          "r_ml": np.linspace(1, 0, 50)})
    final.to_csv(tmp_path / "final_test.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps(validate_config({"bootstrap": {"resamples": 10}})))
    summary = bootstrap_run(tmp_path)
    assert len(summary[summary.family == "point_estimate"]) == 4
    reliability = summary[summary.family == "reliability"]
    assert len(reliability) == 6
    assert reliability.metric.str.contains("s_best").all()


def test_paired_run_comparison_aligns_ids_and_rejects_changed_cohorts():
    from turbopdz.evaluation.reporting import compare_point_runs
    truth = np.linspace(0.1, 2, 30)
    frame = pd.DataFrame({"object_id": [f"obj-{i:03}" for i in range(30)],
                          "z_spec": truth, "z_ml": truth + 0.03})
    result = compare_point_runs(frame, frame.iloc[::-1], n_boot=10)
    assert (result.summary.improvement == 0).all()
    with pytest.raises(ValueError, match="same object cohort"):
        compare_point_runs(frame, frame.iloc[:-1], n_boot=10)
    changed = frame.copy()
    changed.loc[0, "z_spec"] += 0.01
    with pytest.raises(ValueError, match="different spectroscopic"):
        compare_point_runs(frame, changed, n_boot=10)


def test_bootstrap_reports_write_csv_markdown_and_latex(tmp_path):
    from turbopdz.evaluation.reporting import write_tables
    x = np.linspace(0, 1, 30)
    result = paired_bootstrap({"x": x}, {"mean": lambda d: (d["x"].mean(), d["x"].mean())}, n_boot=10)
    write_tables(result.summary, tmp_path)
    assert len(pd.read_csv(tmp_path / "bootstrap_summary.csv")) == 1
    assert "improvement [CI]" in (tmp_path / "bootstrap.md").read_text()
    assert "\\begin{tabular}" in (tmp_path / "bootstrap.tex").read_text()
