"""Numerical, identity, masking and rendering contracts for single-catalog reports."""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from turbopdz.evaluation.diagnostics import (
    align_predictions, point_summary, point_metrics_by_feature, error_scale_table,
    calibration_profile, permutation_metrics, pc_loading_table, numeric_observable,
)
from turbopdz.evaluation.metrics import residual
from turbopdz.preprocessing import Preprocessor
from turbopdz.synthetic import make_synthetic_dataset


@pytest.fixture
def frame():
    result = pd.DataFrame({
        "object_id": ["001", "002", "003", "004"],
        "z_spec": [0.2, 0.4, 1, 1.5], "z_ml": [0.25, 0.45, 0.9, 1.65],
        "z_best": [0.1, 0.6, 0.95, 1.5], "i_cmodel_mag": [20, 21, 22, 23],
        "sigma_ml": [0.1, 0.2, 0.1, 0.2], "s_best": [0.2, 0.2, np.nan, 0.2],
    })
    for name, estimate in (("dz_ml", "z_ml"), ("dz_best", "z_best")):
        result[name] = residual(result[estimate].to_numpy(), result.z_spec.to_numpy())
    return result


def test_alignment_uses_ids_and_null_flags(frame):
    obs = pd.DataFrame({"magnitude": [9e30, 20, 21, 22],
                        "magnitude_isnull": [True, False, False, False]})
    ids = np.array(["004", "003", "002", "001"])
    result = align_predictions(frame.iloc[::-1], obs, ids)
    assert result.object_id.tolist() == ["004", "003", "002", "001"]
    assert np.isnan(result.magnitude.iloc[0])
    assert result.magnitude.iloc[1] == 20
    numeric_ids = align_predictions(frame, obs.assign(object_id=[4, 3, 2, 1]), ids)
    assert numeric_ids.object_id.tolist() == frame.object_id.tolist()
    with pytest.raises(ValueError, match="unique"):
        align_predictions(frame, obs, np.array(["001"] * 4))
    with pytest.raises(ValueError, match="not present"):
        align_predictions(frame, obs, np.array(["x", "y", "z", "w"]))


def test_summary_uses_median_bias_and_shared_mask(frame):
    frame.loc[0, "z_best"] = np.nan
    table = point_summary(frame, cohort="held")
    assert (table.n_objects == 3).all()
    row = table[(table.subset == "all") & (table.estimator == "z_ml")].iloc[0]
    error = residual(frame.z_ml.to_numpy()[1:], frame.z_spec.to_numpy()[1:])
    assert row.bias == pytest.approx(np.median(error))
    assert row.bias != pytest.approx(np.mean(error))


def test_bins_preserve_counts_empty_bins_and_inclusive_final_edge(frame):
    frame["i_cmodel_mag_isnull"] = [False, True, False, False]
    table = point_metrics_by_feature(frame, "i_cmodel_mag", cohort="held", edges=np.arange(19, 25))
    bins = table[(table.kind == "bin") & (table.estimator == "z_ml")]
    assert bins.n_objects.sum() == 3
    assert bins.loc[bins.n_objects == 0, "mae"].isna().all()
    cumulative = table[(table.kind == "cumulative") & (table.estimator == "z_best")]
    assert cumulative.n_objects.iloc[-1] == 3
    assert np.isnan(numeric_observable(frame, "i_cmodel_mag")[1])
    with pytest.raises(ValueError, match="increase"):
        point_metrics_by_feature(frame, "z_spec", cohort="held", edges=[1, 0])


def test_error_scale_comparisons_share_positive_finite_mask(frame):
    table = error_scale_table(frame)
    assert (table[table.redshift_bin == "all"].n_objects == 3).all()
    frame.loc[0, "sigma_ml"] = 0
    table = error_scale_table(frame)
    assert (table[table.redshift_bin == "all"].n_objects == 2).all()
    frame["sigma_ml"] = 0
    empty = error_scale_table(frame)
    assert (empty.n_objects == 0).all()
    assert empty.spearman.isna().all()


def test_profiles_do_not_refit_and_record_empty_bins():
    result = calibration_profile(np.array([1, 2]), np.array([-0.1, 0.3]),
                                  np.array([0.1, 0.15]), np.array([0, 1.5, 3, np.inf]),
                                  cohort="final_test", version="calibrated")
    assert result.n_objects.tolist() == [1, 1, 0]
    assert result.mad1_bin.iloc[0] == 0
    assert result.mad1_bin.iloc[1] == 1
    assert result.median_abs_scaled.iloc[0] == 1
    assert result.median_abs_scaled.iloc[1] == 2
    assert np.isnan(result.mad1_bin.iloc[-1])
    with pytest.raises(ValueError, match="align"):
        calibration_profile(np.ones(2), np.ones(3), np.ones(2), [0, 2],
                            cohort="test", version="raw")
    with pytest.raises(ValueError, match="increase"):
        calibration_profile(np.ones(2), np.ones(2), np.ones(2), [2, 1],
                            cohort="test", version="raw")


def test_multimetric_permutations_reuse_predictions():
    x = np.arange(40).reshape(20, 2).astype(float)
    truth = x.sum(axis=1)
    calls = []
    def predict(values):
        calls.append(1)
        return values.sum(axis=1)
    metrics = {"mae": lambda pred: np.mean(np.abs(pred - truth)),
               "mse": lambda pred: np.mean((pred - truth)**2)}
    groups = {"first": [0], "second": [1]}
    result = permutation_metrics(predict, x, groups, metrics, repeats=3)
    assert len(calls) == 1 + 3 * len(groups)
    assert (result.mean_increase > 0).all()
    assert (result.baseline == 0).all()
    with pytest.raises(ValueError, match="outside"):
        permutation_metrics(predict, x, {"invalid": [3]}, metrics)


@pytest.mark.parametrize("whiten", [False, True])
def test_pc_backprojection_matches_fitted_inverse(whiten):
    data = make_synthetic_dataset(30, 12)
    features = pd.DataFrame({"x": np.arange(30)})
    pp = Preprocessor.fit(data.pdz, features, variance=3)
    pp.pca.whiten = whiten
    table = pc_loading_table(pp, data.z_grid, [0])
    scores = np.zeros((2, 3))
    scores[1, 0] = 1
    reconstructed = pp.sc_pdf.inverse_transform(pp.pca.inverse_transform(scores))
    np.testing.assert_allclose(table.pdz_loading, reconstructed[1] - reconstructed[0], atol=1e-7)


def test_old_training_history_is_explicitly_skipped(tmp_path):
    from turbopdz.analysis import write_training_diagnostics
    source = tmp_path / "zml"
    source.mkdir()
    (source / "best_params.json").write_text(json.dumps({"best_params": {"n_layers": 1}, "final_epochs": 3}))
    result = write_training_diagnostics(tmp_path, "zml")
    assert set(result[result.status == "skipped"].name) == {"final_retrain_loss_curve", "cv_learning_curves", "optuna"}
    assert not list((tmp_path / "plots").rglob("*.pdf"))


def test_known_warning_filter_does_not_change_global_filters():
    from turbopdz.runtime import import_optuna
    pytest.importorskip("optuna")
    original = list(warnings.filters)
    import_optuna()
    assert warnings.filters == original
    with pytest.warns(UserWarning, match="important"):
        warnings.warn("important unrelated warning", UserWarning)


def test_plot_exports_and_empty_diagnostics(frame, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from turbopdz import plotting
    table = point_metrics_by_feature(frame, "i_cmodel_mag", cohort="test")
    path = plotting.save_figure(plotting.plot_binned_metrics(table, feature="i_cmodel_mag"), tmp_path / "metrics.pdf")
    assert path.read_bytes().startswith(b"%PDF-")
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()
    with pytest.raises(ValueError, match="PDF"):
        plotting.save_figure(plt.figure(), tmp_path / "wrong.png")
    plt.close("all")


@pytest.mark.parametrize("metric,value", [("scatter", 0.0), ("outliers", 0.0), ("outliers", 1.0)])
def test_retained_curve_axes_respect_metric_bounds(metric, value):
    pytest.importorskip("matplotlib")
    from turbopdz import plotting
    import matplotlib.pyplot as plt
    table = pd.DataFrame({"retained_fraction": [0.5, 1.0], metric: [value, value]})
    ax = plotting.plot_retained_curves({"r_ml": table}, metric=metric)
    try:
        assert ax.get_ylim()[0] >= 0
        if metric == "outliers":
            assert ax.get_ylim()[1] <= 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), table[metric])
    finally:
        plt.close(ax.figure)


def test_pc_slice_plot_uses_tail_column_and_common_bins(tmp_path):
    pytest.importorskip("matplotlib")
    from turbopdz import plotting
    mean = pd.DataFrame({"pc": ["PC1"] * 6, "tail": ["low"] * 3 + ["high"] * 3,
                         "z": [0, 1, 2] * 2, "mean_pdz": [0.2, 0.7, 0.1, 0.1, 0.7, 0.2]})
    scores = pd.DataFrame({"pc": ["PC1"] * 4, "tail": ["low", "low", "high", "high"],
                           "z_spec": [0.2, 0.4, 1.2, 1.4]})
    summary = pd.DataFrame({"pc": ["PC1"] * 2, "tail": ["low", "high"], "n_objects": [2, 2],
                            "mean_z_spec": [0.3, 1.3], "scatter": [0.1, 0.1], "outliers": [0.1, 0.1]})
    fig = plotting.plot_pc_slices(mean, scores, summary, representation="mass")
    patches = fig.axes[1].patches
    np.testing.assert_allclose(patches[0].get_xy()[:, 0], patches[1].get_xy()[:, 0])
    assert plotting.save_figure(fig, tmp_path / "pc.pdf").exists()


def test_optuna_reports_use_snapshot_and_keep_sealed_database(tmp_path):
    pytest.importorskip("matplotlib")
    optuna = pytest.importorskip("optuna")
    from turbopdz.analysis import write_training_diagnostics
    from turbopdz.artifacts import file_sha256
    source = tmp_path / "zml"
    source.mkdir()
    study = optuna.create_study(study_name="zml", storage=f"sqlite:///{source / 'study.sqlite3'}")
    distributions = {"dropout_rate": optuna.distributions.FloatDistribution(0, 1),
                     "n_layers": optuna.distributions.IntDistribution(1, 1),
                     "n_units_l0": optuna.distributions.IntDistribution(8, 8)}
    for i in range(5):
        study.add_trial(optuna.trial.create_trial(params={"dropout_rate": i / 5, "n_layers": 1, "n_units_l0": 8},
                        distributions=distributions, value=1 + (i / 5)**2))
    study.trials_dataframe().to_csv(source / "trials.csv", index=False)
    (source / "config.json").write_text(json.dumps({"seed": 50, "training": {"hpo": {"study_name": "zml"}}}))
    (source / "best_params.json").write_text(json.dumps({"best_params": {}, "final_epochs": 1}))
    before = file_sha256(source / "study.sqlite3")
    outputs = write_training_diagnostics(tmp_path, "zml")
    assert file_sha256(source / "study.sqlite3") == before
    assert {"optuna_parameter_importance", "optuna_parameter_slices", "optuna_conditional_architectures"} <= set(outputs[outputs.kind == "plot"].name)


def test_population_branch_mass_conventions_and_empty_overlap(frame):
    from turbopdz.data import quadrature_weights
    from turbopdz.evaluation.populations import pca_population_tables
    grid = np.array([0, 0.25, 0.5, 0.8, 1.2, 1.6, 2])
    mass = np.tile([0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1], (4, 1))
    density = mass / quadrature_weights(grid)
    branches = {"true_low": [0, 0.5], "true_high": [1.2, 2],
                "pdz_low": [0, 0.5], "pdz_high": [1.2, 2],
                "predicted_low_max": 0.5, "predicted_high_min": 1.2}
    kwargs = dict(redshift_edges=[0, 0.5, 1, 2], magnitude_edges=[19, 21, 24],
                  branches=branches, investigation_window=[1.4, 2], min_common=15)
    a = pca_population_tables(frame, mass, grid, np.arange(4), representation="mass", **kwargs)
    b = pca_population_tables(frame, density, grid, np.arange(4), representation="density", **kwargs)
    np.testing.assert_allclose(a["objects"].p_low, 0.3)
    np.testing.assert_allclose(a["objects"].p_high, 0.4)
    pd.testing.assert_frame_equal(a["individual_pdz_ambiguity"], b["individual_pdz_ambiguity"])
    assert a["objects"].two_branch_peaks.isna().all()
    assert not len(a["common_redshift_magnitude_cells"])
    assert a["overlap_standardized"].n_common_cells.iloc[0] == 0
    assert np.isnan(a["overlap_standardized"].standardized_mae_low.iloc[0])
    counts = a["redshift_breakdown"].groupby("group").n_objects.sum()
    assert counts.low + counts.other == len(frame)
    for group, rows in a["redshift_breakdown"].groupby("group"):
        assert rows.fraction_of_group.sum() == 1
    with pytest.raises(ValueError, match="Tied"):
        pca_population_tables(frame, mass, grid, np.ones(4), representation="mass", **kwargs)
    with pytest.raises(ValueError, match="mapping"):
        pca_population_tables(frame, mass, grid, np.arange(4), representation="mass",
                              **dict(kwargs, branches=json.dumps(branches)))


def test_population_matched_cells_preserve_requested_threshold(frame):
    from turbopdz.evaluation.populations import pca_population_tables
    values = pca_population_tables(frame, np.ones((4, 4)), np.arange(4), np.arange(4),
        representation="mass", redshift_edges=[0, 2], magnitude_edges=[19, 24], min_common=1)
    row = values["common_redshift_magnitude_cells"].iloc[0]
    assert row.n_low == 1 and row.n_other == 3 and row.weight == 1
    assert values["overlap_standardized"].low_tail_coverage.iloc[0] == 1


def test_workflow_defaults_and_script_have_no_removed_config():
    root = Path(__file__).resolve().parents[1]
    for notebook in (root / "notebooks").glob("[1-5]*.ipynb"):
        cells = json.loads(notebook.read_text())["cells"]
        params = next(cell for cell in cells if "parameters" in cell["metadata"].get("tags", []))
        source = "".join(params["source"])
        assert 'CONFIG_PATH = "configs/synthetic-optuna.yaml"' in source
    assert not (root / "configs/synthetic.yaml").exists()
    assert (root / "examples/run-notebooks.sh").is_file()
