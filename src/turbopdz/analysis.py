"""Regenerate single-catalog reports from saved models, cohorts and predictions.

No helper in this module optimizes or refits a network/calibration. Missing
optional observables or saved training histories are reported in CSV indexes.
Numerical errors and incompatible artifacts are not silently suppressed.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import plotting
from .data import quadrature_weights
from .evaluation.diagnostics import (
    align_predictions, point_summary, point_metrics_by_feature, error_scale_table,
    calibration_profile, permutation_metrics, pc_loading_table, numeric_observable,
)
from .evaluation.metrics import residual, filter_auc, retained_metrics, photoz_metrics


class _Report:
    def __init__(self, root, stage, show):
        self.root, self.stage, self.show = Path(root), stage, show
        self.tables = self.root / "tables" / stage
        self.plots = self.root / "plots" / stage
        self.tables.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)
        self.outputs = []

    def table(self, name, table):
        path = self.tables / f"{name}.csv"
        # NaN is an intentional empty diagnostic cell, not a numerical failure.
        with np.errstate(invalid="ignore"):
            table.to_csv(path, index=False)
        self.outputs.append(dict(kind="table", name=name, path=str(path.relative_to(self.root)), status="saved", reason=""))
        return table

    def figure(self, name, fig):
        path = self.plots / f"{name}.pdf"
        plotting.save_figure(fig, path, show=self.show)
        self.outputs.append(dict(kind="plot", name=name, path=str(path.relative_to(self.root)), status="saved", reason=""))

    def skip(self, name, reason):
        self.outputs.append(dict(kind="optional", name=name, path="", status="skipped", reason=reason))

    def finish(self):
        result = pd.DataFrame(self.outputs, columns=["kind", "name", "path", "status", "reason"])
        result.to_csv(self.tables / "outputs.csv", index=False)
        return result


def _load_frame(root, dataset, cohort):
    frame = pd.read_csv(root / f"{cohort}.csv", dtype={"object_id": str})
    return align_predictions(frame, dataset.observables, dataset.object_id)


def _normalized_pdz(dataset, indices):
    values = dataset.pdz[indices]
    total = values.sum(axis=1) if dataset.representation == "mass" else values @ quadrature_weights(dataset.z_grid)
    return values / total[:, None]


def _pdz_examples(report, dataset, frame, *, name="pdz_examples", count=6):
    if not len(frame):
        report.skip(name, "No objects in this cohort")
        return
    lookup = pd.Series(np.arange(len(dataset.object_id)), index=dataset.object_id.astype(str))
    selected = frame.head(count)
    indices = lookup.loc[selected.object_id.astype(str)].to_numpy()
    curves = _normalized_pdz(dataset, indices)
    rows = [dict(object_id=row.object_id, z=z, pdz=value, z_spec=row.z_spec,
                 z_best=row.z_best, z_ml=getattr(row, "z_ml", np.nan),
                 example_type=getattr(row, "example_type", ""))
            for row, curve in zip(selected.itertuples(), curves)
            for z, value in zip(dataset.z_grid, curve)]
    table = report.table(name, pd.DataFrame(rows))
    report.figure(name, plotting.plot_pdz_examples(table, representation=dataset.representation))


def write_data_diagnostics(directory: str | Path, *, show: bool = False,
                           n_examples: int = 6) -> pd.DataFrame:
    """Save input/cohort summaries, selected-feature statistics and example PDZs."""
    if n_examples < 1:
        raise ValueError("n_examples must be positive")
    from .workflow import load_run
    from .features import feature_table
    from .training.common import feature_config
    root = Path(directory)
    dataset, splits, config = load_run(root)
    report = _Report(root, "data", show)
    report.table("cohort_counts", pd.DataFrame([
        dict(cohort=name, n_objects=len(idx), representation=dataset.representation, seed=config["seed"])
        for name, idx in (("development", splits.development), ("calibration", splits.calibration), ("final_test", splits.test))
    ]))
    for track in ("zml", "sigmaml"):
        features = feature_table(dataset, feature_config(config, track))
        report.table(f"{track}_feature_summary", features.describe().T.rename_axis("feature").reset_index())
    frame = dataset.observables.copy()
    frame["object_id"] = dataset.object_id.astype(str)
    frame["z_spec"] = dataset.z_spec
    reference = config.get("dataset", {}).get("reference_column", "photoz_best")
    frame["z_best"] = numeric_observable(frame, reference)
    report.table("input_summary", frame.select_dtypes(include="number").describe().T.rename_axis("column").reset_index())
    report.figure("catalog_distributions", plotting.plot_catalog_distributions(frame, splits))
    if "i_cmodel_mag" in frame and np.isfinite(numeric_observable(frame, "i_cmodel_mag")).any():
        clean = frame.assign(i_cmodel_mag=numeric_observable(frame, "i_cmodel_mag"))
        report.figure("specz_vs_magnitude", plotting.plot_pair_hexbin(
            clean, [("z_spec", "i_cmodel_mag")], title="Input spectroscopic sample"))
    else:
        report.skip("specz_vs_magnitude", "Magnitude not available or entirely missing")
    order = np.argsort(dataset.z_spec)
    idx = order[np.linspace(0, len(order) - 1, min(n_examples, len(order))).astype(int)]
    _pdz_examples(report, dataset, frame.iloc[idx], count=n_examples)
    return report.finish()


def write_training_diagnostics(directory: str | Path, track: str, *,
                               show: bool = False) -> pd.DataFrame:
    """Export saved loss curves, Optuna history and hyperparameter importance.

    Optuna importance needs at least five completed, non-constant trials.
    It is computed from a temporary copy so sealed study files remain untouched.
    Bundles without saved histories have the corresponding plots marked skipped.
    """
    if track not in {"zml", "sigmaml"}:
        raise ValueError("track must be zml or sigmaml")
    root, source = Path(directory), Path(directory) / track
    report = _Report(root, f"training-{track}", show)
    for filename, name in (("training_history.csv", "final_retrain_loss_curve"),
                           ("cv_learning_curves.csv", "cv_learning_curves")):
        if not (source / filename).exists():
            report.skip(name, "History was not saved by this model; no diagnostic retraining is performed")
            continue
        history = report.table(name, pd.read_csv(source / filename))
        if "trial" in history and (source / "trials.csv").exists():
            trials = pd.read_csv(source / "trials.csv")
            good = trials[(trials.state == "COMPLETE") & np.isfinite(trials.value)]
            history = history[history.trial == good.loc[good.value.idxmin(), "number"]]
        report.figure(name, plotting.plot_learning_curves(history, title=f"{track}: {name}"))
    params = json.loads((source / "best_params.json").read_text())
    report.table("selected_parameters", pd.DataFrame([
        dict(parameter=name, value=value) for name, value in params["best_params"].items()
    ] + [dict(parameter="final_epochs", value=params["final_epochs"])]))
    if track == "sigmaml" and (root / "oof.npz").exists() and (root / "dataset").exists():
        from .workflow import load_run
        from .artifacts import load_manifest, file_sha256
        dataset, splits, _ = load_run(root)
        info = load_manifest(source)
        with np.load(root / "oof.npz", allow_pickle=False) as cache:
            if (not np.array_equal(cache["object_id"], dataset.object_id)
                    or not np.array_equal(cache["fold_id"], splits.fold_id)
                    or str(cache["source_hash"]) != file_sha256(root / "zml/manifest.json")):
                raise ValueError("OOF provenance differs from training diagnostic dataset")
            error = np.abs(residual(cache["prediction"], dataset.z_spec))
        cohorts = np.full(len(error), "development_oof_training", dtype="U32")
        cohorts[splits.calibration], cohorts[splits.test] = "calibration", "final_test"
        targets = report.table("log_error_targets", pd.DataFrame({
            "object_id": dataset.object_id.astype(str), "cohort": cohorts,
            "abs_error": error, "log_error": np.log(error + info["epsilon"])}))
        report.figure("log_error_target_distribution", plotting.plot_log_targets(targets))
    if not (source / "trials.csv").exists():
        report.skip("optuna", "Fixed/reused parameters; no Optuna search belongs to this run")
        return report.finish()
    trials = report.table("optuna_trials", pd.read_csv(source / "trials.csv"))
    report.figure("optimization_history", plotting.plot_optimization(trials, title=f"{track}: Optuna history"))
    good = trials[(trials.state == "COMPLETE") & np.isfinite(trials.value)]
    if len(good) and any(name.startswith("params_") for name in trials):
        report.figure("optuna_parameter_slices", plotting.plot_optuna_slices(trials))
    if ("params_n_layers" in good and good.params_n_layers.notna().any()
            and any(name.startswith("params_n_units_l") for name in good)):
        report.figure("optuna_conditional_architectures", plotting.plot_optuna_architectures(trials))
    if len(good) < 5 or good.value.nunique() < 2:
        report.skip("optuna_parameter_importance", "At least five completed, varying trials required; tiny smoke studies are not informative")
    elif (source / "study.sqlite3").exists():
        from .runtime import import_optuna
        optuna = import_optuna()
        config = json.loads((source / "config.json").read_text())
        hpo = config["training" if track == "zml" else "sigmaml"]["hpo"]
        with tempfile.TemporaryDirectory(prefix="turbopdz-study-") as temporary:
            snapshot = Path(temporary) / "study.sqlite3"
            shutil.copy2(source / "study.sqlite3", snapshot)
            study = optuna.load_study(study_name=hpo.get("study_name", track), storage=f"sqlite:///{snapshot}")
            importance = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=config["seed"]))
        table = report.table("optuna_parameter_importance", pd.DataFrame(
            [dict(parameter=name, importance=value) for name, value in importance.items()]))
        if len(table):
            report.figure("optuna_parameter_importance", plotting.plot_parameter_importance(table))
        else:
            report.skip("optuna_parameter_importance", "No varying parameters")
    else:
        report.skip("optuna_parameter_importance", "Study database was not retained")
    return report.finish()


def _importance(report, dataset, idx, frame, config, pipeline, track, repeats, top_pcs, development):
    from .features import feature_table
    from .objectives import objective_score
    from .evaluation.importance import input_feature_groups
    from .calibration import log_to_scale
    from .training.common import predict_model
    model = pipeline.zml_model if track == "zml" else pipeline.sigma_model
    pp = pipeline.zml_pp if track == "zml" else pipeline.sigma_pp
    info = pipeline.zml_info if track == "zml" else pipeline.sigma_info
    subset = dataset.subset(idx)
    inputs = pp.transform(subset.pdz, feature_table(subset, info["features"]))
    # The frame is deliberately reindexed to model input IDs.
    frame = frame.set_index("object_id").loc[subset.object_id.astype(str)].reset_index()
    if track == "zml":
        refs = info.get("references", {})
        if not refs:
            column = config.get("dataset", {}).get("reference_column", "photoz_best")
            refs = {name: max(value, 1e-6) for name, value in photoz_metrics(
                numeric_observable(dataset.observables, column)[development], dataset.z_spec[development]).items()}
            refs.update(config["selection_objective"].get("scales", {}))
        metrics = {
            "objective": lambda pred: objective_score(pred, subset.z_spec, config["selection_objective"], refs),
            "mae": lambda pred: float(np.mean(np.abs(pred - subset.z_spec))),
        }
        primary = "objective"
    else:
        dz = frame.dz_ml.to_numpy()
        cached = {"prediction": None, "areas": None}
        def areas(pred):
            if pred is not cached["prediction"]:
                reliability = pipeline.calibration.reliability(pipeline.calibration.apply(
                    log_to_scale(pred, pipeline.calibration.epsilon)))
                cached.update(prediction=pred, areas=filter_auc(dz, reliability, higher_is_better=True))
            return cached["areas"]
        metrics = {
            "auc_scatter": lambda pred: areas(pred)["scatter"],
            "auc_outliers": lambda pred: areas(pred)["outliers"],
            "auc_composite": lambda pred: sum(areas(pred).values()) / 2,
            "mae_log_error": lambda pred: float(np.mean(np.abs(pred - np.log(np.abs(dz) + pipeline.calibration.epsilon)))),
        }
        primary = "auc_composite"
    predict = lambda x: predict_model(model, x)
    n_pc = 0 if pp.pca is None else pp.pca.n_components_
    groups = input_feature_groups(pp.feature_names, n_pc)
    table = permutation_metrics(predict, inputs, groups, metrics, repeats=repeats, seed=config["seed"])
    table["cohort"] = "point_holdout" if track == "zml" else "final_test"
    report.table("feature_importance_grouped", table)
    report.figure("feature_importance_grouped", plotting.plot_permutation(table, title=f"{track}: grouped importance"))
    if not n_pc:
        report.skip("pca_interpretation", "PDZ-PCA is disabled for this model")
        return
    groups = {f"PC{i + 1}": [len(pp.feature_names) + i] for i in range(n_pc)}
    pcs = permutation_metrics(predict, inputs, groups, metrics, repeats=repeats, seed=config["seed"])
    pcs["cohort"] = table.cohort.iloc[0]
    report.table("feature_importance_pca", pcs)
    report.figure("feature_importance_pca", plotting.plot_permutation(pcs, title=f"{track}: individual PCA importance"))
    ranked = pcs[pcs.metric == primary].copy()
    ranked["impact"] = ranked.mean_increase.abs()
    selected = [int(name[2:]) - 1 for name in ranked.nlargest(top_pcs, "impact").group]
    loadings = report.table("pca_loadings", pc_loading_table(pp, dataset.z_grid, selected))
    report.figure("pca_component_loadings", plotting.plot_pca_loadings(loadings))
    scores, summaries, means = [], [], []
    normalized = _normalized_pdz(dataset, idx)
    for component in selected:
        values = inputs[:, len(pp.feature_names) + component]
        low, high = np.quantile(values, [0.2, 0.8])
        for tail, mask in (("low", values <= low), ("high", values >= high)):
            name = f"PC{component + 1}"
            scores.extend(dict(object_id=object_id, pc=name, tail=tail, score=score, z_spec=z)
                          for object_id, score, z in zip(subset.object_id[mask].astype(str), values[mask], subset.z_spec[mask]))
            quality = photoz_metrics(frame.z_ml.to_numpy()[mask], subset.z_spec[mask])
            summaries.append(dict(pc=name, tail=tail, score_boundary=low if tail == "low" else high,
                                  n_objects=int(mask.sum()), mean_z_spec=np.mean(subset.z_spec[mask]), **quality))
            means.extend(dict(pc=name, tail=tail, z=z, mean_pdz=p) for z, p in zip(dataset.z_grid, normalized[mask].mean(axis=0)))
    summary = report.table("pc_slice_summary", pd.DataFrame(summaries))
    score_table = report.table("pc_slice_objects", pd.DataFrame(scores))
    mean_table = report.table("pc_slice_mean_pdz", pd.DataFrame(means))
    report.figure("pc_score_high_low", plotting.plot_pc_slices(mean_table, score_table, summary, representation=dataset.representation))


def write_evaluation_diagnostics(directory: str | Path, *, show: bool = False,
                                 run_importance: bool = True, repeats: int = 5,
                                 top_pcs: int = 3, n_examples: int = 6,
                                 n_bins: int = 10) -> pd.DataFrame:
    """Save the complete single-catalog point/error/reliability diagnostic suite.

    zML uses the calibration+test point holdout. sigmaML/rML use the independent
    final test. Calibration plots label calibration and development-OOF training
    profiles separately; none is misrepresented as independent validation.
    """
    from .workflow import load_run
    from .inference import TurboPDZ
    from .features import feature_table
    from .training.common import predict_model
    from .calibration import log_to_scale
    if min(repeats, top_pcs, n_examples, n_bins) < 1:
        raise ValueError("Diagnostic counts must be positive")
    root = Path(directory)
    dataset, splits, config = load_run(root)
    pipeline = TurboPDZ.load(root)
    point, final = (_load_frame(root, dataset, name) for name in ("point_holdout", "final_test"))
    zreport = _Report(root, "zml", show)
    summaries = [point_summary(frame, cohort=cohort) for frame, cohort in ((point, "point_holdout"), (final, "final_test"))]
    zreport.table("summary_table", pd.concat(summaries, ignore_index=True))
    zreport.table("point_predictions", point)
    zreport.figure("redshift_and_residual_histograms", plotting.plot_point_distributions(point, cohort="point_holdout"))
    zreport.figure("specz_vs_photoz_hexbin", plotting.plot_pair_hexbin(
        point, [("z_spec", "z_ml"), ("z_spec", "z_best")], title="Point holdout", shared_axes=True, identity=True))
    zreport.figure("photoz_best_vs_zml_hexbin", plotting.plot_pair_hexbin(
        point, [("z_best", "z_ml")], title="Point holdout", identity=True))
    for column in ("i_cmodel_mag", "z_spec", "z_best"):
        if column not in point or not np.isfinite(numeric_observable(point, column)).any():
            zreport.skip(f"metrics_by_{column}", "Observable not available or entirely missing")
            continue
        table = zreport.table(f"metrics_by_{column}", point_metrics_by_feature(point, column, cohort="point_holdout", n_bins=n_bins))
        zreport.figure(f"metrics_by_{column}", plotting.plot_binned_metrics(table, feature=column))
    worst = point.assign(abs_error=np.abs(point.dz_ml)).sort_values("abs_error", ascending=False)
    _pdz_examples(zreport, dataset, worst, name="worst_zml_pdz", count=n_examples)
    if run_importance:
        _importance(zreport, dataset, np.r_[splits.calibration, splits.test], point, config, pipeline, "zml", repeats, top_pcs, splits.development)
    else:
        zreport.skip("feature_importance", "Disabled by RUN_IMPORTANCE")
    outputs = [zreport.finish()]
    if pipeline.sigma_model is None:
        return pd.concat(outputs, ignore_index=True)
    sreport = _Report(root, "sigmaml", show)
    sreport.table("final_test_predictions", final)
    r_bins = sreport.table("point_metrics_by_reliability", point_metrics_by_feature(
        final, "r_ml", cohort="final_test", edges=np.linspace(0, 1, n_bins + 1)))
    sreport.figure("point_metrics_by_reliability", plotting.plot_binned_metrics(r_bins, feature="r_ml"))
    scale_table = sreport.table("sigma_bins_native_and_controlled", error_scale_table(final))
    sreport.figure("correlations_by_redshift", plotting.plot_error_correlations(scale_table))
    for column in ("i_cmodel_mag", "z_spec"):
        if column not in final or not np.isfinite(numeric_observable(final, column)).any():
            sreport.skip(f"scale_metrics_by_{column}", "Observable unavailable or entirely missing")
            continue
        values = numeric_observable(final, column)
        finite = values[np.isfinite(values)]
        low, high = finite.min(), finite.max()
        boundaries = np.linspace(low, high if high > low else low + 1, n_bins + 1)
        rows = []
        for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            selected = final[(values >= lo) & ((values <= hi) if i == n_bins - 1 else (values < hi))]
            group = error_scale_table(selected)
            group = group[(group.redshift_bin == "all") & (
                ((group.target == "dz_ml") & (group.proxy == "sigma_ml")) |
                ((group.target == "dz_best") & (group.proxy == "s_best")))]
            rows.append(group.assign(feature=column, feature_bin=i, feature_lower=lo,
                                     feature_upper=hi, feature_center=(lo + hi) / 2))
        table = sreport.table(f"scale_metrics_by_{column}", pd.concat(rows, ignore_index=True))
        sreport.figure(f"scale_metrics_by_{column}", plotting.plot_scale_by_feature(table, feature=column))
    for target in ("dz_ml", "dz_best"):
        final[f"abs_{target}"] = np.abs(final[target])
        pairs = [(f"abs_{target}", proxy) for proxy in ("sigma_ml", "s_best") if proxy in final]
        sreport.figure(f"error_scale_vs_abs_{target}", plotting.plot_pair_hexbin(final, pairs, title=f"Final test: |{target}|", logarithmic=True))
    if "s_best" in final:
        sreport.figure("sigma_ml_vs_s_best", plotting.plot_pair_hexbin(final, [("s_best", "sigma_ml")], title="Empirical error-scale proxies", logarithmic=True))
    sreport.figure("scaled_residuals", plotting.plot_scaled_residuals(final))
    if (root / "retained_curves.csv").exists():
        curves = sreport.table("retained_curves", pd.read_csv(root / "retained_curves.csv"))
        aucs = []
        for (scope, target), subset in curves.groupby(["scope", "target"]):
            if scope == "controlled":
                full = subset[subset.requested_fraction == 1]
                for metric in ("scatter", "outliers"):
                    if not np.allclose(full[metric], full[metric].iloc[0], atol=1e-12):
                        raise ValueError("Controlled curves disagree at retained fraction one")
            sreport.figure(f"{scope}_filtering_{target}", plotting.plot_filter_comparison(subset, title=f"{scope}: {target}"))
            for selector, group in subset.groupby("selector"):
                aucs.append(dict(scope=scope, target=target, selector=selector,
                                 **{f"auc_{metric}": np.trapz(group[metric], group.retained_fraction) for metric in ("scatter", "outliers")}))
        sreport.table("filter_auc_native_and_controlled", pd.DataFrame(aucs))
    selectors = {name: higher for name, higher in (("r_ml", True), ("photoz_conf_best", True), ("photoz_risk_best", False)) if name in final}
    needed = ["dz_ml", "dz_best", *selectors]
    joint = np.isfinite(final[needed]).all(axis=1)
    by_redshift = []
    for i, (low, high) in enumerate(zip((0, 0.5, 0.8), (0.5, 0.8, np.inf))):
        cohort = final[joint & (final.z_spec >= low) & (final.z_spec < high)]
        if not len(cohort):
            continue
        for scope, target in (("native", "dz_ml"), ("controlled", "dz_ml"), ("controlled", "dz_best")):
            for selector, higher in selectors.items():
                actual = "dz_best" if scope == "native" and selector != "r_ml" else target
                by_redshift.append(retained_metrics(cohort[actual].to_numpy(), cohort[selector].to_numpy(),
                    higher_is_better=higher).assign(scope=scope, target=target, selector=selector,
                                                  redshift_bin=i, lower=low, upper=high, n_objects=len(cohort)))
    if by_redshift:
        table = sreport.table("filtering_by_redshift", pd.concat(by_redshift, ignore_index=True))
        for (scope, target), subset in table.groupby(["scope", "target"]):
            sreport.figure(f"filtering_by_redshift_{scope}_{target}", plotting.plot_filter_bins(subset))
    selector_rows = [dict(selector=name, n_objects=len(final), n_finite=int(np.isfinite(final[name]).sum()),
                          n_unique=int(final[name].nunique()), minimum=final[name].min(), maximum=final[name].max())
                     for name in selectors]
    sreport.table("selector_diagnostics", pd.DataFrame(selector_rows))
    for target in ("dz_ml", "dz_best"):
        pairs = [(f"abs_{target}", name) for name in selectors if name != "r_ml"]
        if pairs:
            sreport.figure(f"catalog_selector_vs_abs_{target}", plotting.plot_pair_hexbin(final, pairs, title=f"Catalog selector diagnostics: |{target}|"))
    if "photoz_conf_best" in final:
        suspect = final[(final.photoz_conf_best >= 0.95) & (np.abs(final.dz_best) > 0.15)]
        sreport.table("high_confidence_catalog_outliers", suspect)
        _pdz_examples(sreport, dataset, suspect, name="high_confidence_catalog_outlier_pdz", count=n_examples)
    # Evaluate the frozen calibration with boundaries defined by calibration only.
    features = feature_table(dataset, pipeline.sigma_info["features"])
    log_scale = predict_model(pipeline.sigma_model, pipeline.sigma_pp.transform(dataset.pdz, features))
    raw = log_to_scale(log_scale, pipeline.calibration.epsilon)
    calibrated = pipeline.calibration.apply(raw)
    zfeatures = feature_table(dataset, pipeline.zml_info["features"])
    zpred = predict_model(pipeline.zml_model, pipeline.zml_pp.transform(dataset.pdz, zfeatures))
    errors = residual(zpred, dataset.z_spec)
    quantiles = np.linspace(0, 1, min(len(pipeline.calibration.sigma_anchor), len(splits.calibration)) + 1)[1:-1]
    edges = np.unique(np.r_[0, np.quantile(raw[splits.calibration], quantiles), np.inf])
    profiles, thresholds = [], []
    cohorts = [("calibration", splits.calibration), ("final_test", splits.test)]
    if (root / "oof.npz").exists():
        from .artifacts import file_sha256
        with np.load(root / "oof.npz", allow_pickle=False) as cache:
            if (not np.array_equal(cache["object_id"], dataset.object_id)
                    or not np.array_equal(cache["fold_id"], splits.fold_id)
                    or str(cache["source_hash"]) != file_sha256(root / "zml/manifest.json")):
                raise ValueError("OOF provenance differs from diagnostic run")
            errors[splits.development] = residual(cache["prediction"][splits.development], dataset.z_spec[splits.development])
        cohorts.append(("development_oof_training", splits.development))
    for cohort, idx in cohorts:
        for version, scale in (("raw", raw), ("calibrated", calibrated)):
            profiles.append(calibration_profile(raw[idx], errors[idx], scale[idx], edges, cohort=cohort, version=version))
        reliability = pipeline.calibration.reliability(calibrated[idx])
        thresholds.extend(dict(cohort=cohort, threshold=cut, retained_fraction=np.mean(reliability >= cut), n_objects=len(idx))
                          for cut in np.linspace(0, 1, 51))
    anchors = sreport.table("calibration_factors", pd.DataFrame({
        "raw_scale": pipeline.calibration.sigma_anchor, "factor": pipeline.calibration.factors}))
    profile = sreport.table("calibration_profiles", pd.concat(profiles, ignore_index=True))
    sreport.figure("calibration_diagnostics", plotting.plot_calibration(anchors, profile))
    sreport.figure("calibration_bin_deviation_histograms", plotting.plot_calibration_histograms(profile))
    summaries = []
    for (cohort, version), group in profile.groupby(["cohort", "version"]):
        valid = group[group.n_objects > 0]
        summaries.append(dict(cohort=cohort, version=version, n_objects=int(group.n_objects.sum()),
                              n_nonempty_bins=len(valid), per_bin_median=valid.m_bin.median(),
                              per_bin_mean=valid.m_bin.mean(), per_bin_mad_from_1=valid.mad1_bin.median(),
                              per_bin_aad_from_1=valid.mad1_bin.mean(),
                              per_bin_smad_element=valid.smad_element_bin.median()))
    sreport.table("calibration_summary", pd.DataFrame(summaries))
    threshold_table = sreport.table("reliability_thresholds", pd.DataFrame(thresholds))
    sreport.figure("reliability_distribution_and_selection", plotting.plot_reliability(final, threshold_table))
    sreport.table("log_error_targets", pd.DataFrame(dict(object_id=dataset.object_id.astype(str),
                  cohort=np.where(splits.fold_id >= 0, "development_oof_training",
                         np.where(np.isin(np.arange(len(raw)), splits.calibration), "calibration", "final_test")),
                  abs_error=np.abs(errors), log_error=np.log(np.abs(errors) + pipeline.calibration.epsilon),
                  predicted_log_scale=log_scale, raw_scale=raw, sigma_ml=calibrated)))
    if run_importance:
        _importance(sreport, dataset, splits.test, final, config, pipeline, "sigmaml", repeats, top_pcs, splits.development)
    else:
        sreport.skip("feature_importance", "Disabled by RUN_IMPORTANCE")
    outputs.append(sreport.finish())
    return pd.concat(outputs, ignore_index=True)


def write_bootstrap_diagnostics(directory: str | Path, *, show: bool = False) -> pd.DataFrame:
    """Save single-run paired-improvement interval figures and CSV data."""
    root = Path(directory)
    report = _Report(root, "bootstrap", show)
    table = report.table("paired_improvements", pd.read_csv(root / "bootstrap.csv"))
    for family, group in table.groupby("family"):
        report.figure(family, plotting.plot_bootstrap_intervals(group, title=f"{family}: paired improvements"))
    return report.finish()


def write_pca_population_diagnostics(directory: str | Path, *, component: int = 1,
                                      redshift_edges: np.ndarray | None = None,
                                      magnitude_edges: np.ndarray | None = None,
                                      tail_fraction: float = 0.2, min_common: int = 15,
                                      min_rate_count: int = 20, branches: dict | None = None,
                                      investigation_window: tuple[float, float] | None = None,
                                      show: bool = False) -> pd.DataFrame:
    """Export the optional single-component population/degeneracy investigation.

    Component numbers are one-based. Window and overlap settings must be chosen
    explicitly for the dataset; the report does not assign a spectral-desert
    interpretation or adapt cells to get a positive result. Only existing point
    holdout predictions and fitted preprocessing are used, with no retraining.
    """
    from .workflow import load_run
    from .inference import TurboPDZ
    from .features import feature_table
    from .evaluation.populations import pca_population_tables
    root = Path(directory)
    dataset, splits, config = load_run(root)
    pipeline = TurboPDZ.load(root)
    pp = pipeline.zml_pp
    report = _Report(root, f"pca-population-PC{component}", show)
    if pp.pca is None:
        report.skip("population_check", "PCA is disabled for this model")
        return report.finish()
    if not 1 <= component <= pp.pca.n_components_ or min_rate_count < 1:
        raise ValueError("Choose an available one-based component and positive min_rate_count")
    idx = np.r_[splits.calibration, splits.test]
    subset = dataset.subset(idx)
    inputs = pp.transform(subset.pdz, feature_table(subset, pipeline.zml_info["features"]))
    scores = inputs[:, len(pp.feature_names) + component - 1]
    frame = _load_frame(root, dataset, "point_holdout").set_index("object_id").loc[subset.object_id.astype(str)].reset_index()
    if redshift_edges is None:
        values = frame.z_spec.to_numpy()
        lo, hi = values.min(), values.max()
        redshift_edges = np.linspace(lo, hi if hi > lo else lo + 1, 11)
    settings = dict(component=component, redshift_edges=np.asarray(redshift_edges, dtype=float).tolist(),
                    magnitude_edges=None if magnitude_edges is None else np.asarray(magnitude_edges, dtype=float).tolist(),
                    tail_fraction=tail_fraction, min_common=min_common,
                    min_rate_count=min_rate_count, branches=branches,
                    investigation_window=investigation_window, cohort="point_holdout")
    report.table("settings", pd.DataFrame([dict(setting=key, value=json.dumps(value)) for key, value in settings.items()]))
    tables = pca_population_tables(frame, subset.pdz, subset.z_grid, scores,
        representation=subset.representation, redshift_edges=redshift_edges, magnitude_edges=magnitude_edges,
        tail_fraction=tail_fraction, min_common=min_common, branches=branches, investigation_window=investigation_window)
    for name, table in tables.items():
        report.table(name, table.assign(cohort="point_holdout", pc=f"PC{component}"))
    report.figure("specz_vs_zml_by_tail", plotting.plot_pca_population_hexbins(tables["objects"], component=component))
    report.figure("specz_counts_and_outliers", plotting.plot_pca_population_counts(
        tables["redshift_breakdown"], min_count=min_rate_count, investigation_window=investigation_window))
    if "common_redshift_magnitude_cells" not in tables:
        report.skip("overlap_control", "Provide magnitude_edges and i_cmodel_mag to control population cells")
    elif not len(tables["common_redshift_magnitude_cells"]):
        report.skip("overlap_estimate", "No cells pass the prespecified minimum; binning was not relaxed")
    if branches is None:
        report.skip("branch_ambiguity", "Provide dataset-specific true/predicted/PDZ branch windows")
    else:
        if tables["objects"].two_branch_peaks.isna().all():
            report.skip("two_peak_check", "Native grid is nonuniform; bin-index smoothing would not have a fixed redshift width")
        objects = tables["objects"]
        low = objects["tail"] == "low"
        true_low = objects.z_spec.between(*branches["true_low"], inclusive="left")
        true_high = objects.z_spec.between(*branches["true_high"], inclusive="left")
        pred_low = objects.z_ml < branches["predicted_low_max"]
        pred_high = objects.z_ml >= branches["predicted_high_min"]
        rng = np.random.default_rng(config["seed"])
        examples = []
        for name, mask in (("true high -> predicted low", low & true_high & pred_low),
                           ("true high -> predicted high", low & true_high & pred_high),
                           ("true low -> predicted high", low & true_low & pred_high),
                           ("true low -> predicted low", low & true_low & pred_low)):
            candidates = np.flatnonzero(mask)
            if len(candidates):
                examples.append(objects.iloc[[rng.choice(candidates)]].assign(example_type=name))
            else:
                report.skip(name, "No objects in this prespecified example group")
        if examples:
            _pdz_examples(report, dataset, pd.concat(examples), name="individual_branch_pdz_examples", count=4)
    return report.finish()


def write_oof_diagnostics(directory: str | Path, *, show: bool = False) -> pd.DataFrame:
    """Export OOF membership/errors; development summaries are non-nested."""
    from .workflow import load_run
    root = Path(directory)
    dataset, splits, config = load_run(root)
    report = _Report(root, "oof", show)
    from .artifacts import file_sha256
    with np.load(root / "oof.npz", allow_pickle=False) as cache:
        if (not np.array_equal(cache["object_id"], dataset.object_id)
                or not np.array_equal(cache["fold_id"], splits.fold_id)
                or str(cache["source_hash"]) != file_sha256(root / "zml/manifest.json")):
            raise ValueError("OOF provenance differs from saved run")
        prediction = cache["prediction"].copy()
    column = config.get("dataset", {}).get("reference_column", "photoz_best")
    frame = pd.DataFrame(dict(object_id=dataset.object_id.astype(str), z_spec=dataset.z_spec,
                              z_ml=prediction, z_best=numeric_observable(dataset.observables, column),
                              fold_id=splits.fold_id))
    frame["dz_ml"] = residual(prediction, dataset.z_spec)
    frame["dz_best"] = residual(frame.z_best.to_numpy(), dataset.z_spec)
    report.table("predictions_and_membership", frame.assign(
        scope=np.where(splits.fold_id >= 0, "development_oof_non_nested", "final_model_holdout")))
    dev = frame.iloc[splits.development]
    report.table("development_oof_summary", point_summary(dev, cohort="development_oof_non_nested"))
    report.figure("development_oof_distributions", plotting.plot_point_distributions(dev, cohort="development OOF (non-nested)"))
    report.figure("development_oof_specz_comparison", plotting.plot_pair_hexbin(
        dev, [("z_spec", "z_ml"), ("z_spec", "z_best")], shared_axes=True, identity=True,
        title="Development cross-fitted predictions (non-nested HPO)"))
    return report.finish()


def write_applicability_diagnostics(directory: str | Path, *, show: bool = False) -> pd.DataFrame:
    """Save the independent support reference, analogy/projection and test flags."""
    from .workflow import load_run
    from .applicability import ApplicabilityReference, compute_applicability_flags, compute_reference_support_diagnostics
    root = Path(directory)
    dataset, splits, config = load_run(root)
    reference = ApplicabilityReference.load(root / "applicability.npz")
    report = _Report(root, "applicability", show)
    report.table("reference_summary", pd.DataFrame([{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                                                   for key, value in reference.summary().items()}]))
    diagnostics = report.table("reference_support", compute_reference_support_diagnostics(reference))
    fig, _ = plotting.plot_support(reference, diagnostics=diagnostics)
    report.figure("support_distance_and_pca", fig)
    report.figure("neighbor_rule_2d_analogy", plotting.plot_neighbor_rule(reference, seed=config["seed"]))
    if all(name in dataset.observables for name in reference.mag_columns):
        idx = splits.test
        flags = compute_applicability_flags(dataset.observables.iloc[idx], reference).reset_index(drop=True)
        flags.insert(0, "object_id", dataset.object_id[idx].astype(str))
        report.table("final_test_flags", flags)
    else:
        report.skip("final_test_flags", "Compact products lack required photometry; supply row-aligned grizy to apply flags")
    return report.finish()
