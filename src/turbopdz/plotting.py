"""Optional plotting helpers; importing the numerical library needs no matplotlib."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from .applicability import ApplicabilityReference
    from .splits import SplitPlan

METRIC_LABELS = {"mae": "MAE", "bias": "Median bias",
                 "scatter": r"$\sigma_{\rm NMAD}$", "outliers": "Outlier fraction"}


def save_figure(fig: Figure, path: str | Path, *, show: bool = False) -> Path:
    """Save a tightly bounded PDF and optionally display it in a notebook."""
    from pathlib import Path
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    path = Path(path)
    if path.suffix != ".pdf":
        raise ValueError("Scientific figures must be saved as PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"pdf.fonttype": 42}):
        fig.savefig(path, bbox_inches="tight")
    if show:
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return path


def plot_point_distributions(frame: pd.DataFrame, *, cohort: str) -> Figure:
    """Compare redshift and signed-residual distributions on shared objects."""
    import matplotlib.pyplot as plt
    valid = np.isfinite(frame[["z_spec", "z_ml", "z_best"]]).all(axis=1)
    data = frame.loc[valid]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, names in zip(axes, (("z_spec", "z_ml", "z_best"), ("dz_ml", "dz_best"))):
        bins = np.histogram_bin_edges(data[list(names)].to_numpy().ravel(), bins=25)
        for name in names:
            ax.hist(data[name], bins=bins, histtype="step", label=name)
    axes[0].set(xlabel="Redshift", ylabel="Objects")
    axes[1].set(xlabel=r"$\Delta z=(z_{\rm pred}-z_{\rm spec})/(1+z_{\rm spec})$", ylabel="Objects")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"{cohort}: N={len(data)}")
    return fig


def plot_pair_hexbin(frame: pd.DataFrame, pairs: list[tuple[str, str]], *, title: str,
                     shared_axes: bool = False, logarithmic: bool = False,
                     identity: bool = False) -> Figure:
    """Plot finite paired values with correlation annotations and density counts."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    from .evaluation.metrics import error_correlations
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4),
                             sharex=shared_axes, sharey=shared_axes, squeeze=False,
                             layout="constrained")
    if shared_axes:
        common = np.isfinite(frame[list(dict.fromkeys(name for pair in pairs for name in pair))]).all(axis=1)
    else:
        common = np.ones(len(frame), dtype=bool)
    for ax, (xname, yname) in zip(axes.ravel(), pairs):
        x, y = frame[xname].to_numpy(), frame[yname].to_numpy()
        valid = common & np.isfinite(x) & np.isfinite(y)
        if logarithmic:
            valid &= (x > 0) & (y > 0)
        x, y = x[valid], y[valid]
        if len(x):
            artist = ax.hexbin(x, y, gridsize=35, bins="log", mincnt=1,
                               xscale="log" if logarithmic else "linear",
                               yscale="log" if logarithmic else "linear", rasterized=True)
            fig.colorbar(artist, ax=ax, label="Objects")
            corr = error_correlations(x, y)
            ax.set_title(f"N={len(x)}; Spearman={corr['spearman']:.3f}")
            if identity:
                low, high = min(x.min(), y.min()), max(x.max(), y.max())
                ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
        else:
            ax.text(0.5, 0.5, "No finite objects", ha="center", transform=ax.transAxes)
        ax.set(xlabel=xname, ylabel=yname)
        if logarithmic:
            # Narrow scale ranges otherwise label every minor logarithmic tick.
            for axis in (ax.xaxis, ax.yaxis):
                low, high = axis.get_view_interval()
                if low > 0 and high / low < 10:
                    axis.set_major_locator(FixedLocator(np.geomspace(low, high, 3)))
                    axis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.2g}"))
                else:
                    axis.set_major_locator(LogLocator(base=10, numticks=4))
                axis.set_minor_formatter(NullFormatter())
    fig.suptitle(title)
    return fig


def plot_binned_metrics(table: pd.DataFrame, *, feature: str) -> Figure:
    """Show bin/cumulative point metrics and object counts with shared axes."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(10, 13), sharex="col", sharey="row",
                             layout="constrained")
    for j, kind in enumerate(("bin", "cumulative")):
        subset = table[table.kind == kind]
        for estimator, group in subset.groupby("estimator", sort=False):
            x = group.center if kind == "bin" else group.upper
            for i, metric in enumerate(("mae", "bias", "scatter", "outliers")):
                if metric in group:
                    axes[i, j].plot(x, group[metric], marker=".", label=estimator)
                    axes[i, j].set_ylabel(METRIC_LABELS[metric])
            if estimator == "z_ml":
                axes[4, j].plot(x, group.n_objects, marker=".", color="gray")
        axes[0, j].set_title(kind.capitalize())
        axes[0, j].legend()
        axes[1, j].axhline(0, color="gray", linewidth=0.7)
        axes[4, j].set(xlabel=feature, ylabel="Objects")
    return fig


def plot_filter_comparison(curves: pd.DataFrame, *, title: str) -> Figure:
    """Compare scatter and outlier filtering using actual retained fractions."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for metric, ax in zip(("scatter", "outliers"), axes):
        plot_retained_curves(dict(tuple(curves.groupby("selector", sort=False))), metric=metric, ax=ax)
        ax.set_ylabel(METRIC_LABELS[metric])
    fig.suptitle(title)
    return fig


def plot_scaled_residuals(frame: pd.DataFrame) -> Figure:
    """Display standardized-error shapes, not claimed Gaussian coverage."""
    import matplotlib.pyplot as plt
    pairs = [(residual, scale) for residual, scale in (("dz_ml", "sigma_ml"), ("dz_best", "s_best"))
             if scale in frame]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    bins = np.linspace(-6, 6, 61)
    centers = (bins[:-1] + bins[1:]) / 2
    for ax, (target, proxy) in zip(axes.ravel(), pairs):
        valid = np.isfinite(frame[[target, proxy]]).all(axis=1) & (frame[proxy] > 0)
        ratio = (frame.loc[valid, target] / frame.loc[valid, proxy]).to_numpy()
        counts, _ = np.histogram(ratio, bins=bins)
        density = counts / (max(1, len(ratio)) * np.diff(bins))
        ax.stairs(density, bins, label="Observed")
        ax.plot(centers, np.exp(-centers**2 / 2) / np.sqrt(2 * np.pi), "--",
                color="gray", label="Unit Gaussian (reference only)")
        fraction = np.mean(np.abs(ratio) > 3) if len(ratio) else np.nan
        ax.set(xlabel=f"{target}/{proxy}", ylabel="Density",
               title=f"N={len(ratio)}; |ratio|>3: {fraction:.1%}")
        ax.legend(fontsize=8)
    fig.suptitle("Scaled-residual shape diagnostic")
    return fig


def plot_calibration(anchors: pd.DataFrame, profile: pd.DataFrame) -> Figure:
    """Display the frozen calibration factors and raw/calibrated bin profiles."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].plot(anchors.raw_scale, anchors.factor, marker=".", label="Frozen curve")
    raw_profile = profile[(profile.cohort == "calibration") & (profile.version == "raw")]
    axes[0].scatter(raw_profile.raw_mid, raw_profile.m_bin, marker="o", label="Calibration-bin ratios")
    axes[0].set(xscale="log", xlabel="Raw error scale", ylabel="Frozen multiplicative factor")
    axes[0].legend(fontsize=8)
    for (cohort, version), group in profile.groupby(["cohort", "version"]):
        if "raw_mid" not in group:
            continue
        for ax, metric in zip(axes[1:], ("m_bin", "mad1_bin")):
            ax.plot(group.raw_mid, group[metric], marker=".", label=f"{cohort}: {version}")
    for ax, ylabel, reference in zip(axes[1:], ("Median |error| / median scale", "Absolute bin-ratio deviation from 1"), (1, 0)):
        ax.set(xscale="log", xlabel="Calibration-defined raw-scale bins", ylabel=ylabel)
        ax.axhline(reference, linestyle="--", color="gray")
    axes[2].legend(fontsize=8)
    fig.suptitle("Calibration diagnostics; final test is not used for fitting")
    return fig


def plot_calibration_histograms(profile: pd.DataFrame) -> Figure:
    """Compare the distribution of bin-ratio deviations before/after calibration."""
    import matplotlib.pyplot as plt
    cohorts = list(profile.cohort.unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 4), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    values = profile.mad1_bin[np.isfinite(profile.mad1_bin)]
    bins = np.histogram_bin_edges(values, bins=15)
    for ax, cohort in zip(axes.ravel(), cohorts):
        for version, group in profile[profile.cohort == cohort].groupby("version"):
            finite = group.mad1_bin[np.isfinite(group.mad1_bin)]
            ax.hist(finite, bins=bins, histtype="step", label=version)
        ax.set(xlabel="Absolute bin-ratio deviation from 1", ylabel="Nonempty bins", title=cohort)
        ax.legend(fontsize=8)
    return fig


def plot_log_targets(table: pd.DataFrame) -> Figure:
    """Show OOF training and held-out log-error targets on common histogram edges."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bins = np.histogram_bin_edges(table.log_error, bins=30)
    for cohort, group in table.groupby("cohort"):
        ax.hist(group.log_error, bins=bins, histtype="step", label=f"{cohort}: N={len(group)}")
    ax.set(xlabel="log(|Delta z| + epsilon)", ylabel="Objects", title="sigmaML regression targets")
    ax.legend(fontsize=8)
    return fig


def plot_reliability(frame: pd.DataFrame, threshold_table: pd.DataFrame) -> Figure:
    """Compare reliability distributions and threshold-retained fractions."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist(frame.r_ml.dropna(), bins=np.linspace(0, 1, 21), histtype="step")
    for cohort, group in threshold_table.groupby("cohort"):
        axes[1].plot(group.threshold, group.retained_fraction, label=cohort)
    axes[1].plot([0, 1], [1, 0], "--", color="gray", label="Uniform reference")
    axes[0].set(xlabel="r_ml", ylabel="Final-test objects")
    axes[1].set(xlabel="Reliability threshold", ylabel="Retained fraction", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    return fig


def plot_permutation(table: pd.DataFrame, *, title: str, top_n: int = 15) -> Figure:
    """Order each importance panel independently, with highest impact on top."""
    import matplotlib.pyplot as plt
    names = list(table.metric.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False,
                             layout="constrained")
    for ax, name in zip(axes.ravel(), names):
        group = table[table.metric == name].nlargest(top_n, "mean_increase").sort_values("mean_increase")
        ax.barh(group.group, group.mean_increase, xerr=group.std_increase, alpha=0.8)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set(xlabel=f"Increase in {name}", title=name)
    fig.suptitle(title + " (error bars: permutation variability)")
    return fig


def plot_pca_loadings(table: pd.DataFrame) -> Figure:
    """Show signed raw-PDZ modes and normalized absolute loading profiles."""
    import matplotlib.pyplot as plt
    pcs = list(table.pc.unique())
    fig, axes = plt.subplots(len(pcs), 2, figsize=(10, 3 * len(pcs)),
                             squeeze=False, sharex=True, layout="constrained")
    for row, pc in zip(axes, pcs):
        group = table[table.pc == pc]
        row[0].plot(group.z, group.pdz_loading)
        row[0].axhline(0, color="gray", linewidth=0.7)
        row[0].set(ylabel="PDZ mode loading", title=f"{pc}; variance={group.explained_variance.iloc[0]:.1%}")
        norm = np.abs(group.pdz_loading)
        row[1].plot(group.z, norm / max(float(norm.sum()), 1e-12))
        row[1].set(ylabel="Normalized absolute loading")
    for ax in axes[-1]:
        ax.set_xlabel("Native redshift grid")
    return fig


def plot_pc_slices(mean_pdz: pd.DataFrame, scores: pd.DataFrame,
                   summary: pd.DataFrame, *, representation: str) -> Figure:
    """Compare low/high score PDZs and redshift mixtures, retaining PCA signs."""
    import matplotlib.pyplot as plt
    pcs = list(mean_pdz.pc.unique())
    fig, axes = plt.subplots(2, len(pcs), figsize=(5 * len(pcs), 7), squeeze=False,
                             sharex="row", layout="constrained")
    for j, pc in enumerate(pcs):
        bins = np.histogram_bin_edges(scores.loc[scores.pc == pc, "z_spec"], bins=15)
        for tail, color in (("low", "tab:blue"), ("high", "tab:orange")):
            pdz = mean_pdz[(mean_pdz.pc == pc) & (mean_pdz["tail"] == tail)]
            selected = scores[(scores.pc == pc) & (scores["tail"] == tail)]
            row = summary[(summary.pc == pc) & (summary["tail"] == tail)].iloc[0]
            axes[0, j].plot(pdz.z, pdz.mean_pdz, color=color, label=tail)
            axes[1, j].hist(selected.z_spec, bins=bins, histtype="step", color=color,
                             label=f"{tail}: N={row.n_objects}, mean z={row.mean_z_spec:.2f}\n"
                                   f"NMAD={row.scatter:.3f}, outliers={row.outliers:.1%}")
        axes[0, j].set(title=pc, ylabel=f"Mean PDZ ({representation})")
        axes[0, j].legend()
        axes[1, j].set(xlabel="Spectroscopic redshift", ylabel="Objects")
        axes[1, j].legend(fontsize=8)
    fig.suptitle("Bottom/top 20% of stored PCA scores; signs are arbitrary")
    return fig


def plot_learning_curves(table: pd.DataFrame, *, title: str) -> Figure:
    """Plot saved fit-mode loss histories without extra diagnostic retraining."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    if "fold" in table:
        for fold, group in table.groupby("fold"):
            ax.plot(group.epoch, group.loss, alpha=0.6, label=f"Fold {fold + 1}: train")
            if "val_loss" in group:
                ax.plot(group.epoch, group.val_loss, "--", alpha=0.6, label=f"Fold {fold + 1}: validation")
    else:
        ax.plot(table.epoch, table.loss, label="Training loss (fit/dropout mode)")
    ax.set(xlabel="Epoch", ylabel="Network training loss", title=title)
    ax.legend(fontsize=8)
    return fig


def plot_optimization(table: pd.DataFrame, *, title: str) -> Figure:
    """Show completed-trial objective values and their running minimum."""
    import matplotlib.pyplot as plt
    good = table[(table.state == "COMPLETE") & np.isfinite(table.value)].sort_values("number")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.scatter(good.number, good.value, label="Completed trial")
    ax.plot(good.number, good.value.cummin(), label="Best so far")
    ax.set(xlabel="Trial number", ylabel="Selection objective (lower is better)", title=title)
    ax.legend()
    return fig


def plot_neighbor_rule(reference: ApplicabilityReference, *, max_points: int = 1000,
                       seed: int = 50) -> Figure:
    """Illustrate a 2D neighbor-radius analogy, not the actual 4D decision."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(reference.reference_scaled), min(max_points, len(reference.reference_scaled)), replace=False)
    points = PCA(n_components=2).fit_transform(reference.reference_scaled[idx])
    k = min(reference.n_neighbors, len(points) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances = neighbors.kneighbors(points)[0][:, -1]
    boundary = np.quantile(distances, 0.99)
    dense = points[np.argmin(distances)]
    sparse = np.array([points[:, 0].max() + max(1, np.std(points[:, 0])) * 3, np.median(points[:, 1])])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, query, label in zip(axes, (dense, sparse), ("Dense query", "Sparse query")):
        radius = neighbors.kneighbors(query[None], n_neighbors=k)[0][0, -1]
        ax.scatter(points[:, 0], points[:, 1], s=4, color="gray", alpha=0.5)
        ax.scatter(*query, marker="x", color="red")
        ax.add_patch(Circle(query, radius, fill=False, color="red", label=f"d_{k}"))
        ax.add_patch(Circle(query, boundary, fill=False, linestyle="--", color="blue", label="2D reference boundary"))
        ax.set(aspect="equal", xlabel="Projected color PC1", ylabel="Projected color PC2",
               title=label + f": d_{k} {'<=' if radius <= boundary else '>'} T")
        ax.legend(fontsize=8)
    fig.suptitle("Two-dimensional analogy only; flags use the original color space")
    return fig


def plot_retained_curves(curves: dict[str, pd.DataFrame], *, metric: str = "scatter",
                         ax: Axes | None = None) -> Axes:
    """Plot comparable retained-fraction curves produced by evaluation helpers."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for label, table in curves.items():
        ax.plot(table["retained_fraction"], table[metric], label=label)
    ax.set(xlabel="Retained fraction", ylabel=metric, xlim=(0, 1))
    # Autoscaling otherwise invents negative ticks for zero-error/outlier bins.
    if metric in {"mae", "scatter", "outliers"}:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    if metric == "outliers":
        ax.set_ylim(top=min(1, ax.get_ylim()[1]))
    ax.legend()
    return ax


def plot_support(reference: ApplicabilityReference, *, max_points: int = 3000,
                  seed: int = 50, diagnostics: pd.DataFrame | None = None) -> tuple[Figure, np.ndarray]:
    """Show leave-one-out distances and a PCA projection of standardized colors.

    Red points are reference objects outside the LOO support threshold, not
    mislabeled galaxies or objects proven to have poor redshift estimates.
    """
    if reference.reference_scaled.shape[1] < 2:
        raise ValueError("The two-dimensional PCA view requires at least two colors")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from .applicability import compute_reference_support_diagnostics
    if diagnostics is None:
        diagnostics = compute_reference_support_diagnostics(reference)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(reference.reference_scaled), min(max_points, len(reference.reference_scaled)), replace=False))
    projection = PCA(n_components=2).fit_transform(reference.reference_scaled)
    outside = diagnostics["knn_distance"].to_numpy() > reference.support_distance_99
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(diagnostics["knn_distance"], bins=60)
    axes[0].axvline(reference.support_distance_99, color="red", label="99% LOO boundary")
    axes[0].set(xlabel="Leave-one-out neighbor distance", ylabel="Reference objects")
    axes[0].legend()
    axes[1].scatter(projection[idx, 0], projection[idx, 1], c=np.where(outside[idx], "red", "gray"), s=5, alpha=0.6)
    axes[1].set(xlabel="Color-space PC1", ylabel="Color-space PC2")
    fig.tight_layout()
    return fig, axes


def plot_catalog_distributions(frame: pd.DataFrame, splits: SplitPlan) -> Figure:
    """Show input redshift and magnitude distributions for each saved cohort."""
    import matplotlib.pyplot as plt
    from .evaluation.diagnostics import numeric_observable
    columns = ["z_spec"] + (["i_cmodel_mag"] if "i_cmodel_mag" in frame else [])
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4),
                             squeeze=False, layout="constrained")
    for ax, name in zip(axes.ravel(), columns):
        values = numeric_observable(frame, name)
        finite = values[np.isfinite(values)]
        bins = np.histogram_bin_edges(finite, bins=25)
        for cohort, idx in (("development", splits.development), ("calibration", splits.calibration), ("final_test", splits.test)):
            selected = values[idx]
            ax.hist(selected[np.isfinite(selected)], bins=bins, histtype="step", label=f"{cohort}: N={len(idx)}")
        ax.set(xlabel=name, ylabel="Objects")
        ax.legend(fontsize=8)
    return fig


def plot_pdz_examples(table: pd.DataFrame, *, representation: str) -> Figure:
    """Plot native-grid PDZ examples with spectroscopic and point estimates."""
    import matplotlib.pyplot as plt
    groups = list(table.groupby("object_id", sort=False))
    rows = (len(groups) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3 * rows), squeeze=False,
                             sharex=True, layout="constrained")
    for ax, (object_id, group) in zip(axes.ravel(), groups):
        ax.plot(group.z, group.pdz, color="black", label="PDZ")
        for name, color in (("z_spec", "gray"), ("z_best", "tab:orange"), ("z_ml", "tab:blue")):
            value = group[name].iloc[0]
            if np.isfinite(value):
                ax.axvline(value, color=color, linestyle="--", label=name)
        title = f"Object {object_id}"
        if "example_type" in group and group.example_type.iloc[0]:
            title += "\n" + str(group.example_type.iloc[0])
        ax.set(title=title, xlabel="Redshift", ylabel=f"PDZ ({representation})")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    return fig


def plot_parameter_importance(table: pd.DataFrame) -> Figure:
    """Display Optuna fANOVA importance, not astrophysical feature importance."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    group = table.sort_values("importance")
    ax.barh(group.parameter, group.importance)
    ax.set(xlabel="Optuna fANOVA importance", title="Hyperparameter importance")
    return fig


def plot_optuna_slices(table: pd.DataFrame) -> Figure:
    """Show completed-trial objective slices and mark the winning trial.

    Conditional parameters remain missing outside their sampled architectures;
    they are not replaced by zero. Categorical values get explicit tick labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
    import pandas as pd
    good = table[(table.state == "COMPLETE") & np.isfinite(table.value)]
    columns = [name for name in good if name.startswith("params_") and good[name].notna().any()]
    rows = (len(columns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows), squeeze=False,
                             sharey=True, layout="constrained")
    winner = good.loc[good.value.idxmin(), "number"]
    for ax, name in zip(axes.ravel(), columns):
        group = good[good[name].notna()]
        numeric = pd.to_numeric(group[name], errors="coerce")
        if numeric.notna().all():
            x = numeric.to_numpy()
            unique = np.unique(x)
            ax.xaxis.set_major_locator(FixedLocator(unique) if len(unique) <= 6 else MaxNLocator(nbins=3))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.4g}"))
        else:
            labels = list(group[name].astype(str).unique())
            x = np.array([labels.index(value) for value in group[name].astype(str)])
            ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
        ax.scatter(x, group.value, c=group.number, cmap="Blues", edgecolors="gray")
        best = group.number.to_numpy() == winner
        if best.any():
            ax.scatter(x[best], group.value.to_numpy()[best], marker="*", s=160,
                       color="red", label="Best trial")
            ax.legend(fontsize=8)
        ax.set(xlabel=name.removeprefix("params_"), ylabel="Selection objective")
    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Optuna parameter slices; color indicates trial number")
    return fig


def plot_optuna_architectures(table: pd.DataFrame) -> Figure:
    """Compare conditional layer widths within each sampled network depth."""
    import matplotlib.pyplot as plt
    good = table[(table.state == "COMPLETE") & np.isfinite(table.value)]
    depths = sorted(good.params_n_layers.dropna().unique())
    widths = sorted(name for name in good if name.startswith("params_n_units_l"))
    fig, axes = plt.subplots(len(widths), len(depths), figsize=(max(6, 4 * len(depths)), 3 * len(widths)),
                             squeeze=False, sharey=True, layout="constrained")
    winner = good.loc[good.value.idxmin(), "number"]
    for i, width in enumerate(widths):
        for j, depth in enumerate(depths):
            group = good[(good.params_n_layers == depth) & good[width].notna()]
            ax = axes[i, j]
            if len(group):
                ax.scatter(group[width], group.value, c=group.number, cmap="Blues", edgecolors="gray")
                best = group[group.number == winner]
                ax.scatter(best[width], best.value, marker="*", s=160, color="red")
            else:
                ax.text(0.5, 0.5, "Not sampled", ha="center", transform=ax.transAxes)
            ax.set(xlabel=width.removeprefix("params_"), ylabel="Selection objective", title=f"Depth: {int(depth)}")
    fig.suptitle("Conditional network widths; red star marks the winning trial")
    return fig


def plot_error_correlations(table: pd.DataFrame) -> Figure:
    """Compare proxy correlations across redshift bins against common targets."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharey=True)
    selected = table[table.redshift_bin != "all"]
    for j, target in enumerate(("dz_ml", "dz_best")):
        for proxy, group in selected[selected.target == target].groupby("proxy"):
            for i, metric in enumerate(("spearman", "pearson")):
                values = group[metric] if metric in group else np.full(len(group), np.nan)
                axes[i, j].plot(np.arange(len(group)), values, marker="o", label=proxy)
                axes[i, j].set_xticks(np.arange(len(group)), [f"[{lo:g}, {hi:g})" for lo, hi in zip(group.lower, group.upper)])
                axes[i, j].set(xlabel="Spectroscopic-redshift bin", ylabel=metric, ylim=(-1, 1), title=f"|{target}|")
                axes[i, j].axhline(0, color="gray", linewidth=0.7)
        axes[0, j].legend(fontsize=8)
    return fig


def plot_scale_by_feature(table: pd.DataFrame, *, feature: str) -> Figure:
    """Show native error-scale correlations and scaled-residual shape by feature."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    for (target, proxy), group in table.groupby(["target", "proxy"]):
        for ax, metric in zip(axes.ravel(), ("spearman", "pearson", "median_abs_scaled", "frac_abs_scaled_le_1")):
            if metric in group:
                ax.plot(group.feature_center, group[metric], marker=".", label=f"{target}: {proxy}")
            ax.set(xlabel=feature, ylabel=metric)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Native scale diagnostics; fraction within scale is not guaranteed coverage")
    return fig


def plot_filter_bins(table: pd.DataFrame) -> Figure:
    """Show one filtering scope/target across populated redshift bins."""
    import matplotlib.pyplot as plt
    bins = list(table.redshift_bin.unique())
    fig, axes = plt.subplots(len(bins), 2, figsize=(10, 3.5 * len(bins)), squeeze=False,
                             sharex=True, layout="constrained")
    for row, label in zip(axes, bins):
        group = table[table.redshift_bin == label]
        for metric, ax in zip(("scatter", "outliers"), row):
            plot_retained_curves(dict(tuple(group.groupby("selector"))), metric=metric, ax=ax)
            ax.set(title=f"z_spec=[{group.lower.iloc[0]:g}, {group.upper.iloc[0]:g}), N={group.n_objects.iloc[0]}",
                   ylabel=METRIC_LABELS[metric])
    return fig


def plot_bootstrap_intervals(table: pd.DataFrame, *, title: str) -> Figure:
    """Show paired improvement CIs; no model is retrained for these intervals."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(12, max(3, 0.45 * len(table))), layout="constrained")
    y = np.arange(len(table))
    colors = table.status.map({"candidate": "tab:red", "reference": "tab:blue", "unresolved": "gray"})
    for i, row in enumerate(table.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color=colors.iloc[i])
        ax.scatter(row.improvement, i, color=colors.iloc[i], s=15)
    labels = []
    for name in table.metric:
        parts = name.split("__")
        if len(parts) == 4:
            scope, target, selector, metric = parts
            selector = selector.removeprefix("photoz_").removesuffix("_best")
            labels.append(f"{scope}: |{target}| / {selector} / {metric}")
        else:
            labels.append(name)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Paired improvement (positive favors turboPDZ)", title=title)
    ax.legend(handles=[Line2D([0], [0], color=color, label=label) for color, label in (
        ("tab:red", "turboPDZ favored"), ("tab:blue", "Catalog favored"), ("gray", "Unresolved"))],
        loc="upper right", fontsize=8)
    return fig


def plot_pca_population_hexbins(objects: pd.DataFrame, *, component: int) -> Figure:
    """Compare a low-score tail to all other objects on shared point-estimate axes."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True,
                             layout="constrained")
    low, high = objects[["z_spec", "z_ml"]].to_numpy().min(), objects[["z_spec", "z_ml"]].to_numpy().max()
    for ax, mask, name in zip(axes, (objects["tail"] == "low", objects["tail"] != "low"), ("Low-score tail", "Other objects")):
        group = objects.loc[mask]
        artist = ax.hexbin(group.z_spec, group.z_ml, gridsize=40, mincnt=1,
                           bins="log", extent=(low, high, low, high), rasterized=True)
        fig.colorbar(artist, ax=ax, label="Objects")
        ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
        ax.set(xlabel="z_spec", ylabel="z_ml", title=f"{name}: N={len(group)}")
    fig.suptitle(f"PC{component}: population/point-estimate diagnostic")
    return fig


def plot_pca_population_counts(breakdown: pd.DataFrame, *, min_count: int = 20,
                                investigation_window: tuple[float, float] | None = None) -> Figure:
    """Show test-sample counts and tail-specific outlier rates, without completeness claims."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    partition = breakdown[breakdown.group.isin(["low", "other"])]
    total = partition.groupby(["lower", "upper", "center"], as_index=False).n_objects.sum()
    axes[0].bar(total.center, total.n_objects, width=0.95 * (total.upper - total.lower), color="gray")
    for name, group in partition.groupby("group", sort=False):
        axes[1].plot(group.center, group.outlier_rate.where(group.n_objects >= min_count), marker="o", label=name)
    axes[0].set(ylabel="Spectroscopic holdout objects")
    axes[1].set(xlabel="z_spec", ylabel=f"Outlier fraction (N >= {min_count})")
    axes[1].legend()
    if investigation_window is not None:
        for ax in axes:
            ax.axvspan(*investigation_window, color="red", alpha=0.1)
    return fig
