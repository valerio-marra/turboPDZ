import numpy as np
import pandas as pd

from turbopdz.recipes.hsc_bootstrap import (
    _format_ci,
    auc_metric_pairs,
    bootstrap_point_metrics,
    filter_auc_values,
    point_metric_pairs,
    summarize_pairs,
)
from turbopdz.evaluation.metrics import filter_auc


def _toy_reliability_data(n=401):
    x = np.linspace(-1.0, 1.0, n)
    dz_ml = 0.08 * np.sin(7.0 * x) + 0.015 * x
    dz_best = 0.11 * np.cos(5.0 * x) - 0.02 * x
    return {
        "dz_ml": dz_ml,
        "dz_best": dz_best,
        "abs_dz_ml": np.abs(dz_ml),
        "abs_dz_best": np.abs(dz_best),
        "sigma_ml": np.abs(dz_ml) + 0.02 * (x + 1.0),
        "s_best": np.abs(dz_best) + 0.03 * (1.0 - x),
        "r_ml": 1.0 - (x + 1.0) / 2.0,
        "photoz_conf_best": (np.sin(3.0 * x) + 1.0) / 2.0,
        "photoz_risk_best": (np.cos(4.0 * x) + 1.0) / 2.0,
    }


def test_fast_filter_auc_matches_pipeline_reference():
    data = _toy_reliability_data()
    frame = pd.DataFrame(data)
    computed = filter_auc_values(data)
    for selector, direction in (
        ("r_ml", "high"),
        ("photoz_conf_best", "high"),
        ("photoz_risk_best", "low"),
    ):
        for target, residual in (("z_ml", "dz_ml"), ("z_best", "dz_best")):
            areas = filter_auc(frame[residual].to_numpy(), frame[selector].to_numpy(),
                               higher_is_better=direction == "high")
            area_sigma, area_eta = 100*areas["scatter"], 100*areas["outliers"]
            np.testing.assert_allclose(
                computed[(selector, target, "sigma_nmad")], area_sigma
            )
            np.testing.assert_allclose(
                computed[(selector, target, "eta_0_15")], area_eta
            )


def test_point_bootstrap_is_paired_and_reproducible():
    z_spec = np.linspace(0.1, 1.0, 51)
    z_ml = z_spec + 0.01 * np.sin(z_spec)
    z_best = z_spec + 0.04 * np.cos(z_spec)
    data = {
        "z_spec": z_spec,
        "z_ml": z_ml,
        "z_best": z_best,
        "dz_ml": (z_ml - z_spec) / (1.0 + z_spec),
        "dz_best": (z_best - z_spec) / (1.0 + z_spec),
    }
    observed_a, draws_a = bootstrap_point_metrics(data, n_boot=25, seed=123)
    observed_b, draws_b = bootstrap_point_metrics(data, n_boot=25, seed=123)
    assert observed_a == observed_b == point_metric_pairs(data)
    for metric in draws_a:
        np.testing.assert_array_equal(draws_a[metric], draws_b[metric])
        assert np.isfinite(draws_a[metric]).all()
    assert np.all(draws_a["mae"] > 0.0)


def test_summary_uses_positive_is_better_convention():
    observed = {"mae": (0.10, 0.14)}
    supported = np.linspace(0.02, 0.06, 101)
    summary = summarize_pairs(
        cat="demp",
        layer="wide",
        cohort_n=100,
        family="point_estimate",
        observed_pairs=observed,
        draws={"mae": supported},
        ci_level=0.95,
        n_boot=len(supported),
        seed=7,
    )
    row = summary.iloc[0]
    assert np.isclose(row["improvement"], 0.04)
    assert row["status"] == "turbo_supported"


def test_auc_pair_keys_cover_native_and_controlled_comparisons():
    pairs = auc_metric_pairs(_toy_reliability_data())
    assert len(pairs) == 12
    assert "native__photoz_conf_best__sigma_nmad" in pairs
    assert "controlled_z_ml__photoz_risk_best__eta_0_15" in pairs
    assert "controlled_z_best__photoz_conf_best__sigma_nmad" in pairs


def test_latex_highlights_supported_differences_in_either_direction():
    for status, expected_color in (
        ("turbo_supported", "red"),
        ("catalog_supported", "blue"),
        ("unresolved", None),
    ):
        row = pd.Series(
            {"improvement": -0.12, "ci_low": -0.18, "ci_high": -0.06, "status": status}
        )
        rendered = _format_ci(row, digits=2)
        for color in ("red", "blue"):
            assert (rf"\textcolor{{{color}}}{{\textbf{{" in rendered) == (
                color == expected_color
            )
