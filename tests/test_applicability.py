import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from turbopdz.applicability import (
    FLAG_COLUMNS,
    ApplicabilityReference,
    add_applicability_flags,
    compute_reference_support_diagnostics,
    fit_applicability_reference,
)


def _synthetic_photometry(n_rows: int = 1_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    i_mag = rng.normal(22.0, 0.55, n_rows)
    colors = rng.multivariate_normal(
        mean=[1.1, 0.55, 0.30, 0.15],
        cov=np.diag([0.12, 0.08, 0.05, 0.04]) ** 2,
        size=n_rows,
    )
    r_mag = i_mag + colors[:, 1]
    g_mag = r_mag + colors[:, 0]
    z_mag = i_mag - colors[:, 2]
    y_mag = z_mag - colors[:, 3]
    return pd.DataFrame(
        {
            "g_cmodel_mag": g_mag,
            "r_cmodel_mag": r_mag,
            "i_cmodel_mag": i_mag,
            "z_cmodel_mag": z_mag,
            "y_cmodel_mag": y_mag,
        }
    )


class ApplicabilityUtilsTests(unittest.TestCase):
    def test_flags_distinguish_magnitude_color_and_invalid_rows(self) -> None:
        training = _synthetic_photometry()
        reference = fit_applicability_reference(
            training,
            mag_bounds=(18.5, 25.0),
            n_neighbors=10,
        )

        ordinary = training.iloc[[0]].copy()
        too_faint = ordinary.copy()
        too_faint.loc[:, reference.mag_columns] += 8.0
        unusual_color = ordinary.copy()
        unusual_color["g_cmodel_mag"] += 20.0
        invalid = ordinary.copy()
        invalid["g_cmodel_mag"] = np.nan
        flagged_null = ordinary.copy()
        flagged_null["g_cmodel_mag"] = 0.0
        catalog = pd.concat(
            [ordinary, too_faint, unusual_color, invalid, flagged_null],
            ignore_index=True,
        )
        catalog["g_cmodel_mag_isnull"] = False
        catalog.loc[4, "g_cmodel_mag_isnull"] = True

        result = add_applicability_flags(catalog, reference)

        self.assertTrue(all(column in result for column in FLAG_COLUMNS))
        self.assertTrue(result.loc[0, "turbopdz_applicability_flag"])
        self.assertFalse(result.loc[1, "turbopdz_in_mag_domain"])
        self.assertTrue(result.loc[1, "turbopdz_in_specz_support_99"])
        self.assertFalse(result.loc[1, "turbopdz_applicability_flag"])
        self.assertTrue(result.loc[2, "turbopdz_in_mag_domain"])
        self.assertFalse(result.loc[2, "turbopdz_in_specz_support_99"])
        self.assertFalse(result.loc[2, "turbopdz_applicability_flag"])
        self.assertTrue(
            np.isnan(result.loc[3, "turbopdz_specz_support_percentile"])
        )
        self.assertFalse(result.loc[3, "turbopdz_applicability_flag"])
        self.assertTrue(
            np.isnan(result.loc[4, "turbopdz_specz_support_percentile"])
        )
        self.assertFalse(result.loc[4, "turbopdz_in_specz_support_99"])

    def test_reference_round_trip_preserves_results(self) -> None:
        training = _synthetic_photometry()
        reference = fit_applicability_reference(
            training,
            mag_bounds=(19.0, 26.0),
            n_neighbors=8,
            max_reference_size=500,
            random_state=50,
            metadata={"layer": "dud"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = reference.save(Path(temporary_directory) / "reference.npz")
            loaded = ApplicabilityReference.load(path)

        before = add_applicability_flags(training.iloc[:50], reference)
        after = add_applicability_flags(training.iloc[:50], loaded)

        self.assertEqual(loaded.summary()["metadata"]["layer"], "dud")
        self.assertEqual(len(loaded.reference_scaled), 500)
        np.testing.assert_allclose(
            before["turbopdz_specz_support_percentile"],
            after["turbopdz_specz_support_percentile"],
            rtol=0.0,
            atol=0.2,
        )
        np.testing.assert_array_equal(
            before["turbopdz_applicability_flag"],
            after["turbopdz_applicability_flag"],
        )

        diagnostics = compute_reference_support_diagnostics(loaded)
        np.testing.assert_allclose(
            np.sort(diagnostics["knn_distance"]),
            loaded.training_distance_sorted,
            rtol=2e-6,
            atol=1e-7,
        )
        self.assertEqual(len(diagnostics), len(loaded.reference_scaled))

    def test_missing_photometry_column_is_rejected(self) -> None:
        training = _synthetic_photometry().drop(columns="y_cmodel_mag")
        with self.assertRaisesRegex(KeyError, "y_cmodel_mag"):
            fit_applicability_reference(training, mag_bounds=(18.5, 25.0))


if __name__ == "__main__":
    unittest.main()
