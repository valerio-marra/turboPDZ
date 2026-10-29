"""Examples of trusted, importable dataset/feature/objective hooks."""

import numpy as np
import pandas as pd
from turbopdz import PDZDataset


def load_custom_npz(path: str) -> PDZDataset:
    """Adapt a user-created NPZ with native grid, IDs, labels, and magnitudes."""
    with np.load(path, allow_pickle=False) as values:
        observables = pd.DataFrame({"mag_g": values["mag_g"], "mag_r": values["mag_r"],
                                    "catalog_z": values["catalog_z"]})
        return PDZDataset(values["pdz"], values["z_grid"], observables,
                          values["object_id"], values["z_spec"], representation="density")


def observed_color(*, pdz, z_grid, observables) -> pd.DataFrame:
    """A stateless per-object feature: no fitted population statistics or labels."""
    return pd.DataFrame({"g_minus_r": observables["mag_g"] - observables["mag_r"]})


def bias_aware_objective(*, z_pred, z_spec, metrics, references, bias_tolerance=0.01):
    """Example scientific choice, not a claim of universal optimality."""
    return (0.4 * metrics["scatter"] / references["scatter"]
            + 0.4 * metrics["outliers"] / references["outliers"]
            + 0.2 * metrics["absolute_bias"] / bias_tolerance)
