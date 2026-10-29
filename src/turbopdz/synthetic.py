"""Small synthetic fixtures for tutorials/CI, not astrophysical benchmarks."""

from __future__ import annotations
import numpy as np
import pandas as pd

from .data import PDZDataset


def make_synthetic_dataset(n_objects: int = 240, n_grid: int = 48, seed: int = 50) -> PDZDataset:
    """Make reproducible normalized PDZs, imperfect catalog estimates, and grizy."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 2.5, n_objects)
    best = np.clip(truth + rng.normal(0, 0.12, n_objects), 0, 3)
    grid = np.linspace(0, 3, n_grid)
    width = rng.uniform(0.04, 0.2, n_objects)
    pdz = np.exp(-0.5 * ((grid[None, :] - best[:, None]) / width[:, None]) ** 2)
    pdz /= pdz.sum(axis=1, keepdims=True)
    i = rng.uniform(19, 25, n_objects)
    obs = pd.DataFrame({"photoz_best": best, "i_cmodel_mag": i,
                        "g_cmodel_mag": i + 1.4 + rng.normal(0, 0.2, n_objects),
                        "r_cmodel_mag": i + 0.5 + rng.normal(0, 0.1, n_objects),
                        "z_cmodel_mag": i - 0.3 + rng.normal(0, 0.1, n_objects),
                        "y_cmodel_mag": i - 0.45 + rng.normal(0, 0.1, n_objects),
                        "photoz_conf_best": 1 / (1 + width), "photoz_risk_best": width,
                        "s_best": width})
    return PDZDataset(pdz, grid, obs, np.arange(n_objects), truth,
                      metadata={"synthetic": True, "not_for_scientific_claims": True})
