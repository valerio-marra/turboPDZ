"""Read-only regression checks against local HSC reference predictions."""

from __future__ import annotations
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd

from ..inference import TurboPDZ


def read_npz_rows(path: str | Path, key: str, indices: np.ndarray) -> np.ndarray:
    """Read selected rows of a large compressed NPY matrix with bounded memory."""
    positions = np.asarray(indices, dtype=int)
    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported NPY header version")
        if fortran or dtype.hasobject or len(shape) != 2:
            raise ValueError("Expected a C-order numeric PDZ matrix")
        if (positions < 0).any() or (positions >= shape[0]).any():
            raise IndexError("Requested PDZ row outside matrix")
        start = stream.tell()
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        result = np.empty((len(positions), *shape[1:]), dtype=dtype)
        for output_idx in np.argsort(positions, kind="stable"):
            stream.seek(start + int(positions[output_idx]) * row_bytes)
            result[output_idx] = np.frombuffer(stream.read(row_bytes), dtype=dtype).reshape(shape[1:])
    return result


def validate_legacy_pair(project_root: str | Path, spec: dict, bundle: str | Path,
                         *, sample_size: int = 256) -> dict:
    """Compare converted and original final-test outputs, saving no object rows."""
    root = Path(project_root)
    zroot = root / "models_optuna" / spec["zml_run"]
    sroot = root / "models_optuna" / spec["sigmaml_run"]
    zinfo = json.loads((zroot / "run_info.json").read_text())
    with np.load(sroot / "split_idx.npz", allow_pickle=False) as arrays:
        all_idx = arrays["test_idx"].copy()
        n_calibration = len(arrays["calib_idx"])
    rng = np.random.default_rng(50)
    selected = np.sort(rng.choice(len(all_idx), min(sample_size, len(all_idx)), replace=False))
    idx = all_idx[selected]
    pdf_file = root / zinfo["data_files"]["pdf_npz"]
    pdf = read_npz_rows(pdf_file, "pdf_array", idx)
    with np.load(pdf_file, allow_pickle=False) as arrays:
        grid, ids = arrays["z_grid"].copy(), arrays["row_id"][idx]
    table = pd.read_parquet(root / zinfo["data_files"]["table_parquet"], columns=["row_id", "i_cmodel_mag"])
    descriptors = pd.read_parquet(root / zinfo["data_files"]["descriptors_parquet"])
    if not np.array_equal(table.row_id.to_numpy()[idx], ids) or not np.array_equal(descriptors.row_id.to_numpy()[idx], ids):
        raise ValueError("Reference inputs are not row aligned")
    pipeline = TurboPDZ.load(bundle)
    obs = table.iloc[idx].reset_index(drop=True)
    precomputed = pipeline.predict(pdf, grid, obs, object_id=ids,
                                    precomputed_descriptors=descriptors.iloc[idx].reset_index(drop=True))
    computed = pipeline.predict(pdf, grid, obs, object_id=ids)
    with np.load(sroot / "test_outputs.npz", allow_pickle=False) as arrays, np.load(sroot / "reliability_artifacts.npz", allow_pickle=False) as ranking:
        expected = {"z_ml": arrays["zml_pred_test"][selected],
                    "sigma_ml": arrays["sigma_ml_test"][selected], "r_ml": ranking["rml_test"][selected]}
        reference = ranking["sigma_ref_sorted"].copy()
    np.testing.assert_array_equal(pipeline.calibration.sigma_reference, reference)
    original_ranking = 1 - np.searchsorted(reference, expected["sigma_ml"], side="right") / len(reference)
    np.testing.assert_array_equal(original_ranking, expected["r_ml"])
    errors = {}
    for mode, result in (("precomputed", precomputed), ("computed", computed)):
        for name, values in expected.items():
            if name != "r_ml":
                np.testing.assert_allclose(result[name], values, rtol=1e-4 if name == "sigma_ml" else 2e-5,
                                           atol=5e-6, err_msg=f"{spec['cat']}-{spec['layer']} {mode} {name}")
            else:
                # A discontinuous ECDF is checked by its saved definition,
                # not an arbitrary relaxed percentile tolerance. Retain ties
                # and report direct percentile differences separately.
                reconstructed = 1 - np.searchsorted(reference, result["sigma_ml"], side="right") / len(reference)
                np.testing.assert_array_equal(result[name], reconstructed)
            errors[f"max_abs_error_{mode}_{name}"] = float(np.max(np.abs(result[name].to_numpy() - values)))
        errors[f"r_ml_within_five_reference_ranks_{mode}"] = bool(np.max(np.abs(result["r_ml"].to_numpy() - expected["r_ml"])) <= 5/n_calibration)
    return {"cat": spec["cat"], "layer": spec["layer"], "n_sampled_final_test": len(idx),
            "n_final_test": len(all_idx), "n_calibration": n_calibration,
            "passed": True, "scope": "sampled zML/scale numerical equivalence and exact frozen-CDF consistency; no retraining",
            "r_ml_bitwise_reproduction_claimed": False,
            "max_reference_cdf_atom": float(np.unique(reference, return_counts=True)[1].max()/len(reference)), **errors}
