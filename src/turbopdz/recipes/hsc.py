"""HSC-PDR3 adapters and explicit conversion of trusted reference artifacts."""

from __future__ import annotations
from pathlib import Path
import json
import pickle
import shutil

import numpy as np
import pandas as pd

from ..data import PDZDataset
from ..preprocessing import Preprocessor
from ..calibration import Calibration
from ..artifacts import seal_bundle, write_json, file_sha256


def read_products(directory: str | Path, *, require_homogeneous: bool = True) -> PDZDataset:
    """Read existing 1a data products; verify table/descriptor/PDZ row alignment."""
    root = Path(directory)
    info = json.loads((root / "meta.json").read_text())
    if require_homogeneous and info.get("specz_selection") != "homogeneous_true":
        raise ValueError("The HSC paper recipe requires secure homogeneous-flag spectroscopy")
    table = pd.read_parquet(root / "table_pd.parquet")
    descriptors = pd.read_parquet(root / "descriptors_df.parquet")
    with np.load(root / "pdf_data.npz", allow_pickle=False) as arrays:
        ids, pdf, grid = arrays["row_id"], arrays["pdf_array"], arrays["z_grid"]
    if not np.array_equal(table["row_id"], ids) or not np.array_equal(descriptors["row_id"], ids):
        raise ValueError("HSC table/descriptor/PDZ row order differs")
    # Keep scalar observables and benchmarks, not target columns or nested PDFs.
    columns = [name for name in table if name not in {"PDF", "specz_redshift", "row_id", "Dz_best", "|Dz_best|"}]
    obs = table[columns].reset_index(drop=True).copy()
    for name in descriptors:
        if name != "row_id":
            obs[name] = descriptors[name].to_numpy()
    return PDZDataset(pdf, grid, obs, ids, table["specz_redshift"].to_numpy(),
                      representation="mass", metadata={
                          "adapter": "hsc-pdr3-pdz-ml", "label_column": "specz_redshift",
                          "cat": info["cat"], "layer": info["layer"],
                          "specz_selection": info["specz_selection"],
                          "mag_bounds": [info["i_min"], info["i_max"]],
                      })


def select_secure_spectroscopy(table: pd.DataFrame, *, mag_bounds: tuple[float, float],
                               z_bounds: tuple[float, float] = (0.01, 5.0)) -> pd.DataFrame:
    """Apply the paper's secure-redshift, magnitude, and redshift selection."""
    mask = (table["specz_flag_homogeneous"].eq(True)
            & table["i_cmodel_mag"].between(*mag_bounds)
            & table["specz_redshift"].between(*z_bounds))
    for name in ("i_cmodel_mag", "specz_redshift", "specz_flag_homogeneous"):
        null = f"{name}_isnull"
        if null in table:
            mask &= ~table[null].fillna(True).astype(bool)
    return table.loc[mask].copy()


def _portable_preprocessor(path: Path, info: dict, *, trust_pickle: bool) -> tuple[Preprocessor, dict]:
    if not trust_pickle:
        raise ValueError("Legacy pickle can execute code: pass trust_pickle=True only for your own trusted runs")
    with path.open("rb") as handle:
        original = pickle.load(handle)
    names = original.get("tabular_feature_names", info.get("tabular_feature_names"))
    if names is None:
        descriptors = list(info["desc_cols"])
        extra = original["sc_feats"].n_features_in_ - len(descriptors)
        if extra not in {0, 1}:
            raise ValueError("Cannot infer the legacy feature schema")
        names = descriptors + (["i_cmodel_mag"] if extra else [])
    if len(names) != original["sc_feats"].n_features_in_:
        raise ValueError("Legacy feature names do not match scaler dimensions")
    guards = tuple(original.get("pdf_scaler_guard", {}).get("indices", []))
    pp = Preprocessor(tuple(names), np.zeros(len(names)), original["sc_feats"],
                      original["sc_pdf"], original["pca"], guards)
    descriptor_names = list(info["desc_cols"])
    config = {"descriptor_columns": descriptor_names,
              "extra_columns": [name for name in names if name not in descriptor_names],
              "descriptor_missing": "zero", "pdz_pca": True, "missing": "raise"}
    return pp, config


def export_legacy_pair(project_root: str | Path, spec: dict, destination: str | Path,
                       *, trust_pickle: bool = False, include_applicability: bool = True) -> Path:
    """Export fitted HSC models without retraining or copying catalog extracts.

    Only model weights, numeric preprocessing/calibration/support references,
    selected parameters, and sanitized provenance are copied. Legacy pickles
    are read only with explicit trust and never included in the output bundle.
    """
    from ..training.common import tensorflow
    if not trust_pickle:
        raise ValueError("Explicit trust_pickle=True is required for legacy conversion")
    source, target = Path(project_root), Path(destination)
    if target.exists():
        raise FileExistsError(f"Bundle already exists: {target}")
    zsource = source / "models_optuna" / spec["zml_run"]
    ssource = source / "models_optuna" / spec["sigmaml_run"]
    zinfo = json.loads((zsource / "run_info.json").read_text())
    sinfo = json.loads((ssource / "run_info.json").read_text())
    if sinfo["run_tag_zml"] != zinfo["run_tag"]:
        raise ValueError("sigmaML source does not reference the selected zML run")
    for info in (zinfo, sinfo):
        if any(info.get(key) != spec[key] for key in ("cat", "layer", "seed")) or info.get("specz_selection") != "homogeneous_true":
            raise ValueError("Legacy artifact does not match the frozen paper manifest")
    pdf_path = source / zinfo["data_files"]["pdf_npz"]
    with np.load(pdf_path, allow_pickle=False) as arrays:
        grid = arrays["z_grid"].copy()
    target.mkdir(parents=True, exist_ok=False)
    for track, directory, info in (("zml", zsource, zinfo), ("sigmaml", ssource, sinfo)):
        stage = target / track
        stage.mkdir()
        pp, fcfg = _portable_preprocessor(directory / "preprocessors_final.pkl", info, trust_pickle=trust_pickle)
        pp.save(stage / "preprocessor.npz")
        model = tensorflow().keras.models.load_model(directory / "best_final.keras", compile=False)
        model.save(stage / "model.keras")  # strips optimizer state; does not change fitted weights
        np.savez_compressed(stage / "grid.npz", z_grid=grid)
        write_json(stage / "best_params.json", {"best_params": info["best_params"], "final_epochs": info["final_epochs"]})
        metadata = {"track": track, "representation": "mass", "features": fcfg,
                    "cat": spec["cat"], "layer": spec["layer"], "seed": spec["seed"],
                    "source_run_tag": info["run_tag"], "specz_selection": "homogeneous_true",
                    "source_data_checksums": {key: value for key, value in info.get("data_checksums", {}).items() if key != "data_dir"},
                    "fitted_model_unchanged": True, "legacy_sklearn_version": "1.8.0"}
        if track == "sigmaml":
            with np.load(directory / "calibration_curve.npz", allow_pickle=False) as curve, np.load(directory / "reliability_artifacts.npz", allow_pickle=False) as ranking:
                Calibration(curve["sigma_anchor"], curve["m_values"], ranking["sigma_ref_sorted"], info["epsilon"]).save(stage / "calibration.npz")
            metadata.update(parent_zml_manifest_hash=file_sha256(target / "zml" / "manifest.json"),
                            crossfit_scope=info["crossfit_scope"], protocol=info["protocol"])
        seal_bundle(stage, metadata)
    support = source / "data_products" / "applicability" / f"turbopdz_applicability_{spec['layer']}_homospecz.npz"
    if include_applicability and support.is_file():
        shutil.copy2(support, target / "applicability.npz")
    seal_bundle(target, {"kind": "combined-hsc-pdr3", "cat": spec["cat"], "layer": spec["layer"],
                         "paper_run_tags": {"zml": spec["zml_run"], "sigmaml": spec["sigmaml_run"]},
                         "contains_catalog_extracts": False})
    return target
