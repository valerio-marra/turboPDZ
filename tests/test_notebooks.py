"""Static notebook contracts complement the executed Papermill smoke test."""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "recipes").glob("**/*.ipynb")))
def test_notebooks_are_parameterized_output_free_and_valid(path):
    import nbformat
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    params = [cell for cell in nb.cells if "parameters" in cell.metadata.get("tags", [])]
    assert len(params) == 1
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            compile(cell.source, str(path), "exec")


def test_frozen_hsc_configs_resolve_to_local_artifacts():
    from turbopdz.config import load_config
    configs = sorted((ROOT / "recipes/hsc-pdr3/configs").glob("*.yaml"))
    assert len(configs) == 6
    for path in configs:
        config = load_config(path)
        assert config["seed"] == 50
        assert config["training"]["hpo"]["mode"] == "reuse_parameters"
        for track in ("training", "sigmaml"):
            path = Path(config[track]["params_path"])
            assert path.is_relative_to(ROOT / "artifacts" / "hsc-pdr3")
            assert path.name == "best_params.json"


def test_synthetic_optuna_recipe_runs_a_small_search_for_both_tracks():
    from turbopdz.config import load_config
    config = load_config(ROOT / "configs/synthetic-optuna.yaml")
    for track in ("training", "sigmaml"):
        assert config[track]["epochs"] == 3
        assert config[track]["hpo"]["mode"] == "search"
        assert config[track]["hpo"]["trials"] == 2
        assert "units" in config[track]["hpo"]["search_space"]
