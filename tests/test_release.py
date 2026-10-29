"""Release metadata and frozen-model identity checks without training."""

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from turbopdz import __version__
from turbopdz.artifacts import file_sha256, load_manifest

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "artifacts/hsc-pdr3"
SPECS = json.loads((ROOT / "recipes/hsc-pdr3/paper-runs.json").read_text())


def test_release_versions_and_citation_agree():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert metadata["project"]["version"] == citation["version"] == __version__
    assert citation["preferred-citation"]["journal"] == "Publications of the Astronomical Society of Japan"


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: f'{spec["cat"]}-{spec["layer"]}')
def test_frozen_hsc_bundles_match_reference_runs_and_hashes(spec):
    bundle = BUNDLES / f'{spec["cat"]}-{spec["layer"]}'
    if not bundle.exists():
        pytest.skip("Model binaries are separate from Python distributions")
    combined = load_manifest(bundle)
    zml = load_manifest(bundle / "zml")
    sigma = load_manifest(bundle / "sigmaml")
    assert combined["paper_run_tags"] == {"zml": spec["zml_run"], "sigmaml": spec["sigmaml_run"]}
    assert zml["source_run_tag"] == spec["zml_run"]
    assert sigma["source_run_tag"] == spec["sigmaml_run"]
    assert sigma["parent_zml_manifest_hash"] == file_sha256(bundle / "zml/manifest.json")
    assert set(zml["files"]) == {"model.keras", "preprocessor.npz", "grid.npz", "best_params.json"}
    assert set(sigma["files"]) == {*zml["files"], "calibration.npz"}


def test_hsc_distribution_index_is_complete():
    if not (BUNDLES / "SHA256SUMS").exists():
        pytest.skip("Model binaries are separate from Python distributions")
    indexed = {}
    for line in (BUNDLES / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (BUNDLES / relative).resolve()
        assert path.is_relative_to(BUNDLES.resolve())
        assert file_sha256(path) == digest
        indexed[relative] = digest
    actual = {path.relative_to(BUNDLES).as_posix() for path in BUNDLES.rglob("*")
              if path.is_file() and path.suffix in {".json", ".npz", ".keras"}}
    assert actual == set(indexed)


def test_selection_diagnostic_uses_the_frozen_pairs():
    path = ROOT / "recipes/hsc-pdr3/results/hpo-selection/provenance.json"
    cases = json.loads(path.read_text())
    expected = {(spec["cat"], spec["layer"]): spec for spec in SPECS}
    assert len(cases) == len(expected) == 6
    for case in cases:
        spec = expected[(case["cat"], case["layer"])]
        assert (case["zml_run"], case["sigmaml_run"]) == (spec["zml_run"], spec["sigmaml_run"])
        assert 0 < case["n_joint_hpo_unseen_reliability"] <= case["n_sigma_hpo_unseen_reliability"] < case["n_reliability_test"]
