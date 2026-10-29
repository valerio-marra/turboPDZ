"""Versioned, hash-verified artifacts with explicit software provenance."""

from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.metadata
import json


def file_sha256(path: str | Path) -> str:
    """Hash a file incrementally, including large model files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, values: dict) -> None:
    """Write portable metadata, rejecting non-finite numbers."""
    Path(path).write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def software_versions() -> dict[str, str]:
    """Record installed numerical/training versions without importing backends."""
    result = {}
    for name in ("turbopdz", "numpy", "scipy", "pandas", "scikit-learn", "tensorflow", "optuna"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def seal_bundle(directory: str | Path, manifest: dict) -> None:
    """Record hashes of every payload; manifests never include absolute paths."""
    root = Path(directory)
    payload = dict(manifest, bundle_version=1, software=software_versions())
    payload["files"] = {str(path.relative_to(root)): file_sha256(path)
                        for path in sorted(root.rglob("*")) if path.is_file() and path != root / "manifest.json"}
    write_json(root / "manifest.json", payload)


def load_manifest(directory: str | Path, *, verify: bool = True) -> dict:
    """Reject unsupported schemas, unsafe paths, and changed bundle contents."""
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("bundle_version") != 1:
        raise ValueError("Unsupported model bundle version")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact paths must remain within their bundle")
        if verify and (not path.is_file() or file_sha256(path) != expected):
            raise ValueError(f"Missing or changed artifact: {relative}")
    return manifest
