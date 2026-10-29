"""Verify that built distributions contain the current library and workflow."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def check_distribution(root: Path, wheel: Path, sdist: Path) -> None:
    """Compare packaged Python sources and check source-distribution assets."""
    sources = sorted((root / "src/turbopdz").rglob("*.py"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for path in sources:
            relative = path.relative_to(root / "src").as_posix()
            if archive.read(relative) != path.read_bytes():
                raise ValueError(f"Wheel differs from source: {relative}")
        if any(name.startswith(("artifacts/", "data/", "runs/")) for name in names):
            raise ValueError("Wheel must not contain models, catalogs, or generated runs")
    with tarfile.open(sdist) as archive:
        members = {member.name.partition("/")[2]: member for member in archive.getmembers()}
        paths = [root / "README.md", root / "LICENSE", root / "CITATION.cff"]
        for directory in ("notebooks", "configs", "recipes", "docs", "examples", "cluster", "environments", "tools", "tests"):
            paths.extend(path for path in (root / directory).rglob("*")
                         if path.is_file() and path.suffix in {".py", ".ipynb", ".yaml", ".json", ".md", ".csv", ".txt", ".sh", ".slurm"})
        for path in [*sources, *paths]:
            relative = path.relative_to(root).as_posix()
            if relative not in members:
                raise ValueError(f"Source distribution is missing {relative}")
            handle = archive.extractfile(members[relative])
            if handle is None or handle.read() != path.read_bytes():
                raise ValueError(f"Source distribution differs from {relative}")
        if any(name.startswith(("artifacts/", "data/", "runs/")) for name in members):
            raise ValueError("Source distribution must not contain private or model data")
    from email.parser import Parser
    with zipfile.ZipFile(wheel) as archive:
        info, = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(info).decode())
        for field in ("Name", "Version", "Requires-Python", "License-Expression"):
            if not metadata[field]:
                raise ValueError(f"Wheel metadata is missing {field}")
        if metadata["License-Expression"] != "MIT":
            raise ValueError("Unexpected package license metadata")
    print(f"Verified {len(sources)} Python modules, workflow assets and package metadata")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    check_distribution(args.root, args.wheel, args.sdist)


if __name__ == "__main__":
    main()
