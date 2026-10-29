"""Exercise an installed wheel outside the source checkout."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path


def main() -> None:
    import turbopdz

    location = Path(turbopdz.__file__).resolve()
    if "site-packages" not in location.parts:
        raise RuntimeError(f"Expected an installed wheel, imported {location}")
    if turbopdz.__version__ != importlib.metadata.version("turbopdz"):
        raise RuntimeError("Package and distribution versions disagree")
    for name in ("analysis", "runtime", "evaluation.diagnostics", "evaluation.populations",
                 "recipes.hsc_selection", "training.zml", "training.sigmaml", "workflow"):
        importlib.import_module(f"turbopdz.{name}")
    from turbopdz.cli import main as cli
    from turbopdz.features import feature_table
    from turbopdz.synthetic import make_synthetic_dataset

    data = make_synthetic_dataset(80)
    assert len(feature_table(data, {"extra_columns": ["i_cmodel_mag"]})) == 80
    print(f"Installed turboPDZ {turbopdz.__version__}: {location}")
    try:
        cli(["--help"])
    except SystemExit as result:
        if result.code != 0:
            raise


if __name__ == "__main__":
    main()
