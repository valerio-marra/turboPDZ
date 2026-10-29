"""Presentation helpers kept separate from bootstrap calculations."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .bootstrap import paired_bootstrap
from .metrics import photoz_metrics


def improvement_table(summary: pd.DataFrame, *, digits: int = 3) -> pd.DataFrame:
    """Format effects and percentile intervals without assuming symmetric errors."""
    table = summary.copy()
    table["improvement [CI]"] = [f"{row.improvement:.{digits}f} [{row.ci_low:.{digits}f}, {row.ci_high:.{digits}f}]"
                               for row in table.itertuples()]
    return table[[name for name in ("family", "cohort", "metric", "candidate", "reference", "improvement [CI]", "status", "n_objects") if name in table]]


def write_tables(summary: pd.DataFrame, directory: str | Path, *, digits: int = 3) -> None:
    """Write machine-readable CSV plus Markdown and LaTeX comparison tables."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    table = improvement_table(summary, digits=digits)
    summary.to_csv(root / "bootstrap_summary.csv", index=False)
    table.to_csv(root / "bootstrap_display.csv", index=False)
    # Avoid an extra tabulate dependency for Markdown reporting.
    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in table.itertuples(index=False, name=None))
    (root / "bootstrap.md").write_text("\n".join(lines) + "\n")
    (root / "bootstrap.tex").write_text(table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.{digits}f}"))


def compare_point_runs(candidate: pd.DataFrame, reference: pd.DataFrame, **bootstrap_options):
    """Pair different model runs by exact object IDs, never by row position."""
    for frame in (candidate, reference):
        if frame.object_id.duplicated().any():
            raise ValueError("Comparison object IDs must be unique")
    if set(candidate.object_id) != set(reference.object_id):
        raise ValueError("Paired model comparisons require the same object cohort")
    joined = candidate[["object_id", "z_ml", "z_spec"]].merge(
        reference[["object_id", "z_ml", "z_spec"]], on="object_id", validate="one_to_one", suffixes=("_candidate", "_reference"))
    if not np.allclose(joined.z_spec_candidate, joined.z_spec_reference, rtol=0, atol=1e-8):
        raise ValueError("Paired objects have different spectroscopic labels")
    columns = {"candidate": joined.z_ml_candidate.to_numpy(), "reference": joined.z_ml_reference.to_numpy(), "z_spec": joined.z_spec_candidate.to_numpy()}
    comparisons = {metric: (lambda data, metric=metric: (
        photoz_metrics(data["candidate"], data["z_spec"])[metric],
        photoz_metrics(data["reference"], data["z_spec"])[metric]))
        for metric in ("mae", "scatter", "outliers", "absolute_bias")}
    return paired_bootstrap(columns, comparisons, **bootstrap_options)
