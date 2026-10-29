# Repository Guidelines

## Structure

Reusable code lives in `src/turbopdz/`; numbered parameterized notebooks are the
scientific interface. General-purpose YAMLs belong in `configs/`; dataset-specific
adapters, instructions, analyses, and YAMLs belong in `recipes/`, with recipe
configurations under `recipes/<dataset>/configs/`. Keep inference bundles separate
from ignored `data/`, `runs/`, and executed notebook outputs. Do not modify external
research workspaces as part of package development.

## Development and validation

Use Python 3.11 and the environment documented in `docs/environment.md`.
Install with `python -m pip install -e '.[train,notebooks,hsc,test]'`, then run
`python -m pytest` and `python -m build`. Execute the synthetic numbered notebook
workflow through Papermill before modifying cluster workflows. Do not run lengthy
HPO or retrain published models as a routine check.

## Scientific contracts

Preserve native PDZ grids, mass/density conventions, object IDs, feature order,
and cohort membership. Fit preprocessing inside training folds. Calibration and
final test remain disjoint. State explicitly that OOF excludes weights/preprocessing
but does not imply nested HPO. Distinguish empirical error scales from posterior
uncertainties and bootstrap test-sample uncertainty from retraining variability.
Feature/objective changes require a compatible new run/study, not silent reuse.

## Style and changes

Use four spaces, snake_case, type hints and NumPy-style public docstrings. Prefer
small tested helpers over duplicated notebook logic. Keep one parameters cell
per notebook and clear source outputs. Documentation describes current behavior,
usage, and scientific limits, not the project's development history.
Keep batch-execution examples site-neutral, without named computing facilities.
Never load untrusted pickle; legacy
conversion requires explicit trust and exports portable arrays. Do not overwrite
released artifacts or commit private catalogs, credentials, or personal paths.
Commits should explain intent and validation; use short imperative prefixes such
as `feat:`, `fix:`, and `docs:`. Publishing/pushing requires explicit authorization.
