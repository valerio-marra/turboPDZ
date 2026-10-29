# HSC-SSP PDR3 recipe

This recipe covers DEmP, DNNz, and Mizuki across Wide and Deep/UltraDeep layers.
`paper-runs.json` identifies the six seed-50 secure-spectroscopy zML/sigmaML pairs.
The recipe's `configs/` directory contains one YAML per pipeline/layer, using the
same schema as the top-level experiment configurations. Each reuses reference
parameters/epochs and specifies fixed calibration settings.

Place compact data products under `data/hsc-pdr3/{cat}_{layer}/`, or edit the
recipe paths. Each directory needs `table_pd.parquet`, `descriptors_df.parquet`,
`pdf_data.npz`, and `meta.json`. Row-ID alignment and the `homogeneous_true`
selection metadata are checked. The joined catalog data are not distributed.

## Frozen model conversion and verification

Conversion is a maintainer operation requiring the trusted source-run workspace:

```bash
python recipes/hsc-pdr3/export_artifacts.py --project-root /path/to/reference-workspace \
  --trust-legacy-pickle
python recipes/hsc-pdr3/validate_artifacts.py --project-root /path/to/reference-workspace
```

Existing bundles are never overwritten. Source pickle is read only with explicit
trust; released preprocessing uses portable NPZ arrays. Model optimizer state is
removed without refitting weights. Applicability references are copied from the
separate layer-specific training-photometry references when available.

Validation checks both computed and stored descriptors on sampled final-test
objects, compares zML/sigmaML/rML against saved outputs, and exports only aggregate
differences. This is a sampled numerical check, not an all-object validation or
model optimization.

## Reference bootstrap tables

`turbopdz.recipes.hsc_bootstrap` retains the paper's exact saved-run loaders,
cohorts, metric names, signed improvements, and LaTeX table formatting. The
notebook `paper-bootstrap.ipynb` runs this recipe given `LEGACY_PROJECT_ROOT` pointing
to the source artifacts and a separate output directory. Source data and model
artifacts are read-only. Point metrics use the 15% development holdout;
reliability comparisons use the 7.5% final test. The published models exclude
these objects from fitting and calibration, but some were used in earlier
hyperparameter searches; see [model provenance](../../docs/hsc-models.md).

Generic workflows use `turbopdz.evaluation.bootstrap` and return per-estimator
intervals as well as paired improvements. They are intended for new experiments;
the frozen recipe is the reference for exact paper tables.

## Hyperparameter-selection sensitivity

`python -m turbopdz.recipes.hsc_selection` recomputes the published search-pool
membership diagnostic from the saved source-run outputs; no model is refitted.
The [aggregate results and command](results/hpo-selection/README.md) include
independent-subset metrics, conditional bootstrap intervals, and provenance.
These checks support the principal scatter-filtering result without asserting
that reused hyperparameters are optimal for the changed targets.
