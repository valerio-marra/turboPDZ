# Customize a dataset or experiment

Copy an example YAML to a new configuration and use a new run directory. Paths
in YAML are relative to the configuration file, including adapter `path` values.

## Configurations and recipes

A configuration is the YAML specification for one experiment. `configs/` holds
general-purpose examples; each specifies the data adapter, features, objective,
splits, training/search budgets, calibration, and bootstrap settings.

A recipe packages a dataset- or study-specific workflow under `recipes/<name>/`:
input requirements, helper scripts, configurations, and any specialized analysis.
For example, the HSC recipe includes six YAMLs under `recipes/hsc-pdr3/configs/`,
artifact conversion/validation tools, and a bootstrap-table notebook. Recipe YAMLs
use the same loader and schema as top-level configurations; there is no separate
recipe configuration format. Both work with `CONFIG_PATH` in the shared notebooks.
Keep a configuration with its recipe when it depends on that recipe's inputs,
helpers, or reproduction conventions. Moving it requires adjusting relative paths.

## Dataset adapters

Built-in adapters are `synthetic`, `portable` (`PDZDataset.save`), and
`hsc_products` (HSC compact Parquet/NPZ products). A custom `module:function`
must return a `PDZDataset`; see `examples/extensions.py` and `examples/custom.yaml`.
Install your extension module, or explicitly list its trusted parent directory
in `python_paths` (relative to the YAML file). Configurations invoking Python
hooks are executable experiment specifications, not safe untrusted input.
Keep survey selection, null/sentinel policies, units, and column mapping in the
adapter, not the training core. Missing IDs and invalid PDZs must be resolved
explicitly while preserving one common row mask.

## Feature choices

```yaml
features:
  descriptor_groups: [location, shape, peaks]
  extra_columns: []                 # PDZ-only experiment
  pdz_pca: true
  missing: raise                    # or fold-fitted median imputation
  sigmaml:
    extra_columns: [i_cmodel_mag]   # independently override this track
```

Use `descriptor_columns` for exact selection/order; `exclude` removes selected
columns. Add magnitudes, observed colors, or other scalar observables by name.
Custom hooks receive PDZs/grid/observables and return a row-aligned DataFrame.
They must be stateless and must not access spectroscopic labels. Population-fit
extensions must be implemented inside fold preprocessing, with an explicit
portable serialization scheme, not hidden inside a feature hook.

Precomputed descriptors require `descriptor_source: observables`. Their missing
policy is explicit: the HSC compatibility profile uses `descriptor_missing: zero`.
Removing a model input at inference is not a supported ablation: refit the model.

## Objective and search choices

```yaml
selection_objective:
  name: normalized_composite
  weights: {scatter: 0.4, outliers: 0.4, absolute_bias: 0.2}
  bias_tolerance: 0.01
training:
  loss: mae
  hpo: {mode: search, trials: 100}
```

Reference scatter/outlier scales come from the catalog estimator on development
rows only, or from explicit `selection_objective.scales`. Bias uses the declared
tolerance, never division by a nearly zero catalog bias. `name: mae` and
`name: custom` are also supported; custom objectives receive validation
predictions, labels, metrics, and reference scales.

The selection objective is **not** the SGD training loss. Its bias weight changes
model selection, not the network's loss automatically.
With fixed/reused parameters, changing that objective does not optimize new
weights against it; use a new search to select parameters under a new objective.
Customize Optuna's `search_space` using int/float/categorical specifications; see
`training/zml.py:DEFAULT_SEARCH_SPACE`. Supply `sample_weight_column` for explicit
non-negative training weights.

## Parameter reuse and paired experiments

`search` starts a new study; `resume` requires the same semantic fingerprint.
`reuse_parameters` reads an exported `params_path` with `best_params` and
`final_epochs`, then fits fresh weights/PCA. `fixed` uses configured parameters
without Optuna. Both tracks support independent `search` and `resume`
modes, with separate study files and parameter/epoch selection.

Resume checks data, folds, features, objective, loss/search settings, and custom
hook source. A changed feature/objective needs a new study. For a paired no-i
experiment, keep the same dataset and point `splits.split_path` at the baseline
run's `splits.npz`, then select `extra_columns: []` in a new run directory.
`compare_point_runs` aligns saved predictions by object ID and rejects changed
cohorts or labels.

Evaluation/bootstrap of transferred runs uses the cached dataset and saved
model provenance, allowing relocated file paths without changing scientific
settings. Training resumption remains strict about its prepared inputs.

`examples/no-i.yaml` demonstrates a paired synthetic ablation with the baseline
split and frozen parameters. For the analogous HSC experiment, copy a pipeline/layer
configuration, remove the magnitude from `features.zml.extra_columns`, reuse that
baseline's saved split, and use a new run directory. The testing notebook offers
optional grouped permutation importance using the selected zML objective and
redshift MAE; sigmaML diagnostics include log-error MAE and frozen reliability
filtering AUCs. Feature blocks follow the saved schema automatically.

## sigmaML Optuna search

```yaml
sigmaml:
  epochs: 150
  loss: mae
  hpo:
    mode: search
    trials: 100
    study_name: sigmaml
```

This performs one non-nested search on the fixed development folds. Its selection
metric is mean held-fold MAE for `log(abs(Delta z_oof) + epsilon)`, before
calibration. Each fold fits fresh scaling/PCA on its training rows. epsilon and
the zML OOF targets are frozen from development data; neither calibration nor
final-test rows enter fitting, scoring, or early stopping. The selected model is
refitted on the whole development pool for the median best epoch count of the
winning trial. Scale calibration and the reliability CDF are fitted only afterward.
sigmaML selection does not alter the configured zML objective.

`sigmaml.hpo.search_space` accepts the same specifications as zML, but its search,
epochs, loss, callbacks, and parameter overrides are independent. Each track
saves `study.sqlite3`, `trials.csv`, and `best_params.json` under its own directory.
sigmaML resume compatibility additionally checks its parent zML bundle and OOF
file. Inference/export needs neither study database.

For an interrupted search, keep the run directory and change that track's mode
to `resume`. `trials` is the total attempted budget; already recorded trials are
not rerun. Failed/pruned trials also consume that budget. Completed sealed models
are not overwritten. Resume preserves trial history, but does not promise the
same sampler trajectory as an uninterrupted search. Use a new run for scientific
changes; HSC configurations reuse the reference models' parameters.

`configs/synthetic-optuna.yaml` exercises both searches with two trials each.
It is the small integration-test tutorial; `configs/synthetic-serious.yaml` provides
a larger convergence experiment. For a fixed-parameter run,
copy it to your experiment YAML and set both training.hpo.mode and
sigmaml.hpo.mode to fixed; use a fresh run directory.

See [workflow.md](workflow.md) for interactive/Papermill execution and safe
reruns, and [analysis.md](analysis.md) for single-catalog PDF/CSV diagnostics.
