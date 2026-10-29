# Architecture

The numbered notebooks are the scientific interface. `workflow.run_stage`
implements the same stages for notebooks, CLI, and cluster jobs. Shared routines
live under `src/turbopdz`; survey-specific ingestion and reference comparisons live
under `recipes`. General experiment YAMLs live in `configs`; dataset-specific
YAMLs can be kept with their recipe under `recipes/<dataset>/configs`.

## Data and features

`PDZDataset` preserves raw PDZ amplitudes, the native grid, object IDs, observables,
and optional labels. Density inputs are integrated with trapezoidal grid weights
for descriptor moments; discrete masses are normalized by their sum. No input is
silently regridded. Native density heights define density modes/peaks.

Features are selected before fold fitting, but only stateless per-object
calculations happen globally. Scalers, median imputation, and PCA are fitted on
each fold's training rows. Float32-underflowing scaler denominators become unit
scale, and transformed inputs must remain finite. The feature order and missing
policy are stored with each model.

## Training and calibration

Development folds are fixed across Optuna trials. The final zML fits all
development rows for the selected/frozen epoch budget. OOF models fit the
complement of each development fold with fresh preprocessing and frozen
hyperparameters/epochs. Their residuals train the log-error network.

zML and sigmaML can each run one independent Optuna search on these development
folds. zML uses its configured point-estimate objective; sigmaML minimizes
held-fold MAE on the frozen log-OOF-error targets. Both refit preprocessing and
weights on the full development pool for the winning trial's median best epoch
count. sigmaML's study is tied to the exact parent zML bundle and OOF file.
Held-out calibration is never an Optuna validation set.

Hyperparameters may have been selected using the development pool: the default
scope is **weights/preprocessing cross-fitting, not fully nested optimization**.
Calibration/test objects never enter those fits. Scale calibration uses
equal-count bins, multiplicative median factors, and optional log-scale smoothing.
The generic implementation uses configured calibration settings; HSC reference
bundles retain their fitted curves and CDFs.

## Evaluation

The point holdout is development's complement (calibration + final test).
Reliability comparisons use only the independent final test. Both cohort sizes
are recorded; they must not be treated as interchangeable. Controlled selectors
share a residual target and object mask. Inclusive quantile cuts retain ties;
AUC uses the actual retained fractions.

Bootstrap samples the same objects for every metric/model in a cohort and reports
candidate/reference intervals plus paired improvements. Optional group labels
support group-level resampling. Intervals condition on fitted models; training
variability, spectroscopy selection effects, and survey transfer are not included.

## Artifacts and dependencies

Inference loads TensorFlow only when a model is requested, and needs no Optuna
database. Preprocessing/calibration/support arrays use pickle-free NPZ; manifests
hash every payload and record schemas and software versions. Hashes detect
accidental changes, not authenticity. Load bundles and extension code only from
trusted sources. Explicitly trusted legacy pickle is allowed solely during
conversion, never in released inference bundles.
