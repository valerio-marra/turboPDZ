# Validation and numerical reproducibility

Use the [tested environment](environment.md) and run checks from the checkout root:

```bash
python -m pytest -q
python -m build
```

## Automated test coverage

Tests cover native-grid mass/density semantics, unique object IDs, feature order,
null/sentinel handling, common evaluation masks, median bias, empty/inclusive bins,
and numerical preprocessing guards. Training tests exercise small CPU fits,
development-only CV/OOF membership, both Optuna searches and compatible resumes,
independent calibration, parameter reuse, and model reload/export.

Analysis tests cover paired bootstrap, ID-aligned model comparisons, PCA
back-projection and permutation ordering, competing-PDZ branch probabilities,
missing training histories, and physically bounded filtering-plot axes.
Static notebook checks require one parameters cell, valid Python, and no saved
source outputs. Test results apply to the environment used to run them.

Source distributions include notebooks, configurations, recipes, documentation,
and the runner. Model binaries, private catalogs, and generated runs are excluded.
Validate an installed wheel outside the checkout as well as an editable install;
follow the [release checklist](release-checklist.md) before distributing artifacts.

## Synthetic workflow checks

Both synthetic presets execute the seven-stage Papermill workflow, including
separate zML and sigmaML searches, OOF targets, held-out calibration, evaluation,
paired bootstrap, PDF/CSV diagnostics, and independent applicability flags.

- `configs/synthetic-optuna.yaml`: 160 objects, three epochs, two attempted trials
  per network, and 50 bootstrap resamples; a small integration check.
- `configs/synthetic-serious.yaml`: 8,000 objects, five folds, 15 attempted trials
  per network, up to 120 epochs with early stopping, and 2,000 bootstrap resamples;
  a larger convergence check. See [workflow.md](workflow.md).

The Gaussian toy PDZs are centered on noisy catalog estimates. Their widths and
magnitudes are independent of the injected errors. sigmaML can learn the redshift
dependence of normalized errors, but these checks do not establish recovery of
realistic PDZ degeneracies, optimal hyperparameters, or survey performance.
Bootstrap intervals condition on saved models and exclude training variability.

CPU checks do not certify GPU or batch-environment compatibility. Validate devices,
versions, inputs, splits, and reload tolerances in the target environment before
production training. Remote CI results require a completed CI run.

## HSC reference checks

The clean-wheel CPU validation compares each of six zML/sigmaML pairs on 256
sampled final-test objects (1,536 total), using computed and stored descriptors.
Maximum absolute differences with computed descriptors against source outputs are:

| Pipeline/layer | zML | sigmaML | rML |
| --- | ---: | ---: | ---: |
| DEmP-Wide | 8.46e-06 | 1.01e-06 | 2.33e-05 |
| DEmP-DUD | 3.58e-06 | 1.22e-07 | 0 |
| DNNz-Wide | 3.58e-07 | 3.13e-08 | 2.46e-05 |
| DNNz-DUD | 5.96e-07 | 6.41e-08 | 9.62e-05 |
| Mizuki-Wide | 2.15e-05 | 1.26e-07 | 0.0128 |
| Mizuki-DUD | 1.4e-06 | 2.41e-07 | 9.61e-05 |

All pairs pass the sampled zML/error-scale numerical checks and exact
preservation of the frozen reference CDF and ranking definition. This is not
an all-object audit or a claim of bitwise rML reproduction. The largest observed
percentile difference occurs at a Mizuki-Wide CDF atom containing 536 tied
reference values: a scale change of approximately 5.3e-9 crosses the atom and
changes the percentile by 536/41,745 = 0.01284. The frozen rule is preserved;
the validation reports this discrepancy instead of silently widening a tolerance.
Other bundles can also contain CDF ties. See [inference.md](inference.md).

The [release validation report](../recipes/hsc-pdr3/results/bundle-validation-0.1.0.json)
records the clean-install environment, aggregate differences, and counts without
catalog rows. The original conversion report is retained separately in
[`artifacts/hsc-pdr3/validation.json`](../artifacts/hsc-pdr3/validation.json).
Model bundles are tracked under `artifacts/hsc-pdr3/`, separately from the Python
wheel. Their [provenance and license scope](hsc-models.md) accompany the bundles.

To repeat the sampled bundle comparison with the required source-run data:

```bash
python recipes/hsc-pdr3/validate_artifacts.py --project-root /path/to/reference-workspace \
  --sample-size 256 --report /path/to/new-validation.json
```

Sampled bundle agreement is not a full-catalog audit. Scientific resampling
budgets and production-environment checks must be chosen for the intended use.
