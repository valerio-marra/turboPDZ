# turboPDZ

**PDZ-derived point estimates and empirical reliability, with a notebook-first workflow.**

turboPDZ compresses redshift probability distributions through PCA, combines them
with selected descriptors/observables, and learns an optimized point estimate
`z_ml`. A second MLP learns an empirical error-scale proxy `sigma_ml` from
cross-fitted residuals; held-out calibration and percentile ranking yield `r_ml`.
`sigma_ml` is not a posterior standard deviation or a guaranteed coverage interval.

The package provides an installable library, parameterized notebooks, six frozen
HSC-PDR3 pipeline bundles, dataset-specific recipes, and automated tests.

## Install

Clone the repository to obtain the notebooks, configurations, and six HSC model
bundles (about 101 MiB). These bundles are tracked directly; Git LFS is not needed.

```bash
git clone https://github.com/valerio-marra/turboPDZ.git
cd turboPDZ
```

Python 3.11 is the tested interpreter. In a dedicated environment, from this checkout:

```bash
micromamba create -n turbopdz -c conda-forge python=3.11 pip -y
micromamba activate turbopdz

python -m pip install \
  -r requirements-tested.txt \
  -e '.[train,notebooks,hsc,test]'

python -m ipykernel install \
  --sys-prefix \
  --name turbopdz \
  --display-name "Python (turboPDZ)"

python -m pytest -q
```

Use `--user` instead of `--sys-prefix` if you use VS Code's Jupyter extension.

For inference only, use `.[inference]`; metrics/bootstrap/applicability need only
the base package. TensorFlow, Optuna, and plotting backends are imported lazily.
See [the tested environment](docs/environment.md) before moving to a cluster.

## Three entry points

### Apply the HSC reference models

```python
from turbopdz import TurboPDZ

pipeline = TurboPDZ.load("artifacts/hsc-pdr3/dnnz-dud")
results = pipeline.predict(
    pdz=pdz_array,
    z_grid=native_z_grid,
    observables=observables,  # includes i_cmodel_mag for the published models
    object_id=object_ids,
    batch_size=4096,
)
```

The output preserves object order and includes `z_ml`, `sigma_ml`, and `r_ml`.
Add `applicability=True` when supplying the reference's required grizy photometry.
See [inference](docs/inference.md) and `notebooks/0-inference.ipynb`.

### Reproduce the HSC comparisons

The [HSC recipe](recipes/hsc-pdr3/README.md) identifies the exact six fitted
pipeline pairs and keeps the paper's bootstrap/cohort conventions separate from
generic workflows. Frozen models reproduce inference without another HPO run.
Catalog extracts are not distributed; reproduction requires your HSC data products.
[Model provenance and validation limits](docs/hsc-models.md) distinguish frozen
inference, the published analysis, and new training runs.

### Train on another dataset

Run the small synthetic tutorial first, from the checkout root:

```bash
bash examples/run-notebooks.sh configs/synthetic-optuna.yaml runs/synthetic-optuna-01
```

This executes all seven notebooks through Papermill, including both Optuna
searches. Executed copies go to RUN_DIR/executed/, figures to RUN_DIR/plots/
as PDFs, and tables to RUN_DIR/tables/ as CSVs. For an existing completed run:

```bash
python -m papermill notebooks/3-testing.ipynb runs/synthetic-optuna-01/executed/3-testing-updated.ipynb \
  -k turbopdz -p CONFIG_PATH configs/synthetic-optuna.yaml \
  -p RUN_DIR runs/synthetic-optuna-01 --cwd . --log-output
```

The tutorial is a software smoke test, not an astrophysical performance benchmark.
See [customization](docs/customization.md) for adapters, optional magnitudes/colors,
custom objectives, weights, and model-specific feature lists.

For a larger CPU-sized convergence experiment, use an unused run directory:

```bash
bash examples/run-notebooks.sh configs/synthetic-serious.yaml runs/synthetic-serious-01
```

This uses 8,000 objects, five folds, 15 attempted trials per network, up to 120
epochs per fit, and 2,000 paired-bootstrap resamples. Early stopping/pruning can
reduce the actual training. The synthetic generator's PDZ width and magnitude
are independent of the injected redshift noise, so larger budgets do not imply
an information-driven zML improvement over the catalog estimate. See the
[larger synthetic experiment notes](docs/workflow.md#larger-synthetic-experiment).

The canonical synthetic-optuna.yaml example runs two trials per track.
zML uses its configured objective; sigmaML minimizes
development-fold MAE on log-OOF errors, then refits before held-out calibration.
Each track saves a separate resumable study; no nested hyperparameter searches
are performed. HSC configurations reuse the reference models' parameters.

**Best practice:** edit a dataset configuration or a source notebook, execute it,
and inspect the executed copy. New data/features/objectives/training settings
require a new run directory; completed models are never overwritten. Plotting
and bootstrap changes do not require retraining. In an executed notebook,
Papermill's injected cell overrides the original parameters cell.
See [the full workflow and rerun guide](docs/workflow.md) and
[the single-catalog diagnostic inventory](docs/analysis.md).
The optional `4-pca-population-check.ipynb` provides the focused PCA/competing-PDZ
investigation with dataset-specific windows, without another training run.

## Configurations and recipes

`configs/` contains general-purpose experiment YAML files, including the synthetic
examples. A configuration specifies inputs, features, objectives, splits, training
budgets, calibration, and bootstrap settings for the shared notebook workflow.

`recipes/` groups dataset- or study-specific workflows: input requirements,
adapters or helper scripts, reproduction conventions, and their configurations.
For example, `recipes/hsc-pdr3/configs/` contains the six HSC pipeline/layer YAMLs.
These use the same configuration schema as `configs/`; their location keeps them
with the HSC instructions and tools. See [customization](docs/customization.md).

## Scientific safeguards

- Declare PDZ probability **mass** versus sampled **density** and retain the native grid.
- Save unique IDs, data fingerprints, split membership, seeds, and software versions.
- Fit scaling/PCA inside every training fold; use frozen parameters for OOF targets.
- Keep development, scale-calibration, and final-test cohorts disjoint.
- Distinguish the network training loss from the Optuna selection objective.
- Bootstrap paired objects on saved predictions; report the evaluation cohort.
- Reject mismatched grids, feature schemas, artifact hashes, and incompatible studies.

The default protocol is 85% development, 7.5% calibration, and 7.5% final test.
Cross-fitting excludes each object's fold from weights/preprocessing, but is not
nested HPO. A new PDZ dataset requires its own fitted preprocessing and models;
the HSC models are not validated for another survey. Applicability flags assess
observed photometric support, not guaranteed photo-z correctness.

## Contributing

```bash
python -m pytest
python -m build
```

Tests include small CPU training, OOF membership, Optuna compatibility, portable
model reloads, numerical guards, and paired bootstrap. See [architecture](docs/architecture.md)
and [the release checklist](docs/release-checklist.md). The
[release validation report](docs/release-validation.md) records the checks for this version. Test coverage and
model-reproducibility limits are described in [validation](docs/validation.md).

## Reference and availability

Duarte & Marra, *Optimizing the extraction of information from redshift probability
distribution functions*, accepted for publication in *Publications of the
Astronomical Society of Japan* ([arXiv:2607.26822](https://arxiv.org/abs/2607.26822)).

The full HSC-SSP PDR3 value-added catalog containing `z_ml`, `sigma_ml`, `r_ml`,
and applicability flags will be made available after publication. Model bundles
are separate from the Python wheel and contain no raw catalog extracts.
