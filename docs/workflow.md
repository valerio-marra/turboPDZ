# Notebook-first workflow

## One experiment, one configuration, one run directory

Install the package into the same environment used by the notebook kernel:

~~~bash
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
~~~
Use `--user` instead of `--sys-prefix` if you use VS Code's Jupyter extension.

Run commands from the checkout root. Copy a general-purpose configuration from
`configs/`, or a dataset-specific YAML from `recipes/<dataset>/configs/`, to an
experiment configuration. Set inputs, feature groups, objective, seed, and training
budgets there. Paths resolve relative to the YAML file, so adjust them if moving
the configuration. Share the numbered notebook templates across catalogs.

The default protocol is 85% development, 7.5% calibration, and 7.5% independent
final test. There is one non-nested Optuna search per network. Development OOF
predictions exclude each object's fold from weights and preprocessing, not from
the preceding hyperparameter selection.

## Complete terminal run

The checked-in script executes the same seven-notebook loop as manual Papermill:

~~~bash
bash examples/run-notebooks.sh configs/synthetic-optuna.yaml runs/synthetic-optuna-01
~~~

Choose an unused directory (for example synthetic-optuna-02 if 01 exists).
The script stops at the first failure and refuses to overwrite an existing run.
It does not install packages, request cluster resources, or publish results.
Set KERNEL_NAME if your registered kernel has another name.

The equivalent loop is:

~~~bash
(
  set -e
  CONFIG_PATH="configs/synthetic-optuna.yaml"
  RUN_DIR="runs/synthetic-optuna-01"
  if [ -e "$RUN_DIR" ]; then
    echo "Choose a new RUN_DIR: $RUN_DIR already exists."
    exit 1
  fi
  mkdir -p "$RUN_DIR/executed"
  for notebook in \
    1-prepare-data \
    2a-training-zml \
    2b-oof-zml \
    2c-training-sigmaml \
    3-testing \
    4-comparison-bootstrap \
    5-applicability-flag
  do
    python -m papermill "notebooks/$notebook.ipynb" \
      "$RUN_DIR/executed/$notebook.ipynb" \
      -k turbopdz -p CONFIG_PATH "$CONFIG_PATH" -p RUN_DIR "$RUN_DIR" \
      --cwd . --log-output
  done
)
~~~

The tutorial uses 160 objects, three epochs, two trials per network, and 50
bootstrap resamples. It validates execution, not astrophysical performance.
Increase those budgets in a copied configuration for a meaningful experiment.
Fixed training is still available by setting both HPO modes to fixed.

## Larger synthetic experiment

```bash
bash examples/run-notebooks.sh configs/synthetic-serious.yaml runs/synthetic-serious-01
```

The larger preset contains 8,000 objects on a 96-bin native PDZ grid, with
6,800 development, 600 calibration, and 600 independent final-test objects.
Point-estimate comparisons use the 1,200-object calibration-plus-test holdout;
reliability comparisons use only the 600-object final test. Each network gets
15 attempted Optuna trials, five development folds, a 120-epoch ceiling with
early stopping, and a final refit at the selected epoch count. Bootstrap uses
2,000 paired resamples and 95% intervals conditional on the saved models.

Keep its interpretation distinct from a survey benchmark. The toy
generator has Gaussian PDZs centered on a noisy catalog estimate; PDZ width,
magnitude, and colors are independent of the injected redshift noise. zML should
learn a sensible redshift mapping, but substantial gains over that estimate are
not expected. sigmaML can learn the normalized-error dependence on redshift,
whereas the catalog width is not an informative error proxy by construction.
Its reliability advantage therefore tests learning behavior, not real photo-z
performance. No final-test result is used to select or retune the preset.

For CPU execution, optional shell-level thread limits are
`TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4`. They bound CPU parallelism, not scientific settings.
Use the same seven-stage runner and inspection/rerun rules as the small tutorial;
pick a new directory if the example run already exists.

## Interactive use

Start Jupyter Lab; select the registered kernel. In each **source** notebook,
set CONFIG_PATH and RUN_DIR consistently, then execute in numbered order.
Notebooks 1, 2a, 2b, 2c, 3, 4, and 5 export their single-catalog diagnostics
automatically. SHOW_PLOTS=False skips inline display, not PDF/CSV export.
Notebook 3's importance repetitions, number of displayed PCs, bins and
example PDZs are explicit parameters. RUN_IMPORTANCE=False skips expensive
permutation inference, with the reason recorded in the output index.

Executed notebooks are experiment records. They contain an additional
injected-parameters cell; changing only the original parameters cell does not
override injected values. Edit source/configuration files and create a new
executed copy. Exploratory cells can be tried in a copy of an executed notebook.
Restart the kernel after changing imported library code.

## Safe changes and reruns

| Change | What to rerun |
| --- | --- |
| Labels, style, diagnostic bins/examples, importance repetitions | Diagnostic cells or notebook 3; no training |
| Bootstrap resamples/confidence level | Notebook 4 on saved predictions |
| Applicability plots | Diagnostic cell in notebook 5; no refit |
| Dataset, seed, splits, model features, training loss/objective, epoch/search settings | New configuration and new run directory; rerun dependent stages |
| Interrupted HPO, unchanged scientific inputs | Set that track's mode to resume, then rerun only the failed stage and downstream stages |

The trials setting is the total attempted resume budget. A completed sealed model
cannot be resumed/overwritten merely to add trials: start a new experiment. Do not
delete model artifacts to bypass compatibility checks. Full-loop execution is for
new runs, not for regenerating figures in completed runs.

For an existing completed synthetic run, regenerate testing/analysis with:

~~~bash
python -m papermill notebooks/3-testing.ipynb \
  runs/synthetic-optuna-01/executed/3-testing-diagnostics.ipynb \
  -k turbopdz -p CONFIG_PATH configs/synthetic-optuna.yaml \
  -p RUN_DIR runs/synthetic-optuna-01 --cwd . --log-output
~~~

For plot-only regeneration from saved prediction files:

~~~python
from turbopdz.analysis import write_evaluation_diagnostics
write_evaluation_diagnostics("runs/synthetic-optuna-01", show=False)
~~~

These operations overwrite derived PDF/CSV reports, not models. Preserve copies
of reports/executed notebooks if comparing styles or analysis settings. Analysis
category outputs.csv files enumerate this execution; files from earlier
settings may remain in the directory but are not part of its current index.
Models without saved loss histories are reported explicitly; diagnostics do not
invent curves or retrain networks to reconstruct them.

For the optional focused PCA/population investigation, execute
notebooks/4-pca-population-check.ipynb after notebook 3. Choose COMPONENT and
prespecified redshift/magnitude cells; set BRANCHES only when you have meaningful
true-redshift, predicted-redshift and PDZ windows. This reuses saved models and
predictions. Branch windows must be appropriate for the dataset and are not
universal defaults. This notebook is not part of the automatic seven-stage run.
For nested Papermill values, use a parameter YAML file with -f (or -y for an
inline YAML mapping); -p BRANCHES '{...}' passes a string, not a mapping.

## Outputs and compute

- executed/: completed notebook copies, including inline figures.
- plots/category/: PDF figures.
- tables/category/: CSV data, summaries and output indexes.
- zml/, sigmaml/: fitted artifacts and independent Optuna studies.
- dataset/, splits.npz, oof.npz: persistent inputs and cohort membership.
- Root prediction, retained-curve and bootstrap files remain available.

TensorFlow uses eligible GPU operations when a compatible GPU is visible; PCA,
catalog calculations and bootstrap remain CPU based. Check visibility in the
kernel environment:

~~~python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
~~~

On a cluster, request a GPU and configure its TensorFlow/CUDA environment; see
cluster/README.md. GPU execution has not been certified by CPU tests, and no
automatic multi-GPU strategy is configured.

Known cosmetic Keras/tqdm warnings are handled narrowly. Numerical, resource,
experimental and kernel-transport security warnings remain visible. No global
warning suppression is installed.
