# Release validation: 0.1.0

Validated on 29 October 2026 with Python 3.11.16 on macOS arm64, using CPU
TensorFlow 2.19.1. The [environment snapshot](../environments/README.md) records
the dependencies; the [model report](validation.md) defines numerical tolerances.

| Check | Result |
| --- | --- |
| Source-checkout tests | 79 passed |
| Fresh wheel installation in an isolated environment | Passed; `pip check` found no broken requirements |
| Imports and CLI outside the checkout | Passed; imports resolve to the installed wheel |
| Tests against the installed wheel | 79 passed |
| Wheel and source-distribution contents | All 36 Python modules match source; notebook/configuration/recipe assets included |
| Synthetic Papermill workflow | All seven stages completed using the installed wheel |
| Generated diagnostics | 54 PDF figures and 63 CSV tables; all 54 plot pages rendered and visually reviewed |
| Exported synthetic inference | Reloaded predictions and applicability flags match exactly on all 12 final-test objects |
| Frozen HSC bundles | Six pairs passed on 256 sampled final-test objects each; payload hashes unchanged |
| HPO membership diagnostic | All 468 recomputed metric rows and six provenance records match the distributed aggregates |
| Distribution review | No credentials, personal paths, raw catalog extracts, saved notebook outputs, or oversized Git files found |

The earlier larger synthetic experiment also completed all seven stages with
8,000 objects, five folds, 15 attempted trials per network, and 2,000 bootstrap
resamples. Its configuration is `configs/synthetic-serious.yaml`; it was not
rerun as part of these installation checks. Synthetic results check software
behavior and do not establish survey performance.

The frozen HSC report preserves the documented empirical-CDF tie behavior;
passing numerical checks does not imply bitwise `r_ml` reproduction or an
all-object audit. Fitting/calibration separation and the historical HPO-exposure
limits are described in [model provenance](hsc-models.md).

The six bundles are distributed directly in the repository, separately from
Python packages. Source/model metadata and citation identify the accepted PASJ
paper. The full-survey VAC remains a separate planned product.

GitHub CI builds/installs the wheel, runs tests, and executes the small synthetic
notebook workflow on Linux. See the [workflow results](https://github.com/valerio-marra/turboPDZ/actions/workflows/tests.yml)
for the current status. GPU and production batch environments require their own validation.
GitHub/PyPI publication and tagging are separate explicit release actions.
