# Tested environment

The tested CPU environment uses Python 3.11, TensorFlow 2.19.1,
NumPy 1.26.4, SciPy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, and Optuna 4.8.0.
Papermill 2.7.0 and pytest 8.4.2 exercise the notebook and test workflows.

Use a dedicated environment. `pyproject.toml` declares dependency ranges;
`requirements-tested.txt` pins the direct numerical/runtime tools used for this
validation. The full Python dependency snapshot is
[`environments/macos-arm64-py311.txt`](../environments/macos-arm64-py311.txt);
its [environment notes](../environments/README.md) describe the platform scope.
It excludes the package itself, which is installed from the chosen wheel or checkout.
These are version pins, not a cross-platform or hash-locked environment. Generate
a target-specific lock for production runs; manifests record installed versions. NumPy 2 and alternative TensorFlow versions are not validated.

For a cluster, install the package into the site's supported TensorFlow/CUDA
environment and register that environment's kernel. Edit resource requests in
`cluster/papermill.slurm`; it deliberately contains no private paths, login names,
partition, node exclusions, or site-specific CUDA modules. Set `KERNEL_NAME` to
the registered kernel. CPU and GPU predictions can differ at floating-point level;
validate scientifically relevant tolerances rather than demanding bitwise identity.

CPU checks do not establish GPU compatibility.
Before a lengthy run, execute the synthetic workflow and inspect visible devices,
versions, input dimensions, and the saved split/provenance summaries.
