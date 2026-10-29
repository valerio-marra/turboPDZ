#!/usr/bin/env bash
# Run from the checkout root; use a new directory for every training experiment.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: bash examples/run-notebooks.sh CONFIG_PATH RUN_DIR\n' >&2
  exit 2
fi
config_path="$1"
run_dir="$2"
kernel_name="turbopdz"
if [[ -n "${KERNEL_NAME:-}" ]]; then
  kernel_name="$KERNEL_NAME"
fi
if [[ ! -f "$config_path" ]]; then
  printf 'Configuration not found: %s\n' "$config_path" >&2
  exit 1
fi
if [[ -e "$run_dir" ]]; then
  printf 'Run already exists; choose a new directory: %s\n' "$run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/executed"
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
    "$run_dir/executed/$notebook.ipynb" \
    -k "$kernel_name" -p CONFIG_PATH "$config_path" -p RUN_DIR "$run_dir" \
    --cwd . --log-output
done
