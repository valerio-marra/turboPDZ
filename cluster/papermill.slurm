#!/usr/bin/env bash
#SBATCH --job-name=turbopdz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# Add your site's partition/GPU/account/module settings here or via sbatch flags.

set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the installed checkout}"
: "${CONFIG_PATH:?Set CONFIG_PATH to the experiment YAML}"
: "${RUN_DIR:?Set RUN_DIR to a unique shared run directory}"
: "${STAGE:?Set STAGE to prepare/train-zml/oof/train-sigmaml/evaluate/bootstrap/applicability}"
KERNEL_NAME="${KERNEL_NAME:-turbopdz}"

case "$STAGE" in
  prepare) notebook=1-prepare-data ;;
  train-zml) notebook=2a-training-zml ;;
  oof) notebook=2b-oof-zml ;;
  train-sigmaml) notebook=2c-training-sigmaml ;;
  evaluate) notebook=3-testing ;;
  bootstrap) notebook=4-comparison-bootstrap ;;
  applicability) notebook=5-applicability-flag ;;
  *) printf 'Unknown STAGE: %s\n' "$STAGE" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR/executed"
python -m papermill "notebooks/${notebook}.ipynb" \
  "$RUN_DIR/executed/${notebook}-${SLURM_JOB_ID:-local}.ipynb" \
  -k "$KERNEL_NAME" -p CONFIG_PATH "$CONFIG_PATH" -p RUN_DIR "$RUN_DIR" \
  --cwd "$PROJECT_ROOT" --log-output
