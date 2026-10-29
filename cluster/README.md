# Cluster execution

Install/register the package environment first, then customize resources/modules
in `papermill.slurm`. It is a template, not an assertion of compatibility with a
particular cluster. For example:

```bash
sbatch --partition=YOUR_GPU_QUEUE --gres=gpu:1 \
  --export=ALL,PROJECT_ROOT=/path/to/turboPDZ,CONFIG_PATH=recipes/hsc-pdr3/configs/dnnz-dud.yaml,RUN_DIR=/scratch/project/dnnz-dud,STAGE=train-zml,KERNEL_NAME=turbopdz \
  cluster/papermill.slurm
```

Run `prepare` first. Subsequent stages require the previous artifacts; check job
success before manually submitting the next stage. There are no hidden Slurm
dependencies or automatic follow-up jobs. Executed notebooks are written under
the run directory with the job ID, while source notebooks retain empty outputs.
