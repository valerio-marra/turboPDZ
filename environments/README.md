# Validation environment

`macos-arm64-py311.txt` records all installed Python distributions in the CPU
validation environment, excluding turboPDZ itself. It was captured with Python
3.11.16 on macOS arm64. The snapshot includes the numerical, training, notebook,
HSC-adapter, testing, and build dependencies.

For a fresh environment on the same platform:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/macos-arm64-py311.txt
python -m pip install --no-deps .
```

For another platform, use the dependency ranges in `pyproject.toml` and the direct
pins in `requirements-tested.txt`, then capture your own environment. This file
does not pin the operating system, Python binary, wheels, or GPU drivers and is
not a portable GPU lock. Batch templates require validation on the target system.
