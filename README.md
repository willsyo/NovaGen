# Supernova Visual Pipeline (appearance-first)

This package builds a **visual-only** supernova-remnant cache and renders a quick orbit preview.

## Quick start

```bash
cd ~/NovaGen/supernova_visual_pipeline_pkg
python -m snvis.cli configs/remnant_quick.yaml
```

## Run the test

```bash
cd ~/NovaGen/supernova_visual_pipeline_pkg
pytest -q tests/test_e2e.py
```

## Optional MPI run

```bash
mpiexec -n 4 python -m snvis.cli configs/remnant_quick.yaml
```

## Optional CUDA backend

Install CuPy / cupyx and set `render.backend: cuda`. If CUDA dependencies are missing, the code falls back to CPU.
