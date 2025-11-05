# Reproducibility Guide

- Set seeds via config (`seed` field) for Python, NumPy, and PyTorch.
- Enable deterministic kernels with `SeedConfig(deterministic=True)`.
- All experiments log manifests in `runs/` for traceability.
- Use provided `requirements*.txt` to recreate environments.
