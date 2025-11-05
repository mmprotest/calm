# Developer Guide

- Run `pytest` before submitting changes.
- Lint with `ruff` and format with `black` (see `.pre-commit-config.yaml`).
- Tests live in `calm/tests` and cover chunking, autoencoder, energy modeling, sampling, and smoke runs.
- CI (GitHub Actions) executes style checks and unit tests using the tiny config.
