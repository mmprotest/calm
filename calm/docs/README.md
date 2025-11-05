# Continuous Autoregressive Language Model (CALM)

This repository implements an end-to-end Continuous Autoregressive Language Model (CALM) stack.

## Highlights

- K-token variational autoencoder with β ≈ 0.001 for robust chunk latents.
- Single-step Energy Transformer head trained with the likelihood-free Energy Score.
- Likelihood-free temperature sampling (exact and batch modes).
- BrierLM evaluation for calibrated discrete reconstructions.
- Batteries-included training scripts, configs, docs, and CI.

Refer to the accompanying documents for detailed instructions.
