# Method Notes

This implementation follows the Continuous Autoregressive Language Model (CALM) design:

- **Chunk VAE** compresses K tokens into one latent vector with a β-VAE objective (β ≈ 0.001).
- **Energy Transformer** models latent trajectories with a single-step energy function.
- **Energy Score** provides a likelihood-free training signal using Monte Carlo estimates.
- **Likelihood-free temperature sampling** offers exact and approximate selection from candidate latents.
- **BrierLM** evaluates reconstruction calibration over decoded chunks.
