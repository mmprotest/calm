# CALM: Continuous Autoregressive Language Models

A production-ready reference implementation of Continuous Autoregressive Language Models (CALM). The system compresses groups of K discrete tokens into a single continuous latent with a robust variational autoencoder, models latent dynamics with a single-step Energy Transformer trained via a likelihood-free Energy Score, and generates text using likelihood-free temperature sampling. This README walks you through installation, configuration, training, evaluation, and troubleshooting so you can reproduce results end to end.

## Table of contents

- [Project highlights](#project-highlights)
- [Repository layout](#repository-layout)
- [Prerequisites and installation](#prerequisites-and-installation)
- [Quickstart on a laptop or CPU-only box](#quickstart-on-a-laptop-or-cpu-only-box)
- [Working with data](#working-with-data)
- [Training workflows](#training-workflows)
- [Evaluation](#evaluation)
- [Text generation](#text-generation)
- [Configuration system](#configuration-system)
- [Experiment tracking and checkpoints](#experiment-tracking-and-checkpoints)
- [Testing and continuous integration](#testing-and-continuous-integration)
- [Troubleshooting](#troubleshooting)
- [Additional documentation](#additional-documentation)
- [Citation](#citation)

## Project highlights
- Chunk variational autoencoder with β ≈ 0.001 that reconstructs K-token spans with ≥99 percent accuracy on the tiny configuration.
- Single-step Energy Transformer head that consumes latent sequences and produces samples in one pass to maintain the K-fold speedup promised by CALM.
- Likelihood-free training objectives: Energy Score for the generative head, likelihood-free temperature sampling for decoding, and BrierLM for calibrated evaluation.
- Modular, typed PyTorch 2.3 codebase with torch.compile hooks where safe, AMP, deterministic seeding, gradient clipping, checkpointing, and resume support.
- Comprehensive developer experience: YAML configs, CLI entry points, documentation, diagrams, unit tests, pre-commit hooks, and GitHub Actions CI.

## Repository layout
```
calm/                    Core Python package
  config/                YAML configuration presets
  data/                  Tokenization, chunking, dataset adapters
  models/                Autoencoder, energy transformer, losses, metrics
  train/                 Training, evaluation, and generation entry points
  scripts/               Utility scripts for data prep, profiling, export
  tests/                 Pytest suite covering critical components
  docs/                  Extended documentation and guides
  assets/diagrams/       SVG diagrams referenced in docs
requirements.txt         Runtime dependencies
requirements-dev.txt     Developer dependencies (tests, linters)
pyproject.toml           Packaging metadata and CLI entry points
setup.cfg               Tooling defaults (pytest, coverage, etc.)
README.md                You are here
```

## Prerequisites and installation
1. Install Python 3.10 or newer. On Linux or macOS you can use `pyenv`, `conda`, or system Python. On Windows use the official installer or WSL2.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
3. Upgrade pip and install runtime dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Optional but recommended for development:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```
4. Verify PyTorch detects your hardware:
   ```python
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
   If CUDA is unavailable the scripts will fall back to CPU.

## Quickstart on a laptop or CPU-only box
The tiny configuration exercises every component with minimal compute and downloads the `wikitext-2` dataset from Hugging Face.

1. Prepare data (downloads tokenizer and dataset cache into the Hugging Face default directories):
   ```bash
   python -m calm.scripts.prepare_data --dataset wikitext-2 --tokenizer gpt2
   ```
2. Train the chunk VAE with the tiny config:
   ```bash
   python -m calm.train.train_autoencoder --config calm/config/tiny.yaml --data wikitext-2
   ```
3. Train the CALM Energy Transformer head while reusing the trained autoencoder weights:
   ```bash
   python -m calm.train.train_calm --config calm/config/tiny.yaml --autoencoder_checkpoint path/to/ae.ckpt
   ```
4. Generate text from a prompt using the trained models:
   ```bash
   python -m calm.train.generate --prompt "The future of language models" --config calm/config/tiny.yaml --temperature 0.8
   ```
5. Evaluate calibration with BrierLM:
   ```bash
   python -m calm.train.evaluate --config calm/config/tiny.yaml --metrics brierlm
   ```

Expect each command to print progress with `tqdm`, log summaries to JSONL under `runs/`, and emit checkpoints in `checkpoints/`.

## Working with data
- **Hugging Face datasets**: specify `--data DATASET_NAME` (for example `wikitext-2`). Tokenizers default to `gpt2`; override with `--tokenizer your/tokenizer`.
- **Plain text folders**: point `--data_dir path/to/texts` to a directory containing `.txt` files. The data adapter shuffles and streams files to chunk builders.
- **Chunking**: token IDs are grouped into spans of length `K` (default 4). Training drops incomplete tails; inference pads with the EOS token. Adjust `K` via config or CLI (`--chunk_size`).
- **Caching**: datasets and tokenizers are cached in the Hugging Face home directory (`~/.cache/huggingface` by default). Set the `HF_HOME` environment variable to customize.

## Training workflows
### Autoencoder
- Script: `python -m calm.train.train_autoencoder`
- Key flags: `--config`, `--data`, `--data_dir`, `--seed`, `--precision`.
- Outputs: checkpoints saved under `checkpoints/autoencoder/` with both loss-minimizing and accuracy-maximizing snapshots. Validation accuracy must reach ≥99 percent on the tiny split.
- Tips: use `--freeze_embeddings_steps N` to stabilize early training, `--beta_schedule cosine` for KL annealing, and `--compile` to enable `torch.compile` when CUDA 12+ is available.

### CALM Energy Transformer head
- Script: `python -m calm.train.train_calm`
- Inputs: pretrained autoencoder checkpoint (pass via `--autoencoder_checkpoint`). Optionally finetune the decoder with `--finetune_autoencoder_decoder`.
- Training objective: Energy Score over next-chunk latent posteriors. Control the number of target samples `M` and model samples `N` via `--targets_m` and `--samples_n` (defaults in configs).
- Validation: runs periodic BrierLM evaluation and prints sample generations. Early stopping is available through callbacks configured in YAML.
- Performance tips: enable AMP with `--precision amp`, adjust context length via `--context_len`, and set `--grad_clip` to keep gradients stable.

## Evaluation
`python -m calm.train.evaluate` computes metrics on validation or test splits. Pass `--metrics brierlm` for calibration, `--num_generations` for qualitative samples, and `--exact_temperature` to audit the sampler. Results are written to `runs/eval/*.jsonl` and printed to stdout.

## Text generation
`python -m calm.train.generate` supports interactive or batch prompting.
- Choose sampling temperature with `--temperature T` where `0 < T ≤ 1`.
- Select candidate pool size with `--num_candidates` and restrict to the top K candidates via `--topk`.
- Switch between the exact Algorithm 1 sampler (`--exact_temperature`) and the faster batch approximation (default).
- Control maximum tokens with `--max_tokens` and specify output destination with `--output_file`.

## Configuration system
- Default hyperparameters live in `calm/config/defaults.yaml`. Specialized presets extend it via YAML anchors.
- `calm/config/base_AE.yaml` and `calm/config/base_Calm.yaml` provide reusable component blocks.
- `calm/config/tiny.yaml` overrides settings for CPU-friendly experiments.
- Override any option from the CLI using `--key value` notation (for example `--trainer.max_epochs 1`). All resolved configs are dumped next to run logs for reproducibility.

## Experiment tracking and checkpoints
- All training scripts create a manifest under `runs/<timestamp>/` containing the resolved config, metrics JSONL, and tensorboard summaries.
- Checkpoints live in `checkpoints/<run_name>/` and store model state dicts, optimizer and scheduler state, AMP scaler state, and extra metadata.
- Resume training with `--resume path/to/checkpoint.ckpt`. To continue CALM training with new data while keeping the autoencoder fixed, pass `--freeze_autoencoder`.

## Testing and continuous integration
- Run unit tests locally with:
  ```bash
  pytest calm/tests
  ```
- Format and lint the codebase using pre-commit:
  ```bash
  pre-commit run --all-files
  ```
- GitHub Actions (`.github/workflows/ci.yml`) installs dependencies, runs ruff and black, executes the pytest suite with the tiny config, and exercises a smoke training pass.

## Troubleshooting
| Symptom | Fix |
| --- | --- |
| Hugging Face authentication errors | Run `huggingface-cli login` or set the `HF_HOME` cache directory to a writable path. |
| CUDA not detected | Ensure that you installed the correct PyTorch wheel for your CUDA version. Fallback to CPU by omitting `--device cuda`. |
| Autoencoder accuracy below target | Increase epochs, decrease learning rate, or enable `--freeze_embeddings_steps 500`. Verify that `K` matches your training data. |
| Energy Score diverges | Reduce `--samples_n`, use gradient clipping via config, and confirm that autoencoder checkpoints are loaded correctly. |
| Generation repeats or stalls | Lower temperature, increase `--num_candidates`, or switch to the exact temperature sampler for debugging. |
| Out of memory | Lower batch size (`--batch_size`), shorten context length, or disable compilation. |

## Additional documentation
Detailed guides live under `calm/docs/`:
- `docs/README.md`: project overview.
- `docs/QUICKSTART.md`: scripted walkthroughs.
- `docs/METHOD_NOTES.md`: architecture, Energy Score derivation, and temperature sampling algorithm.
- `docs/CONFIG_GUIDE.md`: explanation of every configuration knob.
- `docs/REPRO_GUIDE.md`: reproducibility checklist and environment specification.
- `docs/DEV_GUIDE.md`: contributing guidelines and developer tips.

## Citation
If you use this implementation in research or production, please cite the CALM paper and this repository. Adapt as needed:
```
@software{calm_reference_implementation,
  title = {CALM: Continuous Autoregressive Language Models Reference Implementation},
  author = {CALM Engineering Team},
  year = {2024},
  url = {https://github.com/your-org/calm}
}
```
