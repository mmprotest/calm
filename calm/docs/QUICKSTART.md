# Quickstart

```bash
pip install -e .
python -m calm.train.train_autoencoder --config config/tiny.yaml
python -m calm.train.train_calm --config config/tiny.yaml
python -m calm.train.generate --prompt "The future of language models" --max_tokens 64 --temperature 0.8
```
