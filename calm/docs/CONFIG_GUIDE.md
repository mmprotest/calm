# Configuration Guide

Configurations live in `calm/config`. Each YAML file merges into defaults when the `inherit` key is used. Key groups:

- `autoencoder`: defines chunk size `K`, model width, learning rate, and β weight.
- `calm`: controls the Energy Transformer depth, number of samples, Energy Score settings, and evaluation cadence.
- `logging` and `checkpoint`: path management for reproducibility.
