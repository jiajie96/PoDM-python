# PoDM

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of **PoDM (Point-of-Departure Diffusion Model)** for structural image synthesis, demonstrated on the BIKED bicycle design dataset at 256x256 resolution.

PoDM uses an EDM-style noise schedule with a U-Net denoiser featuring residual blocks and self-attention. The codebase also includes **DragFDM**, a latent-space dragging method for interactive image manipulation using diffusion feature maps.

## Architecture

| Component | Description |
|-----------|-------------|
| U-Net Denoiser | ResNet blocks + self-attention at configurable resolutions |
| Timestep Embedding | Positional (sine/cosine) with continuous noise conditioning |
| EMA | Exponential moving average of model weights |
| Sampler | EDM sampler with optional stochastic churning |

## Repository Structure

```
PoDM-python/
├── diffusion_model.py         # U-Net denoiser (ResnetBlock, AttnBlock, Model)
├── train.py                   # Training entry point
├── sample.py                  # Sampling / generation from a trained checkpoint
├── drag.py                    # DragFDM: interactive latent-space manipulation
├── evaluate_fid.py            # FID evaluation with EDM sampler
├── interpolate.py             # SLERP latent-space interpolation
├── biked_256.yml              # Default config for BIKED 256x256
├── image_analysis/            # Statistical analysis scripts (KS, AD, KLD, Wasserstein)
│   ├── biked_*_paper.py       # BIKED dataset analyses
│   ├── cifar-10_*.py          # CIFAR-10 analyses
│   ├── FFHQ_*.py              # FFHQ analyses
│   ├── noise_scheduling_*.py  # Noise schedule visualisation
│   └── score_function_*.py    # Score function comparisons (EDM vs PoDM)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/jiajie96/PoDM-python.git
cd PoDM-python

conda create -n podm python=3.10
conda activate podm

pip install -e ".[dev]"
```

## Data Preparation

Pre-process the BIKED dataset into `.npy` files of shape `(N, 256, 256, 1)`, values in `[0, 1]`:

```
data/
├── biked_train_256.npy
├── biked_val_256.npy
└── biked_test_256.npy
```

## Training

```bash
python train.py \
    --config biked_256.yml \
    --train_path data/biked_train_256.npy \
    --val_path   data/biked_val_256.npy \
    --test_path  data/biked_test_256.npy \
    --log_dir    result_diffusion_model \
    --gpu_ids    0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `biked_256.yml` | YAML config file |
| `--train_path` | required | Training `.npy` file |
| `--val_path` | required | Validation `.npy` file |
| `--test_path` | required | Test `.npy` file |
| `--log_dir` | `result_diffusion_model` | Output / checkpoint directory |
| `--gpu_ids` | `0` | Comma-separated GPU IDs |
| `--resume_ckpt` | `None` | Checkpoint `.pth` path to resume from |

## Sampling

```bash
python sample.py \
    --config biked_256.yml \
    --ckpt_path result_diffusion_model/ckpt_109000.pth \
    --output_dir generated_images \
    --timesteps 18 \
    --rho 7
```

## DragFDM (Interactive Manipulation)

```bash
python drag.py \
    --config biked_256.yml \
    --ckpt_path result_diffusion_model/ckpt_109000.pth \
    --test_path data/biked_test_256.npy \
    --image_index 0
```

## Interpolation

```bash
python interpolate.py \
    --config biked_256.yml \
    --ckpt_path result_diffusion_model/ckpt_109000.pth \
    --test_path data/biked_test_256.npy \
    --idx_a 0 --idx_b 7 \
    --num_steps 11
```

## FID Evaluation

```bash
python evaluate_fid.py \
    --config biked_256.yml \
    --ckpt_path result_diffusion_model/ckpt_109000.pth \
    --test_path data/biked_test_256.npy \
    --output_dir fid_samples \
    --num_samples 1000
```

## Acknowledgements

This work builds on [EDM](https://arxiv.org/abs/2206.00364) (Karras et al.) and [DDPM](https://arxiv.org/abs/2006.11239) (Ho et al.). The DragFDM component is inspired by [DragDiffusion](https://arxiv.org/abs/2306.14435).
