# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `.gitignore` covering Python artifacts, ML checkpoints, data files, and notebooks
- `pyproject.toml` with `hatchling` build backend and pinned runtime dependencies
- GitHub Actions workflows for lint (`ci.yml`) and release (`release.yml`)
- Pre-commit configuration with `pre-commit-hooks` and `ruff`
- `CONTRIBUTING.md` with setup, branching, commit, and PR guidelines
- `CHANGELOG.md` (this file)
- `requirements.txt` with pinned dependencies
- Converted all Jupyter notebooks to plain Python scripts (`train.py`, `sample.py`, `drag.py`, `evaluate_fid.py`, `interpolate.py`)

### Changed
- Rewritten README with badges, architecture table, CLI reference, and usage examples

## [0.1.0] - 2024-07-29

### Added
- Initial implementation of PoDM diffusion model (U-Net with ResNet blocks and self-attention)
- EDM-style noise schedule and sampler
- DragFDM interactive manipulation via latent-space optimisation
- SLERP interpolation between latent codes
- FID evaluation pipeline
- BIKED 256x256 configuration (`biked_256.yml`)
- Statistical image analysis scripts (KS, Anderson-Darling, KLD, Wasserstein)

[Unreleased]: https://github.com/jiajie96/PoDM-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jiajie96/PoDM-python/releases/tag/v0.1.0
