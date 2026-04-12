# Contributing

## Setup

```bash
git clone https://github.com/jiajie96/PoDM-python.git
cd PoDM-python

conda create -n podm python=3.10
conda activate podm

pip install -e ".[dev]"
pre-commit install
```

## Branch Naming

| Type | Pattern |
|------|---------|
| Feature | `feat/<short-description>` |
| Bug fix | `fix/<short-description>` |
| Docs | `docs/<short-description>` |
| Refactor | `refactor/<short-description>` |

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

[optional body]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

## Lint

```bash
ruff check .
ruff check --fix .
```

## Pull Requests

1. Fork the repository
2. Create a branch from `main`
3. Make your changes with clear, conventional commit messages
4. Open a PR against `main` — describe the motivation and summarise what changed

## Reporting Issues

Open an issue with:
- A concise description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Environment details (Python version, PyTorch version, OS, GPU)
