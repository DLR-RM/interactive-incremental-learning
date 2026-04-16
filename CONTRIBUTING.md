# Contributing

Thank you for your interest in contributing to this project!

## Bug Reports & Feature Requests

Please open an issue on [GitHub](https://github.com/DLR-RM/interactive-incremental-learning/issues) with:
- A clear description of the bug or feature
- Steps to reproduce (for bugs)
- Expected vs. actual behavior

## Development Setup

```bash
# Clone the repository
git clone https://github.com/DLR-RM/interactive-incremental-learning.git
cd interactive-incremental-learning

# Create conda environment
conda env create -f requirements.yaml
conda activate tpkmp

# Or install with pip (editable mode with test dependencies)
pip install -e ".[tests]"
```

## Code Style

This project uses:
- **[Black](https://github.com/psf/black)** for code formatting (line length: 127)
- **[Ruff](https://github.com/astral-sh/ruff)** for linting
- **[mypy](https://mypy-lang.org/)** for static type checking

Run all checks before submitting a PR:

```bash
make commit-checks   # format + type check + lint
```

Or run individually:

```bash
make format          # Auto-format code
make type            # Run mypy type checking
make lint            # Run ruff linting
make check-codestyle # Check formatting without modifying
```

## Testing

All changes must pass the existing test suite:

```bash
make pytest
```

If you add new functionality, please include corresponding tests.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes
3. Run `make commit-checks` and `make pytest` to verify everything passes
4. Submit a pull request with a clear description of your changes
