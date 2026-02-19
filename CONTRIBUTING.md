# Contributing to nnunet-tracker

Thank you for your interest in contributing to nnunet-tracker!

## Reporting Issues

- Use [GitHub Issues](https://github.com/sathiesh/nnunet-tracker/issues) to report bugs or request features.
- Include Python version, nnU-Net version, and MLflow version when reporting bugs.
- Provide a minimal reproducible example if possible.

## Development Setup

```bash
git clone https://github.com/sathiesh/nnunet-tracker.git
cd nnunet-tracker
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --tb=short --cov=nnunet_tracker
```

Tests run without nnU-Net installed (fully mocked). The test suite requires Python 3.10+.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Keep commits focused and write clear commit messages.
5. Open a PR against `main` with a description of your changes.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
