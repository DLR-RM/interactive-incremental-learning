SHELL=/bin/bash
LINT_PATHS=interactive_incremental_learning/ tests/ setup.py

.PHONY: help pytest mypy type lint format check-codestyle commit-checks clean

help:
	@echo "Available targets:"
	@echo "  pytest          Run tests with coverage"
	@echo "  type / mypy     Run mypy type checking"
	@echo "  lint            Run ruff linting"
	@echo "  format          Auto-format code (ruff + black)"
	@echo "  check-codestyle Check formatting without modifying"
	@echo "  commit-checks   Run format + type + lint"
	@echo "  clean           Remove generated artifacts"

pytest:
	python3 -m pytest tests/ --cov-config pyproject.toml --cov-report html --cov-report term --cov=. -v --color=yes

mypy:
	mypy ${LINT_PATHS}

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

clean:
	rm -rf htmlcov/ .coverage __pycache__/ .mypy_cache/ .ruff_cache/ .pytest_cache/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
