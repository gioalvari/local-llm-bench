.PHONY: install check test full

install:
	uv sync --extra dev

install-mlx:
	uv sync --extra dev --extra mlx

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

test:
	uv run pytest

full: check
	uv run pytest --cov=localllm_bench --cov-branch --cov-report=term-missing --cov-fail-under=80
