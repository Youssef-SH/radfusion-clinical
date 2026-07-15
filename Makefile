.PHONY: sync test lint format check inspect

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint test

inspect:
	uv run python scripts/inspect_dicom.py "$(FILE)"