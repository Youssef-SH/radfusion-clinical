.PHONY: sync lint format format-check test check inspect rsna-manifest pre-commit

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

check: lint format-check test

inspect:
	uv run python scripts/inspect_dicom.py "$(FILE)"

rsna-manifest:
	uv run python -m radfusion.data.rsna_manifest

pre-commit:
	uv run pre-commit run --all-files
