.PHONY: sync lock-check lint format format-check test check inspect rsna-manifest rsna-audit \
	train pre-commit clean

# Cleanup searches preserve repository metadata, environments, and source data.
CLEAN_FIND_PRUNE = \( -path './.git' -o -path './.venv' -o -path './data/raw' \) -prune -o

sync:
	uv sync

lock-check:
	uv lock --check

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

check: lock-check lint format-check test

inspect:
	uv run python scripts/inspect_dicom.py "$(FILE)"

rsna-manifest:
	uv run python -m radfusion.data.rsna_manifest

rsna-audit:
	uv run python -m radfusion.data.rsna_audit

train:
	@test -n "$(CONFIG)" || (echo "CONFIG=path/to/experiment.yaml is required"; exit 2)
	@test -f "$(CONFIG)" || (echo "Experiment config not found: $(CONFIG)"; exit 2)
	uv run python -m radfusion.training.train --config "$(CONFIG)"

pre-commit:
	uv run pre-commit run --all-files

clean:
	@set -eu; \
	output_count=0; \
	for path in reports models mlruns mlartifacts; do \
		if [ -e "$$path" ]; then rm -rf -- "$$path"; output_count=$$((output_count + 1)); fi; \
	done; \
	cache_count=0; \
	for path in .pytest_cache .ruff_cache .mypy_cache; do \
		if [ -e "$$path" ]; then rm -rf -- "$$path"; cache_count=$$((cache_count + 1)); fi; \
	done; \
	pycache_count=$$(find . $(CLEAN_FIND_PRUNE) \
		-type d -name '__pycache__' -print | wc -l); \
	find . $(CLEAN_FIND_PRUNE) \
		-type d -name '__pycache__' -prune -exec rm -rf -- {} +; \
	pyc_count=$$(find . $(CLEAN_FIND_PRUNE) \
		-type f -name '*.pyc' -print | wc -l); \
	find . $(CLEAN_FIND_PRUNE) \
		-type f -name '*.pyc' -exec rm -f -- {} +; \
	bundle_count=0; current_count=0; \
	if [ -d data/manifests ]; then \
		bundle_count=$$(find data/manifests -type d -name 'build-*' -print | wc -l); \
		find data/manifests -type d -name 'build-*' -prune -exec rm -rf -- {} +; \
		current_count=$$(find data/manifests \( -type f -o -type l \) -name CURRENT -print | wc -l); \
		find data/manifests \( -type f -o -type l \) -name CURRENT -exec rm -f -- {} +; \
		find data/manifests -depth -type d -empty ! -path data/manifests -exec rmdir -- {} \;; \
	fi; \
	printf 'Removed %s output directories, %s cache directories, %s __pycache__ directories, %s .pyc files, %s bundles, and %s CURRENT pointers.\n' \
		"$$output_count" "$$cache_count" "$$pycache_count" "$$pyc_count" "$$bundle_count" "$$current_count"
