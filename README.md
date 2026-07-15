# RadFusion-Clinical

RadFusion-Clinical investigates whether structured clinical information improves pneumonia
prediction from chest radiographs. The repository implements deterministic ingestion of the RSNA
Pneumonia Detection Challenge Stage 2 training set.

> Research prototype only. Not intended for clinical decision-making.

## Implemented capabilities

- Exact parsing and joins across RSNA labels, detailed classes, and labeled training images
- Metadata-only DICOM header extraction without pixel decoding
- Aggregate DICOM and data-quality metadata without patient-level report content
- Task-agnostic samples, normalized task labels, and bounding-box annotation artifacts
- Exact Arrow schemas and validation before and after Parquet serialization
- Deterministic ordering and source, Arrow IPC, and serialized-file SHA-256 hashes
- Immutable bundle publication with an atomically updated `CURRENT` marker
- Ruff, pytest, pre-commit, and continuous-integration checks

## Setup

The project requires Python 3.13 and [uv](https://docs.astral.sh/uv/). The default environment does
not install data-acquisition tooling.

```bash
uv sync --locked
uv run pre-commit install
```

## Data prerequisite

Obtain the RSNA Pneumonia Detection Challenge data under its original access terms and extract it
to `data/raw/rsna/extracted/`. The required filenames and directory layout are documented in
[`data/README.md`](data/README.md). Raw data and generated manifests are ignored by Git.

## Commands

```bash
make rsna-manifest   # build samples, labels, annotations, and aggregate metadata
make check           # Ruff lint, Ruff format check, and unit/contract tests
make pre-commit      # run repository hooks against all tracked files
make inspect FILE=path/to/image.dcm
```

Generated files:

```text
data/manifests/rsna/CURRENT
data/manifests/rsna/builds/<bundle-id>/rsna_samples.parquet
data/manifests/rsna/builds/<bundle-id>/rsna_labels.parquet
data/manifests/rsna/builds/<bundle-id>/rsna_annotations.parquet
data/manifests/rsna/builds/<bundle-id>/rsna_manifest_metadata.json
```

## Repository layout

```text
src/radfusion/data/   ingestion, schemas, validation, and hashing
tests/                unit and contract tests
docs/                 architecture, data contracts, dataset, privacy, and reproducibility
data/                 ignored local inputs and generated artifacts
scripts/              small inspection utilities
```

## Privacy boundary

Do not commit DICOMs, patient-level manifests, credentials, or real patient examples. See
[`docs/privacy.md`](docs/privacy.md).

## Limitations

- Includes the labeled RSNA Stage 2 training set.
- Leaves `split` null until a validated patient-level split policy is defined.
- Treats the RSNA challenge target as a radiology-derived label, not confirmed clinical diagnosis.
- Preserves and flags five source ages above 120 years.
- Currently focuses on deterministic RSNA ingestion. Training, evaluation, serving, and additional dataset adapters are introduced in later milestones.

Architecture, artifact contracts, data handling, and rebuild guarantees are documented under
[`docs/`](docs/).
