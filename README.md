# RadFusion-Clinical

RadFusion-Clinical is a reproducible machine-learning benchmark and experimentation framework for
radiographic pneumonia prediction, centered on the RSNA Pneumonia Detection Challenge. It provides
deterministic data preparation, patient-disjoint evaluation, and reproducible metadata-based
baselines.

> This is a research and educational prototype. It is not a medical device and must not be used for clinical decision-making.

## Implemented capabilities

- Validated joins across RSNA labels, classes, and DICOM images
- DICOM header extraction and aggregate data-quality reporting
- Typed samples, labels, bounding-box annotations, and patient-disjoint splits
- SHA-256 source inventory for every labeled DICOM
- Content-addressed immutable bundles with exact schemas and integrity validation
- Metadata preprocessing fitted on the training split and fixed Logistic Regression and LightGBM
  baselines
- Separate validation and explicit test-evaluation runs with MLflow lineage
- Compact run-qualified model packages with exact config and model bytes
- Ruff, pytest, pre-commit, and continuous-integration checks

## Setup

The project requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Data-acquisition tooling is
available through the optional `acquisition` dependency group.

```bash
uv sync --locked
uv run pre-commit install
```

## Data prerequisite

Obtain the RSNA Pneumonia Detection Challenge data under its original access terms and extract it
to `data/raw/rsna/extracted/`. The required filenames and directory layout are documented in
[`data/README.md`](data/README.md).

## Commands

```bash
make rsna-manifest   # publish an RSNA bundle
make rsna-audit      # generate reports under reports/rsna/audit/<bundle-id>
make train CONFIG=configs/metadata_logistic.yaml
make train CONFIG=configs/metadata_lightgbm.yaml
make evaluate RUN_ID=<training-run-id>
make compare         # regenerate CSV and Markdown comparison views from MLflow
make clean           # remove reproducible outputs while preserving raw datasets
make check           # lock consistency, Ruff checks, and unit/contract tests
make pre-commit      # run repository hooks against all tracked files
make inspect FILE=path/to/image.dcm
```

`<training-run-id>` denotes the run ID printed by `make train`. Model packages are stored under
`models/`, generated reports under `reports/`, MLflow metadata in `mlflow.db`, and small MLflow
training-configuration artifacts under `mlartifacts/`.

Every executable experiment is declared by a validated YAML file under `configs/`. See
[`docs/training.md`](docs/training.md) for the training workflow.

## Cleaning generated artifacts

Run `make clean` to remove generated reports, models, MLflow database and artifacts, Python
caches, and bundles. Git ignores these outputs, and the pipeline regenerates them from source
data. Raw source datasets remain under `data/raw/`.

## Repository layout

```text
src/radfusion/data/        ingestion, splits, audits, schemas, validation, and hashing
src/radfusion/models/      fixed estimator definitions
src/radfusion/training/    reusable training entry points
src/radfusion/evaluation/  metrics and aggregate evaluation plots
configs/                   experiment definitions
tests/                     unit, contract, and local integration tests
docs/                      architecture, data contracts, privacy, and reproducibility
data/                      ignored local inputs and generated artifacts
scripts/                   small inspection utilities
```

## Privacy boundary

Keep DICOMs, patient-level bundle artifacts, credentials, and real patient examples outside
version control. See [`docs/privacy.md`](docs/privacy.md).

## Limitations

- Implemented scope covers the labeled RSNA Stage 2 training set and metadata-only baselines.
- The labels are derived from public radiology-labeling pipelines and are not equivalent to
  confirmed clinical diagnosis.

See [`docs/architecture.md`](docs/architecture.md) for system structure,
[`docs/data_contract.md`](docs/data_contract.md) for artifact contracts, and
[`docs/reproducibility.md`](docs/reproducibility.md) for reconstruction details. RSNA-specific
facts are documented in [`docs/datasets/rsna.md`](docs/datasets/rsna.md).
