# Metadata experiments

Each executable experiment is defined by one strict YAML file:

```yaml
config_version: 1
name: metadata_logistic_regression

dataset:
  registry_key: rsna
  manifest_directory: data/manifests
  bundle_id: build-<sha256>
  task_id: pneumonia

model:
  registry_key: metadata_logistic
  parameters: {}
  fit_parameters: {}

training:
  seed: 42
  report_directory: reports
  model_directory: models/rsna

evaluation:
  sensitivity_target: 0.90
  calibration_bins: 15
  latency_warmup_calls: 100
  latency_measured_calls: 1000

mlflow:
  experiment_name: radfusion-rsna
```

The loader rejects missing, unknown, and duplicate keys. Each experiment names an exact immutable
bundle ID. `training.seed` is the single randomness authority.

The executable configs are `configs/metadata_logistic.yaml` and
`configs/metadata_lightgbm.yaml`.

## Feature boundary

The RSNA adapter exposes these model features:

- `age_years`
- `age_is_implausible`
- `sex`
- `view_position`
- `pixel_spacing_row_mm`
- `pixel_spacing_col_mm`

Preprocessing rejects additional columns. Sample IDs, patient IDs, image paths, targets,
partitions, and lineage remain separate from the feature frame. Imputation, categories,
missingness indicators, scaling, and class weighting are derived from training data.

## Execution

```bash
make train CONFIG=configs/metadata_logistic.yaml
make evaluate RUN_ID=<training-run-id>
make compare
```

`<training-run-id>` denotes the run ID printed by `make train`.

Training validates the complete pinned bundle, then performs projected and filtered reads for
train and validation only. It fits preprocessing and the estimator on train, uses validation for
LightGBM early stopping, selects both operating thresholds on validation, and publishes the fitted
model.

Test evaluation is a separate run. Before reading test data, it cross-checks the package,
configuration, model bytes, pinned bundle, source-run tags, validation-derived thresholds, and
LightGBM best iteration. It then reads only test and applies the verified choices unchanged.

The built-in dataset and model adapters are held in immutable mappings. One tabular runner owns
training; the explicit evaluator owns test evaluation.

## Tracking and outputs

Model packages under `models/` and complete reports under `reports/` are the authoritative
physical outputs. MLflow stores the run ledger, parameters, scalar metrics, provenance, lineage,
completion state, references to project-owned outputs, and the exact loaded training
configuration. Run metadata lives in `mlflow.db`; training configuration artifacts live under
`mlartifacts/`. Inspect local runs with:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db
```

A training run logs the exact loaded YAML before dataset access and records resolved split and
label-policy lineage before fitting. Training and test-evaluation runs become complete only after
their local run-qualified outputs are published. Training runs publish model packages; test runs
link to their source training runs and publish test reports.

Local training packages contain:

```text
models/rsna/runs/<training-run-id>/
  model.skops
  resolved_config.yaml
  model_manifest.json
```

The manifest records the model hash, training run, pinned bundle and split assignment, task,
positive class, config hash, seed, Git commit, dependency lock, best iteration, and thresholds.

Each training or test-evaluation run publishes aggregate reports under
`reports/rsna/runs/<run-id>/`:

```text
metrics.json
evaluation_report.md
confusion_summary.md
roc_curve.png
precision_recall_curve.png
calibration_curve.png
confusion_matrix_youden_j.png
confusion_matrix_target_sensitivity.png
```

Publication requires exactly this set after privacy validation. `make compare` deterministically
regenerates `reports/model_comparison_table.csv` and `.md` from complete, finite MLflow records.
Failed, unfinished, and incomplete runs are excluded.

The training, evaluation, and comparison CLIs default to `sqlite:///mlflow.db` and accept
`--tracking-uri` when an isolated local SQLite database is required.

Metric definitions and experimental protocols are documented in
[`reproducibility.md`](reproducibility.md).
