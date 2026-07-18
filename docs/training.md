# Configuration-driven training

Run one experiment from one YAML file:

```bash
make train CONFIG=configs/metadata_logistic.yaml
make train CONFIG=configs/metadata_lightgbm.yaml
```

The loader rejects unknown or duplicate keys, missing fields, invalid types, unsupported policies,
unknown registry keys, and estimator parameters that conflict with repository-level controls.
Executable release configs require a clean Git tree.

## Configuration

Each YAML defines:

- a registered dataset adapter and task;
- a registered model adapter and fixed estimator parameters;
- one training seed and output locations;
- threshold, calibration, and latency settings;
- an MLflow experiment and tracking location.

`training.seed` controls estimator randomness. The model adapter derives all estimator seed fields
from it. The class-weighting policy also has one source: Logistic Regression receives balanced
weights, while LightGBM derives `scale_pos_weight` from training labels.

The executable baseline configs are:

- `configs/metadata_logistic.yaml`
- `configs/metadata_lightgbm.yaml`

The image and fusion YAML files are non-executable configuration examples.

## Execution flow

The CLI loads the config, registers built-in datasets and models once, and invokes one runner. The
runner:

1. loads the validated bundle through the dataset adapter;
2. fits preprocessing and the estimator from training data;
3. uses validation data for LightGBM stopping and operating-point selection;
4. evaluates the fixed pipeline on the test split;
5. publishes aggregate reports and an immutable model run;
6. records the execution in MLflow;
7. updates the current comparison table under an interprocess lock.

LightGBM monitors validation average precision as its sole early-stopping metric. Its configured
500 estimators are an upper bound, and prediction uses the fitted best iteration.

## Registries and extension points

`DatasetRegistry` resolves bundle loading and frame construction. `ModelRegistry` resolves
estimator construction and fitting. Registration occurs through `register_builtin_components()`;
module import side effects do not register components.

A compatible tabular model requires:

1. a model adapter implementing the fitting interface;
2. one registration in the built-in model registry;
3. an experiment YAML that names the registry key.

The shared runner owns evaluation, publication, and tracking. Dataset adapters return validated,
deterministically ordered train, validation, and test data with bundle, split, and label lineage.

## Outputs

Reports are published under `reports/<dataset>/models/<output-name>/`. `metrics.json` is the
structured metric source; Markdown and CSV views are generated from the same in-memory values.

Models are published under:

```text
models/<dataset>/<output-name>/
  CURRENT
  runs/<mlflow-run-id>/
    model.skops
    lineage.json
```

Run directories are immutable. `lineage.json` binds the model to its config, bundle, split
identities, label policy, seed, Git state, dependency lock, derived parameters, and physical model
identity.

MLflow retains full execution history. `reports/model_comparison_table.csv` contains the current
row for each semantic experiment identity. See [`reproducibility.md`](reproducibility.md) for the
evaluation and provenance protocols.
