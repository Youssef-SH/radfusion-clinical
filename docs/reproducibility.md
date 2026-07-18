# Reproducibility

## Environment

The project targets Python 3.13. `uv.lock` defines the dependency environment.

```bash
uv sync --locked
```

Each experiment records the lock-file SHA-256, Python version, operating system, CPU architecture
and model, and versions of NumPy, PyArrow, scikit-learn, LightGBM, MLflow, and skops.

## Rebuild the RSNA bundle

Place Stage 2 source data under `data/raw/rsna/extracted/`, then run:

```bash
make rsna-manifest
```

Custom paths and split recipes are available through the CLI:

```bash
uv run python -m radfusion.data.rsna_manifest \
  --dataset-root /approved/local/rsna/extracted \
  --output-directory /approved/local/manifests \
  --split-seed 42 \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15
```

The build authenticates both source CSV files and every labeled DICOM by SHA-256. It validates
each generated artifact before publishing the immutable bundle and updating `CURRENT`.

The default split recipe groups samples by patient and stratifies on the binary challenge target.
Within each target stratum, patients are ordered by SHA-256 of the UTF-8 bytes
`<seed>\0<patient_id>`, with patient ID as the collision tie-break. Allocation guarantees one
patient per positive-ratio destination when feasible, then applies largest remainder in canonical
train, validation, test order. For smaller strata, patients fill the highest-ratio destinations,
with canonical order breaking equal-ratio ties.

The split recipe ID hashes the complete policy. The cohort fingerprint hashes canonical patient
membership and targets. The split assignment ID hashes the recipe, cohort, and assignments.

## Generate audits and experiments

```bash
make rsna-audit
make train CONFIG=configs/metadata_logistic.yaml
make train CONFIG=configs/metadata_lightgbm.yaml
```

Preprocessing is fitted on training data. Validation selects the LightGBM stopping point and both
operating thresholds. The fixed pipeline is then evaluated on the patient-disjoint internal test
split.

Release configs require a clean Git tree. Every run records Git commit and dirty status. A
development run from a dirty tree archives the binary tracked diff and relevant nonignored
untracked source, config, test, and documentation files. A SHA-256 identity of that source state is
recorded with the run. Raw data, credentials, environments, and generated outputs are excluded
from the snapshot.

## Probability and operating-point metrics

Models expose class labels and probabilities. Evaluation locates the column labeled `1` and
rejects invalid class or probability contracts.

Average precision is computed with `average_precision_score`. Probability metrics also include
ROC-AUC and Brier score. Threshold-dependent precision, recall, specificity, F1, and confusion
counts are grouped by operating point:

- the Youden-J comparative threshold maximizes validation sensitivity minus false-positive rate;
- the target-sensitivity threshold is the highest validation threshold meeting the configured
  sensitivity, which is 0.90 in the executable configs.

The thresholds are applied unchanged to test probabilities. The Youden-J point is a comparative
benchmark operating point, not a clinical optimum.

## Calibration

Expected calibration error uses the configured number of equal-width bins over `[0, 1]`. Bins are
lower-inclusive; the final bin includes 1. Empty bins contribute zero. Each non-empty bin
contributes its sample fraction times the absolute difference between mean predicted probability
and observed positive fraction. Executable configs use 15 bins, and the plot uses the same count
and strategy.

Calibration slope and intercept come from an L2 logistic regression of the target on predicted
log-odds. Probabilities are clipped to `[1e-6, 1 - 1e-6]` before the logit transform. The fit uses
`C=1e6`, the `lbfgs` solver, 2,000 maximum iterations, and an intercept.

Calibration statistics describe raw class-weighted model outputs. They are point estimates;
bootstrap resampling is not used.

## Latency and model size

Latency covers the complete preprocessing and probability pipeline on CPU. It is the median of
1,000 individually timed single-sample calls after 100 warm-up calls. The benchmark always uses
the first sample in deterministic test order. Results are reported in milliseconds and remain
dependent on hardware and system load.

Model size is the exact serialized `model.skops` byte count divided by 1,048,576 and reported as
MiB.

## Artifact lineage

Bundle identity and acceptance rules are defined in [`data_contract.md`](data_contract.md).
Immutable model lineage and comparison-table roles are defined in [`training.md`](training.md).

MLflow stores the complete run history. Each run records config hash, bundle and split identities,
label policy, Git state, lock hash, resolved model parameters, deterministic LightGBM settings,
environment, metrics, reports, and the skops model.

## Quality gates

```bash
uv lock --check
uv sync --locked
make check
make pre-commit
git diff --check
```

The default suite uses synthetic data. Local RSNA integration tests run when the source dataset is
available:

```bash
uv run pytest -m integration
```

Generated bundles, reports, models, and MLflow state can be deleted with `make clean` and rebuilt.
Raw datasets remain user-managed external inputs.
