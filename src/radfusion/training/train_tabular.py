"""Train and evaluate one configured tabular experiment."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radfusion-matplotlib"))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from radfusion.evaluation.latency import LATENCY_SAMPLE_POLICY, benchmark_single_sample_latency_ms
from radfusion.evaluation.metrics import (
    CALIBRATION_BINNING_STRATEGY,
    BinaryMetrics,
    evaluate_binary,
    target_sensitivity_threshold,
    validated_binary_targets,
    youden_j_threshold,
)
from radfusion.evaluation.plots import write_evaluation_plots
from radfusion.evaluation.probabilities import positive_class_probabilities
from radfusion.training.config import ExperimentConfig
from radfusion.training.registry import DATASET_REGISTRY, MODEL_REGISTRY
from radfusion.utils.mlflow_utils import (
    configure_mlflow,
    environment_provenance,
    git_revision,
    tracked_run,
    uv_lock_sha256,
    write_dirty_source_snapshot,
)
from radfusion.utils.model_publication import PublishedModel, publish_model_run
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory
from radfusion.utils.skops_io import load_skops, save_skops, trusted_types_for_file

MODEL_REPORT_BASE_FILENAMES = frozenset(
    {
        "confusion_matrix_youden_j.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "evaluation_report.md",
        "confusion_summary.md",
        "metrics.json",
    }
)


@dataclass(frozen=True)
class ModelResult:
    """Aggregate outputs and lineage for one trained baseline."""

    model_name: str
    run_id: str
    youden_j_threshold: float
    target_sensitivity_threshold: float | None
    validation_youden_j: BinaryMetrics
    test_youden_j: BinaryMetrics
    validation_target_sensitivity: BinaryMetrics | None
    test_target_sensitivity: BinaryMetrics | None
    model_path: Path
    model_artifact_sha256: str
    artifact_directory: Path
    latency_ms: float
    model_size_mib: float


def train_configured_experiment(config: ExperimentConfig) -> ModelResult:
    """Train and evaluate one configured experiment."""
    if not config.executable:
        raise ValueError(f"Experiment {config.name!r} is documented but not executable")
    commit, dirty = git_revision()
    if config.training.require_clean_git and dirty:
        raise ValueError("Release experiment configuration requires a clean Git tree")
    dataset = DATASET_REGISTRY.get(config.dataset.registry_key).load(config.dataset)
    model_implementation = MODEL_REGISTRY.get(config.model.registry_key)
    frame = dataset.frame
    configure_mlflow(
        experiment_name=config.mlflow.experiment_name,
        tracking_directory=config.mlflow.tracking_directory,
    )
    train = frame.loc[frame["split_name"] == "train"]
    validation = frame.loc[frame["split_name"] == "validation"]
    train_features, train_targets = _features_and_target(train)
    validation_features, validation_targets = _features_and_target(validation)
    fitted = model_implementation.fit(
        config.model,
        config.training.seed,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
    )
    lineage = {
        "bundle_id": dataset.bundle_id,
        "split_recipe_id": dataset.split_recipe_id,
        "cohort_fingerprint": dataset.cohort_fingerprint,
        "split_assignment_id": dataset.split_assignment_id,
        "label_policy_version": dataset.label_policy_version,
        "git_commit": commit,
        "git_dirty": dirty,
        "uv_lock_sha256": uv_lock_sha256(),
        "train_positive_count": int((train_targets == 1).sum()),
        "train_negative_count": int((train_targets == 0).sum()),
        "config_path": config.source_path.as_posix(),
        "config_sha256": config.source_sha256,
        "derived_parameters": dict(fitted.derived_parameters),
    }
    result = _evaluate_and_log(
        config=config,
        model=fitted.pipeline,
        frame=frame,
        lineage=lineage,
    )
    comparison_path = config.training.report_directory / "model_comparison_table.csv"
    _write_comparison_table(result, config, lineage, comparison_path)
    mlflow.tracking.MlflowClient().log_artifact(result.run_id, str(comparison_path))
    return result


def _features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    targets = validated_binary_targets(frame["target"].to_numpy())
    return frame.drop(columns=["target"]).copy(), targets


def _evaluate_and_log(
    *,
    config: ExperimentConfig,
    model: Pipeline,
    frame: pd.DataFrame,
    lineage: dict[str, Any],
) -> ModelResult:
    model_name = config.model.output_name
    validation = frame.loc[frame["split_name"] == "validation"]
    test = frame.loc[frame["split_name"] == "test"]
    validation_features, validation_targets = _features_and_target(validation)
    test_features, test_targets = _features_and_target(test)
    validation_probabilities = positive_class_probabilities(model, validation_features)
    youden_threshold = youden_j_threshold(validation_targets, validation_probabilities)
    try:
        target_threshold = target_sensitivity_threshold(
            validation_targets,
            validation_probabilities,
            sensitivity=config.evaluation.sensitivity_target,
        )
    except ValueError:
        target_threshold = None
    validation_youden = evaluate_binary(
        validation_targets,
        validation_probabilities,
        threshold=youden_threshold,
        calibration_bins=config.evaluation.calibration_bins,
    )
    validation_target = (
        evaluate_binary(
            validation_targets,
            validation_probabilities,
            threshold=target_threshold,
            calibration_bins=config.evaluation.calibration_bins,
        )
        if target_threshold is not None
        else None
    )
    test_probabilities = positive_class_probabilities(model, test_features)
    latency_ms = benchmark_single_sample_latency_ms(
        model,
        test_features,
        warmup_calls=config.evaluation.latency_warmup_calls,
        measured_calls=config.evaluation.latency_measured_calls,
    )
    test_youden = evaluate_binary(
        test_targets,
        test_probabilities,
        threshold=youden_threshold,
        calibration_bins=config.evaluation.calibration_bins,
    )
    test_target = (
        evaluate_binary(
            test_targets,
            test_probabilities,
            threshold=target_threshold,
            calibration_bins=config.evaluation.calibration_bins,
        )
        if target_threshold is not None
        else None
    )

    artifact_directory = (
        config.training.report_directory / config.dataset.registry_key / "models" / model_name
    )
    artifact_stage = staging_directory(artifact_directory)
    temporary_model_root = Path(tempfile.mkdtemp(prefix="radfusion-model-"))
    source_snapshot_root: Path | None = None
    try:
        write_evaluation_plots(
            test_targets,
            test_probabilities,
            youden_j_threshold=youden_threshold,
            target_sensitivity_threshold=target_threshold,
            calibration_bins=config.evaluation.calibration_bins,
            output_directory=artifact_stage,
        )
        metrics_document = _metrics_document(
            config,
            youden_threshold,
            target_threshold,
            validation_youden,
            test_youden,
            validation_target,
            test_target,
        )
        (artifact_stage / "metrics.json").write_text(
            json.dumps(metrics_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_evaluation_report(
            artifact_stage / "evaluation_report.md", model_name, metrics_document
        )
        _write_confusion_summary(artifact_stage / "confusion_summary.md", metrics_document)
        expected_reports = set(MODEL_REPORT_BASE_FILENAMES)
        if target_threshold is not None:
            expected_reports.add("confusion_matrix_target_sensitivity.png")
        actual_reports = {path.name for path in artifact_stage.iterdir() if path.is_file()}
        if actual_reports != expected_reports:
            raise ValueError(
                f"Model report staging set mismatch: expected={sorted(expected_reports)}, "
                f"actual={sorted(actual_reports)}"
            )
        validate_public_reports(
            artifact_stage.iterdir(),
            forbidden_source_values=_frame_source_identifiers(frame),
        )

        temporary_model = save_skops(model, temporary_model_root / "model.skops")
        restored_model = load_skops(temporary_model)
        restored_probabilities = positive_class_probabilities(restored_model, test_features)
        if not np.array_equal(restored_probabilities, test_probabilities):
            raise ValueError("Serialized model probabilities differ from the fitted model")
        temporary_model_size_mib = temporary_model.stat().st_size / (1024.0 * 1024.0)
        source_snapshot: Path | None = None
        lineage["git_source_state_sha256"] = "clean"
        if lineage["git_dirty"]:
            source_snapshot_root = Path(tempfile.mkdtemp(prefix="radfusion-source-parent-"))
            source_snapshot = write_dirty_source_snapshot(source_snapshot_root / "source_state")
            snapshot_document = json.loads(
                (source_snapshot / "snapshot_manifest.json").read_text(encoding="utf-8")
            )
            lineage["git_source_state_sha256"] = snapshot_document["source_state_sha256"]
        tags = _logged_tags(config, lineage)
        parameters = _logged_parameters(config, lineage)
        published: PublishedModel | None = None
        with tracked_run(run_name=config.name, tags=tags, parameters=parameters) as run_id:
            mlflow.log_metrics(
                _mlflow_metrics(
                    metrics_document,
                    latency_ms=latency_ms,
                    model_size_mib=temporary_model_size_mib,
                )
            )
            mlflow.log_artifacts(str(artifact_stage))
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
                skops_trusted_types=trusted_types_for_file(temporary_model),
            )
            if source_snapshot is not None:
                mlflow.log_artifacts(str(source_snapshot), artifact_path="source_state")
            published = publish_model_run(
                None,
                model_root=config.training.model_directory,
                model_key=model_name,
                mlflow_run_id=run_id,
                serialized_model_path=temporary_model,
                lineage={
                    "dataset_id": config.dataset.registry_key,
                    "task_id": config.dataset.task_id,
                    "config_source_sha256": config.source_sha256,
                    "bundle_id": lineage["bundle_id"],
                    "cohort_fingerprint": lineage["cohort_fingerprint"],
                    "split_recipe_id": lineage["split_recipe_id"],
                    "split_assignment_id": lineage["split_assignment_id"],
                    "label_policy_version": lineage["label_policy_version"],
                    "training_seed": config.training.seed,
                    "git_commit": lineage["git_commit"],
                    "git_dirty": lineage["git_dirty"],
                    "git_source_state_sha256": lineage["git_source_state_sha256"],
                    "uv_lock_sha256": lineage["uv_lock_sha256"],
                    "derived_parameters": lineage["derived_parameters"],
                },
            )
            mlflow.set_tags(
                {
                    "local_model_path": published.model_path.as_posix(),
                    "local_model_artifact_sha256": published.model_artifact_sha256,
                    "model_size_unit": "MiB",
                }
            )
        if published is None:
            raise RuntimeError("Model publication did not complete")
        publish_directory(artifact_stage, artifact_directory)
    finally:
        if artifact_stage.exists():
            shutil.rmtree(artifact_stage)
        shutil.rmtree(temporary_model_root, ignore_errors=True)
        if source_snapshot_root is not None:
            shutil.rmtree(source_snapshot_root, ignore_errors=True)

    return ModelResult(
        model_name=model_name,
        run_id=run_id,
        youden_j_threshold=youden_threshold,
        target_sensitivity_threshold=target_threshold,
        validation_youden_j=validation_youden,
        test_youden_j=test_youden,
        validation_target_sensitivity=validation_target,
        test_target_sensitivity=test_target,
        model_path=published.model_path,
        model_artifact_sha256=published.model_artifact_sha256,
        artifact_directory=artifact_directory,
        latency_ms=latency_ms,
        model_size_mib=published.model_size_mib,
    )


def _metrics_document(
    config: ExperimentConfig,
    youden_threshold: float,
    target_threshold: float | None,
    validation_youden: BinaryMetrics,
    test_youden: BinaryMetrics,
    validation_target: BinaryMetrics | None,
    test_target: BinaryMetrics | None,
) -> dict[str, Any]:
    return {
        "calibration": {
            "calibration_bins": config.evaluation.calibration_bins,
            "calibration_binning_strategy": CALIBRATION_BINNING_STRATEGY,
        },
        "probability_metrics": {
            "validation": validation_youden.probability.as_dict(),
            "test": test_youden.probability.as_dict(),
        },
        "operating_points": {
            "youden_j": {
                "youden_j_threshold": youden_threshold,
                "validation": validation_youden.operating_point.as_dict(),
                "test": test_youden.operating_point.as_dict(),
            },
            "target_sensitivity": {
                "configured_target_sensitivity": config.evaluation.sensitivity_target,
                "target_sensitivity_threshold": target_threshold,
                "validation": (
                    validation_target.operating_point.as_dict()
                    if validation_target is not None
                    else None
                ),
                "test": (
                    test_target.operating_point.as_dict() if test_target is not None else None
                ),
            },
        },
    }


def _logged_tags(config: ExperimentConfig, lineage: dict[str, Any]) -> dict[str, str]:
    return {
        "dataset": config.dataset.registry_key,
        "dataset_bundle_id": str(lineage["bundle_id"]),
        "split_recipe_id": str(lineage["split_recipe_id"]),
        "cohort_fingerprint": str(lineage["cohort_fingerprint"]),
        "split_assignment_id": str(lineage["split_assignment_id"]),
        "label_policy_version": str(lineage["label_policy_version"]),
        "task": config.dataset.task_id,
        "model_type": config.model.output_name,
        "seed": str(config.training.seed),
        "git_commit": str(lineage["git_commit"]),
        "git_dirty": str(lineage["git_dirty"]).lower(),
        "git_source_state_sha256": str(lineage["git_source_state_sha256"]),
        "experiment_config_sha256": config.source_sha256,
        "uv_lock_sha256": str(lineage["uv_lock_sha256"]),
    }


def _logged_parameters(config: ExperimentConfig, lineage: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        "tabular_model_type": config.model.output_name,
        "class_weighting": config.model.class_weighting,
        "threshold_policy": config.evaluation.threshold_policy,
        "configured_target_sensitivity": config.evaluation.sensitivity_target,
        "calibration_bins": config.evaluation.calibration_bins,
        "calibration_binning_strategy": CALIBRATION_BINNING_STRATEGY,
        "latency_sample_policy": LATENCY_SAMPLE_POLICY,
        "latency_warmup_calls": config.evaluation.latency_warmup_calls,
        "latency_measured_calls": config.evaluation.latency_measured_calls,
        "training_seed": config.training.seed,
        "train_positive_count": lineage["train_positive_count"],
        "train_negative_count": lineage["train_negative_count"],
        "experiment_config_path": lineage["config_path"],
        "experiment_config_sha256": lineage["config_sha256"],
        "uv_lock_sha256": lineage["uv_lock_sha256"],
        **dict(config.model.parameters),
        **dict(config.model.fit_parameters),
        **lineage["derived_parameters"],
        **environment_provenance(),
    }
    if "scale_pos_weight" in lineage["derived_parameters"]:
        parameters["class_weighting"] = (
            f"scale_pos_weight={lineage['derived_parameters']['scale_pos_weight']:.12g}"
        )
    return parameters


def _mlflow_metrics(
    document: dict[str, Any], *, latency_ms: float, model_size_mib: float
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name, values in document["probability_metrics"].items():
        metrics.update({f"{split_name}_{key}": float(value) for key, value in values.items()})
    for policy_name, policy in document["operating_points"].items():
        threshold_key = f"{policy_name}_threshold"
        if policy.get(threshold_key) is not None:
            metrics[threshold_key] = float(policy[threshold_key])
        for split_name in ("validation", "test"):
            if policy[split_name] is not None:
                metrics.update(
                    {
                        f"{split_name}_{policy_name}_{key}": float(value)
                        for key, value in policy[split_name].items()
                    }
                )
    metrics["latency_ms"] = latency_ms
    metrics["model_size_mib"] = model_size_mib
    return metrics


def _write_comparison_table(
    result: ModelResult,
    config: ExperimentConfig,
    lineage: dict[str, Any],
    path: Path,
) -> None:
    record = {
        "dataset": config.dataset.registry_key,
        "task": config.dataset.task_id,
        "model_name": result.model_name,
        "modality": config.model.modality,
        "experiment_config_sha256": config.source_sha256,
        "dataset_bundle_id": lineage["bundle_id"],
        "split_assignment_id": lineage["split_assignment_id"],
        "split_recipe_id": lineage["split_recipe_id"],
        "cohort_fingerprint": lineage["cohort_fingerprint"],
        "seed": config.training.seed,
        "git_commit": lineage["git_commit"],
        "git_dirty": lineage["git_dirty"],
        "git_source_state_sha256": lineage["git_source_state_sha256"],
        "uv_lock_sha256": lineage["uv_lock_sha256"],
        "mlflow_run_id": result.run_id,
        "local_model_path": result.model_path.as_posix(),
        "local_model_sha256": result.model_artifact_sha256,
        "youden_j_threshold": result.youden_j_threshold,
        "target_sensitivity": config.evaluation.sensitivity_target,
        "target_sensitivity_threshold": result.target_sensitivity_threshold,
        "calibration_bins": config.evaluation.calibration_bins,
        "calibration_binning_strategy": CALIBRATION_BINNING_STRATEGY,
        "validation_average_precision": result.validation_youden_j.probability.average_precision,
        **{
            f"test_{key}": value
            for key, value in result.test_youden_j.probability.as_dict().items()
        },
        **{
            f"test_youden_j_{key}": value
            for key, value in result.test_youden_j.operating_point.as_dict().items()
        },
        **(
            {
                f"test_target_sensitivity_{key}": value
                for key, value in result.test_target_sensitivity.operating_point.as_dict().items()
            }
            if result.test_target_sensitivity is not None
            else {}
        ),
        "latency_ms": result.latency_ms,
        "model_size_mib": result.model_size_mib,
        **environment_provenance(),
    }
    identity_columns = [
        "dataset",
        "task",
        "model_name",
        "experiment_config_sha256",
        "dataset_bundle_id",
        "split_assignment_id",
        "seed",
        "git_commit",
        "git_source_state_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        table = pd.DataFrame.from_records([record])
        if path.is_file():
            existing = pd.read_csv(path)
            columns = [
                *existing.columns,
                *(column for column in table.columns if column not in existing),
            ]
            existing = existing.reindex(columns=columns)
            table = table.reindex(columns=columns)
            identity = np.ones(len(existing), dtype=bool)
            for column in identity_columns:
                identity &= existing[column].astype(str) == str(record[column])
            table = pd.concat([existing.loc[~identity], table], ignore_index=True)
        table = table.sort_values(identity_columns, kind="stable")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                table.to_csv(stream, index=False, float_format="%.10f", lineterminator="\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_evaluation_report(path: Path, model_name: str, document: dict[str, Any]) -> None:
    probability = document["probability_metrics"]
    operating = document["operating_points"]
    lines = [
        f"# {model_name} evaluation",
        "",
        "The model outputs raw class-weighted probabilities for the RSNA radiographic "
        "challenge target.",
        "",
        "| Split | Average precision | ROC-AUC | Brier | ECE | Calibration slope | "
        "Calibration intercept |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("validation", "test"):
        values = probability[split_name]
        lines.append(
            f"| {split_name} | {values['average_precision']:.6f} | {values['roc_auc']:.6f} | "
            f"{values['brier_score']:.6f} | {values['expected_calibration_error']:.6f} | "
            f"{values['calibration_slope']:.6f} | {values['calibration_intercept']:.6f} |"
        )
    for policy_name, title in (
        ("youden_j", "Youden-J comparative operating point"),
        ("target_sensitivity", "Target-sensitivity operating point"),
    ):
        policy = operating[policy_name]
        threshold = policy.get(f"{policy_name}_threshold")
        if threshold is None:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Validation-derived threshold: `{threshold:.10f}`.",
                "",
                "| Split | Precision | Recall | Specificity | F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for split_name in ("validation", "test"):
            values = policy[split_name]
            lines.append(
                f"| {split_name} | {values['precision']:.6f} | {values['recall']:.6f} | "
                f"{values['specificity']:.6f} | {values['f1']:.6f} |"
            )
    calibration = document["calibration"]
    lines.extend(
        [
            "",
            f"Calibration uses {calibration['calibration_bins']} equal-width probability bins. "
            "The full calculation and latency protocol are defined in `docs/reproducibility.md`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_confusion_summary(path: Path, document: dict[str, Any]) -> None:
    lines = ["# Aggregate test confusion summary", ""]
    for policy_name, title in (
        ("youden_j", "Youden-J comparative operating point"),
        ("target_sensitivity", "Target-sensitivity operating point"),
    ):
        policy = document["operating_points"][policy_name]
        if policy["test"] is None:
            continue
        values = policy["test"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- True negatives: {values['true_negative']:,}",
                f"- False positives: {values['false_positive']:,}",
                f"- False negatives: {values['false_negative']:,}",
                f"- True positives: {values['true_positive']:,}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _frame_source_identifiers(frame: pd.DataFrame) -> set[str]:
    identifiers: set[str] = set()
    for column in ("patient_id", "sample_id", "image_id", "image_path"):
        if column in frame:
            identifiers.update(str(value) for value in frame[column].dropna())
    if "image_path" in frame:
        identifiers.update(Path(value).name for value in frame["image_path"].dropna())
        identifiers.update(Path(value).stem for value in frame["image_path"].dropna())
    return identifiers
