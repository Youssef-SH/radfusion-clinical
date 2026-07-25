"""Regenerate experiment comparison views from completed MLflow runs."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from radfusion.utils.mlflow_utils import DEFAULT_TRACKING_URI, configure_mlflow

COMPARISON_COLUMNS = (
    "run_id",
    "parent_training_run_id",
    "experiment_name",
    "model",
    "modality",
    "task",
    "model_package_id",
    "bundle_id",
    "split_assignment_id",
    "seed",
    "evaluation_scope",
    "average_precision",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
    "youden_j_threshold",
    "youden_j_precision",
    "youden_j_recall",
    "youden_j_specificity",
    "youden_j_f1",
    "target_sensitivity_threshold",
    "target_sensitivity_precision",
    "target_sensitivity_recall",
    "target_sensitivity_specificity",
    "target_sensitivity_f1",
    "latency_ms",
    "model_size_mib",
)
_COMMON_TAGS = (
    "experiment_name",
    "model",
    "model_package_id",
    "dataset_bundle_id",
    "split_assignment_id",
    "seed",
    "evaluation_scope",
    "run_kind",
    "task",
)
_PREFIXED_METRICS = (
    "average_precision",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
    "youden_j_threshold",
    "youden_j_precision",
    "youden_j_recall",
    "youden_j_specificity",
    "youden_j_f1",
    "target_sensitivity_threshold",
    "target_sensitivity_precision",
    "target_sensitivity_recall",
    "target_sensitivity_specificity",
    "target_sensitivity_f1",
    "latency_ms",
)
_BOUNDED_METRICS = frozenset(
    {
        "average_precision",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "youden_j_threshold",
        "youden_j_precision",
        "youden_j_recall",
        "youden_j_specificity",
        "youden_j_f1",
        "target_sensitivity_threshold",
        "target_sensitivity_precision",
        "target_sensitivity_recall",
        "target_sensitivity_specificity",
        "target_sensitivity_f1",
    }
)


def regenerate_comparison(
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    output_directory: str | Path = "reports",
) -> tuple[Path, Path, int]:
    """Write deterministic CSV and Markdown views from complete MLflow runs."""
    client = configure_mlflow(tracking_uri=tracking_uri)
    experiment_ids = [item.experiment_id for item in client.search_experiments()]
    runs = (
        client.search_runs(
            experiment_ids=experiment_ids,
            filter_string="attributes.status = 'FINISHED'",
            max_results=50_000,
        )
        if experiment_ids
        else []
    )
    candidates = []
    for run in runs:
        if record := _comparison_record(run):
            candidates.append(record)
    training = {
        record["run_id"]: record
        for record in candidates
        if record["evaluation_scope"] == "validation"
    }
    records = [
        record
        for record in candidates
        if (record["evaluation_scope"] == "validation" and record["modality"] == "metadata")
        or _has_matching_training_parent(record, training)
    ]
    records.sort(
        key=lambda item: (
            item["experiment_name"],
            item["model"],
            item["evaluation_scope"],
            item["run_id"],
        )
    )
    table = pd.DataFrame.from_records(records, columns=COMPARISON_COLUMNS)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "model_comparison_table.csv"
    markdown_path = output / "model_comparison_table.md"
    _atomic_write_csv(table, csv_path)
    _atomic_write_text(
        markdown_path,
        "# Experiment comparison\n\n"
        + ("No completed runs are available.\n" if table.empty else _markdown_table(table)),
    )
    return csv_path, markdown_path, len(table)


def _has_matching_training_parent(
    test_record: dict[str, object],
    training_records: dict[object, dict[str, object]],
) -> bool:
    parent = training_records.get(test_record["parent_training_run_id"])
    return parent is not None and all(
        test_record[field] == parent[field]
        for field in (
            "experiment_name",
            "model",
            "modality",
            "task",
            "model_package_id",
            "bundle_id",
            "split_assignment_id",
            "seed",
        )
    )


def _comparison_record(run) -> dict[str, object] | None:
    tags = run.data.tags
    if tags.get("run_complete") != "true" or any(
        not isinstance(tags.get(key), str) or not tags[key].strip() for key in _COMMON_TAGS
    ):
        return None
    scope = tags["evaluation_scope"]
    kind = tags["run_kind"]
    if (kind, scope) not in {
        ("training", "validation"),
        ("test_evaluation", "test"),
    }:
        return None
    parent = tags.get("source_training_run_id", "")
    if scope == "test" and (not isinstance(parent, str) or not parent.strip()):
        return None
    metrics = {name: run.data.metrics.get(f"{scope}_{name}") for name in _PREFIXED_METRICS}
    metrics["model_size_mib"] = run.data.metrics.get("model_size_mib")
    modality = tags.get("modality", "metadata")
    if modality not in {"metadata", "image"} or not _valid_metrics(metrics, modality=modality):
        return None
    return {
        "run_id": run.info.run_id,
        "parent_training_run_id": parent,
        "experiment_name": tags["experiment_name"],
        "model": tags["model"],
        "modality": modality,
        "task": tags["task"],
        "model_package_id": tags["model_package_id"],
        "bundle_id": tags["dataset_bundle_id"],
        "split_assignment_id": tags["split_assignment_id"],
        "seed": tags["seed"],
        "evaluation_scope": scope,
        **metrics,
    }


def _valid_metrics(metrics: dict[str, object], *, modality: str) -> bool:
    for name, value in metrics.items():
        if name == "latency_ms" and modality == "image" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            return False
        if name in _BOUNDED_METRICS and not 0.0 <= value <= 1.0:
            return False
    return (modality == "image" or metrics["latency_ms"] >= 0.0) and metrics["model_size_mib"] > 0.0


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False, float_format="%.10f", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown_table(table: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow SQLite tracking URI",
    )
    parser.add_argument("--output-directory", type=Path, default=Path("reports"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate both comparison views and print their row count."""
    args = _parser().parse_args(argv)
    try:
        csv_path, markdown_path, rows = regenerate_comparison(
            tracking_uri=args.tracking_uri,
            output_directory=args.output_directory,
        )
    except (MlflowException, SQLAlchemyError, OSError, ValueError) as exc:
        print(f"Comparison generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Generated {rows} rows in {csv_path} and {markdown_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
