"""Publish and validate compact run-qualified local model packages."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radfusion.data.hashing import sha256_file
from radfusion.utils.skops_io import trusted_types_for_file

MODEL_FILENAME = "model.skops"
CONFIG_FILENAME = "resolved_config.yaml"
MANIFEST_FILENAME = "model_manifest.json"
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "model_sha256",
        "training_mlflow_run_id",
        "bundle_id",
        "split_assignment_id",
        "task",
        "positive_class",
        "source_config_sha256",
        "seed",
        "git_commit",
        "dependency_lock_sha256",
        "best_iteration",
        "thresholds",
    }
)


@dataclass(frozen=True)
class PublishedModel:
    """Paths and physical identity for one local training-run package."""

    run_directory: Path
    model_path: Path
    config_path: Path
    manifest_path: Path
    model_sha256: str
    model_size_mib: float


def publish_model_run(
    *,
    model_root: str | Path,
    mlflow_run_id: str,
    serialized_model_path: str | Path,
    source_config_bytes: bytes,
    manifest: Mapping[str, Any],
) -> PublishedModel:
    """Publish one complete model package atomically."""
    _validate_component(mlflow_run_id, "mlflow_run_id")
    runs_root = Path(model_root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final = runs_root / mlflow_run_id
    stage = Path(tempfile.mkdtemp(prefix=f".{mlflow_run_id}-", dir=runs_root))
    try:
        model_path = stage / MODEL_FILENAME
        config_path = stage / CONFIG_FILENAME
        shutil.copyfile(serialized_model_path, model_path)
        config_path.write_bytes(source_config_bytes)
        trusted_types_for_file(model_path)
        document = {
            **dict(manifest),
            "training_mlflow_run_id": mlflow_run_id,
            "model_sha256": sha256_file(model_path),
        }
        _validate_manifest(document, mlflow_run_id, model_path, config_path)
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if final.exists():
            existing = validate_published_model(final)
            if existing != document:
                raise FileExistsError(f"Model run exists with conflicting content: {final}")
        else:
            os.replace(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    model_path = final / MODEL_FILENAME
    return PublishedModel(
        run_directory=final,
        model_path=model_path,
        config_path=final / CONFIG_FILENAME,
        manifest_path=final / MANIFEST_FILENAME,
        model_sha256=sha256_file(model_path),
        model_size_mib=model_path.stat().st_size / (1024.0 * 1024.0),
    )


def validate_published_model(run_directory: str | Path) -> dict[str, Any]:
    """Validate one run-qualified model package and return its manifest."""
    directory = Path(run_directory)
    if directory.parent.name != "runs" or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Model run must be a physical directory beneath runs")
    expected = {MODEL_FILENAME, CONFIG_FILENAME, MANIFEST_FILENAME}
    with os.scandir(directory) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != expected:
        raise ValueError("Model run contains an unexpected artifact set")
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Model run entries must be regular non-symlink files")
    try:
        document = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Model manifest is unreadable") from exc
    _validate_manifest(
        document,
        directory.name,
        directory / MODEL_FILENAME,
        directory / CONFIG_FILENAME,
    )
    trusted_types_for_file(directory / MODEL_FILENAME)
    return document


def _validate_manifest(
    document: Mapping[str, Any],
    run_id: str,
    model_path: Path,
    config_path: Path,
) -> None:
    if set(document) != REQUIRED_MANIFEST_FIELDS:
        raise ValueError("Model manifest contains an unexpected field set")
    if document["training_mlflow_run_id"] != run_id:
        raise ValueError("Model manifest run ID does not match its path")
    for field in ("bundle_id", "split_assignment_id", "task", "git_commit"):
        if not isinstance(document[field], str) or not document[field]:
            raise ValueError(f"Model manifest {field} must be a non-empty string")
    for field in ("model_sha256", "source_config_sha256", "dependency_lock_sha256"):
        if not _is_sha256(document[field]):
            raise ValueError(f"Model manifest {field} must be a lowercase SHA-256")
    if document["model_sha256"] != sha256_file(model_path):
        raise ValueError("Model SHA-256 does not match model bytes")
    if document["source_config_sha256"] != sha256_file(config_path):
        raise ValueError("Source config SHA-256 does not match archived config bytes")
    if document["positive_class"] != 1:
        raise ValueError("Model manifest positive class must be 1")
    if isinstance(document["seed"], bool) or not isinstance(document["seed"], int):
        raise ValueError("Model manifest seed must be an integer")
    best_iteration = document["best_iteration"]
    if best_iteration is not None and (
        isinstance(best_iteration, bool)
        or not isinstance(best_iteration, int)
        or best_iteration <= 0
    ):
        raise ValueError("Model manifest best_iteration must be null or positive")
    thresholds = document["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "youden_j",
        "target_sensitivity",
    }:
        raise ValueError("Model manifest thresholds are invalid")
    for value in thresholds.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Model manifest thresholds must be finite probabilities")


def _validate_component(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"{field} must be one safe path component")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
