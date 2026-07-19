"""Publish immutable local model runs with validated lineage."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radfusion.data.hashing import sha256_file
from radfusion.utils.skops_io import save_skops, trusted_types_for_file

MODEL_FILENAME = "model.skops"
LINEAGE_FILENAME = "lineage.json"
REQUIRED_LINEAGE_FIELDS = frozenset(
    {
        "mlflow_run_id",
        "dataset_id",
        "task_id",
        "model_key",
        "config_source_sha256",
        "bundle_id",
        "split_recipe_id",
        "split_assignment_id",
        "label_policy_version",
        "training_seed",
        "git_commit",
        "git_dirty",
        "git_source_state_sha256",
        "uv_lock_sha256",
        "model_artifact_sha256",
        "model_size_mib",
        "positive_class_label",
        "derived_parameters",
        "generation_timestamp_utc",
    }
)


@dataclass(frozen=True)
class PublishedModel:
    """Paths and physical identity for one immutable local model run."""

    run_directory: Path
    model_path: Path
    lineage_path: Path
    model_artifact_sha256: str
    model_size_mib: float


def publish_model_run(
    model: Any | None,
    *,
    model_root: str | Path,
    model_key: str,
    mlflow_run_id: str,
    lineage: Mapping[str, Any],
    serialized_model_path: str | Path | None = None,
) -> PublishedModel:
    """Publish one run atomically and update its model-key CURRENT pointer."""
    _validate_path_component(model_key, "model_key")
    _validate_path_component(mlflow_run_id, "mlflow_run_id")
    key_root = Path(model_root) / model_key
    runs_root = key_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    if key_root.is_symlink() or runs_root.is_symlink():
        raise ValueError("Model publication directories must not be symbolic links")
    final = runs_root / mlflow_run_id
    stage = Path(tempfile.mkdtemp(prefix=f".{mlflow_run_id}-", dir=runs_root))
    try:
        model_path = stage / MODEL_FILENAME
        if serialized_model_path is None:
            if model is None:
                raise ValueError("A model or serialized_model_path is required")
            save_skops(model, model_path)
        else:
            shutil.copy2(Path(serialized_model_path), model_path)
        trusted_types_for_file(model_path)
        model_hash = sha256_file(model_path)
        model_size_mib = model_path.stat().st_size / (1024.0 * 1024.0)
        document = {
            **dict(lineage),
            "mlflow_run_id": mlflow_run_id,
            "model_key": model_key,
            "model_artifact_sha256": model_hash,
            "model_size_mib": model_size_mib,
            "positive_class_label": 1,
            "generation_timestamp_utc": datetime.now(UTC).isoformat(),
        }
        _validate_lineage(document, mlflow_run_id, model_key, model_path)
        (stage / LINEAGE_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if final.exists() or final.is_symlink():
            existing = validate_published_model(final)
            if not _same_publication(existing, document):
                raise FileExistsError(
                    f"Immutable model run exists with conflicting content: {final}"
                )
        else:
            os.replace(stage, final)
        _update_current(key_root / "CURRENT", mlflow_run_id)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return PublishedModel(
        run_directory=final,
        model_path=final / MODEL_FILENAME,
        lineage_path=final / LINEAGE_FILENAME,
        model_artifact_sha256=model_hash,
        model_size_mib=model_size_mib,
    )


def validate_published_model(run_directory: str | Path) -> dict[str, Any]:
    """Validate one immutable model directory and return its lineage."""
    directory = Path(run_directory)
    if directory.parent.name != "runs":
        raise ValueError("Model run path must be nested beneath a runs directory")
    if directory.parent.is_symlink() or directory.parent.parent.is_symlink():
        raise ValueError("Model run parent directories must not be symbolic links")
    physical_model_key = directory.parent.parent.name
    _validate_path_component(directory.name, "mlflow_run_id")
    _validate_path_component(physical_model_key, "model_key")
    _require_exact_regular_entries(directory)
    model_path = directory / MODEL_FILENAME
    lineage_path = directory / LINEAGE_FILENAME
    try:
        document = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Model lineage is unreadable: {lineage_path}") from exc
    _validate_lineage(document, directory.name, physical_model_key, model_path)
    trusted_types_for_file(model_path)
    return document


def _require_exact_regular_entries(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Model run is not a physical directory: {directory}")
    with os.scandir(directory) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != {MODEL_FILENAME, LINEAGE_FILENAME}:
        raise ValueError(f"Model run contains an unexpected artifact set: {directory}")
    for entry in inspected:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError(f"Model run entry must be a regular non-symlink file: {entry.name}")


def _same_publication(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    ignored = {"generation_timestamp_utc"}
    return {key: value for key, value in existing.items() if key not in ignored} == {
        key: value for key, value in candidate.items() if key not in ignored
    }


def _validate_lineage(
    document: Mapping[str, Any], run_id: str, model_key: str | None, model_path: Path
) -> None:
    missing = sorted(REQUIRED_LINEAGE_FIELDS - document.keys())
    if missing:
        raise ValueError(f"Model lineage is missing fields: {missing}")
    if document["mlflow_run_id"] != run_id or document["model_key"] != model_key:
        raise ValueError("Model lineage does not match its immutable run path")
    for field in (
        "mlflow_run_id",
        "dataset_id",
        "task_id",
        "model_key",
        "bundle_id",
        "split_recipe_id",
        "split_assignment_id",
        "label_policy_version",
        "git_commit",
    ):
        if not isinstance(document[field], str) or not document[field]:
            raise ValueError(f"Model lineage {field} must be a non-empty string")
    for field in ("config_source_sha256", "uv_lock_sha256", "model_artifact_sha256"):
        if not _is_sha256(document[field]):
            raise ValueError(f"Model lineage {field} must be a lowercase SHA-256")
    if isinstance(document["training_seed"], bool) or not isinstance(
        document["training_seed"], int
    ):
        raise ValueError("Model lineage training_seed must be an integer")
    if not isinstance(document["git_dirty"], bool):
        raise ValueError("Model lineage git_dirty must be boolean")
    source_state = document["git_source_state_sha256"]
    if (document["git_dirty"] and not _is_sha256(source_state)) or (
        not document["git_dirty"] and source_state != "clean"
    ):
        raise ValueError("Model lineage source-state identity does not match git_dirty")
    if not isinstance(document["derived_parameters"], dict):
        raise ValueError("Model lineage derived_parameters must be a mapping")
    if document["positive_class_label"] != 1:
        raise ValueError("Model lineage positive class must be 1")
    if document["model_artifact_sha256"] != sha256_file(model_path):
        raise ValueError("Model artifact SHA-256 does not match lineage")
    actual_mib = model_path.stat().st_size / (1024.0 * 1024.0)
    if (
        isinstance(document["model_size_mib"], bool)
        or not isinstance(document["model_size_mib"], int | float)
        or not math.isfinite(document["model_size_mib"])
        or document["model_size_mib"] <= 0
        or document["model_size_mib"] != actual_mib
    ):
        raise ValueError("Model artifact size does not match lineage")
    try:
        timestamp = datetime.fromisoformat(document["generation_timestamp_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Model lineage generation timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise ValueError("Model lineage generation timestamp must include a timezone")


def _validate_path_component(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be one safe path component")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _update_current(path: Path, run_id: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".CURRENT-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(run_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
