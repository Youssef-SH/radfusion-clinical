"""Generate and publish immutable, validated RSNA artifact bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom

from radfusion.data.artifact_validation import (
    validate_annotation_table,
    validate_label_table,
    validate_sample_table,
)
from radfusion.data.hashing import arrow_ipc_sha256, sha256_file
from radfusion.data.rsna_dicom import AuditAccumulator, read_dicom_metadata
from radfusion.data.rsna_source import (
    RSNA_CLASS_VALUES,
    BoundingBox,
    ManifestBuildError,
    RsnaPaths,
    canonical_image_path,
    discover_dicoms,
    load_source_samples,
    resolve_image_path,
    validate_box_bounds,
    validate_identifier_sets,
)
from radfusion.data.schemas import (
    DATASET_ID,
    MANIFEST_SCHEMA_VERSION,
    PNEUMONIA_LABEL_POLICY_VERSION,
    PNEUMONIA_LABEL_SOURCE,
    PNEUMONIA_TASK_ID,
    RSNA_ANNOTATION_SCHEMA,
    RSNA_CLASS_LABEL_POLICY_VERSION,
    RSNA_CLASS_LABEL_SOURCE,
    RSNA_CLASS_TASK_ID,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
    require_exact_schema,
)

DATASET_VERSION = "rsna-pneumonia-detection-challenge-stage-2"
SAMPLES_FILENAME = "rsna_samples.parquet"
LABELS_FILENAME = "rsna_labels.parquet"
ANNOTATIONS_FILENAME = "rsna_annotations.parquet"
METADATA_FILENAME = "rsna_manifest_metadata.json"
CURRENT_FILENAME = "CURRENT"
BUNDLES_DIRECTORY = "builds"


@dataclass(frozen=True)
class SampleRecord:
    dataset_id: str
    sample_id: str
    patient_id: str
    study_id: str | None
    image_id: str
    image_path: str
    split: str | None
    age_years: float | None
    sex: str | None
    view_position: str | None
    pixel_spacing_row_mm: float | None
    pixel_spacing_col_mm: float | None


@dataclass(frozen=True)
class LabelRecord:
    dataset_id: str
    sample_id: str
    task_id: str
    label_value: int
    label_status: str
    label_source: str
    label_policy_version: str


@dataclass(frozen=True)
class AnnotationRecord:
    dataset_id: str
    sample_id: str
    annotation_id: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class BuildResult:
    samples: pa.Table
    labels: pa.Table
    annotations: pa.Table
    metadata: Mapping[str, Any]
    image_dimensions: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class BundlePaths:
    bundle_id: str
    bundle_directory: Path
    samples_path: Path
    labels_path: Path
    annotations_path: Path
    metadata_path: Path
    current_path: Path


@dataclass(frozen=True)
class WriteResult:
    paths: BundlePaths
    arrow_ipc_sha256: Mapping[str, str]


def build_rsna_artifacts(dataset_root: str | Path) -> BuildResult:
    """Build validated, deterministic in-memory RSNA artifacts."""
    paths = RsnaPaths.from_root(dataset_root)
    paths.validate()
    source_samples = load_source_samples(paths)
    dicom_paths = discover_dicoms(paths.images)
    validate_identifier_sets(set(source_samples), set(dicom_paths), "labels/classes", "DICOMs")

    sample_records: list[SampleRecord] = []
    label_records: list[LabelRecord] = []
    annotation_records: list[AnnotationRecord] = []
    dimensions: dict[str, tuple[int, int]] = {}
    audit = AuditAccumulator.empty()

    for source_id in sorted(source_samples):
        source = source_samples[source_id]
        relative_image_path = canonical_image_path(source_id)
        image_path = resolve_image_path(paths.root, relative_image_path)
        if image_path != dicom_paths[source_id]:
            raise ManifestBuildError(
                f"Discovered DICOM path differs from canonical path for source ID {source_id!r}"
            )
        dicom = read_dicom_metadata(image_path)
        if dicom.patient_id != source_id:
            raise ManifestBuildError(
                f"DICOM PatientID {dicom.patient_id!r} does not match filename/source "
                f"identifier {source_id!r}"
            )
        audit.add(dicom)

        sample_id = f"{DATASET_ID}:{source_id}"
        sample_records.append(
            SampleRecord(
                dataset_id=DATASET_ID,
                sample_id=sample_id,
                patient_id=source_id,
                study_id=None,
                image_id=source_id,
                image_path=relative_image_path.as_posix(),
                split=None,
                age_years=dicom.age.value_years,
                sex=dicom.sex,
                view_position=dicom.view_position,
                pixel_spacing_row_mm=dicom.pixel_spacing_row_mm,
                pixel_spacing_col_mm=dicom.pixel_spacing_col_mm,
            )
        )
        label_records.extend(_label_records(sample_id, source.target, source.rsna_class))

        if dicom.rows is None or dicom.columns is None:
            raise ManifestBuildError(f"Missing image dimensions for DICOM {source_id!r}")
        dimensions[sample_id] = (dicom.rows, dicom.columns)
        annotation_records.extend(
            _annotation_records(
                sample_id,
                source_id,
                source.target,
                source.boxes,
                dimensions[sample_id],
            )
        )

    samples = pa.Table.from_pylist(
        [asdict(record) for record in sample_records], RSNA_SAMPLE_SCHEMA
    )
    labels = pa.Table.from_pylist(
        [asdict(record) for record in sorted(label_records, key=_label_sort_key)],
        RSNA_LABEL_SCHEMA,
    )
    annotations = pa.Table.from_pylist(
        [asdict(record) for record in annotation_records], RSNA_ANNOTATION_SCHEMA
    )
    validate_sample_table(samples, paths.root)
    validate_label_table(labels, samples)
    validate_annotation_table(annotations, samples, labels, dimensions)
    metadata = _build_aggregate_metadata(paths, samples, labels, annotations, audit)
    return BuildResult(samples, labels, annotations, metadata, dimensions)


def write_bundle(result: BuildResult, output_directory: str | Path) -> WriteResult:
    """Publish an immutable bundle and atomically update its CURRENT marker."""
    output_root = Path(output_directory)
    dataset_root = output_root / DATASET_ID
    builds_root = dataset_root / BUNDLES_DIRECTORY
    builds_root.mkdir(parents=True, exist_ok=True)
    current_path = dataset_root / CURRENT_FILENAME

    arrow_hashes = {
        SAMPLES_FILENAME: arrow_ipc_sha256(result.samples),
        LABELS_FILENAME: arrow_ipc_sha256(result.labels),
        ANNOTATIONS_FILENAME: arrow_ipc_sha256(result.annotations),
    }
    bundle_id = _bundle_id(arrow_hashes, result.metadata["source_file_hashes"])
    final_directory = builds_root / bundle_id
    stage_directory = Path(tempfile.mkdtemp(prefix=".staging-", dir=builds_root))

    try:
        staged_paths = _bundle_paths(bundle_id, stage_directory, current_path)
        pq.write_table(result.samples, staged_paths.samples_path, compression="zstd")
        pq.write_table(result.labels, staged_paths.labels_path, compression="zstd")
        pq.write_table(result.annotations, staged_paths.annotations_path, compression="zstd")
        _validate_parquet_round_trip(
            staged_paths.samples_path, result.samples, RSNA_SAMPLE_SCHEMA, "samples"
        )
        _validate_parquet_round_trip(
            staged_paths.labels_path, result.labels, RSNA_LABEL_SCHEMA, "labels"
        )
        _validate_parquet_round_trip(
            staged_paths.annotations_path,
            result.annotations,
            RSNA_ANNOTATION_SCHEMA,
            "annotations",
        )
        metadata = _finalize_metadata(result.metadata, bundle_id, staged_paths, arrow_hashes)
        staged_paths.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_bundle_directory(stage_directory, expected_bundle_id=bundle_id)

        if final_directory.exists():
            validate_bundle_directory(final_directory, expected_bundle_id=bundle_id)
            shutil.rmtree(stage_directory)
        else:
            os.replace(stage_directory, final_directory)
        final_paths = _bundle_paths(bundle_id, final_directory, current_path)
        validate_bundle_directory(final_directory, expected_bundle_id=bundle_id)
        _update_current_marker(current_path, bundle_id)
        loaded = load_current_bundle(output_root)
        if loaded.bundle_id != bundle_id:
            raise ManifestBuildError("CURRENT did not resolve to the newly published bundle")
    finally:
        if stage_directory.exists():
            shutil.rmtree(stage_directory)

    return WriteResult(final_paths, arrow_hashes)


def build_and_write(dataset_root: str | Path, output_directory: str | Path) -> WriteResult:
    return write_bundle(build_rsna_artifacts(dataset_root), output_directory)


def load_current_bundle(output_directory: str | Path) -> BundlePaths:
    """Resolve CURRENT only after verifying metadata and every declared artifact hash."""
    dataset_root = Path(output_directory) / DATASET_ID
    current_path = dataset_root / CURRENT_FILENAME
    if not current_path.is_file():
        raise ManifestBuildError(f"Missing CURRENT marker: {current_path}")
    bundle_id = current_path.read_text(encoding="utf-8").strip()
    if not bundle_id or Path(bundle_id).name != bundle_id:
        raise ManifestBuildError("CURRENT contains an invalid bundle identifier")
    bundle_directory = dataset_root / BUNDLES_DIRECTORY / bundle_id
    validate_bundle_directory(bundle_directory, expected_bundle_id=bundle_id)
    return _bundle_paths(bundle_id, bundle_directory, current_path)


def validate_bundle_directory(
    bundle_directory: str | Path, *, expected_bundle_id: str | None = None
) -> dict[str, Any]:
    """Require complete metadata plus matching file and Arrow IPC hashes for a bundle."""
    directory = Path(bundle_directory)
    metadata_path = directory / METADATA_FILENAME
    if not metadata_path.is_file():
        raise ManifestBuildError(f"Bundle metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"Bundle metadata is unreadable: {metadata_path}") from exc
    bundle_id = metadata.get("bundle", {}).get("bundle_id")
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ManifestBuildError(
            f"Bundle metadata ID {bundle_id!r} does not match {expected_bundle_id!r}"
        )
    expected_files = {
        SAMPLES_FILENAME: RSNA_SAMPLE_SCHEMA,
        LABELS_FILENAME: RSNA_LABEL_SCHEMA,
        ANNOTATIONS_FILENAME: RSNA_ANNOTATION_SCHEMA,
    }
    hashes = metadata.get("generated_artifact_hashes", {})
    for filename, schema in expected_files.items():
        path = directory / filename
        declared = hashes.get(filename)
        if not path.is_file() or not isinstance(declared, dict):
            raise ManifestBuildError(f"Bundle is incomplete: {filename}")
        if sha256_file(path) != declared.get("file_sha256"):
            raise ManifestBuildError(f"File hash mismatch for {filename}")
        table = pq.read_table(path)
        try:
            require_exact_schema(table, schema, filename)
        except ValueError as exc:
            raise ManifestBuildError(str(exc)) from exc
        if arrow_ipc_sha256(table) != declared.get("arrow_ipc_sha256"):
            raise ManifestBuildError(f"Arrow IPC hash mismatch for {filename}")
    return metadata


def _label_records(sample_id: str, target: int, rsna_class: str) -> list[LabelRecord]:
    return [
        LabelRecord(
            DATASET_ID,
            sample_id,
            PNEUMONIA_TASK_ID,
            target,
            "observed",
            PNEUMONIA_LABEL_SOURCE,
            PNEUMONIA_LABEL_POLICY_VERSION,
        ),
        LabelRecord(
            DATASET_ID,
            sample_id,
            RSNA_CLASS_TASK_ID,
            RSNA_CLASS_VALUES[rsna_class],
            "observed",
            RSNA_CLASS_LABEL_SOURCE,
            RSNA_CLASS_LABEL_POLICY_VERSION,
        ),
    ]


def _annotation_records(
    sample_id: str,
    image_id: str,
    target: int,
    boxes: tuple[BoundingBox, ...],
    dimensions: tuple[int, int],
) -> list[AnnotationRecord]:
    if target == 0:
        if boxes:
            raise ManifestBuildError(f"Negative sample {sample_id!r} must not contain boxes")
        return []
    records: list[AnnotationRecord] = []
    for index, box in enumerate(boxes):
        validate_box_bounds(box, dimensions, sample_id)
        records.append(
            AnnotationRecord(
                DATASET_ID,
                sample_id,
                f"{DATASET_ID}:{image_id}:bbox:{index:04d}",
                box.x,
                box.y,
                box.width,
                box.height,
            )
        )
    return records


def _build_aggregate_metadata(
    paths: RsnaPaths,
    samples: pa.Table,
    labels: pa.Table,
    annotations: pa.Table,
    audit: AuditAccumulator,
) -> dict[str, Any]:
    sample_rows = samples.to_pylist()
    label_rows = labels.to_pylist()
    pneumonia = Counter(
        str(row["label_value"]) for row in label_rows if row["task_id"] == PNEUMONIA_TASK_ID
    )
    rsna_classes = Counter(
        str(row["label_value"]) for row in label_rows if row["task_id"] == RSNA_CLASS_TASK_ID
    )
    class_names = {str(value): name for name, value in RSNA_CLASS_VALUES.items()}
    sex = Counter(row["sex"] or "<missing>" for row in sample_rows)
    views = Counter(row["view_position"] or "<missing>" for row in sample_rows)
    spacing = Counter(
        f"{row['pixel_spacing_row_mm']:.12g},{row['pixel_spacing_col_mm']:.12g}"
        if row["pixel_spacing_row_mm"] is not None
        else "<missing>"
        for row in sample_rows
    )
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "label_policy_versions": {
            PNEUMONIA_TASK_ID: PNEUMONIA_LABEL_POLICY_VERSION,
            RSNA_CLASS_TASK_ID: RSNA_CLASS_LABEL_POLICY_VERSION,
        },
        "sample_count": samples.num_rows,
        "label_count": labels.num_rows,
        "positive_count": pneumonia["1"],
        "negative_count": pneumonia["0"],
        "annotation_count": annotations.num_rows,
        "image_dimensions_summary": {
            "rows": _sorted_counter(audit.dicom_values.get("Rows", Counter())),
            "columns": _sorted_counter(audit.dicom_values.get("Columns", Counter())),
        },
        "photometric_interpretation_summary": _sorted_counter(
            audit.dicom_values.get("PhotometricInterpretation", Counter())
        ),
        "transfer_syntax_summary": _sorted_counter(
            audit.dicom_values.get("TransferSyntaxUID", Counter())
        ),
        "pixel_spacing_summary": {
            "unique_pair_count": len(spacing),
            "value_counts": _sorted_counter(spacing),
        },
        "age_parsing_summary": {
            "status_counts": _sorted_counter(audit.age_status),
            "source_format_counts": _sorted_counter(audit.age_source_format),
            "warning_counts": _sorted_counter(audit.age_warnings),
        },
        "implausible_age_count": audit.implausible_age_count,
        "sex_distribution": _sorted_counter(sex),
        "view_distribution": _sorted_counter(views),
        "rsna_class_distribution": {
            class_names[key]: rsna_classes[key] for key in sorted(rsna_classes)
        },
        "dicom_audit": {
            "field_value_counts": {
                keyword: _sorted_counter(counter)
                for keyword, counter in sorted(audit.dicom_values.items())
            },
            "uid_uniqueness": {
                "sop_instance_uid_unique_count": len(audit.sop_instance_uids),
                "study_instance_uid_unique_count": len(audit.study_instance_uids),
                "series_instance_uid_unique_count": len(audit.series_instance_uids),
                "media_storage_sop_instance_uid_unique_count": len(
                    audit.media_storage_sop_instance_uids
                ),
                "media_storage_matches_sop_count": audit.media_sop_matches,
            },
        },
        "source_file_hashes": {
            paths.labels.name: {"algorithm": "sha256", "file_sha256": sha256_file(paths.labels)},
            paths.class_info.name: {
                "algorithm": "sha256",
                "file_sha256": sha256_file(paths.class_info),
            },
        },
        "hash_policy": {
            "source_files": "SHA-256 over source CSV bytes",
            "arrow_ipc_sha256": "SHA-256 over ordered Arrow IPC stream including exact schema",
            "arrow_ipc_pyarrow_version": pa.__version__,
            "arrow_ipc_stability_scope": (
                "Deterministic only for the recorded PyArrow version; cross-version stability "
                "is not claimed"
            ),
            "artifact_files": "SHA-256 over generated Parquet bytes",
            "dicom_files": "Individual DICOM files are not hashed in M1",
        },
    }


def _finalize_metadata(
    base: Mapping[str, Any],
    bundle_id: str,
    paths: BundlePaths,
    arrow_hashes: Mapping[str, str],
) -> dict[str, Any]:
    artifact_paths = {
        SAMPLES_FILENAME: paths.samples_path,
        LABELS_FILENAME: paths.labels_path,
        ANNOTATIONS_FILENAME: paths.annotations_path,
    }
    return {
        **base,
        "bundle": {
            "bundle_id": bundle_id,
            "publication_model": "immutable-build-directory-with-atomic-CURRENT-marker",
        },
        "generated_artifact_hashes": {
            filename: {
                "arrow_ipc_sha256": arrow_hashes[filename],
                "file_sha256": sha256_file(path),
            }
            for filename, path in artifact_paths.items()
        },
        "generation": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": "python -m radfusion.data.rsna_manifest",
            "python_version": platform.python_version(),
            "pydicom_version": pydicom.__version__,
            "pyarrow_version": pa.__version__,
            "pandas_version": pd.__version__,
        },
    }


def _bundle_id(arrow_hashes: Mapping[str, str], source_hashes: Mapping[str, Any]) -> str:
    identity = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "arrow_ipc_sha256": dict(sorted(arrow_hashes.items())),
        "source_file_hashes": source_hashes,
        "pyarrow_version": pa.__version__,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"build-{digest}"


def _bundle_paths(bundle_id: str, directory: Path, current_path: Path) -> BundlePaths:
    return BundlePaths(
        bundle_id,
        directory,
        directory / SAMPLES_FILENAME,
        directory / LABELS_FILENAME,
        directory / ANNOTATIONS_FILENAME,
        directory / METADATA_FILENAME,
        current_path,
    )


def _update_current_marker(current_path: Path, bundle_id: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".CURRENT-", suffix=".tmp", dir=current_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(bundle_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, current_path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_parquet_round_trip(
    path: Path, expected: pa.Table, schema: pa.Schema, artifact_name: str
) -> None:
    restored = pq.read_table(path)
    try:
        require_exact_schema(restored, schema, artifact_name)
    except ValueError as exc:
        raise ManifestBuildError(str(exc)) from exc
    if restored.num_rows != expected.num_rows or not restored.equals(expected):
        raise ManifestBuildError(f"{artifact_name} changed during Parquet round-trip")


def _label_sort_key(record: LabelRecord) -> tuple[str, str]:
    return record.sample_id, record.task_id


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}
