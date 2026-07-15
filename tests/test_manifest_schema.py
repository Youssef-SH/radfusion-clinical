from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from radfusion.data.artifact_validation import (
    validate_annotation_table,
    validate_label_table,
    validate_sample_table,
)
from radfusion.data.hashing import arrow_ipc_sha256
from radfusion.data.rsna_artifacts import (
    CURRENT_FILENAME,
    SAMPLES_FILENAME,
    BuildResult,
    build_rsna_artifacts,
    load_current_bundle,
    write_bundle,
)
from radfusion.data.rsna_manifest import main
from radfusion.data.rsna_source import ManifestBuildError, aggregate_labels
from radfusion.data.schemas import (
    PNEUMONIA_TASK_ID,
    RSNA_ANNOTATION_SCHEMA,
    RSNA_CLASS_TASK_ID,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
)


def _write_header(
    path: Path,
    patient_id: str | None,
    *,
    age: str | None = "057Y",
    sex: str | None = "F",
    view: str | None = "PA",
    spacing: tuple[float, float] | None = (0.168, 0.168),
    rows: int = 1024,
    columns: int = 1024,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    sop_instance_uid = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    if patient_id is not None:
        dataset.PatientID = patient_id
    if age is not None:
        dataset.PatientAge = age
    if sex is not None:
        dataset.PatientSex = sex
    if view is not None:
        dataset.ViewPosition = view
    if spacing is not None:
        dataset.PixelSpacing = list(spacing)
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Modality = "CR"
    dataset.BodyPartExamined = "CHEST"
    dataset.save_as(path)


def _write_sources(
    root: Path,
    *,
    positive: bool = True,
    patient_id: str = "positive",
    age: str | None = "057Y",
    sex: str | None = "F",
    view: str | None = "PA",
    spacing: tuple[float, float] | None = (0.168, 0.168),
) -> Path:
    images = root / "stage_2_train_images"
    images.mkdir(parents=True)
    labels = [
        {
            "patientId": "negative",
            "x": None,
            "y": None,
            "width": None,
            "height": None,
            "Target": 0,
        }
    ]
    classes = [{"patientId": "negative", "class": "Normal"}]
    _write_header(images / "negative.dcm", "negative")
    if positive:
        labels.extend(
            [
                {
                    "patientId": patient_id,
                    "x": 1,
                    "y": 2,
                    "width": 3,
                    "height": 4,
                    "Target": 1,
                },
                {
                    "patientId": patient_id,
                    "x": 5,
                    "y": 6,
                    "width": 7,
                    "height": 8,
                    "Target": 1,
                },
            ]
        )
        classes.extend(
            [
                {"patientId": patient_id, "class": "Lung Opacity"},
                {"patientId": patient_id, "class": "Lung Opacity"},
            ]
        )
        _write_header(
            images / f"{patient_id}.dcm",
            patient_id,
            age=age,
            sex=sex,
            view=view,
            spacing=spacing,
        )
    pd.DataFrame(labels).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return root


def _tables(tmp_path: Path) -> tuple[Path, BuildResult]:
    root = _write_sources(tmp_path / "extracted")
    return root, build_rsna_artifacts(root)


def _replace_table_row(table: pa.Table, row_index: int, **changes: object) -> pa.Table:
    rows = table.to_pylist()
    rows[row_index].update(changes)
    return pa.Table.from_pylist(rows, schema=table.schema)


def test_happy_path_builds_canonical_samples_and_multiple_annotations(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)

    assert result.samples.schema == RSNA_SAMPLE_SCHEMA
    assert result.labels.schema == RSNA_LABEL_SCHEMA
    assert result.annotations.schema == RSNA_ANNOTATION_SCHEMA
    assert result.samples.num_rows == 2
    assert result.labels.num_rows == 4
    assert result.annotations.num_rows == 2
    positive = result.samples.to_pylist()[1]
    assert positive["sample_id"] == "rsna:positive"
    assert positive["study_id"] is None
    assert positive["split"] is None
    assert positive["image_path"] == "stage_2_train_images/positive.dcm"
    assert "target_pneumonia" not in positive
    assert "rsna_class" not in positive
    assert not Path(positive["image_path"]).is_absolute()
    validate_sample_table(result.samples, root)
    validate_label_table(result.labels, result.samples)

    positive_labels = {
        row["task_id"]: row["label_value"]
        for row in result.labels.to_pylist()
        if row["sample_id"] == "rsna:positive"
    }
    assert positive_labels == {PNEUMONIA_TASK_ID: 1, RSNA_CLASS_TASK_ID: 2}


def test_all_negative_fixture_has_empty_typed_annotations(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted", positive=False)
    result = build_rsna_artifacts(root)

    assert result.samples.num_rows == 1
    assert result.labels.num_rows == 2
    assert result.annotations.num_rows == 0
    assert result.annotations.schema == RSNA_ANNOTATION_SCHEMA


@pytest.mark.parametrize("column", ["patientId", "x", "Target"])
def test_missing_label_column_fails(tmp_path: Path, column: str) -> None:
    root = _write_sources(tmp_path / "extracted")
    labels_path = root / "stage_2_train_labels.csv"
    pd.read_csv(labels_path).drop(columns=column).to_csv(labels_path, index=False)

    with pytest.raises(ManifestBuildError, match="missing required columns"):
        build_rsna_artifacts(root)


def test_missing_class_column_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    class_path = root / "stage_2_detailed_class_info.csv"
    pd.read_csv(class_path).drop(columns="class").to_csv(class_path, index=False)

    with pytest.raises(ManifestBuildError, match="missing required columns"):
        build_rsna_artifacts(root)


def test_inconsistent_targets_fail(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    labels_path = root / "stage_2_train_labels.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[labels.index[-1], "Target"] = 0
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ManifestBuildError, match="consistent binary"):
        build_rsna_artifacts(root)


def test_inconsistent_classes_fail(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    class_path = root / "stage_2_detailed_class_info.csv"
    classes = pd.read_csv(class_path)
    classes.loc[classes.index[-1], "class"] = "Normal"
    classes.to_csv(class_path, index=False)

    with pytest.raises(ManifestBuildError, match="one consistent"):
        build_rsna_artifacts(root)


@pytest.mark.parametrize("missing_from", ["labels", "classes", "dicoms"])
def test_source_identifier_mismatches_fail(tmp_path: Path, missing_from: str) -> None:
    root = _write_sources(tmp_path / "extracted")
    if missing_from == "labels":
        path = root / "stage_2_train_labels.csv"
        frame = pd.read_csv(path)
        frame[frame["patientId"] != "positive"].to_csv(path, index=False)
    elif missing_from == "classes":
        path = root / "stage_2_detailed_class_info.csv"
        frame = pd.read_csv(path)
        frame[frame["patientId"] != "positive"].to_csv(path, index=False)
    else:
        (root / "stage_2_train_images" / "positive.dcm").unlink()

    with pytest.raises(ManifestBuildError, match="Source identifier mismatch"):
        build_rsna_artifacts(root)


def test_extra_dicom_without_label_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "extra.dcm", "extra")

    with pytest.raises(ManifestBuildError, match="Source identifier mismatch"):
        build_rsna_artifacts(root)


def test_duplicate_image_identifier_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.DCM", "positive")

    with pytest.raises(ManifestBuildError, match="Duplicate DICOM"):
        build_rsna_artifacts(root)


def test_missing_dicom_patient_id_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.dcm", None)

    with pytest.raises(ManifestBuildError, match="PatientID is missing"):
        build_rsna_artifacts(root)


def test_filename_dicom_patient_id_mismatch_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.dcm", "different")

    with pytest.raises(ManifestBuildError, match="does not match"):
        build_rsna_artifacts(root)


def test_unreadable_dicom_fails_clearly(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    (root / "stage_2_train_images" / "positive.dcm").write_bytes(b"not a dicom")

    with pytest.raises(ManifestBuildError, match="Could not read DICOM metadata"):
        build_rsna_artifacts(root)


def test_missing_optional_metadata_is_preserved_as_null(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted", age=None, sex=None, view=None, spacing=None)
    result = build_rsna_artifacts(root)
    positive = result.samples.to_pylist()[1]

    assert positive["age_years"] is None
    assert positive["sex"] is None
    assert positive["view_position"] is None
    assert positive["pixel_spacing_row_mm"] is None
    assert result.metadata["age_parsing_summary"]["status_counts"]["missing"] == 1


def test_malformed_and_implausible_ages_are_reported_in_aggregate(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="Invalid value for VR AS"):
        malformed_root = _write_sources(tmp_path / "malformed", age="BAD")
    malformed = build_rsna_artifacts(malformed_root)
    assert malformed.samples.to_pylist()[1]["age_years"] is None
    assert malformed.metadata["age_parsing_summary"]["status_counts"]["malformed"] == 1

    with pytest.warns(UserWarning, match="Invalid value for VR AS"):
        implausible_root = _write_sources(tmp_path / "implausible", age="155")
    implausible = build_rsna_artifacts(implausible_root)
    assert implausible.samples.to_pylist()[1]["age_years"] == 155.0
    assert implausible.metadata["implausible_age_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("sex", "X", "PatientSex"), ("view", "LL", "ViewPosition")],
)
def test_invalid_categories_fail(tmp_path: Path, field: str, value: str, message: str) -> None:
    kwargs = {field: value}
    root = _write_sources(tmp_path / "extracted", **kwargs)

    with pytest.raises(ManifestBuildError, match=message):
        build_rsna_artifacts(root)


def test_target_class_incompatibility_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    class_path = root / "stage_2_detailed_class_info.csv"
    classes = pd.read_csv(class_path)
    classes.loc[classes["patientId"] == "positive", "class"] = "Normal"
    classes.to_csv(class_path, index=False)

    with pytest.raises(ManifestBuildError, match="Target/class incompatibility"):
        build_rsna_artifacts(root)


def _labels_frame(boxes: list[dict[str, object]], target: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{"patientId": "sample", "Target": target, **box} for box in boxes])


def test_partial_positive_box_fails() -> None:
    frame = _labels_frame([{"x": 1, "y": 2, "width": 3, "height": None}])
    with pytest.raises(ManifestBuildError, match="incomplete"):
        aggregate_labels(frame)


def test_coordinates_on_negative_fail() -> None:
    frame = _labels_frame([{"x": 1, "y": None, "width": None, "height": None}], target=0)
    with pytest.raises(ManifestBuildError, match="Negative"):
        aggregate_labels(frame)


@pytest.mark.parametrize(
    "box",
    [
        {"x": -1, "y": 0, "width": 1, "height": 1},
        {"x": 0, "y": -1, "width": 1, "height": 1},
        {"x": 0, "y": 0, "width": 0, "height": 1},
        {"x": 0, "y": 0, "width": 1, "height": -1},
        {"x": 0, "y": 0, "width": float("inf"), "height": 1},
    ],
)
def test_invalid_box_geometry_fails(box: dict[str, object]) -> None:
    with pytest.raises(ManifestBuildError, match="geometry|non-finite"):
        aggregate_labels(_labels_frame([box]))


def test_duplicate_boxes_fail() -> None:
    box = {"x": 1, "y": 2, "width": 3, "height": 4}
    with pytest.raises(ManifestBuildError, match="Duplicate bounding box"):
        aggregate_labels(_labels_frame([box, box]))


def test_out_of_bounds_box_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    labels_path = root / "stage_2_train_labels.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[labels["patientId"] == "positive", "x"] = 1023
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ManifestBuildError, match="exceeds image bounds"):
        build_rsna_artifacts(root)


def test_nonpositive_image_dimensions_fail(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.dcm", "positive", rows=0)

    with pytest.raises(ManifestBuildError, match="Rows and Columns must be positive"):
        build_rsna_artifacts(root)


@pytest.mark.parametrize("spacing", [(0.0, 0.1), (-0.1, 0.1), (float("inf"), 0.1)])
def test_invalid_pixel_spacing_fails(tmp_path: Path, spacing: tuple[float, float]) -> None:
    root = _write_sources(tmp_path / "extracted", spacing=spacing)

    with pytest.raises(ManifestBuildError, match="Pixel spacing"):
        build_rsna_artifacts(root)


def test_sample_schema_rejects_unexpected_column(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    invalid = result.samples.append_column("extra", pa.array([1, 2], type=pa.int8()))

    with pytest.raises(ManifestBuildError, match="schema mismatch"):
        validate_sample_table(invalid, root)


def test_annotation_schema_rejects_unexpected_column(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    invalid = result.annotations.append_column("extra", pa.array([1, 2], type=pa.int8()))

    with pytest.raises(ManifestBuildError, match="schema mismatch"):
        validate_annotation_table(invalid, result.samples, result.labels, result.image_dimensions)


def test_label_schema_rejects_unexpected_column(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    invalid = result.labels.append_column(
        "extra", pa.array([1] * result.labels.num_rows, type=pa.int8())
    )

    with pytest.raises(ManifestBuildError, match="schema mismatch"):
        validate_label_table(invalid, result.samples)


def test_exactly_one_pneumonia_label_is_required_per_sample(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    rows = [
        row
        for row in result.labels.to_pylist()
        if not (row["sample_id"] == "rsna:positive" and row["task_id"] == PNEUMONIA_TASK_ID)
    ]
    missing = pa.Table.from_pylist(rows, RSNA_LABEL_SCHEMA)

    with pytest.raises(ManifestBuildError, match="exactly pneumonia and rsna_class"):
        validate_label_table(missing, result.samples)


def test_annotation_relationships_and_identifier_are_enforced(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    negative_annotation = _replace_table_row(
        result.annotations,
        0,
        sample_id="rsna:negative",
        annotation_id="rsna:negative:bbox:0000",
    )
    with pytest.raises(ManifestBuildError, match="Negative sample"):
        validate_annotation_table(
            negative_annotation,
            result.samples,
            result.labels,
            result.image_dimensions,
        )

    invalid_id = _replace_table_row(result.annotations, 0, annotation_id="arbitrary")
    with pytest.raises(ManifestBuildError, match="deterministic annotation_id"):
        validate_annotation_table(
            invalid_id, result.samples, result.labels, result.image_dimensions
        )


def test_every_positive_sample_requires_an_annotation(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    empty = pa.Table.from_pylist([], RSNA_ANNOTATION_SCHEMA)

    with pytest.raises(ManifestBuildError, match="Every positive sample"):
        validate_annotation_table(empty, result.samples, result.labels, result.image_dimensions)


def test_patient_id_is_not_required_to_be_unique(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    rows = result.samples.to_pylist()
    duplicate_patient = {
        **rows[1],
        "sample_id": "rsna:second-sample",
        "image_id": "second-sample",
        "image_path": "stage_2_train_images/second-sample.dcm",
    }
    _write_header(root / duplicate_patient["image_path"], "positive")
    table = pa.Table.from_pylist(
        sorted([*rows, duplicate_patient], key=lambda row: row["sample_id"]),
        RSNA_SAMPLE_SCHEMA,
    )

    validate_sample_table(table, root)


def test_path_traversal_and_absolute_paths_fail(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    for invalid_path in ("../positive.dcm", str((root / "positive.dcm").resolve())):
        invalid = _replace_table_row(result.samples, 0, image_path=invalid_path)
        with pytest.raises(ManifestBuildError, match="relative POSIX path|escapes"):
            validate_sample_table(invalid, root)


def test_parquet_round_trip_is_exact_and_nested_free(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    restored_samples = pq.read_table(written.paths.samples_path)
    restored_labels = pq.read_table(written.paths.labels_path)
    restored_annotations = pq.read_table(written.paths.annotations_path)
    assert restored_samples.equals(result.samples)
    assert restored_labels.equals(result.labels)
    assert restored_annotations.equals(result.annotations)
    assert restored_samples.schema == RSNA_SAMPLE_SCHEMA
    assert restored_labels.schema == RSNA_LABEL_SCHEMA
    assert restored_annotations.schema == RSNA_ANNOTATION_SCHEMA
    assert not any(pa.types.is_nested(field.type) for field in restored_annotations.schema)


def test_arrow_ipc_hashes_are_deterministic(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(result, tmp_path / "second")

    assert first.arrow_ipc_sha256 == second.arrow_ipc_sha256
    assert first.paths.bundle_id == second.paths.bundle_id
    assert first.arrow_ipc_sha256[SAMPLES_FILENAME] == arrow_ipc_sha256(result.samples)


def test_staging_failure_preserves_current_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    first = write_bundle(result, output)
    current_before = first.paths.current_path.read_text(encoding="utf-8")

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise ManifestBuildError("staged validation failed")

    monkeypatch.setattr(
        "radfusion.data.rsna_artifacts._validate_parquet_round_trip", fail_validation
    )
    with pytest.raises(ManifestBuildError, match="staged validation failed"):
        write_bundle(result, output)
    assert first.paths.current_path.read_text(encoding="utf-8") == current_before
    assert load_current_bundle(output).bundle_id == first.paths.bundle_id


def test_consumer_rejects_bundle_with_hash_mismatch(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    written.paths.labels_path.write_bytes(b"corrupt")

    with pytest.raises(ManifestBuildError, match="hash mismatch"):
        load_current_bundle(tmp_path / "manifests")


def test_cli_success_and_failure_exit_codes(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    output = tmp_path / "manifests"

    assert main(["--dataset-root", str(root), "--output-directory", str(output)]) == 0
    current = load_current_bundle(output)
    assert current.current_path.name == CURRENT_FILENAME
    assert current.samples_path.is_file()
    assert current.labels_path.is_file()
    metadata = json.loads(current.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_count"] == 2
    assert metadata["label_count"] == 4
    assert metadata["hash_policy"]["arrow_ipc_pyarrow_version"] == pa.__version__
    assert (
        "cross-version stability is not claimed"
        in metadata["hash_policy"]["arrow_ipc_stability_scope"]
    )
    assert main(["--dataset-root", str(tmp_path / "missing")]) == 1


@pytest.mark.integration
def test_real_rsna_aggregate_contract() -> None:
    root = Path("data/raw/rsna/extracted")
    if not (root / "stage_2_train_images").is_dir():
        pytest.skip("Local RSNA data is not available")

    result = build_rsna_artifacts(root)

    assert result.samples.num_rows == 26_684
    assert result.labels.num_rows == 53_368
    assert result.annotations.num_rows == 9_555
    targets = Counter(
        row["label_value"]
        for row in result.labels.to_pylist()
        if row["task_id"] == PNEUMONIA_TASK_ID
    )
    assert targets == {0: 20_672, 1: 6_012}
