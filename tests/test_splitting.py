from __future__ import annotations

from collections import Counter, defaultdict

import pyarrow as pa
import pytest

from radfusion.data.rsna_source import ManifestBuildError
from radfusion.data.schemas import RSNA_LABEL_SCHEMA, RSNA_SAMPLE_SCHEMA, RSNA_SPLIT_SCHEMA
from radfusion.data.splitting import (
    SPLIT_NAMES,
    SplitConfig,
    cohort_fingerprint,
    create_patient_stratified_splits,
    split_assignment_id,
    validate_split_table,
)


def _cohort(patient_count: int = 20) -> tuple[pa.Table, pa.Table]:
    samples: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for target in (0, 1):
        for index in range(patient_count):
            patient_id = f"patient-{target}-{index:02d}"
            sample_count = 2 if index < 3 else 1
            for sample_index in range(sample_count):
                sample_id = f"rsna:{patient_id}-{sample_index}"
                samples.append(
                    {
                        "dataset_id": "rsna",
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "study_id": None,
                        "image_id": f"{patient_id}-{sample_index}",
                        "image_path": f"stage_2_train_images/{patient_id}-{sample_index}.dcm",
                        "split": None,
                        "age_years": 50.0,
                        "sex": "F",
                        "view_position": "PA",
                        "pixel_spacing_row_mm": 0.168,
                        "pixel_spacing_col_mm": 0.168,
                    }
                )
                labels.append(
                    {
                        "dataset_id": "rsna",
                        "sample_id": sample_id,
                        "task_id": "pneumonia",
                        "label_value": target,
                        "label_status": "observed",
                        "label_source": "rsna-stage-2-challenge-target",
                        "label_policy_version": "rsna-stage-2-target-v1",
                    }
                )
    return (
        pa.Table.from_pylist(sorted(samples, key=lambda row: row["sample_id"]), RSNA_SAMPLE_SCHEMA),
        pa.Table.from_pylist(sorted(labels, key=lambda row: row["sample_id"]), RSNA_LABEL_SCHEMA),
    )


def test_split_is_deterministic_disjoint_stratified_and_complete() -> None:
    samples, labels = _cohort()
    first = create_patient_stratified_splits(samples, labels)
    second = create_patient_stratified_splits(samples, labels)

    assert first.schema == RSNA_SPLIT_SCHEMA
    assert first.equals(second)
    assert first.num_rows == samples.num_rows
    assignments = {row["sample_id"]: row["split_name"] for row in first.to_pylist()}
    patients: dict[str, set[str]] = defaultdict(set)
    patient_targets: dict[str, int] = {}
    targets = {row["sample_id"]: row["label_value"] for row in labels.to_pylist()}
    for row in samples.to_pylist():
        patients[row["patient_id"]].add(assignments[row["sample_id"]])
        patient_targets[row["patient_id"]] = targets[row["sample_id"]]
    assert all(len(names) == 1 for names in patients.values())
    counts = Counter(
        (next(iter(names)), patient_targets[patient_id]) for patient_id, names in patients.items()
    )
    assert counts == {
        ("train", 0): 14,
        ("validation", 0): 3,
        ("test", 0): 3,
        ("train", 1): 14,
        ("validation", 1): 3,
        ("test", 1): 3,
    }


def test_split_seed_changes_recipe_and_assignment_identities() -> None:
    samples, labels = _cohort()
    first = create_patient_stratified_splits(samples, labels, SplitConfig(seed=42))
    second = create_patient_stratified_splits(samples, labels, SplitConfig(seed=43))

    assert (
        first.column("split_recipe_id").to_pylist()[0]
        != second.column("split_recipe_id").to_pylist()[0]
    )
    assert (
        first.column("split_assignment_id").to_pylist()[0]
        != second.column("split_assignment_id").to_pylist()[0]
    )
    assert first.column("split_name").to_pylist() != second.column("split_name").to_pylist()


@pytest.mark.parametrize(
    "config",
    [
        SplitConfig(train_ratio=0.0, validation_ratio=0.5, test_ratio=0.5),
        SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.2),
    ],
)
def test_invalid_split_ratios_fail(config: SplitConfig) -> None:
    samples, labels = _cohort()
    with pytest.raises(ManifestBuildError, match="ratios"):
        create_patient_stratified_splits(samples, labels, config)


def test_split_validation_rejects_patient_overlap() -> None:
    samples, labels = _cohort()
    splits = create_patient_stratified_splits(samples, labels)
    sample_rows = samples.to_pylist()
    duplicate_patient = sample_rows[0]["patient_id"]
    patient_sample_ids = [
        row["sample_id"] for row in sample_rows if row["patient_id"] == duplicate_patient
    ]
    assert len(patient_sample_ids) == 2
    rows = splits.to_pylist()
    for row in rows:
        if row["sample_id"] == patient_sample_ids[1]:
            current = row["split_name"]
            row["split_name"] = next(name for name in SPLIT_NAMES if name != current)
    invalid = pa.Table.from_pylist(rows, RSNA_SPLIT_SCHEMA)

    assignments = {row["sample_id"]: row["split_name"] for row in rows}
    assignment_id = split_assignment_id(
        rows[0]["split_recipe_id"], rows[0]["cohort_fingerprint"], assignments
    )
    for row in rows:
        row["split_assignment_id"] = assignment_id
    invalid = pa.Table.from_pylist(rows, RSNA_SPLIT_SCHEMA)

    with pytest.raises(ManifestBuildError, match="multiple splits"):
        validate_split_table(invalid, samples, labels)


@pytest.mark.parametrize(
    ("total", "expected"),
    [(1, (1, 0, 0)), (2, (1, 1, 0)), (3, (1, 1, 1)), (4, (2, 1, 1)), (20, (14, 3, 3))],
)
def test_split_allocation_is_feasible_and_deterministic(total: int, expected) -> None:
    samples, labels = _cohort(total)
    splits = create_patient_stratified_splits(samples, labels)
    assignments = {row["sample_id"]: row["split_name"] for row in splits.to_pylist()}
    patient_assignments: dict[tuple[int, str], set[str]] = defaultdict(set)
    targets = {row["sample_id"]: row["label_value"] for row in labels.to_pylist()}
    for row in samples.to_pylist():
        patient_assignments[(targets[row["sample_id"]], assignments[row["sample_id"]])].add(
            row["patient_id"]
        )

    for target in (0, 1):
        assert tuple(len(patient_assignments[(target, name)]) for name in SPLIT_NAMES) == expected


def test_same_cohort_recipe_and_row_reordering_preserve_all_identities() -> None:
    samples, labels = _cohort()
    reordered_samples = pa.Table.from_pylist(
        list(reversed(samples.to_pylist())), RSNA_SAMPLE_SCHEMA
    )
    reordered_labels = pa.Table.from_pylist(list(reversed(labels.to_pylist())), RSNA_LABEL_SCHEMA)
    first = create_patient_stratified_splits(samples, labels)
    second = create_patient_stratified_splits(reordered_samples, reordered_labels)
    assert first.equals(second)
    assert cohort_fingerprint(samples, labels) == cohort_fingerprint(
        reordered_samples, reordered_labels
    )


def test_same_recipe_on_different_cohorts_changes_cohort_and_assignment_identities() -> None:
    samples, labels = _cohort()
    sample_rows = samples.to_pylist()[1:]
    retained = {row["sample_id"] for row in sample_rows}
    label_rows = [row for row in labels.to_pylist() if row["sample_id"] in retained]
    smaller_samples = pa.Table.from_pylist(sample_rows, RSNA_SAMPLE_SCHEMA)
    smaller_labels = pa.Table.from_pylist(label_rows, RSNA_LABEL_SCHEMA)
    first = create_patient_stratified_splits(samples, labels)
    second = create_patient_stratified_splits(smaller_samples, smaller_labels)
    assert first.column("split_recipe_id")[0] == second.column("split_recipe_id")[0]
    assert first.column("cohort_fingerprint")[0] != second.column("cohort_fingerprint")[0]
    assert first.column("split_assignment_id")[0] != second.column("split_assignment_id")[0]


def test_assignment_and_declared_recipe_tampering_are_detected() -> None:
    samples, labels = _cohort()
    splits = create_patient_stratified_splits(samples, labels)
    rows = splits.to_pylist()
    rows[0]["split_name"] = "test" if rows[0]["split_name"] != "test" else "train"
    with pytest.raises(ManifestBuildError, match="assignment identity"):
        validate_split_table(pa.Table.from_pylist(rows, RSNA_SPLIT_SCHEMA), samples, labels)

    with pytest.raises(ManifestBuildError, match="recipe identity"):
        validate_split_table(splits, samples, labels, config=SplitConfig(seed=43))
