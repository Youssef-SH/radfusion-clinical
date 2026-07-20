"""Load approved RSNA metadata partitions from an exact bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from radfusion.data.rsna_artifacts import (
    BUNDLES_DIRECTORY,
    LABELS_FILENAME,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    validate_bundle_directory,
)
from radfusion.data.rsna_source import ManifestBuildError
from radfusion.data.tabular_preprocess import SOURCE_FEATURES
from radfusion.evaluation.metrics import validated_binary_targets
from radfusion.training.config import DatasetConfig
from radfusion.training.interfaces import DatasetLineage, DatasetPartition, DatasetRunData


class RsnaDataset:
    """Expose approved metadata features from a pinned RSNA bundle."""

    def load_train_validation(self, config: DatasetConfig) -> DatasetRunData:
        """Load train and validation without reading the test partition."""
        bundle, metadata = _load_pinned_bundle(config)
        frame = _task_frame(bundle, config.task_id, partitions=("train", "validation"))
        train = _partition(frame, "train")
        validation = _partition(frame, "validation")
        return DatasetRunData(
            train=train,
            validation=validation,
            lineage=_lineage(config, metadata),
        )

    def load_lineage(self, config: DatasetConfig) -> DatasetLineage:
        """Validate the pinned bundle and return lineage without reading partitions."""
        _, metadata = _load_pinned_bundle(config)
        return _lineage(config, metadata)

    def load_test(self, config: DatasetConfig) -> tuple[DatasetPartition, DatasetLineage]:
        """Load the test partition from the same pinned bundle."""
        bundle, metadata = _load_pinned_bundle(config)
        frame = _task_frame(bundle, config.task_id, partitions=("test",))
        test = _partition(frame, "test")
        return test, _lineage(config, metadata)


@dataclass(frozen=True)
class _PinnedBundlePaths:
    samples_path: Path
    labels_path: Path
    splits_path: Path


def _load_pinned_bundle(
    config: DatasetConfig,
) -> tuple[_PinnedBundlePaths, dict[str, object]]:
    dataset_root = config.manifest_directory / config.registry_key
    bundle_directory = dataset_root / BUNDLES_DIRECTORY / config.bundle_id
    metadata = validate_bundle_directory(
        bundle_directory,
        expected_bundle_id=config.bundle_id,
    )
    return (
        _PinnedBundlePaths(
            bundle_directory / SAMPLES_FILENAME,
            bundle_directory / LABELS_FILENAME,
            bundle_directory / SPLITS_FILENAME,
        ),
        metadata,
    )


def _task_frame(
    bundle: _PinnedBundlePaths,
    task_id: str,
    *,
    partitions: tuple[str, ...],
) -> pd.DataFrame:
    assignments = pq.read_table(
        bundle.splits_path,
        columns=["sample_id", "split_name"],
        filters=[("split_name", "in", list(partitions))],
    ).to_pandas()
    selected_ids = assignments["sample_id"].astype(str).tolist()
    sample_columns = ["sample_id", "patient_id", *SOURCE_FEATURES]
    samples = pq.read_table(
        bundle.samples_path,
        columns=sample_columns,
        filters=[("sample_id", "in", selected_ids)],
    ).to_pandas()
    target = pq.read_table(
        bundle.labels_path,
        columns=["sample_id", "label_value"],
        filters=[
            ("task_id", "=", task_id),
            ("sample_id", "in", selected_ids),
        ],
    ).to_pandas()
    if target.empty:
        raise ManifestBuildError(f"Bundle does not contain configured task {task_id!r}")
    return (
        samples.merge(assignments, on="sample_id", validate="one_to_one")
        .merge(
            target.rename(columns={"label_value": "target"}), on="sample_id", validate="one_to_one"
        )
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )


def _partition(frame: pd.DataFrame, name: str) -> DatasetPartition:
    selected = frame.loc[frame["split_name"] == name]
    if selected.empty:
        raise ManifestBuildError(f"Bundle partition {name!r} is empty")
    targets = validated_binary_targets(selected["target"].to_numpy())
    features = selected.loc[:, SOURCE_FEATURES].copy()
    if tuple(features.columns) != SOURCE_FEATURES:
        raise ManifestBuildError("RSNA metadata feature contract is invalid")
    return DatasetPartition(
        features=features,
        targets=targets,
        sample_ids=tuple(selected["sample_id"].astype(str)),
        patient_ids=tuple(selected["patient_id"].astype(str)),
        partition=name,
    )


def _lineage(
    config: DatasetConfig,
    metadata: dict[str, object],
) -> DatasetLineage:
    split = metadata["split"]
    tasks = metadata["tasks"]
    return DatasetLineage(
        bundle_id=config.bundle_id,
        split_assignment_id=str(split["split_assignment_id"]),
        label_policy_version=str(tasks[config.task_id]["label_policy_version"]),
        task_id=config.task_id,
    )
