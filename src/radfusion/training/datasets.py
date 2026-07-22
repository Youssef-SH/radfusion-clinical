"""Load approved RSNA model inputs from an exact bundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from radfusion.data.dicom_loader import DicomRecord, read_dicom
from radfusion.data.rsna_artifacts import (
    BUNDLES_DIRECTORY,
    LABELS_FILENAME,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    validate_bundle_directory,
)
from radfusion.data.rsna_source import ManifestBuildError, resolve_image_path
from radfusion.data.tabular_preprocess import SOURCE_FEATURES
from radfusion.evaluation.metrics import validated_binary_targets
from radfusion.training.config import DatasetConfig
from radfusion.training.interfaces import DatasetLineage, DatasetPartition, DatasetRunData

_IMAGE_FRAME_COLUMNS = ("sample_id", "patient_id", "image_path", "split_name", "target")


class ImageSample(TypedDict):
    """One lazily decoded image example."""

    image: torch.Tensor
    target: torch.Tensor
    sample_id: str
    patient_id: str


@dataclass(frozen=True)
class _ImageRow:
    sample_id: str
    patient_id: str
    image_path: PurePosixPath
    target: int


class RsnaImageDataset(Dataset[ImageSample]):
    """Lazily decode a validated, deterministically ordered RSNA partition."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        dataset_root: str | Path,
        partition: str,
        transform: Callable[[np.ndarray], torch.Tensor],
        decoder: Callable[[str | Path], tuple[np.ndarray, DicomRecord]] = read_dicom,
    ) -> None:
        if tuple(frame.columns) != _IMAGE_FRAME_COLUMNS:
            raise ManifestBuildError(
                f"RSNA image frame columns must be exactly {_IMAGE_FRAME_COLUMNS}"
            )
        if partition not in {"train", "validation", "test"}:
            raise ManifestBuildError(f"Unsupported RSNA image partition: {partition!r}")
        if frame.empty:
            raise ManifestBuildError(f"RSNA image partition {partition!r} is empty")

        self._dataset_root = Path(dataset_root)
        self._transform = transform
        self._decoder = decoder
        rows: list[_ImageRow] = []
        for row in frame.to_dict(orient="records"):
            sample_id = _nonempty_text(row["sample_id"], "sample_id")
            patient_id = _nonempty_text(row["patient_id"], "patient_id")
            if row["split_name"] != partition:
                raise ManifestBuildError("RSNA image frame does not match the requested partition")
            relative_path = _validated_image_path(row["image_path"])
            resolve_image_path(self._dataset_root, relative_path)
            target = row["target"]
            if (
                isinstance(target, bool)
                or not isinstance(target, int | np.integer)
                or target not in {0, 1}
            ):
                raise ManifestBuildError(f"Invalid binary target for sample {sample_id!r}")
            rows.append(_ImageRow(sample_id, patient_id, relative_path, int(target)))
        sample_ids = [row.sample_id for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ManifestBuildError("RSNA image sample_id values must be unique")
        if sample_ids != sorted(sample_ids):
            raise ManifestBuildError("RSNA image samples must be ordered by sample_id")
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> ImageSample:
        row = self._rows[index]
        path = resolve_image_path(self._dataset_root, row.image_path)
        pixels, record = self._decoder(path)
        if record.patient_id != row.patient_id:
            raise ManifestBuildError(
                f"Decoded DICOM patient does not match sample {row.sample_id!r}"
            )
        image = self._transform(pixels)
        if (
            not isinstance(image, torch.Tensor)
            or image.dtype != torch.float32
            or image.shape != (1, 224, 224)
            or not torch.isfinite(image).all()
        ):
            raise ManifestBuildError(
                f"Image transform returned an invalid tensor for sample {row.sample_id!r}"
            )
        return {
            "image": image,
            "target": torch.tensor(float(row.target), dtype=torch.float32),
            "sample_id": row.sample_id,
            "patient_id": row.patient_id,
        }


class RsnaDataset:
    """Expose approved RSNA model partitions from a pinned bundle."""

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

    def load_image_partition_frame(
        self,
        config: DatasetConfig,
        partition: str,
    ) -> tuple[pd.DataFrame, DatasetLineage]:
        """Load approved image rows without decoding DICOM pixels."""
        if partition not in {"train", "validation", "test"}:
            raise ManifestBuildError(f"Unsupported RSNA image partition: {partition!r}")
        bundle, metadata = _load_pinned_bundle(config)
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=(partition,),
            feature_columns=("image_path",),
        )
        return frame.loc[:, _IMAGE_FRAME_COLUMNS].copy(), _lineage(config, metadata)


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
    feature_columns: tuple[str, ...] = SOURCE_FEATURES,
) -> pd.DataFrame:
    assignments = pq.read_table(
        bundle.splits_path,
        columns=["sample_id", "split_name"],
        filters=[("split_name", "in", list(partitions))],
    ).to_pandas()
    selected_ids = assignments["sample_id"].astype(str).tolist()
    sample_columns = ["sample_id", "patient_id", *feature_columns]
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


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestBuildError(f"RSNA image {field} must be a non-empty string")
    return value


def _validated_image_path(value: object) -> PurePosixPath:
    text = _nonempty_text(value, "image_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise ManifestBuildError(f"Invalid normalized relative image path: {text!r}")
    return path
