"""Register dataset adapters for configured experiments."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq

from radfusion.data.rsna_artifacts import load_current_bundle
from radfusion.data.rsna_source import ManifestBuildError
from radfusion.data.splitting import SPLIT_NAMES
from radfusion.training.config import DatasetConfig
from radfusion.training.interfaces import DatasetRunData


class RsnaDataset:
    """Load validated RSNA bundle artifacts for training."""

    def load(self, config: DatasetConfig) -> DatasetRunData:
        """Return the configured RSNA task frame and lineage."""
        bundle = load_current_bundle(config.manifest_directory)
        samples = pq.read_table(bundle.samples_path)
        labels = pq.read_table(bundle.labels_path)
        splits = pq.read_table(bundle.splits_path)
        metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
        frame = _training_frame(
            samples.to_pandas(), labels.to_pandas(), splits.to_pandas(), config.task_id
        )
        _validate_training_frame(frame)
        return DatasetRunData(
            frame=frame,
            bundle_id=bundle.bundle_id,
            split_recipe_id=str(metadata["split"]["split_recipe_id"]),
            split_assignment_id=str(metadata["split"]["split_assignment_id"]),
            label_policy_version=str(metadata["tasks"][config.task_id]["label_policy_version"]),
        )


def _training_frame(
    samples: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, task_id: str
) -> pd.DataFrame:
    target = labels.loc[labels["task_id"] == task_id, ["sample_id", "label_value"]]
    target = target.rename(columns={"label_value": "target"})
    assignments = splits[["sample_id", "split_name"]]
    return (
        samples.merge(assignments, on="sample_id", validate="one_to_one")
        .merge(target, on="sample_id", validate="one_to_one")
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )


def _validate_training_frame(frame: pd.DataFrame) -> None:
    if set(frame["split_name"]) != set(SPLIT_NAMES):
        raise ManifestBuildError("Training requires train, validation, and test splits")
    for split_name in SPLIT_NAMES:
        targets = set(frame.loc[frame["split_name"] == split_name, "target"])
        if targets != {0, 1}:
            raise ManifestBuildError(f"Split {split_name!r} must contain both target classes")
