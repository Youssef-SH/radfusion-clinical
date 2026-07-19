"""Define typed boundaries for registered training components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from radfusion.training.config import DatasetConfig, ModelConfig


@dataclass(frozen=True)
class DatasetRunData:
    """Validated model-ready dataset frame and lineage."""

    frame: pd.DataFrame
    bundle_id: str
    split_recipe_id: str
    split_assignment_id: str
    label_policy_version: str


@dataclass(frozen=True)
class ModelFitResult:
    """Fitted pipeline and model-derived logging parameters."""

    pipeline: Pipeline
    derived_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derived_parameters",
            MappingProxyType(dict(self.derived_parameters)),
        )


class DatasetImplementation(Protocol):
    """Registered dataset adapter used by the experiment runner."""

    def load(self, config: DatasetConfig) -> DatasetRunData:
        """Load and validate one configured dataset bundle."""


class ModelImplementation(Protocol):
    """Registered model implementation used by the experiment runner."""

    def fit(
        self,
        config: ModelConfig,
        training_seed: int,
        train_features: pd.DataFrame,
        train_targets: np.ndarray,
        validation_features: pd.DataFrame,
        validation_targets: np.ndarray,
    ) -> ModelFitResult:
        """Fit one model from training data with validation monitoring."""
