"""Validate fitted class contracts and extract positive-class probabilities."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ProbabilityEstimator(Protocol):
    """Fitted estimator that exposes class labels and class probabilities."""

    classes_: Any

    def predict_proba(self, features: Any) -> Any:
        """Return class probabilities."""


def positive_class_probabilities(estimator: ProbabilityEstimator, features: Any) -> np.ndarray:
    """Return validated probabilities for the exact positive label ``1``."""
    classes = np.asarray(getattr(estimator, "classes_", None))
    if classes.ndim != 1 or len(classes) != 2 or len(np.unique(classes)) != 2:
        raise ValueError("Estimator must expose two unique one-dimensional class labels")
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Estimator classes must be exactly {0, 1}")
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Estimator must expose the positive class label 1 exactly once")
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or probabilities.shape[1] != len(classes)
    ):
        raise ValueError("predict_proba returned an invalid class-probability matrix")
    try:
        feature_count = len(features)
    except TypeError:
        feature_count = None
    if feature_count is not None and probabilities.shape[0] != feature_count:
        raise ValueError("Class-probability row count does not match the input")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Class probabilities must be finite and within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Class-probability rows must sum to 1")
    positive = probabilities[:, int(positive_columns[0])]
    return positive
