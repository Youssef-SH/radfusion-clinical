from __future__ import annotations

import numpy as np
import pytest

from radfusion.evaluation.probabilities import positive_class_probabilities


class _Estimator:
    def __init__(self, classes, probabilities) -> None:
        self.classes_ = np.asarray(classes)
        self._probabilities = probabilities

    def predict_proba(self, features):
        return self._probabilities


def test_positive_probability_uses_the_column_labeled_one() -> None:
    estimator = _Estimator([1, 0], [[0.8, 0.2], [0.3, 0.7]])
    np.testing.assert_array_equal(positive_class_probabilities(estimator, ["a", "b"]), [0.8, 0.3])


@pytest.mark.parametrize("classes", [[0], [0, 2], [0, 0], [[0, 1]]])
def test_positive_probability_rejects_invalid_class_contracts(classes) -> None:
    with pytest.raises(ValueError, match="class|classes"):
        positive_class_probabilities(_Estimator(classes, [[0.5, 0.5]]), ["a"])


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.5]],
        [0.5, 0.5],
        [],
        [[float("nan"), 0.5]],
        [[-0.1, 1.1]],
        [[0.4, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ],
)
def test_positive_probability_rejects_invalid_probability_outputs(probabilities) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        positive_class_probabilities(_Estimator([0, 1], probabilities), ["a"])
