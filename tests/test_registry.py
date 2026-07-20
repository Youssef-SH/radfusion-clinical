from __future__ import annotations

import pytest

from radfusion.training.registry import DATASETS, MODELS, RegistryError, get_dataset, get_model


def test_builtin_component_mappings_are_immutable_and_complete() -> None:
    assert tuple(DATASETS) == ("rsna",)
    assert tuple(MODELS) == ("metadata_logistic", "metadata_lightgbm")
    assert get_dataset("rsna") is DATASETS["rsna"]
    assert get_model("metadata_logistic") is MODELS["metadata_logistic"]
    with pytest.raises(TypeError):
        DATASETS["other"] = object()  # type: ignore[index]


@pytest.mark.parametrize(
    ("lookup", "message"),
    [(get_dataset, "dataset"), (get_model, "model")],
)
def test_unknown_builtin_component_keys_fail(lookup, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        lookup("missing")
