from __future__ import annotations

import pytest

from radfusion.training.registry import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    Registry,
    RegistryError,
    register_builtin_components,
)


def test_registry_resolves_registered_implementation() -> None:
    registry: Registry[object] = Registry("test")
    implementation = object()

    registry.register("example", implementation)

    assert registry.get("example") is implementation
    assert registry.keys() == ("example",)


def test_registry_rejects_duplicate_and_unknown_keys() -> None:
    registry: Registry[object] = Registry("test")
    registry.register("example", object())

    with pytest.raises(RegistryError, match="already registered"):
        registry.register("example", object())
    with pytest.raises(RegistryError, match="Unknown test registry key"):
        registry.get("missing")


def test_current_dataset_and_model_implementations_are_registered() -> None:
    register_builtin_components()
    assert DATASET_REGISTRY.keys() == ("rsna",)
    assert MODEL_REGISTRY.keys() == ("metadata_lightgbm", "metadata_logistic")


def test_builtin_registration_is_idempotent() -> None:
    register_builtin_components()
    register_builtin_components()

    assert DATASET_REGISTRY.keys() == ("rsna",)
    assert MODEL_REGISTRY.keys() == ("metadata_lightgbm", "metadata_logistic")
