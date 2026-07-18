"""Provide explicit registries for datasets and experiment models."""

from __future__ import annotations

from radfusion.training.interfaces import DatasetImplementation, ModelImplementation


class RegistryError(LookupError):
    """Raised for duplicate or missing registry entries."""


class Registry[T]:
    """Map stable configuration keys to registered implementations."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        """Register one implementation under a stable key."""
        if not key:
            raise RegistryError(f"{self._kind} registry key must not be empty")
        if key in self._entries:
            raise RegistryError(f"{self._kind.title()} registry key is already registered: {key!r}")
        self._entries[key] = value

    def get(self, key: str) -> T:
        """Return a registered implementation."""
        try:
            return self._entries[key]
        except KeyError as exc:
            raise RegistryError(f"Unknown {self._kind} registry key: {key!r}") from exc

    def keys(self) -> tuple[str, ...]:
        """Return registered keys in deterministic order."""
        return tuple(sorted(self._entries))


class DatasetRegistry(Registry[DatasetImplementation]):
    """Registry of dataset loading and frame-construction adapters."""

    def __init__(self) -> None:
        super().__init__("dataset")


class ModelRegistry(Registry[ModelImplementation]):
    """Registry of model-specific fitting implementations."""

    def __init__(self) -> None:
        super().__init__("model")


DATASET_REGISTRY = DatasetRegistry()
MODEL_REGISTRY = ModelRegistry()
_BUILTINS_REGISTERED = False


def register_builtin_components() -> None:
    """Register built-in dataset and model adapters exactly once."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from radfusion.models.tabular_baseline import MetadataLightgbmModel, MetadataLogisticModel
    from radfusion.training.datasets import RsnaDataset

    DATASET_REGISTRY.register("rsna", RsnaDataset())
    MODEL_REGISTRY.register("metadata_logistic", MetadataLogisticModel())
    MODEL_REGISTRY.register("metadata_lightgbm", MetadataLightgbmModel())
    _BUILTINS_REGISTERED = True
