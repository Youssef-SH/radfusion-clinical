"""Load and validate typed experiment configuration files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _StrictSafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


MODEL_RANDOMNESS_KEYS = frozenset(
    {
        "random_state",
        "seed",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
        "drop_seed",
        "extra_seed",
    }
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset adapter and bundle settings."""

    registry_key: str
    manifest_directory: Path
    task_id: str


@dataclass(frozen=True)
class ModelConfig:
    """Registered model and its declarative parameters."""

    registry_key: str
    output_name: str
    modality: str
    class_weighting: str
    parameters: MappingProxyType[str, Any]
    fit_parameters: MappingProxyType[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    """Training seed and generated-output locations."""

    seed: int
    report_directory: Path
    model_directory: Path
    require_clean_git: bool


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation policies shared by model implementations."""

    threshold_policy: str
    sensitivity_target: float
    calibration_bins: int
    latency_warmup_calls: int
    latency_measured_calls: int


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow experiment and tracking settings."""

    experiment_name: str
    tracking_directory: Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete immutable experiment definition."""

    config_version: int
    name: str
    executable: bool
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    source_path: Path
    source_sha256: str


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    source = Path(path)
    try:
        source_bytes = source.read_bytes()
        document = yaml.load(source_bytes.decode("utf-8"), Loader=_StrictSafeLoader)
    except OSError as exc:
        raise ConfigError(f"Experiment config is unreadable: {source}") from exc
    except UnicodeError as exc:
        raise ConfigError(f"Experiment config is not valid UTF-8: {source}") from exc
    except yaml.YAMLError as exc:
        detail = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(f"Experiment config is invalid YAML: {source}: {detail}") from exc
    root = _mapping(document, "config")
    _keys(
        root,
        required={
            "config_version",
            "name",
            "executable",
            "dataset",
            "model",
            "training",
            "evaluation",
            "mlflow",
        },
        context="config",
    )
    config_version = _integer(root["config_version"], "config_version")
    if config_version != 1:
        raise ConfigError(f"Unsupported config_version: {config_version}")
    return ExperimentConfig(
        config_version=config_version,
        name=_text(root["name"], "name"),
        executable=_boolean(root["executable"], "executable"),
        dataset=_dataset_config(root["dataset"]),
        model=_model_config(root["model"]),
        training=_training_config(root["training"]),
        evaluation=_evaluation_config(root["evaluation"]),
        mlflow=_mlflow_config(root["mlflow"]),
        source_path=source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def _dataset_config(value: object) -> DatasetConfig:
    data = _mapping(value, "dataset")
    _keys(data, required={"registry_key", "manifest_directory", "task_id"}, context="dataset")
    return DatasetConfig(
        registry_key=_path_component(data["registry_key"], "dataset.registry_key"),
        manifest_directory=Path(_text(data["manifest_directory"], "dataset.manifest_directory")),
        task_id=_text(data["task_id"], "dataset.task_id"),
    )


def _model_config(value: object) -> ModelConfig:
    data = _mapping(value, "model")
    _keys(
        data,
        required={
            "registry_key",
            "output_name",
            "modality",
            "class_weighting",
            "parameters",
            "fit_parameters",
        },
        context="model",
    )
    parameters = _mapping(data["parameters"], "model.parameters")
    randomness_conflicts = sorted(MODEL_RANDOMNESS_KEYS & parameters.keys())
    if randomness_conflicts:
        raise ConfigError(
            "model.parameters contains randomness controls reserved for training.seed: "
            f"{randomness_conflicts}"
        )
    return ModelConfig(
        registry_key=_text(data["registry_key"], "model.registry_key"),
        output_name=_path_component(data["output_name"], "model.output_name"),
        modality=_text(data["modality"], "model.modality"),
        class_weighting=_text(data["class_weighting"], "model.class_weighting"),
        parameters=MappingProxyType(parameters),
        fit_parameters=MappingProxyType(_mapping(data["fit_parameters"], "model.fit_parameters")),
    )


def _training_config(value: object) -> TrainingConfig:
    data = _mapping(value, "training")
    _keys(
        data,
        required={"seed", "report_directory", "model_directory", "require_clean_git"},
        context="training",
    )
    seed = _integer(data["seed"], "training.seed")
    if not 0 <= seed <= 2**31 - 1:
        raise ConfigError("training.seed must be between 0 and 2147483647")
    return TrainingConfig(
        seed=seed,
        report_directory=Path(_text(data["report_directory"], "training.report_directory")),
        model_directory=Path(_text(data["model_directory"], "training.model_directory")),
        require_clean_git=_boolean(data["require_clean_git"], "training.require_clean_git"),
    )


def _evaluation_config(value: object) -> EvaluationConfig:
    data = _mapping(value, "evaluation")
    _keys(
        data,
        required={
            "threshold_policy",
            "sensitivity_target",
            "calibration_bins",
            "latency_warmup_calls",
            "latency_measured_calls",
        },
        context="evaluation",
    )
    sensitivity = _number(data["sensitivity_target"], "evaluation.sensitivity_target")
    bins = _integer(data["calibration_bins"], "evaluation.calibration_bins")
    warmup = _integer(data["latency_warmup_calls"], "evaluation.latency_warmup_calls")
    measured = _integer(data["latency_measured_calls"], "evaluation.latency_measured_calls")
    if not 0 < sensitivity <= 1:
        raise ConfigError("evaluation.sensitivity_target must be in (0, 1]")
    if bins <= 0 or warmup < 0 or measured <= 0:
        raise ConfigError("Evaluation bin and latency call counts are invalid")
    threshold_policy = _text(data["threshold_policy"], "evaluation.threshold_policy")
    if threshold_policy != "youden_j":
        raise ConfigError(f"Unsupported threshold policy: {threshold_policy}")
    return EvaluationConfig(threshold_policy, sensitivity, bins, warmup, measured)


def _mlflow_config(value: object) -> MLflowConfig:
    data = _mapping(value, "mlflow")
    _keys(data, required={"experiment_name", "tracking_directory"}, context="mlflow")
    return MLflowConfig(
        experiment_name=_text(data["experiment_name"], "mlflow.experiment_name"),
        tracking_directory=Path(_text(data["tracking_directory"], "mlflow.tracking_directory")),
    )


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{context} must be a mapping with string keys")
    return dict(value)


def _keys(data: dict[str, Any], *, required: set[str], context: str) -> None:
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise ConfigError(f"{context} is missing keys: {missing}")
    if unknown:
        raise ConfigError(f"{context} has unknown keys: {unknown}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _path_component(value: object, context: str) -> str:
    text = _text(value, context)
    if (
        text in {".", ".."}
        or Path(text).is_absolute()
        or "/" in text
        or "\\" in text
        or Path(text).name != text
    ):
        raise ConfigError(f"{context} must be one safe path component")
    return text


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{context} must be numeric")
    return float(value)


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be boolean")
    return value
