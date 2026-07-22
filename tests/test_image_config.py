from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from radfusion.training.config import ConfigError, load_experiment_config
from radfusion.training.train import main as train_main


def _image_document() -> dict[str, object]:
    return yaml.safe_load(Path("configs/image_densenet.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "image.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_image_config_is_strict_single_seed_and_non_executable() -> None:
    config = load_experiment_config("configs/image_densenet.yaml")

    assert config.executable is False
    assert config.training.seed == 42
    assert not hasattr(config.training, "seeds")
    assert config.dataset.dataset_root == Path("data/raw/rsna/extracted")
    assert config.model.modality == "image"
    assert dict(config.model.parameters) == {
        "encoder_name": "densenet121",
        "weights": "densenet121-res224-chex",
        "image_size": 224,
        "embedding_dimension": 1024,
        "class_weighting": "train_pos_weight",
    }
    assert config.image is not None
    assert config.image.rotation_degrees == 7.0
    assert config.image.translation_fraction == 0.05
    assert config.image.pin_memory_policy == "auto"


def test_metadata_configs_remain_executable_and_unchanged() -> None:
    for path in ("configs/metadata_logistic.yaml", "configs/metadata_lightgbm.yaml"):
        config = load_experiment_config(path)
        assert config.executable is True
        assert config.model.modality == "metadata"
        assert config.image is None
        assert config.dataset.dataset_root is None


def test_image_foundation_config_cannot_start_training(monkeypatch, capsys) -> None:
    def reject_training(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr("radfusion.training.train.train_configured_experiment", reject_training)

    assert train_main(["--config", "configs/image_densenet.yaml"]) == 1
    assert "not executable yet" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document["training"].update({"seeds": [42]}), "unknown keys"),
        (lambda document: document["image"].update({"unknown": 1}), "unknown keys"),
        (lambda document: document["image"].pop("batch_size"), "missing keys"),
    ],
)
def test_image_config_rejects_unknown_and_missing_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    document = _image_document()
    mutation(document)

    with pytest.raises(ConfigError, match=message):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["model"].update({"registry_key": "metadata_logistic"}),
            "image_densenet",
        ),
        (
            lambda document: document["model"].update({"registry_key": "metadata_lightgbm"}),
            "image_densenet",
        ),
        (lambda document: document["dataset"].pop("dataset_root"), "dataset_root"),
        (lambda document: document.pop("image"), "image configuration"),
        (
            lambda document: document["model"].update({"fit_parameters": {"epochs": 1}}),
            "empty model.fit_parameters",
        ),
        (lambda document: document.update({"executable": True}), "non-executable"),
        (
            lambda document: (
                document["model"].update(
                    {"modality": "metadata", "registry_key": "image_densenet"}
                ),
                document.pop("image"),
                document["dataset"].pop("dataset_root"),
            ),
            "registered metadata model",
        ),
        (
            lambda document: (
                document["model"].update(
                    {"modality": "metadata", "registry_key": "metadata_logistic"}
                ),
                document.pop("image"),
            ),
            "do not accept dataset.dataset_root",
        ),
        (
            lambda document: (
                document["model"].update(
                    {"modality": "metadata", "registry_key": "metadata_logistic"}
                ),
                document["dataset"].pop("dataset_root"),
            ),
            "requires model.modality='image'",
        ),
    ],
)
def test_modality_cross_field_contract_is_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    document = _image_document()
    mutation(document)

    with pytest.raises(ConfigError, match=message):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("encoder_name", "resnet50", "DenseNet121"),
        ("weights", "other", "densenet121-res224-chex"),
        ("image_size", True, "integer"),
        ("embedding_dimension", 512, "1024"),
        ("class_weighting", "none", "train_pos_weight"),
    ],
)
def test_image_model_contract_is_fixed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    document = _image_document()
    document["model"]["parameters"][field] = value

    with pytest.raises(ConfigError, match=message):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", True, "integer"),
        ("head_learning_rate", float("nan"), "finite"),
        ("head_learning_rate", float("inf"), "finite"),
        ("encoder_learning_rate", float("-inf"), "finite"),
        ("translation_fraction", 1.1, "within"),
        ("device", "mps", "one of"),
        ("optimizer", "sgd", "one of"),
        ("pin_memory_policy", "always", "one of"),
        ("scheduler_factor", 1.0, "within"),
        ("early_stopping_patience", -1, "nonnegative"),
    ],
)
def test_image_runtime_and_optimization_values_are_strict(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    document = _image_document()
    document["image"][field] = value

    with pytest.raises(ConfigError, match=message):
        load_experiment_config(_write(tmp_path, document))
