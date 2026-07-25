from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from radfusion.training.config import (
    ConfigError,
    image_semantic_config_sha256,
    load_experiment_config,
)
from radfusion.training.train import main as train_main


def _image_document() -> dict[str, object]:
    return yaml.safe_load(Path("configs/image_densenet.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "image.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_image_config_is_strict_single_seed_and_executable() -> None:
    config = load_experiment_config("configs/image_densenet.yaml")

    assert config.executable is True
    assert config.training.seed == 42
    assert not hasattr(config.training, "seeds")
    assert config.dataset.dataset_root == Path("data/raw/rsna/extracted")
    assert (
        config.dataset.bundle_metadata_sha256
        == "246f00dc185e6a2935e17317684de9de5026cd6d7e25843de138c09952428e75"
    )
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


def test_image_semantic_config_identity_excludes_paths_but_binds_training_meaning(
    tmp_path: Path,
) -> None:
    baseline = load_experiment_config("configs/image_densenet.yaml")
    path_changed = _image_document()
    path_changed["dataset"]["dataset_root"] = "/different/raw/root"
    path_changed["dataset"]["manifest_directory"] = "/different/manifests"
    path_changed["training"]["model_directory"] = "/different/models"
    path_changed["training"]["report_directory"] = "/different/reports"
    changed_path = load_experiment_config(_write(tmp_path, path_changed))
    meaning_changed = _image_document()
    meaning_changed["training"]["seed"] = 17
    meaning_path = tmp_path / "meaning.yaml"
    meaning_path.write_text(yaml.safe_dump(meaning_changed, sort_keys=False), encoding="utf-8")
    changed_meaning = load_experiment_config(meaning_path)

    assert image_semantic_config_sha256(changed_path) == image_semantic_config_sha256(baseline)
    assert image_semantic_config_sha256(changed_meaning) != image_semantic_config_sha256(baseline)


@pytest.mark.parametrize("value", [None, "A" * 64, "a" * 63, "g" * 64])
def test_image_config_requires_exact_bundle_metadata_sha256(
    tmp_path: Path, value: str | None
) -> None:
    document = _image_document()
    if value is None:
        document["dataset"].pop("bundle_metadata_sha256")
    else:
        document["dataset"]["bundle_metadata_sha256"] = value

    with pytest.raises(ConfigError, match="bundle_metadata_sha256"):
        load_experiment_config(_write(tmp_path, document))


def test_bundle_metadata_pin_changes_image_semantic_identity(tmp_path: Path) -> None:
    baseline = load_experiment_config("configs/image_densenet.yaml")
    document = _image_document()
    document["dataset"]["bundle_metadata_sha256"] = "a" * 64
    changed = load_experiment_config(_write(tmp_path, document))

    assert image_semantic_config_sha256(changed) != image_semantic_config_sha256(baseline)


def test_operational_image_fields_do_not_change_semantic_identity(tmp_path: Path) -> None:
    baseline = load_experiment_config("configs/image_densenet.yaml")
    document = _image_document()
    document["executable"] = False
    document["evaluation"]["latency_warmup_calls"] = 1
    document["evaluation"]["latency_measured_calls"] = 2
    changed = load_experiment_config(_write(tmp_path, document))

    assert image_semantic_config_sha256(changed) == image_semantic_config_sha256(baseline)
    assert changed.source_sha256 != baseline.source_sha256


def test_metadata_configs_remain_executable_and_unchanged() -> None:
    for path in ("configs/metadata_logistic.yaml", "configs/metadata_lightgbm.yaml"):
        config = load_experiment_config(path)
        assert config.executable is True
        assert config.model.modality == "metadata"
        assert config.image is None
        assert config.dataset.dataset_root is None


def test_image_config_dispatches_to_image_runner(monkeypatch, capsys) -> None:
    captured = {}

    def fake_training(config, *, tracking_uri):
        captured.update(config=config, tracking_uri=tracking_uri)
        return type(
            "Result",
            (),
            {
                "model_name": "image_densenet",
                "run_id": "image-run",
                "validation_probability": type("Metrics", (), {"average_precision": 0.5})(),
                "model_path": Path("models/rsna/runs/image-run/model.pt"),
                "artifact_directory": Path("reports/rsna/runs/image-run"),
            },
        )()

    monkeypatch.setattr("radfusion.training.train.train_image_experiment", fake_training)

    assert train_main(["--config", "configs/image_densenet.yaml"]) == 0
    assert captured["config"].model.modality == "image"
    assert captured["tracking_uri"] == "sqlite:///mlflow.db"
    assert '"mlflow_run_id": "image-run"' in capsys.readouterr().out


def test_non_executable_image_config_is_rejected_by_execution_cli(tmp_path: Path, capsys) -> None:
    document = _image_document()
    document["executable"] = False

    assert train_main(["--config", str(_write(tmp_path, document))]) == 1
    assert "not executable" in capsys.readouterr().err


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
