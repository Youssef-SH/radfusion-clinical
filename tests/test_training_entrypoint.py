from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from radfusion.training.config import load_experiment_config
from radfusion.training.train import main
from radfusion.training.train_tabular import train_configured_experiment


def test_training_entrypoint_invokes_runner_from_config(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_train(config):
        captured["config"] = config
        return SimpleNamespace(
            model_name="metadata_logistic_regression",
            run_id="run-test",
            test_youden_j=SimpleNamespace(probability=SimpleNamespace(average_precision=0.4)),
            model_path=Path("models/rsna/model/runs/run-test/model.skops"),
            artifact_directory=Path("reports/rsna/models/model"),
        )

    monkeypatch.setattr("radfusion.training.train.train_configured_experiment", fake_train)

    assert main(["--config", "configs/metadata_logistic.yaml"]) == 0
    assert captured["config"].model.registry_key == "metadata_logistic"
    assert '"config": "configs/metadata_logistic.yaml"' in capsys.readouterr().out


def test_release_configuration_rejects_dirty_source_before_loading_data(monkeypatch) -> None:
    config = load_experiment_config("configs/metadata_logistic.yaml")
    monkeypatch.setattr("radfusion.training.train_tabular.git_revision", lambda: ("commit", True))

    with pytest.raises(ValueError, match="clean Git tree"):
        train_configured_experiment(config)
