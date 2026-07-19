from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from radfusion.training.config import load_experiment_config
from radfusion.training.datasets import RsnaDataset
from radfusion.training.interfaces import DatasetRunData
from radfusion.training.registry import register_builtin_components
from radfusion.training.train_tabular import _evaluate_and_log, train_configured_experiment
from radfusion.utils.model_publication import validate_published_model
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory
from radfusion.utils.skops_io import load_skops


def _temporary_publication_paths(parent: Path, name: str) -> list[Path]:
    return sorted(
        [
            *parent.glob(f".{name}-staging-*"),
            *parent.glob(f".{name}-backup-*"),
        ]
    )


def test_successful_directory_publication_replaces_complete_previous_output(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reports"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    stage = staging_directory(destination)
    (stage / "new.txt").write_text("new", encoding="utf-8")

    publish_directory(stage, destination)

    assert {path.name for path in destination.iterdir()} == {"new.txt"}
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    assert _temporary_publication_paths(tmp_path, destination.name) == []


def test_failed_directory_publication_restores_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "reports"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    stage = staging_directory(destination)
    (stage / "partial.txt").write_text("partial", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_stage_publish(source: str | Path, target: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        real_replace(source, target)

    monkeypatch.setattr("radfusion.utils.publication.os.replace", fail_stage_publish)

    with pytest.raises(OSError, match="publication failed"):
        publish_directory(stage, destination)

    assert {path.name for path in destination.iterdir()} == {"old.txt"}
    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert _temporary_publication_paths(tmp_path, destination.name) == []


class _ProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = features["score"].to_numpy(dtype=np.float64)
        return np.column_stack([1.0 - scores, scores])


def test_model_report_generation_failure_preserves_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config("configs/metadata_logistic.yaml")
    report_root = tmp_path / "reports"
    config = replace(
        config,
        training=replace(
            config.training,
            report_directory=report_root,
            model_directory=tmp_path / "models",
        ),
    )
    destination = report_root / "rsna" / "models" / config.model.output_name
    destination.mkdir(parents=True)
    (destination / "previous.txt").write_text("complete", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "split_name": ["validation"] * 4 + ["test"] * 4,
            "target": [0, 0, 1, 1] * 2,
            "score": [0.1, 0.3, 0.6, 0.9] * 2,
        }
    )

    def fail_plots(*args: object, **kwargs: object) -> None:
        output = Path(kwargs["output_directory"])
        (output / "confusion_matrix.png").write_bytes(b"partial")
        raise RuntimeError("plot generation failed")

    monkeypatch.setattr(
        "radfusion.training.train_tabular.benchmark_single_sample_latency_ms", lambda *a, **k: 1.0
    )
    monkeypatch.setattr("radfusion.training.train_tabular.write_evaluation_plots", fail_plots)

    with pytest.raises(RuntimeError, match="plot generation failed"):
        _evaluate_and_log(
            config=config,
            model=_ProbabilityModel(),
            frame=frame,
            lineage={},
        )

    assert {path.name for path in destination.iterdir()} == {"previous.txt"}
    assert not list(destination.parent.glob(f".{destination.name}-staging-*"))
    assert not list(destination.parent.glob(f".{destination.name}-backup-*"))


def test_synthetic_logistic_experiment_completes_full_training_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame_rows: list[dict[str, object]] = []
    for split_index, split_name in enumerate(("train", "validation", "test")):
        for index in range(8):
            target = index % 2
            name = f"synthetic-{split_name}-{index}"
            frame_rows.append(
                {
                    "sample_id": f"rsna:{name}",
                    "patient_id": name,
                    "image_id": name,
                    "image_path": f"stage_2_train_images/{name}.dcm",
                    "image_rows": 1024,
                    "image_columns": 1024,
                    "age_years": 30.0 + 20.0 * target + split_index,
                    "age_is_implausible": False,
                    "sex": "F" if split_name == "train" else "M",
                    "view_position": "PA" if split_name == "train" else "AP",
                    "pixel_spacing_row_mm": 0.14 + 0.02 * target,
                    "pixel_spacing_col_mm": 0.14 + 0.02 * target,
                    "split_name": split_name,
                    "target": target,
                }
            )
    frame = pd.DataFrame.from_records(frame_rows).sort_values("sample_id", kind="stable")
    dataset = DatasetRunData(
        frame=frame,
        bundle_id="build-synthetic",
        split_recipe_id="recipe-synthetic",
        split_assignment_id="assignment-synthetic",
        label_policy_version="label-synthetic",
    )
    monkeypatch.setattr(RsnaDataset, "load", lambda self, config: dataset)
    monkeypatch.setattr(
        "radfusion.training.train_tabular.git_revision", lambda: ("commit-synthetic", False)
    )
    register_builtin_components()

    config_text = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    report_root = tmp_path / "reports"
    model_root = tmp_path / "models" / "rsna"
    tracking_root = tmp_path / "mlruns"
    config_text = config_text.replace(
        "  report_directory: reports", f"  report_directory: {report_root}"
    )
    config_text = config_text.replace(
        "  model_directory: models/rsna", f"  model_directory: {model_root}"
    )
    config_text = config_text.replace("  require_clean_git: true", "  require_clean_git: false")
    config_text = config_text.replace("  latency_warmup_calls: 100", "  latency_warmup_calls: 1")
    config_text = config_text.replace(
        "  latency_measured_calls: 1000", "  latency_measured_calls: 3"
    )
    config_text = config_text.replace(
        "  tracking_directory: mlruns", f"  tracking_directory: {tracking_root}"
    )
    config_path = tmp_path / "synthetic-logistic.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    config = load_experiment_config(config_path)

    result = train_configured_experiment(config)

    assert result.run_id == result.model_path.parent.name
    assert result.artifact_directory.is_dir()
    report_names = {path.name for path in result.artifact_directory.iterdir()}
    assert {
        "metrics.json",
        "evaluation_report.md",
        "confusion_summary.md",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
    } <= report_names
    validate_public_reports(
        result.artifact_directory.iterdir(),
        forbidden_source_values={
            str(value)
            for column in ("patient_id", "sample_id", "image_id", "image_path")
            for value in frame[column]
        },
    )
    metrics = json.loads((result.artifact_directory / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["probability_metrics"]["test"]) >= {
        "average_precision",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "calibration_slope",
        "calibration_intercept",
    }

    lineage = validate_published_model(result.model_path.parent)
    assert lineage["mlflow_run_id"] == result.run_id
    assert lineage["model_key"] == config.model.output_name
    assert lineage["config_source_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert lineage["bundle_id"] == dataset.bundle_id
    assert lineage["split_recipe_id"] == dataset.split_recipe_id
    assert lineage["split_assignment_id"] == dataset.split_assignment_id
    assert lineage["training_seed"] == config.training.seed
    assert lineage["git_commit"] == "commit-synthetic"
    assert lineage["git_source_state_sha256"] == "clean"
    assert lineage["model_artifact_sha256"] == result.model_artifact_sha256
    restored = load_skops(result.model_path)
    categories = (
        restored.named_steps["preprocess"]
        .named_steps["columns"]
        .named_transformers_["categorical"]
        .named_steps["encode"]
        .categories_
    )
    assert "M" not in categories[0]
    test_features = frame.loc[frame["split_name"] == "test"].drop(columns=["target"])
    assert np.isfinite(restored.predict_proba(test_features)).all()

    run = mlflow.get_run(result.run_id)
    assert run.data.tags["dataset_bundle_id"] == dataset.bundle_id
    assert run.data.tags["split_assignment_id"] == dataset.split_assignment_id
    assert run.data.tags["experiment_config_sha256"] == config.source_sha256
    assert run.data.tags["local_model_artifact_sha256"] == result.model_artifact_sha256
    assert run.data.params["training_seed"] == str(config.training.seed)
    assert "test_average_precision" in run.data.metrics
    assert "test_expected_calibration_error" in run.data.metrics
    artifact_names = {
        artifact.path for artifact in mlflow.tracking.MlflowClient().list_artifacts(result.run_id)
    }
    assert {"metrics.json", "evaluation_report.md", "confusion_summary.md"} <= artifact_names

    comparison = pd.read_csv(report_root / "model_comparison_table.csv").iloc[0]
    assert comparison["mlflow_run_id"] == result.run_id
    assert comparison["dataset_bundle_id"] == dataset.bundle_id
    assert comparison["split_assignment_id"] == dataset.split_assignment_id
    assert comparison["experiment_config_sha256"] == config.source_sha256
    assert comparison["local_model_sha256"] == result.model_artifact_sha256
