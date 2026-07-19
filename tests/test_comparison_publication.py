from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from radfusion.evaluation.metrics import evaluate_binary
from radfusion.training.config import load_experiment_config
from radfusion.training.train_tabular import ModelResult, _write_comparison_table


def _result(tmp_path: Path, *, run_id: str = "run-test") -> ModelResult:
    metrics = evaluate_binary([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9], threshold=0.5)
    return ModelResult(
        model_name="metadata_logistic_regression",
        run_id=run_id,
        youden_j_threshold=0.5,
        target_sensitivity_threshold=0.4,
        validation_youden_j=metrics,
        test_youden_j=metrics,
        validation_target_sensitivity=metrics,
        test_target_sensitivity=metrics,
        model_path=tmp_path / "model" / "runs" / run_id / "model.skops",
        model_artifact_sha256="model-sha",
        artifact_directory=tmp_path / "artifacts",
        latency_ms=1.0,
        model_size_mib=2.0,
    )


def _lineage() -> dict[str, object]:
    return {
        "bundle_id": "bundle",
        "split_assignment_id": "assignment",
        "split_recipe_id": "recipe",
        "git_commit": "commit",
        "git_dirty": False,
        "git_source_state_sha256": "clean",
        "uv_lock_sha256": "lock",
    }


def test_comparison_table_uses_semantic_experiment_key_and_exact_lineage(tmp_path: Path) -> None:
    path = tmp_path / "model_comparison_table.csv"
    config = load_experiment_config("configs/metadata_logistic.yaml")
    _write_comparison_table(_result(tmp_path), config, _lineage(), path)
    _write_comparison_table(_result(tmp_path, run_id="replacement"), config, _lineage(), path)

    table = pd.read_csv(path)
    assert len(table) == 1
    current = table.iloc[0]
    assert current["experiment_config_sha256"] == config.source_sha256
    assert current["mlflow_run_id"] == "replacement"
    assert current["split_assignment_id"] == "assignment"
    assert current["local_model_sha256"] == "model-sha"
    assert "test_target_sensitivity_recall" in table.columns
    assert "validation_average_precision" in table.columns
    assert "pr_auc" not in " ".join(table.columns).lower()
    assert path.with_suffix(".csv.lock").is_file()


def test_comparison_table_keeps_distinct_semantic_experiments(tmp_path: Path) -> None:
    path = tmp_path / "model_comparison_table.csv"
    config = load_experiment_config("configs/metadata_logistic.yaml")
    _write_comparison_table(_result(tmp_path), config, _lineage(), path)
    second_lineage = {**_lineage(), "split_assignment_id": "assignment-2"}
    _write_comparison_table(_result(tmp_path, run_id="run-2"), config, second_lineage, path)
    assert len(pd.read_csv(path)) == 2


def test_comparison_table_distinguishes_dirty_source_states(tmp_path: Path) -> None:
    path = tmp_path / "model_comparison_table.csv"
    config = load_experiment_config("configs/metadata_logistic.yaml")
    first = {**_lineage(), "git_dirty": True, "git_source_state_sha256": "source-a"}
    second = {**first, "git_source_state_sha256": "source-b"}

    _write_comparison_table(_result(tmp_path), config, first, path)
    _write_comparison_table(_result(tmp_path, run_id="run-2"), config, second, path)

    assert len(pd.read_csv(path)) == 2


def test_comparison_table_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model_comparison_table.csv"
    previous = "dataset,task,model_name\nother,task,previous\n"
    path.write_text(previous, encoding="utf-8")
    config = load_experiment_config("configs/metadata_logistic.yaml")

    def fail_write(*args, **kwargs) -> None:
        raise RuntimeError("CSV generation failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    with pytest.raises(RuntimeError, match="CSV generation failed"):
        _write_comparison_table(_result(tmp_path), config, _lineage(), path)
    assert path.read_text(encoding="utf-8") == previous
    assert not list(tmp_path.glob(".model_comparison_table.csv-*.tmp"))


def test_comparison_table_preserves_unrelated_columns(tmp_path: Path) -> None:
    path = tmp_path / "model_comparison_table.csv"
    pd.DataFrame([{"dataset": "other", "private_note": "preserve"}]).to_csv(path, index=False)

    _write_comparison_table(
        _result(tmp_path),
        load_experiment_config("configs/metadata_logistic.yaml"),
        _lineage(),
        path,
    )

    table = pd.read_csv(path)
    assert table.loc[table["dataset"] == "other", "private_note"].iloc[0] == "preserve"
