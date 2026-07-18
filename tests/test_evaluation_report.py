from __future__ import annotations

from pathlib import Path

from radfusion.evaluation.metrics import evaluate_binary, target_sensitivity_threshold
from radfusion.training.config import load_experiment_config
from radfusion.training.train_tabular import _metrics_document, _write_evaluation_report


def test_report_renders_configured_sensitivity_target_and_exact_metric_names(
    tmp_path: Path,
) -> None:
    targets = [0, 0, 1, 1]
    probabilities = [0.1, 0.3, 0.6, 0.9]
    target_threshold = target_sensitivity_threshold(targets, probabilities, sensitivity=0.9)
    metrics = evaluate_binary(targets, probabilities, threshold=0.5, calibration_bins=15)
    target_metrics = evaluate_binary(
        targets, probabilities, threshold=target_threshold, calibration_bins=15
    )
    config = load_experiment_config("configs/metadata_logistic.yaml")
    document = _metrics_document(
        config,
        0.5,
        target_threshold,
        metrics,
        metrics,
        target_metrics,
        target_metrics,
    )
    path = tmp_path / "evaluation_report.md"

    _write_evaluation_report(path, "test_model", document)

    report = path.read_text(encoding="utf-8")
    assert "Average precision" in report
    assert "PR-AUC" not in report
    assert "Youden-J comparative operating point" in report
    assert "Target-sensitivity operating point" in report
    assert "15 equal-width probability bins" in report
