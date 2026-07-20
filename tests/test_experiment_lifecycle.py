from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from radfusion.data.rsna_artifacts import build_and_write
from radfusion.data.tabular_preprocess import SOURCE_FEATURES
from radfusion.training.config import load_experiment_config
from radfusion.training.datasets import RsnaDataset
from radfusion.training.evaluate import _verify_training_lineage, evaluate_training_run
from radfusion.training.interfaces import DatasetLineage, DatasetPartition, DatasetRunData
from radfusion.training.train_tabular import train_configured_experiment
from radfusion.utils.mlflow_utils import configure_mlflow
from radfusion.utils.model_publication import validate_published_model

_SHA256 = "a" * 64


def _partition(name: str) -> DatasetPartition:
    size = 12
    targets = np.asarray([0, 1] * (size // 2), dtype=np.int8)
    features = pd.DataFrame(
        {
            "age_years": np.linspace(25.0, 75.0, size),
            "age_is_implausible": [False] * size,
            "sex": ["F", "M"] * (size // 2),
            "view_position": ["PA", "AP"] * (size // 2),
            "pixel_spacing_row_mm": np.linspace(0.14, 0.19, size),
            "pixel_spacing_col_mm": np.linspace(0.14, 0.19, size),
        },
        columns=SOURCE_FEATURES,
    )
    return DatasetPartition(
        features=features,
        targets=targets,
        sample_ids=tuple(f"rsna:{name}-{index}" for index in range(size)),
        patient_ids=tuple(f"{name}-{index}" for index in range(size)),
        partition=name,
    )


def _config(
    tmp_path: Path,
    *,
    bundle_id: str = "build-synthetic",
    manifest_directory: Path | None = None,
    filename: str = "metadata_logistic.yaml",
):
    document = yaml.safe_load((Path("configs") / filename).read_text(encoding="utf-8"))
    document["dataset"]["bundle_id"] = bundle_id
    document["training"]["report_directory"] = str(tmp_path / "reports")
    document["training"]["model_directory"] = str(tmp_path / "models" / "rsna")
    document["evaluation"]["latency_warmup_calls"] = 1
    document["evaluation"]["latency_measured_calls"] = 3
    if manifest_directory is not None:
        document["dataset"]["manifest_directory"] = str(manifest_directory)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_experiment_config(path)


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _client(tmp_path: Path):
    return configure_mlflow(tracking_uri=_tracking_uri(tmp_path))


def _train(config, tmp_path: Path):
    return train_configured_experiment(config, tracking_uri=_tracking_uri(tmp_path))


def _install_dataset(monkeypatch: pytest.MonkeyPatch) -> tuple[DatasetRunData, DatasetPartition]:
    lineage = DatasetLineage(
        bundle_id="build-synthetic",
        split_assignment_id="assignment-synthetic",
        label_policy_version="label-synthetic",
        task_id="pneumonia",
    )
    data = DatasetRunData(
        train=_partition("train"),
        validation=_partition("validation"),
        lineage=lineage,
    )
    test = _partition("test")
    monkeypatch.setattr(
        RsnaDataset,
        "load_train_validation",
        lambda self, config: data,
    )
    monkeypatch.setattr(
        RsnaDataset,
        "load_test",
        lambda self, config: (test, lineage),
    )
    monkeypatch.setattr(
        RsnaDataset,
        "load_lineage",
        lambda self, config: lineage,
    )
    return data, test


def _fixed_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "radfusion.training.train_tabular.git_revision",
        lambda: ("commit-synthetic", True),
    )
    monkeypatch.setattr("radfusion.training.train_tabular.uv_lock_sha256", lambda: _SHA256)
    monkeypatch.setattr("radfusion.training.evaluate.uv_lock_sha256", lambda: _SHA256)


@pytest.mark.parametrize("filename", ["metadata_logistic.yaml", "metadata_lightgbm.yaml"])
def test_train_then_explicit_test_evaluation_uses_separate_partitions_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    data, _ = _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path, filename=filename)
    test_loads = 0
    original_test_loader = RsnaDataset.load_test

    def count_test_load(self, dataset_config):
        nonlocal test_loads
        test_loads += 1
        return original_test_loader(self, dataset_config)

    monkeypatch.setattr(RsnaDataset, "load_test", count_test_load)
    training = _train(config, tmp_path)
    assert test_loads == 0
    assert tuple(data.train.features.columns) == SOURCE_FEATURES
    assert not hasattr(data, "test")

    configure_mlflow(tracking_uri=f"sqlite:///{(tmp_path / 'other.db').as_posix()}")
    evaluation = evaluate_training_run(
        training.run_id,
        tracking_uri=_tracking_uri(tmp_path),
    )
    assert test_loads == 1
    training_run = mlflow.get_run(training.run_id)
    evaluation_run = mlflow.get_run(evaluation.run_id)
    assert training_run.data.tags["evaluation_scope"] == "validation"
    assert evaluation_run.data.tags["evaluation_scope"] == "test"
    assert evaluation_run.data.tags["source_training_run_id"] == training.run_id
    assert "test_average_precision" not in training_run.data.metrics
    assert "test_average_precision" in evaluation_run.data.metrics
    assert float(evaluation_run.data.tags["threshold_youden_j"]) == training.thresholds["youden_j"]
    assert evaluation_run.data.params["best_iteration"] == training_run.data.params.get(
        "best_iteration", "not_applicable"
    )

    manifest = validate_published_model(training.model_path.parent)
    assert set(manifest["thresholds"]) == {"youden_j", "target_sensitivity"}
    metrics = json.loads(
        (evaluation.artifact_directory / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["evaluation_scope"] == "test"

    client = _client(tmp_path)
    downloaded = Path(
        client.download_artifacts(training.run_id, "model/model.skops", tmp_path / "download")
    )
    assert downloaded.read_bytes() == training.model_path.read_bytes()


def test_fit_failure_leaves_failed_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)

    class FailingModel:
        def fit(self, *args, **kwargs):
            assert mlflow.active_run() is not None
            raise RuntimeError("fit failed")

    monkeypatch.setattr("radfusion.training.train_tabular.get_model", lambda _: FailingModel())
    with pytest.raises(RuntimeError, match="fit failed"):
        _train(config, tmp_path)
    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id]
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"
    assert run.data.tags["split_assignment_id"] == "assignment-synthetic"
    assert run.data.tags["label_policy_version"] == "label-synthetic"
    downloaded = Path(
        _client(tmp_path).download_artifacts(
            run.info.run_id,
            "config/resolved_config.yaml",
            tmp_path / "failed-config-download",
        )
    )
    assert downloaded.read_bytes() == config.source_bytes


def test_required_model_publication_failure_leaves_failed_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "radfusion.training.train_tabular.publish_model_run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("publication failed")),
    )

    with pytest.raises(RuntimeError, match="publication failed"):
        _train(config, tmp_path)
    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id]
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] != "true"


def test_incomplete_report_set_fails_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    from radfusion.training import train_tabular

    write_reports = train_tabular.write_run_reports

    def write_incomplete_reports(*args, **kwargs):
        write_reports(*args, **kwargs)
        Path(args[0], "calibration_curve.png").unlink()

    monkeypatch.setattr(train_tabular, "write_run_reports", write_incomplete_reports)
    with pytest.raises(ValueError, match="report set"):
        _train(config, tmp_path)

    run = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id]
    )[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"


def test_training_report_publication_failure_does_not_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "radfusion.training.train_tabular.publish_directory",
        lambda *args: (_ for _ in ()).throw(RuntimeError("report publication failed")),
    )

    with pytest.raises(RuntimeError, match="report publication failed"):
        _train(config, tmp_path)

    run = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id]
    )[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"


def test_training_archives_and_logs_loaded_config_bytes_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    original = config.source_bytes
    config.source_path.write_bytes(b"mutated-after-load: true\n")

    training = _train(config, tmp_path)

    assert (training.model_path.parent / "resolved_config.yaml").read_bytes() == original
    downloaded = Path(
        _client(tmp_path).download_artifacts(
            training.run_id,
            "config/resolved_config.yaml",
            tmp_path / "config-download",
        )
    )
    assert downloaded.read_bytes() == original


@pytest.mark.parametrize(
    "tag_name",
    [
        "dataset_bundle_id",
        "split_assignment_id",
        "task",
        "model",
        "seed",
        "git_commit",
        "source_config_sha256",
        "dependency_lock_sha256",
        "local_model_sha256",
    ],
)
def test_evaluator_rejects_each_training_lineage_tag_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag_name: str,
) -> None:
    config = _config(tmp_path)
    run_id = "training-run"
    model_path = config.training.model_directory / "runs" / run_id / "model.skops"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"serialized-model")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "training_mlflow_run_id": run_id,
        "source_config_sha256": config.source_sha256,
        "bundle_id": config.dataset.bundle_id,
        "split_assignment_id": "assignment-synthetic",
        "task": config.dataset.task_id,
        "seed": config.training.seed,
        "git_commit": "commit-synthetic",
        "dependency_lock_sha256": _SHA256,
        "positive_class": 1,
        "model_sha256": model_hash,
    }
    tags = {
        "dataset_bundle_id": manifest["bundle_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task": manifest["task"],
        "model": config.model.registry_key,
        "seed": str(manifest["seed"]),
        "git_commit": manifest["git_commit"],
        "source_config_sha256": manifest["source_config_sha256"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "local_model_sha256": manifest["model_sha256"],
    }
    tags[tag_name] = "mismatch"
    run = SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags=tags),
    )
    monkeypatch.setattr("radfusion.training.evaluate.uv_lock_sha256", lambda: _SHA256)

    with pytest.raises(ValueError, match=tag_name):
        _verify_training_lineage(run, config, manifest, model_path)


@pytest.mark.parametrize("policy", ["youden_j", "target_sensitivity"])
def test_evaluator_rejects_validation_threshold_mismatch_before_test_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    manifest_path = training.model_path.parent / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["thresholds"][policy] = 0.0 if manifest["thresholds"][policy] > 0.5 else 1.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    test_loads = 0

    def reject_test_load(self, dataset_config):
        nonlocal test_loads
        test_loads += 1
        raise AssertionError("test data must not be loaded")

    monkeypatch.setattr(RsnaDataset, "load_test", reject_test_load)
    with pytest.raises(ValueError, match=f"validation_{policy}_threshold mismatch"):
        evaluate_training_run(
            training.run_id,
            tracking_uri=_tracking_uri(tmp_path),
        )
    assert test_loads == 0


def test_evaluator_rejects_lightgbm_best_iteration_mismatch_before_test_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path, filename="metadata_lightgbm.yaml")
    training = _train(config, tmp_path)
    manifest_path = training.model_path.parent / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["best_iteration"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        RsnaDataset,
        "load_test",
        lambda self, dataset_config: (_ for _ in ()).throw(
            AssertionError("test data must not be loaded")
        ),
    )

    with pytest.raises(ValueError, match="best_iteration mismatch"):
        evaluate_training_run(
            training.run_id,
            tracking_uri=_tracking_uri(tmp_path),
        )


def test_evaluation_report_publication_failure_does_not_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    monkeypatch.setattr(
        "radfusion.training.evaluate.publish_directory",
        lambda *args: (_ for _ in ()).throw(RuntimeError("report publication failed")),
    )

    with pytest.raises(RuntimeError, match="report publication failed"):
        evaluate_training_run(
            training.run_id,
            tracking_uri=_tracking_uri(tmp_path),
        )

    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id],
        filter_string="tags.run_kind = 'test_evaluation'",
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] != "true"


@pytest.mark.integration
def test_synthetic_raw_source_to_bundle_training_and_explicit_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = _write_raw_source(tmp_path / "raw")
    manifest_root = tmp_path / "manifests"
    bundle = build_and_write(raw_root, manifest_root)
    monkeypatch.setattr(
        "radfusion.training.train_tabular.git_revision",
        lambda: ("commit-synthetic", True),
    )
    config = _config(
        tmp_path,
        bundle_id=bundle.paths.bundle_id,
        manifest_directory=manifest_root,
    )

    tracking_uri = _tracking_uri(tmp_path)
    training = train_configured_experiment(config, tracking_uri=tracking_uri)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "radfusion.training.evaluate",
            "--run-id",
            training.run_id,
            "--tracking-uri",
            tracking_uri,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    evaluation_run_id = output["test_evaluation_run_id"]
    evaluation_directory = Path(output["artifact_directory"])

    assert training.artifact_directory.is_dir()
    assert evaluation_directory.is_dir()
    assert mlflow.get_run(training.run_id).data.tags["evaluation_scope"] == "validation"
    assert _client(tmp_path).get_run(evaluation_run_id).data.tags["evaluation_scope"] == "test"


def _write_raw_source(root: Path) -> Path:
    images = root / "stage_2_train_images"
    images.mkdir(parents=True)
    labels = []
    classes = []
    for target in (0, 1):
        for index in range(6):
            patient_id = f"patient-{target}-{index}"
            _write_dicom(images / f"{patient_id}.dcm", patient_id, age=35 + target * 20 + index)
            labels.append(
                {
                    "patientId": patient_id,
                    "x": 10 if target else None,
                    "y": 20 if target else None,
                    "width": 30 if target else None,
                    "height": 40 if target else None,
                    "Target": target,
                }
            )
            classes.append(
                {
                    "patientId": patient_id,
                    "class": "Lung Opacity" if target else "Normal",
                }
            )
    pd.DataFrame(labels).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return root


def _write_dicom(path: Path, patient_id: str, *, age: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = patient_id
    dataset.PatientAge = f"{age:03d}Y"
    dataset.PatientSex = "F" if age % 2 else "M"
    dataset.ViewPosition = "PA"
    dataset.PixelSpacing = [0.168, 0.168]
    dataset.Rows = 1024
    dataset.Columns = 1024
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Modality = "CR"
    dataset.BodyPartExamined = "CHEST"
    dataset.save_as(path)
