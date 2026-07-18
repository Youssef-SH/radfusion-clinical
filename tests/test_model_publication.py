from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import radfusion.utils.model_publication as model_publication
from radfusion.utils.model_publication import publish_model_run, validate_published_model
from radfusion.utils.skops_io import load_skops, save_skops

_SHA256 = "a" * 64


def _lineage() -> dict[str, object]:
    return {
        "dataset_id": "rsna",
        "task_id": "pneumonia",
        "config_source_sha256": _SHA256,
        "bundle_id": "build-test",
        "cohort_fingerprint": "cohort-test",
        "split_recipe_id": "recipe-test",
        "split_assignment_id": "assignment-test",
        "label_policy_version": "label-test",
        "training_seed": 42,
        "git_commit": "commit-test",
        "git_dirty": False,
        "git_source_state_sha256": "clean",
        "uv_lock_sha256": _SHA256,
        "derived_parameters": {"best_iteration": None},
    }


def test_model_run_publication_is_immutable_traceable_and_loadable(tmp_path) -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    targets = np.asarray([0, 0, 1, 1])
    model = LogisticRegression().fit(features, targets)
    published = publish_model_run(
        model,
        model_root=tmp_path / "models" / "rsna",
        model_key="metadata_logistic",
        mlflow_run_id="run-test",
        lineage=_lineage(),
    )

    assert published.run_directory.name == "run-test"
    assert (tmp_path / "models" / "rsna" / "metadata_logistic" / "CURRENT").read_text(
        encoding="utf-8"
    ) == "run-test\n"
    lineage = validate_published_model(published.run_directory)
    assert lineage["model_artifact_sha256"] == published.model_artifact_sha256
    restored = load_skops(published.model_path)
    np.testing.assert_array_equal(restored.predict_proba(features), model.predict_proba(features))

    retried = publish_model_run(
        None,
        model_root=tmp_path / "models" / "rsna",
        model_key="metadata_logistic",
        mlflow_run_id="run-test",
        serialized_model_path=published.model_path,
        lineage=_lineage(),
    )
    assert retried == published


def test_model_lineage_tampering_is_rejected(tmp_path) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    published = publish_model_run(
        model,
        model_root=tmp_path,
        model_key="model",
        mlflow_run_id="run",
        lineage=_lineage(),
    )
    document = json.loads(published.lineage_path.read_text(encoding="utf-8"))
    document["model_artifact_sha256"] = "tampered"
    published.lineage_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_published_model(published.run_directory)


@pytest.mark.parametrize(("model_key", "run_id"), [("../model", "run"), ("model", "../run")])
def test_model_publication_rejects_unsafe_path_components(
    tmp_path, model_key: str, run_id: str
) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="safe path component"):
        publish_model_run(
            model,
            model_root=tmp_path,
            model_key=model_key,
            mlflow_run_id=run_id,
            lineage=_lineage(),
        )


@pytest.mark.parametrize("entry_kind", ["file", "directory", "symlink"])
def test_model_validation_rejects_unexpected_entries(tmp_path, entry_kind: str) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    published = publish_model_run(
        model,
        model_root=tmp_path,
        model_key="model",
        mlflow_run_id="run",
        lineage=_lineage(),
    )
    unexpected = published.run_directory / "unexpected"
    if entry_kind == "file":
        unexpected.write_text("unexpected\n", encoding="utf-8")
    elif entry_kind == "directory":
        unexpected.mkdir()
    else:
        unexpected.symlink_to(published.model_path.name)

    with pytest.raises(ValueError, match="unexpected artifact set"):
        validate_published_model(published.run_directory)


@pytest.mark.parametrize("filename", ["model.skops", "lineage.json"])
def test_model_validation_rejects_required_symlinks(tmp_path, filename: str) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    published = publish_model_run(
        model,
        model_root=tmp_path,
        model_key="model",
        mlflow_run_id="run",
        lineage=_lineage(),
    )
    required = published.run_directory / filename
    external = tmp_path / f"external-{filename}"
    required.rename(external)
    required.symlink_to(external)

    with pytest.raises(ValueError, match="regular non-symlink"):
        validate_published_model(published.run_directory)


def test_model_validation_uses_the_physical_parent_model_key(tmp_path) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    published = publish_model_run(
        model,
        model_root=tmp_path,
        model_key="original-model",
        mlflow_run_id="run",
        lineage=_lineage(),
    )
    moved = tmp_path / "different-model" / "runs" / "run"
    moved.parent.mkdir(parents=True)
    published.run_directory.rename(moved)

    with pytest.raises(ValueError, match="does not match its immutable run path"):
        validate_published_model(moved)


def test_model_publication_recovers_after_current_update_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    serialized = save_skops(model, tmp_path / "serialized.skops")
    model_root = tmp_path / "models"
    real_update = model_publication._update_current
    attempts = 0

    def fail_once(path, run_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("CURRENT update failed")
        real_update(path, run_id)

    monkeypatch.setattr("radfusion.utils.model_publication._update_current", fail_once)
    with pytest.raises(OSError, match="CURRENT update failed"):
        publish_model_run(
            None,
            model_root=model_root,
            model_key="model",
            mlflow_run_id="run",
            serialized_model_path=serialized,
            lineage=_lineage(),
        )

    run_directory = model_root / "model" / "runs" / "run"
    validate_published_model(run_directory)
    original_contents = {path.name: path.read_bytes() for path in sorted(run_directory.iterdir())}

    recovered = publish_model_run(
        None,
        model_root=model_root,
        model_key="model",
        mlflow_run_id="run",
        serialized_model_path=serialized,
        lineage=_lineage(),
    )

    assert recovered.run_directory == run_directory
    assert (model_root / "model" / "CURRENT").read_text(encoding="utf-8") == "run\n"
    assert {path.name: path.read_bytes() for path in sorted(run_directory.iterdir())} == (
        original_contents
    )


def test_model_publication_rejects_conflicting_retry_content(tmp_path) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    serialized = save_skops(model, tmp_path / "serialized.skops")
    publish_model_run(
        None,
        model_root=tmp_path / "models",
        model_key="model",
        mlflow_run_id="run",
        serialized_model_path=serialized,
        lineage=_lineage(),
    )
    conflicting = {**_lineage(), "task_id": "different-task"}

    with pytest.raises(FileExistsError, match="conflicting content"):
        publish_model_run(
            None,
            model_root=tmp_path / "models",
            model_key="model",
            mlflow_run_id="run",
            serialized_model_path=serialized,
            lineage=conflicting,
        )
