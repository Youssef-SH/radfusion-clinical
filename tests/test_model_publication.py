from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from radfusion.utils.model_publication import (
    REQUIRED_MANIFEST_FIELDS,
    publish_model_run,
    validate_published_model,
)
from radfusion.utils.skops_io import load_skops, save_skops

_SHA256 = "a" * 64


def _manifest() -> dict[str, object]:
    return {
        "bundle_id": "build-test",
        "split_assignment_id": "assignment-test",
        "task": "pneumonia",
        "positive_class": 1,
        "source_config_sha256": _SHA256,
        "seed": 42,
        "git_commit": "commit-test",
        "dependency_lock_sha256": _SHA256,
        "best_iteration": None,
        "thresholds": {"youden_j": 0.5, "target_sensitivity": 0.3},
    }


def _publish(tmp_path: Path):
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model = LogisticRegression().fit(features, [0, 0, 1, 1])
    serialized = save_skops(model, tmp_path / "source.skops")
    config = tmp_path / "source.yaml"
    config.write_text("config_version: 1\n", encoding="utf-8")
    manifest = {
        **_manifest(),
        "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    published = publish_model_run(
        model_root=tmp_path / "models" / "rsna",
        mlflow_run_id="run-test",
        serialized_model_path=serialized,
        source_config_bytes=config.read_bytes(),
        manifest=manifest,
    )
    return features, model, published


def test_model_package_is_compact_run_qualified_and_loadable(tmp_path: Path) -> None:
    features, model, published = _publish(tmp_path)

    assert published.run_directory == tmp_path / "models" / "rsna" / "runs" / "run-test"
    assert {path.name for path in published.run_directory.iterdir()} == {
        "model.skops",
        "resolved_config.yaml",
        "model_manifest.json",
    }
    document = validate_published_model(published.run_directory)
    assert set(document) == REQUIRED_MANIFEST_FIELDS
    np.testing.assert_array_equal(
        load_skops(published.model_path).predict_proba(features),
        model.predict_proba(features),
    )


def test_model_package_preserves_exact_serialized_and_config_bytes(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    assert published.model_path.read_bytes() == (tmp_path / "source.skops").read_bytes()
    assert published.config_path.read_bytes() == (tmp_path / "source.yaml").read_bytes()


def test_model_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    document["model_sha256"] = "tampered"
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_published_model(published.run_directory)


@pytest.mark.parametrize("run_id", ["../run", "a/b", ".", ".."])
def test_model_publication_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    serialized = save_skops(model, tmp_path / "source.skops")
    config = tmp_path / "source.yaml"
    config.write_text("test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="safe path component"):
        publish_model_run(
            model_root=tmp_path / "models",
            mlflow_run_id=run_id,
            serialized_model_path=serialized,
            source_config_bytes=config.read_bytes(),
            manifest={
                **_manifest(),
                "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            },
        )


def test_conflicting_model_publication_retry_is_rejected(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    with pytest.raises(FileExistsError, match="conflicting"):
        publish_model_run(
            model_root=tmp_path / "models" / "rsna",
            mlflow_run_id="run-test",
            serialized_model_path=published.model_path,
            source_config_bytes=published.config_path.read_bytes(),
            manifest={
                **_manifest(),
                "bundle_id": "different",
                "source_config_sha256": hashlib.sha256(
                    published.config_path.read_bytes()
                ).hexdigest(),
            },
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("youden_j", True),
        ("youden_j", "0.5"),
        ("youden_j", float("nan")),
        ("youden_j", float("inf")),
        ("youden_j", -0.1),
        ("youden_j", 1.1),
        ("missing", None),
        ("extra", 0.5),
    ],
)
def test_model_manifest_rejects_invalid_thresholds(
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    thresholds = document["thresholds"]
    if mutation == "missing":
        thresholds.pop("youden_j")
    elif mutation == "extra":
        thresholds["unexpected"] = value
    else:
        thresholds[mutation] = value
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold"):
        validate_published_model(published.run_directory)
