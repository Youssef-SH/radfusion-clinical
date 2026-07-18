"""Configure lightweight local MLflow experiment tracking."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm
import mlflow
import numpy
import pyarrow
import sklearn
import skops

from radfusion.data.hashing import sha256_file

_SOURCE_ROOTS = ("src/", "tests/", "configs/", "docs/")
_SOURCE_FILES = {
    "AGENTS.md",
    "README.md",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".pre-commit-config.yaml",
}


def configure_mlflow(*, experiment_name: str, tracking_directory: str | Path = "mlruns") -> None:
    """Configure a local file-backed MLflow experiment."""
    tracking_root = Path(tracking_directory).resolve()
    tracking_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_root.as_uri())
    mlflow.set_experiment(experiment_name)


@contextmanager
def tracked_run(
    *,
    run_name: str,
    tags: dict[str, str],
    parameters: dict[str, Any],
) -> Iterator[str]:
    """Start an MLflow run and log normalized tags and parameters."""
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.set_tag("run_date", datetime.now(UTC).date().isoformat())
        mlflow.log_params({key: _parameter_value(value) for key, value in parameters.items()})
        yield run.info.run_id


def git_revision() -> tuple[str, bool]:
    """Return the current Git commit and dirty-worktree flag."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def uv_lock_sha256(path: str | Path = "uv.lock") -> str:
    """Return the SHA-256 of the exact dependency lock."""
    lock = Path(path)
    if not lock.is_file():
        raise FileNotFoundError(f"Dependency lock is missing: {lock}")
    return sha256_file(lock)


def write_dirty_source_snapshot(destination: str | Path) -> Path:
    """Archive the tracked diff and relevant nonignored untracked source files."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    (output / "tracked.diff").write_bytes(diff)
    untracked_output = output / "untracked"
    paths = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    archived: dict[str, str] = {}
    for raw_path in sorted(paths):
        if not raw_path:
            continue
        path_text = raw_path.decode("utf-8")
        if not (_is_source_path(path_text)):
            continue
        source = Path(path_text)
        if not source.is_file() or source.is_symlink():
            continue
        target = untracked_output / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        archived[path_text] = sha256_file(target)
    identity = {
        "tracked_diff_sha256": sha256_file(output / "tracked.diff"),
        "untracked_file_sha256": archived,
    }
    source_state_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "git_commit": git_revision()[0],
        **identity,
        "untracked_files": sorted(archived),
        "source_state_sha256": source_state_sha256,
    }
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def environment_provenance() -> dict[str, str]:
    """Return runtime provenance for experiment tracking."""
    return {
        "environment_python_version": platform.python_version(),
        "environment_operating_system": platform.platform(),
        "environment_cpu_architecture": platform.machine(),
        "environment_cpu_model": cpu_model(),
        "environment_numpy_version": numpy.__version__,
        "environment_pyarrow_version": pyarrow.__version__,
        "environment_scikit_learn_version": sklearn.__version__,
        "environment_lightgbm_version": lightgbm.__version__,
        "environment_mlflow_version": mlflow.__version__,
        "environment_skops_version": skops.__version__,
    }


def cpu_model() -> str:
    """Return the detected CPU model, falling back to ``unknown``."""
    try:
        system = platform.system()
        if system == "Linux":
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            for line in cpuinfo.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip().lower() in {"model name", "hardware"}:
                    if model := value.strip():
                        return model
        elif system == "Darwin":
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if model := completed.stdout.strip():
                return model
        elif system == "Windows":
            if model := os.environ.get("PROCESSOR_IDENTIFIER", "").strip():
                return model
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        return platform.processor().strip() or "unknown"
    except (OSError, ValueError):
        return "unknown"


def _parameter_value(value: Any) -> str | float | int | bool:
    if value is None:
        return "not_applicable"
    if isinstance(value, str | float | int | bool):
        return value
    return str(value)


def _is_source_path(path: str) -> bool:
    return path in _SOURCE_FILES or path.startswith(_SOURCE_ROOTS)
