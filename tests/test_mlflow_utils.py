from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from radfusion.utils.mlflow_utils import uv_lock_sha256, write_dirty_source_snapshot


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def test_uv_lock_hash_uses_exact_file_bytes(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    content = b"version = 1\n"
    lock.write_bytes(content)
    assert uv_lock_sha256(lock) == hashlib.sha256(content).hexdigest()


def test_dirty_source_snapshot_captures_reconstructing_state_and_excludes_ignored_files(
    tmp_path: Path, monkeypatch
) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "README.md")
    _git(
        tmp_path,
        "-c",
        "user.name=RadFusion Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("secret\n", encoding="utf-8")
    (tmp_path / "unrelated.bin").write_bytes(b"not source")
    monkeypatch.chdir(tmp_path)

    snapshot = write_dirty_source_snapshot(tmp_path / "snapshot")
    document = json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))

    assert b"changed" in (snapshot / "tracked.diff").read_bytes()
    assert document["untracked_files"] == ["src/new.py"]
    assert document["untracked_file_sha256"] == {
        "src/new.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()
    }
    assert len(document["source_state_sha256"]) == 64
    assert (snapshot / "untracked" / "src" / "new.py").is_file()
    assert not (snapshot / "untracked" / "ignored.txt").exists()
    assert not (snapshot / "untracked" / "unrelated.bin").exists()
