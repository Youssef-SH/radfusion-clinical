"""Publish generated directories atomically with rollback."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def staging_directory(destination: str | Path) -> Path:
    """Create a sibling staging directory for a generated destination."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}-staging-", dir=target.parent))


def publish_directory(stage: str | Path, destination: str | Path) -> None:
    """Publish a complete staged directory, restoring the previous version on failure."""
    staged = Path(stage)
    target = Path(destination)
    if not staged.is_dir():
        raise ValueError(f"Staged publication directory does not exist: {staged}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent))
    backup.rmdir()
    previous_moved = False
    try:
        if target.exists():
            os.replace(target, backup)
            previous_moved = True
        try:
            os.replace(staged, target)
        except BaseException:
            if previous_moved:
                os.replace(backup, target)
                previous_moved = False
            raise
        if previous_moved:
            shutil.rmtree(backup)
            previous_moved = False
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if backup.exists():
            if previous_moved and not target.exists():
                os.replace(backup, target)
            else:
                shutil.rmtree(backup)
