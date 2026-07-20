"""Hash source files and Arrow artifacts deterministically."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Hash file bytes with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def arrow_ipc_sha256(table: pa.Table) -> str:
    """Hash canonical ordered Arrow content and its exact schema with SHA-256."""
    canonical = pa.Table.from_arrays(
        [pa.array(table.column(field.name).to_pylist(), type=field.type) for field in table.schema],
        schema=table.schema,
    )
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, canonical.schema) as writer:
        writer.write_table(canonical)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()
