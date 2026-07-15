# Reproducibility

## Environment

The project targets Python 3.13. Dependencies are resolved by uv and recorded in `uv.lock`.

```bash
uv sync --locked
```

Install local Git hooks once:

```bash
uv run pre-commit install
```

## Rebuild the RSNA artifacts

Place the extracted Stage 2 data under `data/raw/rsna/extracted/`, then run:

```bash
make rsna-manifest
```

Custom locations are supported:

```bash
uv run python -m radfusion.data.rsna_manifest \
  --dataset-root /approved/local/rsna/extracted \
  --output-directory /approved/local/manifests
```

Paths inside the sample artifact remain relative to the dataset root. A custom absolute input path
is never written into a sample row.

The build command publishes an immutable directory and atomically updates
`data/manifests/rsna/CURRENT`. Bundle loaders should resolve the current build through the
validation API rather than selecting a build directory directly.

## Deterministic behavior

The adapter validates all source identifiers and orders samples and annotations by stable IDs.
Explicit Arrow schemas prevent inference drift. Each Parquet file is read back and compared before
atomic replacement.

Aggregate metadata records:

- SHA-256 hashes of the two source CSV files
- `arrow_ipc_sha256` values for samples, labels, and annotations
- serialized Parquet file hashes
- schema, dataset, and task-label versions
- package versions and UTC generation time

Repeated builds from identical inputs and the recorded PyArrow version produce the same bundle ID
and `arrow_ipc_sha256` values. Cross-version Arrow IPC stability is not claimed. The metadata
generation timestamp is excluded from bundle identity.

## Quality checks

```bash
make check
uv run pre-commit run --all-files
```

The ordinary test suite uses synthetic headers and does not require RSNA data. A marked local
integration test is available with:

```bash
uv run pytest -m integration
```

## Reproducibility boundary

Code and locked dependencies are version-controlled artifacts. Rebuilding the manifests also
requires separately acquired RSNA source data under its original terms. The repository does not
provide or download credentials and cannot reproduce data that the user is not authorized to
access.
