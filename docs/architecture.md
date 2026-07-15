# Architecture

## Dataset adapter boundary

Each dataset adapter converts source-specific files into stable, typed artifacts. Downstream code
must consume validated bundles rather than repeat source joins or parse raw DICOM metadata.

The RSNA adapter is separated by responsibility:

```text
rsna_source.py          CSV parsing, source joins, DICOM discovery
rsna_dicom.py           selected DICOM metadata and aggregate audit
artifact_validation.py cross-table contracts
rsna_artifacts.py       table generation and immutable bundle publication
rsna_manifest.py        command-line entry point
```

## Layered artifacts

`rsna_samples.parquet` contains one row per prediction sample. It is task-agnostic: outcomes and
RSNA-specific label taxonomies are not sample columns. Its primary key is `sample_id`; `patient_id`
is a leakage-grouping key and is not assumed to be unique across samples.

`rsna_labels.parquet` contains one row per sample and task. It records an observed `pneumonia`
challenge target and an auxiliary `rsna_class` task. The auxiliary task encodes the three-class
RSNA taxonomy without adding dataset-specific columns to the canonical sample schema.

`rsna_annotations.parquet` is a one-to-many table keyed by `sample_id`. Negative pneumonia samples
have no annotation rows.

## Bundle publication

Builds are immutable and content-addressed:

```text
data/manifests/rsna/
  CURRENT
  builds/
    build-<sha256>/
      rsna_samples.parquet
      rsna_labels.parquet
      rsna_annotations.parquet
      rsna_manifest_metadata.json
```

All tables are staged, round-trip checked, and hashed before a build directory is published.
`CURRENT` is replaced atomically only after the complete bundle validates. Bundle loaders resolve
`CURRENT`, require metadata and every declared file, then verify serialized-file and Arrow IPC
hashes. An incomplete or altered bundle is never accepted as current.

## Path resolution

Image paths are normalized POSIX paths relative to an explicit dataset root:

```text
stage_2_train_images/<image_id>.dcm
```

Downstream code receives the dataset root through configuration and resolves paths beneath it.
Absolute paths and parent traversal are invalid.

## Component boundary

The adapter owns source parsing through bundle publication. Downstream code starts from a validated
current bundle and does not depend on RSNA source-file structure.
