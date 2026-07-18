# Data contract

Manifest schema version `0.1.0` defines the current development contract. Version changes require
an explicit contract decision. The bundle ID and declared hashes identify exact content.

Each bundle contains five Parquet artifacts and one JSON manifest. Column order, Arrow physical
types, nullability, and deterministic row order are contractual.

## Canonical samples

File: `rsna_samples.parquet`. Cardinality: one row per labeled RSNA prediction sample.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Dataset-qualified key, `rsna:<image_id>` |
| `patient_id` | `string` | no | Patient grouping key |
| `study_id` | `string` | yes | Study grouping key; null for RSNA |
| `image_id` | `string` | no | Source image identifier and filename stem |
| `image_path` | `string` | no | Dataset-root-relative POSIX path |
| `split` | `string` | yes | Null for RSNA; assignments use the split artifact |
| `age_years` | `float64` | yes | Parsed DICOM age in years |
| `sex` | `string` | yes | `F` or `M` |
| `view_position` | `string` | yes | `AP` or `PA` |
| `pixel_spacing_row_mm` | `float64` | yes | Positive DICOM row spacing |
| `pixel_spacing_col_mm` | `float64` | yes | Positive DICOM column spacing |

`sample_id` and `image_id` are unique. `image_path` has the form
`stage_2_train_images/<image_id>.dcm` and resolves beneath the configured raw dataset root.

## Task labels

File: `rsna_labels.parquet`. Cardinality: one row per sample and task.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Foreign key to samples |
| `task_id` | `string` | no | `pneumonia` or `rsna_class` |
| `label_value` | `int8` | no | Task-specific value |
| `label_status` | `string` | no | `observed` |
| `label_source` | `string` | no | Stable source identifier |
| `label_policy_version` | `string` | no | Task-label policy identity |

The `pneumonia` task is the binary RSNA challenge target. The auxiliary `rsna_class` task encodes
`Normal` as 0, `No Lung Opacity / Not Normal` as 1, and `Lung Opacity` as 2. Every sample has both
tasks. A positive binary target pairs with class 2; a negative target pairs with class 0 or 1.

## Localization annotations

File: `rsna_annotations.parquet`. Cardinality: one row per positive bounding box.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Foreign key to samples |
| `annotation_id` | `string` | no | Deterministic box key |
| `x` | `float64` | no | Left coordinate in source pixels |
| `y` | `float64` | no | Top coordinate in source pixels |
| `width` | `float64` | no | Positive width in source pixels |
| `height` | `float64` | no | Positive height in source pixels |

Coordinates are finite, nonnegative at the origin, positive in extent, and unique within a sample.
Each positive sample has at least one box; negative samples have none.

During bundle construction, every source image must exist and expose valid dimensions, and every
box must fit within those dimensions. Portable bundle validation enforces the stored geometry and
cross-table invariants without requiring access to the source DICOM files.

## Patient-level splits

File: `rsna_splits.parquet`. Cardinality: one row per sample.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Foreign key to samples |
| `split_name` | `string` | no | `train`, `validation`, or `test` |
| `split_recipe_id` | `string` | no | Identity of the complete assignment policy |
| `cohort_fingerprint` | `string` | no | Identity of canonical patient membership and targets |
| `split_assignment_id` | `string` | no | Identity of recipe, cohort, and assignments |
| `split_source` | `string` | no | Origin of the assignment |

All rows carry the same three identities so the table remains self-describing. The manifest
declares the same identities and the full recipe.

The cohort fingerprint includes dataset ID, exact sample and label schemas, ordered patient and
sample membership, and the patient-consistent binary target. The assignment ID includes the recipe
ID, cohort fingerprint, and canonical sample-to-split mapping.

Validation recomputes all identities and assignments. It also requires complete coverage, sorted
unique sample keys, valid split names, one split per patient, and both classes in every split when
the cohort permits.

## Source DICOM inventory

File: `rsna_source_inventory.parquet`. Cardinality: one row per sample.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Foreign key to samples |
| `relative_path` | `string` | no | Portable source path beneath the dataset root |
| `byte_size` | `int64` | no | Source file size in bytes |
| `sha256` | `string` | no | SHA-256 of exact source DICOM bytes |

Rows are sorted by `sample_id`. Sample IDs and relative paths are unique, paths match the sample
artifact, byte sizes are positive, and hashes are lowercase 64-character hexadecimal values.
This protected patient-level artifact is excluded from public reports.

## Manifest

File: `rsna_manifest_metadata.json`.

The manifest declares the schema, dataset and label policies, aggregate counts, split recipe and
identities, DICOM audit summaries, source hashes, generated-artifact hashes, bundle identity, and
runtime provenance. Each source CSV and DICOM is authenticated by SHA-256.

## Bundle acceptance and identity

A loader accepts a bundle only when:

1. the directory contains exactly the manifest and five required Parquet artifacts;
2. every direct entry is a regular file stored physically in the bundle directory;
3. the manifest schema is supported;
4. the manifest and directory name declare the requested bundle ID;
5. every required artifact has its exact Arrow schema;
6. every declared Parquet SHA-256 and Arrow IPC SHA-256 matches;
7. the recomputed bundle ID matches;
8. cross-table, split-identity, and source-inventory invariants pass.

Unexpected entries, directories, symbolic links, and special filesystem objects are rejected.

The bundle ID covers ordered Arrow hashes, serialized Parquet hashes, source inputs, and semantic
build policy. Timestamps, commands, tool versions, and publication metadata are provenance fields.

Samples, splits, and source inventory are ordered by `sample_id`; labels by
`(sample_id, task_id)`; annotations by `annotation_id`. Arrow IPC identity includes the exact
schema and is reproducible within the recorded PyArrow version.

Published bundle directories are immutable. `CURRENT` is replaced atomically after complete
validation.
