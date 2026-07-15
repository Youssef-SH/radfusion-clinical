# Data contract

Schema version `1.0.0` defines four bundle files. Column order and Arrow physical types are part
of the contract.

## Canonical samples

File: `rsna_samples.parquet`

Cardinality: one row per labeled RSNA prediction sample.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | Stable dataset identifier; `rsna` |
| `sample_id` | `string` | no | Dataset-qualified primary key, `rsna:<image_id>` |
| `patient_id` | `string` | no | Patient grouping key; not universally unique |
| `study_id` | `string` | yes | Null for RSNA; reserved for real study grouping |
| `image_id` | `string` | no | RSNA image identifier and filename stem |
| `image_path` | `string` | no | Dataset-root-relative POSIX path |
| `split` | `string` | yes | Null, `train`, `validation`, or `test` |
| `age_years` | `float64` | yes | Parsed DICOM age in years |
| `sex` | `string` | yes | `F` or `M` |
| `view_position` | `string` | yes | `AP` or `PA` |
| `pixel_spacing_row_mm` | `float64` | yes | Positive row spacing from DICOM `PixelSpacing` |
| `pixel_spacing_col_mm` | `float64` | yes | Positive column spacing from DICOM `PixelSpacing` |

Samples contain no task labels. `sample_id` and `image_id` are unique in this artifact;
`patient_id` intentionally has no uniqueness constraint.

## Task labels

File: `rsna_labels.parquet`

Cardinality: exactly one row for each sample/task pair.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Foreign key to canonical samples |
| `task_id` | `string` | no | `pneumonia` or `rsna_class` |
| `label_value` | `int8` | no | Task-specific encoded value |
| `label_status` | `string` | no | `observed` |
| `label_source` | `string` | no | Stable source identifier |
| `label_policy_version` | `string` | no | Versioned task-label policy |

The `pneumonia` task uses values 0 and 1 from the RSNA Stage 2 challenge target. The auxiliary
`rsna_class` task uses:

```text
0  Normal
1  No Lung Opacity / Not Normal
2  Lung Opacity
```

Every sample has exactly one row for each task. `pneumonia=1` is compatible only with
`rsna_class=2`; `pneumonia=0` cannot have `rsna_class=2`.

## Localization annotations

File: `rsna_annotations.parquet`

Cardinality: one row per pneumonia bounding box; negative pneumonia samples have no rows.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `dataset_id` | `string` | no | `rsna` |
| `sample_id` | `string` | no | Foreign key to canonical samples |
| `annotation_id` | `string` | no | Deterministic box identifier |
| `x` | `float64` | no | Left coordinate in source pixels |
| `y` | `float64` | no | Top coordinate in source pixels |
| `width` | `float64` | no | Box width in source pixels |
| `height` | `float64` | no | Box height in source pixels |

Coordinates must be finite, nonnegative at the origin, strictly positive in extent, unique within
a sample, and bounded by source dimensions. Every positive pneumonia sample has at least one box.

## Aggregate metadata

`rsna_manifest_metadata.json` records schema and policy versions, aggregate counts and DICOM
coverage, age parsing, source hashes, artifact hashes, bundle identity, and tool versions. It
contains no patient identifiers or row-level values.

## Bundle acceptance

`data/manifests/rsna/CURRENT` contains one build directory name. A consumer accepts that build only
when:

1. the metadata JSON exists and declares the same bundle ID;
2. all three Parquet files exist;
3. each serialized file SHA-256 matches metadata;
4. each exact Arrow schema matches;
5. each `arrow_ipc_sha256` matches.

The build directory is immutable. `CURRENT` is atomically replaced only after validation.

## Determinism and hashing

Samples are sorted by `sample_id`, labels by `(sample_id, task_id)`, and annotations by
`annotation_id`. Source CSV hashes cover file bytes. `arrow_ipc_sha256` is SHA-256 over an ordered
Arrow IPC stream including schema. Its metadata records the PyArrow version; stability across
PyArrow versions is not claimed. Parquet file hashes cover serialized bytes.
