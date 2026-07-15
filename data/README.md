# Local data workspace

This directory holds local source data and generated patient-level artifacts. Its contents are
ignored except for this file and directory placeholders.

Expected RSNA layout:

```text
data/
  raw/
    rsna/
      archive/
        rsna-pneumonia-detection-challenge.zip
      extracted/
        stage_2_train_labels.csv
        stage_2_detailed_class_info.csv
        stage_2_sample_submission.csv
        stage_2_train_images/
        stage_2_test_images/
  interim/
  processed/
  manifests/
    rsna/
      CURRENT
      builds/
        build-<sha256>/
          rsna_samples.parquet
          rsna_labels.parquet
          rsna_annotations.parquet
          rsna_manifest_metadata.json
```

After accepting the competition terms and configuring the Kaggle CLI outside the repository, the
archive can be downloaded with the command documented in `docs/datasets/rsna.md`. Extract it into
the layout above and run:

```bash
make rsna-manifest
```

Never force-add raw images, source CSVs, generated manifests, or credentials to Git. These files
contain patient-level information even when public identifiers are deidentified.
