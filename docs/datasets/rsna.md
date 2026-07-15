# RSNA Pneumonia Detection Challenge

## Source and access

The adapter uses the Stage 2 files from the RSNA Pneumonia Detection Challenge. Access is through
the competition host and remains subject to the dataset's terms. The repository does not
redistribute source CSV files or DICOM images.

The optional acquisition dependency group provides the Kaggle CLI so you can download the
competition data after accepting the rules and configuring authentication outside this repository:

```bash
uv run --group acquisition kaggle competitions download \
  -c rsna-pneumonia-detection-challenge \
  -p data/raw/rsna/archive
```

Extract the archive into `data/raw/rsna/extracted/`.

## Verified local layout

```text
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_sample_submission.csv
stage_2_train_images/    26,684 labeled DICOMs
stage_2_test_images/      3,000 unlabeled DICOMs
```

There are 29,684 DICOM files in total. The adapter includes only the 26,684 images with training
labels. Competition test images have no released target and are not assigned invented labels or
splits.

## Labels

`stage_2_train_labels.csv` contains `patientId, x, y, width, height, Target`. It has 30,227 rows for
26,684 images. Repeated positive rows represent multiple boxes. The normalized annotation artifact
contains 9,555 boxes for 6,012 positive samples. The remaining 20,672 samples are negative and have
no annotation rows.

`stage_2_detailed_class_info.csv` provides one consistent class per source identifier:

- `Normal`
- `No Lung Opacity / Not Normal`
- `Lung Opacity`

The class is encoded as the auxiliary `rsna_class` task in the label artifact. `Target=1` must
correspond to `Lung Opacity`; `Target=0` must not.

The challenge target is derived from the public competition labeling process. It is not equivalent
to microbiologically confirmed clinical pneumonia.

## DICOM characteristics

All 26,684 labeled headers were read without decoding pixels. Verified aggregate properties:

- 1024 × 1024 pixels
- `MONOCHROME2`
- one transfer syntax
- AP or PA view
- F or M patient sex
- 18 distinct pixel-spacing pairs

Patient age, sex, projection, and pixel spacing enter the sample artifact. Image dimensions,
photometric interpretation, transfer syntax, bit depth, compression, modality, body part, and UID
consistency are aggregate audit fields. Names, raw DICOM UIDs, dates, referring physician fields,
and implementation metadata are excluded from row-level artifacts.

## Known data-quality findings

RSNA age values use bare integers interpreted as years under a compatibility rule. Standard DICOM
day, week, month, and year forms are also supported. Five observed values exceed 120 years. They
are preserved and counted as implausible; they are not clipped or silently corrected.
