# Privacy and data handling

The version-controlled repository contains code, documentation, configuration, and synthetic
fixtures. Raw and generated patient-level data remains in ignored local storage under the terms of
its source dataset.

## Version-control boundary

Keep these materials outside version control:

- DICOM images and other medical images;
- source tables containing patient-level rows;
- generated bundles and source inventories;
- model and experiment artifacts;
- clinical-note or laboratory rows;
- credentials, access tokens, and private keys;
- screenshots containing patient-level values.

Review staged files against this policy before each commit. `.gitignore` provides baseline
filtering.

## Reports

Public reports contain aggregate counts, distributions, quality findings, and model metrics.
Publication checks reject source patient IDs, sample IDs, image names and paths, UUID-shaped
identifiers, and DICOM UID-shaped values. Model evaluation reports are also logged to MLflow after
privacy validation.

Public examples and test fixtures use synthetic data. External transfer of raw or derived data
requires the dataset terms and project data-handling policy to permit the destination and use.
