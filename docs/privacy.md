# Privacy and data handling

The repository contains code and aggregate metadata, not clinical data. Access and use follow each
dataset's terms.

## Never commit

- DICOM images or other medical images
- Source CSV files containing patient-level rows
- Generated sample or feature manifests
- Laboratory or clinical-note rows
- Model or experiment artifacts that contain patient data
- Access tokens, credentials, or private keys
- Screenshots containing patient-level values

The `.gitignore` is a guard, not the privacy policy. Review staged files before every commit.

## Data use

Keep raw and generated patient-level data in ignored local storage. Do not send patient-level data
to external AI, analytics, artifact-storage, or collaboration services unless the dataset terms and
project data-handling policy explicitly permit it. Public examples and API fixtures must be
synthetic.

Technical reports may contain aggregate counts, distributions, quality findings, and metrics only
when individuals cannot be reconstructed. Raw identifiers and row-level examples are excluded.

Follow each dataset's access and storage requirements. A dataset adapter does not grant permission
to redistribute inputs or derived artifacts.
