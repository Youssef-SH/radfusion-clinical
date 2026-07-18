# Architecture

## Terminology

- **Dataset:** an external source collection with a stable logical identity.
- **Build:** one execution of bundle construction. A successful build publishes one bundle.
- **Bundle:** an immutable, validated set of typed artifacts stored under `build-<sha256>`.
- **Manifest:** `rsna_manifest_metadata.json`, the bundle document that declares identity,
  contents, hashes, policies, and provenance. The `rsna-manifest` command runs a build.

The lifecycle is: source dataset → build execution → immutable bundle → validated consumers.
`CURRENT` points to the active bundle without changing any published bundle.

## Component boundaries

| Component | Responsibility |
| --- | --- |
| Source adapter | Parse source tables, discover DICOM files, and extract selected headers |
| Artifact builder | Normalize typed records and construct patient-level assignments |
| Validator | Enforce schemas, relationships, identities, paths, and ordering |
| Bundle publisher | Publish immutable bundles and update `CURRENT` atomically |
| Audit generator | Produce aggregate dataset reports from a validated bundle |
| Dataset registry | Resolve a configured bundle adapter |
| Model registry | Resolve a configured model adapter |
| Experiment runner | Coordinate fitting, evaluation, publication, and tracking |
| Evaluation utilities | Compute probabilities, metrics, thresholds, latency, and plots |

Dataset adapters isolate source-specific behavior. Training consumes validated bundles and never
parses raw dataset files. Model adapters own estimator construction and fitting. One runner serves
all registered experiment types.

## Data flow

```text
RSNA source files
    → dataset adapter
    → validated tables and manifest
    → immutable bundle
    → audit or experiment runner
    → aggregate reports, immutable models, and MLflow runs
```

Artifact schemas are defined in [`data_contract.md`](data_contract.md). Experiment composition is
defined in [`training.md`](training.md). Reconstruction and evaluation protocols are defined in
[`reproducibility.md`](reproducibility.md).
