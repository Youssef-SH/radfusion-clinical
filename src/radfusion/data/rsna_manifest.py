"""Provide the command-line entry point for RSNA artifact bundle generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from radfusion.data.rsna_artifacts import build_rsna_artifacts, write_bundle
from radfusion.data.rsna_source import ManifestBuildError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/rsna/extracted"),
        help="Directory containing the extracted RSNA Stage 2 files",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/manifests"),
        help="Root directory for versioned RSNA artifact bundles",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_rsna_artifacts(args.dataset_root)
        written = write_bundle(result, args.output_directory)
    except (ManifestBuildError, OSError) as exc:
        print(f"RSNA manifest build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "bundle_id": written.paths.bundle_id,
                "bundle_directory": str(written.paths.bundle_directory),
                "current_marker": str(written.paths.current_path),
                "sample_count": result.samples.num_rows,
                "label_count": result.labels.num_rows,
                "annotation_count": result.annotations.num_rows,
                "arrow_ipc_sha256": dict(written.arrow_ipc_sha256),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
