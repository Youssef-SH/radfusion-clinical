"""Run one experiment from a validated YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from radfusion.training.config import ConfigError, load_experiment_config
from radfusion.training.registry import RegistryError, register_builtin_components
from radfusion.training.train_tabular import train_configured_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load one config, execute it, and print aggregate run lineage."""
    args = _parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        register_builtin_components()
        result = train_configured_experiment(config)
    except (ConfigError, RegistryError, OSError, ValueError, KeyError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "experiment": config.name,
                "config": config.source_path.as_posix(),
                "model_name": result.model_name,
                "mlflow_run_id": result.run_id,
                "test_average_precision": result.test_youden_j.probability.average_precision,
                "model_path": result.model_path.as_posix(),
                "artifact_directory": result.artifact_directory.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
