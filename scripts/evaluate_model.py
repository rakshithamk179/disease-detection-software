#!/usr/bin/env python3
"""Script to evaluate model performance on validation data or locked test data."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.evaluate import evaluate_model_pipeline
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model on validation data (development) or locked test data (final)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["val", "test_locked", "test"],
        help="Dataset split to evaluate ('val' or 'test_locked').",
    )
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Mandatory confirmation flag required to evaluate the locked test set.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to model .keras file (default: models/best_model.keras).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["resolved_paths"]
    logger = setup_logger(log_dir=paths["logs_dir"])

    split_target = "test_locked" if args.split in ["test", "test_locked"] else "val"

    try:
        evaluate_model_pipeline(
            config=config,
            split=split_target,
            final_evaluation=args.final_evaluation,
            model_path=Path(args.model_path) if args.model_path else None,
        )
    except PermissionError as pe:
        print(f"\n[EVALUATION PERMISSION ERROR]: {str(pe)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[EVALUATION ERROR]: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
