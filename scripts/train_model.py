#!/usr/bin/env python3
"""Script to execute two-stage transfer learning using strictly train and validation data."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.train import execute_training
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train tomato leaf disease classifier using train and validation sets only."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "stage1"],
        help="Training stage to execute ('all' for both stages, 'stage1' for head-only).",
    )
    parser.add_argument(
        "--skip-fine-tune",
        action="store_true",
        help="Skip Stage 2 fine-tuning and use Stage 1 best model as production model.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="Number of epochs for Stage 1 head training.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Number of epochs for Stage 2 fine-tuning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing Stage 1 model weights if available.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["resolved_paths"]
    logger = setup_logger(log_dir=paths["logs_dir"])

    print("=" * 80)
    print("STARTING MODEL TRAINING")
    print(f"Stage mode:       {args.stage}")
    print(f"Skip fine-tune:   {args.skip_fine_tune}")
    print(f"Stage 1 epochs:   {args.stage1_epochs or config.get('training', {}).get('stage1_epochs', 30)}")
    print(f"Fine-tune epochs: {args.fine_tune_epochs or config.get('training', {}).get('fine_tune_epochs', 15)}")
    print(f"Batch size:       {args.batch_size or config.get('training', {}).get('batch_size', 16)}")
    print(f"Resume:           {args.resume}")
    print("=" * 80)

    try:
        execute_training(
            config=config,
            stage=args.stage,
            skip_fine_tune=args.skip_fine_tune,
            stage1_epochs=args.stage1_epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    except Exception as e:
        print(f"\n[TRAINING ERROR]: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
