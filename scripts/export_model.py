#!/usr/bin/env python3
"""Script to export trained Keras model to TensorFlow Lite formats."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.export_tflite import export_models_to_tflite
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained model to float32 and dynamic-range quantized TFLite models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to trained .keras model (default: models/best_model.keras).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["resolved_paths"]
    logger = setup_logger(log_dir=paths["logs_dir"])

    print("=" * 80)
    print("STARTING TENSORFLOW LITE MODEL EXPORT")
    print("=" * 80)

    try:
        report = export_models_to_tflite(
            config=config,
            keras_model_path=Path(args.model_path) if args.model_path else None,
        )
        print(f"[SUCCESS] TFLite models successfully exported to {paths['models_dir']}/")
        print(f"Report available at: {paths['reports_dir'] / 'tflite_export_report.md'}")
    except Exception as e:
        print(f"\n[EXPORT ERROR]: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
