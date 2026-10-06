#!/usr/bin/env python3
"""Script to run inference on a single tomato leaf image."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.inference import predict_single_image
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disease class for a single tomato leaf image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input tomato leaf image file.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated visualization with prediction banner to reports/figures/.",
    )
    parser.add_argument(
        "--output-annotated",
        type=str,
        default=None,
        help="Custom destination path for annotated image.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["resolved_paths"]
    logger = setup_logger(log_dir=paths["logs_dir"])

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[ERROR] Specified image does not exist: {image_path}")
        sys.exit(1)

    try:
        result = predict_single_image(
            image_path=image_path,
            config=config,
            save_annotated=args.save_annotated,
            output_annotated_path=Path(args.output_annotated) if args.output_annotated else None,
        )

        print("=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)
        print(f"Image File:          {result['image_path']}")
        print(f"Predicted Class:     {result['predicted_class']}")
        print(f"Confidence:          {result['confidence_percentage']}")
        print(f"Inference Latency:   {result['inference_time_ms']} ms")
        print("\nClass Probabilities:")
        for c, p in result["class_probabilities"].items():
            print(f"  - {c}: {p * 100:.2f}%")

        if result["low_confidence_warning"]:
            print(f"\n[WARNING] {result['low_confidence_warning']}")

        print(f"\n[SCIENTIFIC NOTE] {result['scientific_integrity_note']}")

        if "annotated_image_path" in result:
            print(f"\nAnnotated visualization saved: {result['annotated_image_path']}")

        print("\nFull JSON Response:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n[INFERENCE ERROR]: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
