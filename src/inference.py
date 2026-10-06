"""Inference pipeline for single image prediction with latency tracking and annotation."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from src.utils import load_json, save_json


logger = logging.getLogger("tomato_disease_classifier.inference")

MANDATORY_SEVERITY_DISCLAIMER = (
    "Confidence indicates classifier certainty for this image; "
    "it does not measure infected leaf area or disease severity."
)


def predict_single_image(
    image_path: Path,
    config: Dict[str, Any],
    model: Optional[tf.keras.Model] = None,
    class_names: Optional[List[str]] = None,
    save_annotated: bool = False,
    output_annotated_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run inference on an unseen tomato leaf image.

    Applies identical preprocessing to training without data augmentation.
    """
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    paths = config["resolved_paths"]
    models_dir = paths["models_dir"]
    logs_dir = paths["logs_dir"]
    reports_dir = paths["reports_dir"]

    # 1. Load class names
    if class_names is None:
        class_names_path = models_dir / "class_names.json"
        if not class_names_path.is_file():
            raise FileNotFoundError(f"Class names file not found at: {class_names_path}")
        class_names = load_json(class_names_path)

    # 2. Load model
    model_source_path = models_dir / "best_model.keras"
    if model is None:
        if not model_source_path.is_file():
            # Fallback to stage 1 model if available
            alt_path = models_dir / "stage1_best_model.keras"
            if alt_path.is_file():
                model_source_path = alt_path
            else:
                raise FileNotFoundError(f"No trained model found at {model_source_path} or {alt_path}")
        model = tf.keras.models.load_model(model_source_path)

    output_dim = model.output_shape[-1]
    if output_dim != len(class_names):
        raise ValueError(
            f"Model output dimension ({output_dim}) does not match class_names list length ({len(class_names)})."
        )

    # 3. Preprocess image
    i_cfg = config.get("image", {})
    target_size = (i_cfg.get("width", 224), i_cfg.get("height", 224))

    # Read image using PIL to ensure RGB mode
    with Image.open(image_path) as pil_img:
        pil_rgb = pil_img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(pil_rgb, dtype=np.float32)
        batch_input = np.expand_dims(img_array, axis=0)

    # 4. Measure inference latency
    t0 = time.perf_counter()
    raw_probs = model(batch_input, training=False).numpy()[0]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    pred_idx = int(np.argmax(raw_probs))
    pred_class = class_names[pred_idx]
    confidence = float(raw_probs[pred_idx])

    probs_dict = {class_names[i]: round(float(raw_probs[i]), 4) for i in range(len(class_names))}

    # 5. Low confidence check
    low_conf_threshold = config.get("model", {}).get("low_confidence_threshold", 0.50)
    low_confidence_warning = None
    if confidence < low_conf_threshold:
        low_confidence_warning = (
            f"LOW CONFIDENCE: Classifier certainty ({confidence * 100:.1f}%) is below the {low_conf_threshold * 100:.0f}% threshold. "
            "The image may be ambiguous, poorly illuminated, non-tomato foliage, or out-of-distribution."
        )

    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "class_probabilities": probs_dict,
        "inference_time_ms": latency_ms,
        "model_path": str(model_source_path),
        "low_confidence_warning": low_confidence_warning,
        "scientific_integrity_note": MANDATORY_SEVERITY_DISCLAIMER,
    }

    # Save prediction log
    predictions_log_dir = logs_dir / "predictions"
    predictions_log_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = predictions_log_dir / f"pred_{image_path.stem}_{ts_slug}.json"
    save_json(result, log_file)

    # 6. Save annotated image if requested
    if save_annotated:
        annotated_dest = output_annotated_path or (reports_dir / "figures" / f"annotated_{image_path.name}")
        annotated_dest.parent.mkdir(parents=True, exist_ok=True)
        annotate_image(
            src_image_path=image_path,
            dest_image_path=annotated_dest,
            predicted_class=pred_class,
            confidence=confidence,
            has_warning=bool(low_confidence_warning),
        )
        result["annotated_image_path"] = str(annotated_dest)

    return result


def annotate_image(
    src_image_path: Path,
    dest_image_path: Path,
    predicted_class: str,
    confidence: float,
    has_warning: bool = False,
) -> None:
    """Create an annotated visualization with prediction banner and warnings."""
    img_bgr = cv2.imread(str(src_image_path))
    if img_bgr is None:
        # Fallback if cv2 fails to read
        with Image.open(src_image_path) as p_img:
            img_bgr = cv2.cvtColor(np.array(p_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]

    # Banner background
    banner_h = max(60, int(h * 0.14))
    overlay = img_bgr.copy()
    color = (0, 140, 255) if has_warning else (40, 160, 40)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)

    # Text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.0, w / 600.0))
    thickness = 2

    label_text = f"Pred: {predicted_class} ({confidence * 100:.1f}%)"
    cv2.putText(img_bgr, label_text, (15, int(banner_h * 0.45)), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    sub_text = "WARNING: Low confidence (<50%)" if has_warning else "Classifier Certainty (Not Disease Severity)"
    sub_color = (0, 200, 255) if has_warning else (180, 220, 180)
    cv2.putText(img_bgr, sub_text, (15, int(banner_h * 0.85)), font, scale * 0.7, sub_color, 1, cv2.LINE_AA)

    cv2.imwrite(str(dest_image_path), img_bgr)
