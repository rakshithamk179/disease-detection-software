"""TensorFlow Lite export module with float32 and dynamic-range quantization,

and validation-only consistency auditing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import tensorflow as tf

from src.utils import load_json, save_json, save_markdown


logger = logging.getLogger("tomato_disease_classifier.tflite")


def run_tflite_inference(tflite_bytes: bytes, input_data: np.ndarray) -> np.ndarray:
    """Execute inference with a TFLite flatbuffer."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output


def export_models_to_tflite(
    config: Dict[str, Any],
    keras_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export frozen Keras model to float32 and dynamic-range quantized TFLite models.

    Compares consistency using VALIDATION images only (never locked test images).
    """
    paths = config["resolved_paths"]
    models_dir = paths["models_dir"]
    reports_dir = paths["reports_dir"]
    val_dir = paths["val_dir"]

    model_file = Path(keras_model_path) if keras_model_path else models_dir / "best_model.keras"
    if not model_file.is_file():
        alt = models_dir / "stage1_best_model.keras"
        if alt.is_file():
            model_file = alt
        else:
            raise FileNotFoundError(f"Keras model not found at {model_file}")

    print(f"\n[INFO] Loading Keras model for TFLite export from: {model_file}")
    keras_model = tf.keras.models.load_model(model_file)
    keras_size_bytes = model_file.stat().st_size

    # 1. Convert to Float32 TFLite
    print("[INFO] Converting to Float32 TFLite...")
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_f32_content = converter_f32.convert()
    f32_path = models_dir / "tomato_disease_mobilenetv3_float32.tflite"
    with open(f32_path, "wb") as f:
        f.write(tflite_f32_content)
    f32_size_bytes = len(tflite_f32_content)

    # 2. Convert to Dynamic-Range Quantized TFLite (Post-Training Quantization without representative calibration)
    print("[INFO] Converting to Dynamic-Range Quantized TFLite...")
    converter_dr = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter_dr.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dr_content = converter_dr.convert()
    dr_path = models_dir / "tomato_disease_mobilenetv3_dynamic_range.tflite"
    with open(dr_path, "wb") as f:
        f.write(tflite_dr_content)
    dr_size_bytes = len(tflite_dr_content)

    # 3. Consistency check using VALIDATION samples only (strict test isolation)
    print("[INFO] Checking numerical consistency using VALIDATION images...")
    consistency_results: List[Dict[str, Any]] = []

    # Find up to 5 validation images across classes
    val_samples: List[Path] = []
    if val_dir.is_dir():
        for c_dir in sorted(val_dir.iterdir()):
            if c_dir.is_dir():
                for f in sorted(c_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        val_samples.append(f)
                        if len(val_samples) >= 5:
                            break
            if len(val_samples) >= 5:
                break

    img_size = (config["image"]["width"], config["image"]["height"])
    for s_path in val_samples:
        with Image.open(s_path) as p_img:
            resized = p_img.convert("RGB").resize(img_size, Image.Resampling.BILINEAR)
            inp = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)

        keras_pred = keras_model(inp, training=False).numpy()[0]
        f32_pred = run_tflite_inference(tflite_f32_content, inp)[0]
        dr_pred = run_tflite_inference(tflite_dr_content, inp)[0]

        max_diff_f32 = float(np.max(np.abs(keras_pred - f32_pred)))
        max_diff_dr = float(np.max(np.abs(keras_pred - dr_pred)))

        consistency_results.append({
            "sample_image": s_path.name,
            "keras_top_class": int(np.argmax(keras_pred)),
            "f32_top_class": int(np.argmax(f32_pred)),
            "dr_top_class": int(np.argmax(dr_pred)),
            "classes_match": bool(np.argmax(keras_pred) == np.argmax(f32_pred) == np.argmax(dr_pred)),
            "max_abs_diff_float32": round(max_diff_f32, 6),
            "max_abs_diff_dynamic_range": round(max_diff_dr, 6),
        })

    report_data = {
        "source_keras_model": str(model_file),
        "keras_size_bytes": keras_size_bytes,
        "keras_size_mb": round(keras_size_bytes / (1024 * 1024), 2),
        "float32_tflite": {
            "path": str(f32_path),
            "size_bytes": f32_size_bytes,
            "size_mb": round(f32_size_bytes / (1024 * 1024), 2),
            "compression_ratio": round(keras_size_bytes / f32_size_bytes, 2) if f32_size_bytes else 1.0,
        },
        "dynamic_range_tflite": {
            "path": str(dr_path),
            "size_bytes": dr_size_bytes,
            "size_mb": round(dr_size_bytes / (1024 * 1024), 2),
            "compression_percentage": f"{round((1.0 - dr_size_bytes / keras_size_bytes) * 100, 1)}%",
            "compression_ratio": round(keras_size_bytes / dr_size_bytes, 2) if dr_size_bytes else 1.0,
        },
        "validation_consistency_samples": consistency_results,
        "preprocessing_contract": {
            "input_resolution": f"{img_size[0]}x{img_size[1]}x3",
            "color_format": "RGB",
            "normalization": "MobileNetV3 standard (handled inside model graph or [0, 255] float32 input)",
        },
        "raspberry_pi_deployment_notes": (
            "Deploy 'tomato_disease_mobilenetv3_dynamic_range.tflite' with tflite_runtime on Raspberry Pi OS. "
            "Input images should be captured via OpenCV/picamera, resized to 224x224, converted to RGB float32, and passed to interpreter."
        ),
    }

    save_json(report_data, reports_dir / "tflite_export_report.json")
    md_report = generate_markdown_tflite_report(report_data)
    save_markdown(md_report, reports_dir / "tflite_export_report.md")

    print("\n" + "=" * 80)
    print("TFLITE EXPORT COMPLETED:")
    print(f"- Keras Model Size: {report_data['keras_size_mb']} MB")
    print(f"- Float32 TFLite Size: {report_data['float32_tflite']['size_mb']} MB")
    print(f"- Dynamic-Range Quantized Size: {report_data['dynamic_range_tflite']['size_mb']} MB ({report_data['dynamic_range_tflite']['compression_percentage']} compression)")
    print(f"Report saved: {reports_dir / 'tflite_export_report.md'}")
    print("=" * 80 + "\n")

    return report_data


def generate_markdown_tflite_report(data: Dict[str, Any]) -> str:
    """Generate Markdown report for TFLite export."""
    f32 = data["float32_tflite"]
    dr = data["dynamic_range_tflite"]

    md = [
        "# TensorFlow Lite Model Export & Edge Deployment Report",
        "",
        "## Model Size & Compression Summary",
        f"- **Original Keras Model**: `{data['keras_size_mb']} MB` ({data['keras_size_bytes']:,} bytes)",
        f"- **Float32 TFLite**: `{f32['size_mb']} MB` (Compression ratio: `{f32['compression_ratio']}x`)",
        f"- **Dynamic-Range Quantized TFLite**: `{dr['size_mb']} MB` (Compression ratio: `{dr['compression_ratio']}x`, Space saved: `{dr['compression_percentage']}`)",
        "",
        "## Validation Consistency Verification",
        "> [!NOTE] Consistency checks were performed strictly on VALIDATION split images. Locked test images were NOT used.",
        "",
        "| Sample File | Keras Top Class | F32 Top Class | DR Top Class | Match | Max F32 Abs Diff | Max DR Abs Diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for s in data["validation_consistency_samples"]:
        md.append(
            f"| `{s['sample_image']}` | {s['keras_top_class']} | {s['f32_top_class']} | {s['dr_top_class']} | "
            f"{'✅' if s['classes_match'] else '❌'} | {s['max_abs_diff_float32']:.6f} | {s['max_abs_diff_dynamic_range']:.6f} |"
        )

    md.extend([
        "",
        "## Preprocessing Contract for Edge Deployment",
        f"- **Input Dimensions**: `{data['preprocessing_contract']['input_resolution']}`",
        f"- **Color Space**: `{data['preprocessing_contract']['color_format']}`",
        f"- **Scaling**: `{data['preprocessing_contract']['normalization']}`",
        "",
        "## Raspberry Pi Deployment Guide",
        f"{data['raspberry_pi_deployment_notes']}",
        "",
        "```python",
        "import numpy as np",
        "from tflite_runtime.interpreter import Interpreter",
        "from PIL import Image",
        "",
        "# Load TFLite dynamic range model",
        "interpreter = Interpreter(model_path='models/tomato_disease_mobilenetv3_dynamic_range.tflite')",
        "interpreter.allocate_tensors()",
        "input_index = interpreter.get_input_details()[0]['index']",
        "output_index = interpreter.get_output_details()[0]['index']",
        "",
        "# Prepare image",
        "img = Image.open('leaf.jpg').convert('RGB').resize((224, 224))",
        "input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)",
        "interpreter.set_tensor(input_index, input_data)",
        "interpreter.invoke()",
        "probs = interpreter.get_tensor(output_index)[0]",
        "```",
        "",
    ])

    return "\n".join(md)
