"""Evaluation module for validation assessment and strictly locked final test evaluation."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset_loader import load_locked_test_dataset, load_validation_dataset
from src.reporting import compute_comprehensive_metrics, plot_confusion_matrices
from src.split_dataset import verify_locked_test_set
from src.utils import load_json, save_json, save_markdown


logger = logging.getLogger("tomato_disease_classifier.evaluate")


def evaluate_model_pipeline(
    config: Dict[str, Any],
    split: str = "val",
    final_evaluation: bool = False,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate trained model on either the validation split or the locked test split.

    Strictly requires --final-evaluation flag for test_locked split.
    """
    paths = config["resolved_paths"]
    reports_dir = paths["reports_dir"]
    models_dir = paths["models_dir"]
    train_dir = paths["train_dir"]
    val_dir = paths["val_dir"]
    test_dir = paths["locked_test_dir"]
    test_lock_manifest_path = paths["test_lock_manifest"]

    model_target_path = Path(model_path) if model_path else models_dir / "best_model.keras"
    if not model_target_path.is_file():
        raise FileNotFoundError(f"Model weights not found at: {model_target_path}")

    class_names_path = models_dir / "class_names.json"
    if not class_names_path.is_file():
        raise FileNotFoundError(f"Class names file not found at: {class_names_path}")
    class_names = load_json(class_names_path)

    # Load model
    print(f"\n[INFO] Loading model from: {model_target_path}")
    model = tf.keras.models.load_model(model_target_path)
    output_classes = model.output_shape[-1]
    if output_classes != len(class_names):
        raise ValueError(f"Model output classes ({output_classes}) does not match class_names ({len(class_names)}).")

    img_size = (config["image"]["width"], config["image"]["height"])
    batch_size = config["training"].get("batch_size", 16)

    # =========================================================================
    # LOCKED TEST SET EVALUATION
    # =========================================================================
    if split in ["test", "test_locked"]:
        if not final_evaluation:
            raise PermissionError(
                "\n" + "!" * 80 + "\n"
                "ACCESS DENIED: Evaluation on the locked test set requires the '--final-evaluation' flag.\n"
                "The test set is reserved exclusively for final evaluation of the frozen, selected model.\n"
                "Do not evaluate the test set during hyperparameter tuning or model development.\n"
                + "!" * 80
            )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION WARNING:")
        print("You are evaluating the frozen selected model on the locked test set.")
        print("Do not use these results to continue tuning the current model.")
        print("If you change architecture, data preprocessing, augmentation, thresholds,")
        print("or hyperparameters after seeing these results, create a fresh locked test split")
        print("for the next final evaluation.")
        print("=" * 80 + "\n")

        # Verify test lock integrity before proceeding
        is_lock_valid, lock_data = verify_locked_test_set(
            test_lock_manifest_path=test_lock_manifest_path,
            processed_dir=train_dir.parent,
            reports_dir=reports_dir,
        )
        if not is_lock_valid:
            raise RuntimeError(
                f"Test evaluation REFUSED: Test lock verification failed. "
                f"Errors: {lock_data['errors']}"
            )

        # Load test metadata manifest to attach image_id and source_group
        manifest_path = paths["manifest_path"]
        manifest_lookup: Dict[str, Dict[str, Any]] = {}
        if manifest_path.is_file():
            try:
                m_df = pd.read_csv(manifest_path)
                for _, r in m_df.iterrows():
                    fn = str(r.get("filename", ""))
                    cl = str(r.get("class_label", ""))
                    manifest_lookup[f"{cl}/{fn}"] = r.to_dict()
                    manifest_lookup[fn] = r.to_dict()
            except Exception:
                pass

        # Collect all test image files deterministically
        test_images_list = []
        for c in class_names:
            c_dir = test_dir / c
            if c_dir.is_dir():
                for f in sorted(c_dir.iterdir()):
                    if f.is_file():
                        test_images_list.append((f, c, class_names.index(c)))

        if not test_images_list:
            raise ValueError(f"No test images found in {test_dir}")

        prediction_records: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        print(f"[INFO] Running final inference on {len(test_images_list)} locked test images...")

        for file_path, true_class, true_idx in test_images_list:
            # Load and preprocess single image identically
            raw_img = tf.keras.utils.load_img(file_path, target_size=img_size)
            img_arr = tf.keras.utils.img_to_array(raw_img)
            img_batch = np.expand_dims(img_arr, axis=0)

            t0 = time.perf_counter()
            probs = model(img_batch, training=False).numpy()[0]
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            conf = float(probs[pred_idx])

            y_true_all.append(true_idx)
            y_pred_all.append(pred_idx)

            key = f"{true_class}/{file_path.name}"
            meta = manifest_lookup.get(key, manifest_lookup.get(file_path.name, {}))

            rec = {
                "image_path": str(file_path.relative_to(test_dir.parent)).replace("\\", "/"),
                "image_id": meta.get("image_id", file_path.stem),
                "true_label": true_class,
                "predicted_label": pred_class,
                "confidence": round(conf, 4),
                "correct": bool(true_class == pred_class),
                "source_group": meta.get("source_group", "unknown"),
            }
            # Add probability columns per class
            for c_i, c_name in enumerate(class_names):
                rec[f"prob_{c_name}"] = round(float(probs[c_i]), 4)
            prediction_records.append(rec)

        # Build prediction dataframe
        pred_df = pd.DataFrame(prediction_records)
        pred_df.to_csv(reports_dir / "final_test_predictions.csv", index=False)

        # Compute metrics
        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        metrics = compute_comprehensive_metrics(y_true_arr, y_pred_arr, class_names)
        cm = np.array(metrics["confusion_matrix"])
        cm_norm = np.array(metrics["confusion_matrix_normalized"])

        plot_confusion_matrices(cm, cm_norm, class_names, "final_test", reports_dir)

        # Classification report dict
        clf_rep = classification_report(y_true_arr, y_pred_arr, target_names=class_names, output_dict=True, zero_division=0)
        save_json(clf_rep, reports_dir / "final_test_classification_report.json")

        latency_summary = {
            "mean_ms": round(float(np.mean(latencies_ms)), 2),
            "median_ms": round(float(np.median(latencies_ms)), 2),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "throughput_fps": round(float(1000.0 / np.mean(latencies_ms)), 1) if np.mean(latencies_ms) > 0 else 0,
            "total_evaluated_images": len(test_images_list),
        }
        metrics["inference_latency"] = latency_summary
        save_json(metrics, reports_dir / "final_test_metrics.json")

        # Generate final_test_summary.md with mandatory honesty statements
        min_class_count = min(metrics["class_wise"][c]["support"] for c in class_names)
        has_small_test = min_class_count < 20

        md_summary = generate_final_test_summary_md(
            metrics=metrics,
            latency=latency_summary,
            class_names=class_names,
            has_small_test=has_small_test,
            min_class_count=min_class_count,
        )
        save_markdown(md_summary, reports_dir / "final_test_summary.md")

        print("\n" + "=" * 80)
        print("FINAL LOCKED TEST EVALUATION RESULTS:")
        print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Mean Latency: {latency_summary['mean_ms']} ms/image ({latency_summary['throughput_fps']} FPS)")
        print(f"Reports saved in {reports_dir}/")
        print("=" * 80 + "\n")

        return metrics

    # =========================================================================
    # VALIDATION SET EVALUATION (Development-Time)
    # =========================================================================
    elif split == "val":
        print(f"\n[INFO] Evaluating model on VALIDATION set ({val_dir})...")
        val_ds = load_validation_dataset(
            val_dir=val_dir,
            image_size=img_size,
            batch_size=batch_size,
            class_names=class_names,
        )

        val_eval = model.evaluate(val_ds, verbose=0)
        val_loss, val_acc = float(val_eval[0]), float(val_eval[1])

        y_true_list = []
        y_pred_list = []
        for imgs, labels in val_ds:
            preds = model.predict(imgs, verbose=0)
            y_true_list.extend(labels.numpy().tolist())
            y_pred_list.extend(np.argmax(preds, axis=1).tolist())

        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)

        metrics = compute_comprehensive_metrics(y_true_arr, y_pred_arr, class_names, loss_val=val_loss)
        cm = np.array(metrics["confusion_matrix"])
        cm_norm = np.array(metrics["confusion_matrix_normalized"])

        plot_confusion_matrices(cm, cm_norm, class_names, "validation", reports_dir)

        clf_rep = classification_report(y_true_arr, y_pred_arr, target_names=class_names, output_dict=True, zero_division=0)
        save_json(clf_rep, reports_dir / "validation_classification_report.json")
        save_json(metrics, reports_dir / "validation_metrics.json")

        # Check performance warnings
        warnings = []
        if metrics["accuracy"] < 0.75:
            warnings.append(f"Validation accuracy ({metrics['accuracy']:.4f}) is below acceptable baseline (0.75).")
        if metrics["macro_f1"] < 0.70:
            warnings.append(f"Validation macro F1 ({metrics['macro_f1']:.4f}) is below 0.70.")
        for c in class_names:
            rec = metrics["class_wise"][c]["recall"]
            if rec < 0.60:
                warnings.append(f"Class '{c}' recall ({rec:.4f}) is below 0.60.")
            sup = metrics["class_wise"][c]["support"]
            if sup < 10:
                warnings.append(f"Validation support for '{c}' ({sup}) is too small (<10).")

        pred_counts = pd.Series(y_pred_arr).value_counts(normalize=True)
        if (pred_counts > 0.70).any():
            dom_idx = pred_counts.index[0]
            warnings.append(f"Prediction collapse warning: Class '{class_names[dom_idx]}' dominates {pred_counts.iloc[0]*100:.1f}% of all predictions.")

        if warnings:
            print("\n" + "!" * 80)
            print("VALIDATION PERFORMANCE WARNINGS:")
            for w in warnings:
                print(f"  - ⚠️  {w}")
            print("!" * 80 + "\n")
        else:
            print(f"[PASS] Validation metrics look healthy: Accuracy={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1']:.4f}")

        return metrics

    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'val' or 'test_locked'.")


def generate_final_test_summary_md(
    metrics: Dict[str, Any],
    latency: Dict[str, Any],
    class_names: List[str],
    has_small_test: bool,
    min_class_count: int,
) -> str:
    """Generate comprehensive, scientifically honest final test report."""
    total_test = sum(metrics["class_wise"][c]["support"] for c in class_names)
    acc_pct = metrics["accuracy"] * 100
    macro_f1 = metrics["macro_f1"]

    md = [
        "# Final Locked Test Evaluation Report",
        "",
        "> [!IMPORTANT]",
        f"> On a locked internal test set containing {total_test} images across three classes, "
        f"the model achieved {acc_pct:.2f}% accuracy and {macro_f1:.4f} macro F1-score. "
        "The test split was separated using group-aware cryptographic locking. "
        "These results estimate internal generalization only and do not by themselves demonstrate field or drone deployment performance.",
        "",
        "## Scientific Integrity Statement",
        "The model achieved the reported score on a locked internal test set. This result does not establish field or drone deployment performance unless the test images were independently collected under representative field/drone conditions.",
        "",
    ]

    if has_small_test:
        md.extend([
            "> [!WARNING]",
            f"> **WARNING: The test set contains fewer than 20 images per class (minimum found: {min_class_count}). The result is statistically unstable and should be presented as preliminary.**",
            "",
        ])

    md.extend([
        "## Overall Summary Metrics",
        f"- **Locked Test Accuracy**: `{metrics['accuracy']:.4f}` ({acc_pct:.2f}%)",
        f"- **Macro Precision**: `{metrics['macro_precision']:.4f}`",
        f"- **Macro Recall**: `{metrics['macro_recall']:.4f}`",
        f"- **Macro F1-Score**: `{metrics['macro_f1']:.4f}`",
        f"- **Weighted F1-Score**: `{metrics['weighted_f1']:.4f}`",
        f"- **Mean Latency (CPU)**: `{latency['mean_ms']} ms` (~`{latency['throughput_fps']} FPS`)",
        "",
        "## Class-Wise Performance",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| --- | --- | --- | --- | --- |",
    ])

    for c in class_names:
        stats = metrics["class_wise"][c]
        md.append(f"| `{c}` | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1_score']:.4f} | {stats['support']} |")

    md.extend([
        "",
        "## Confusion Matrix",
        "| True \\ Pred | " + " | ".join(f"Pred {c}" for c in class_names) + " |",
        "| --- | " + " | ".join(["---"] * len(class_names)) + " |",
    ])

    for i, c in enumerate(class_names):
        row_vals = " | ".join(str(metrics["confusion_matrix"][i][j]) for j in range(len(class_names)))
        md.append(f"| **True {c}** | {row_vals} |")

    md.extend([
        "",
        "## Output Files Generated",
        "- Metrics JSON: `reports/final_test_metrics.json`",
        "- Classification Report JSON: `reports/final_test_classification_report.json`",
        "- Per-Image Predictions CSV: `reports/final_test_predictions.csv`",
        "- Confusion Matrix: `reports/final_test_confusion_matrix.png`",
        "- Normalized Confusion Matrix: `reports/final_test_confusion_matrix_normalized.png`",
        "",
    ])

    return "\n".join(md)
