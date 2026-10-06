"""Production model training with two-stage transfer learning, validation-driven selection,

and strict locked test-set exclusion.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from src.dataset_loader import load_training_dataset, load_validation_dataset
from src.model import build_classifier_model, prepare_model_for_finetuning
from src.reporting import compute_comprehensive_metrics, plot_training_history
from src.split_dataset import verify_locked_test_set
from src.utils import save_json, save_markdown


logger = logging.getLogger("tomato_disease_classifier.train")


def determine_class_weights(
    train_dir: Path,
    class_names: List[str],
    ratio_threshold: float = 1.2,
    logs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[int, float]], Dict[str, Any]]:
    """Calculate class weights if training distribution imbalance ratio exceeds threshold."""
    train_dir = Path(train_dir)
    class_counts: Dict[str, int] = {}
    y_train_indices: List[int] = []

    for idx, c_name in enumerate(class_names):
        c_folder = train_dir / c_name
        cnt = len([f for f in c_folder.iterdir() if f.is_file()]) if c_folder.is_dir() else 0
        class_counts[c_name] = cnt
        y_train_indices.extend([idx] * cnt)

    counts_list = list(class_counts.values())
    if not counts_list or min(counts_list) == 0:
        info = {
            "applied": False,
            "reason": "Empty or missing class folders in train directory",
            "class_counts": class_counts,
            "weights": None,
        }
        if logs_dir:
            save_json(info, Path(logs_dir) / "class_weights.json")
        return None, info

    max_c = max(counts_list)
    min_c = min(counts_list)
    imbalance_ratio = float(max_c / min_c)

    if imbalance_ratio > ratio_threshold:
        unique_classes = np.arange(len(class_names))
        weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=np.array(y_train_indices),
        )
        weights_dict = {int(cls_idx): float(round(w, 4)) for cls_idx, w in zip(unique_classes, weights_arr)}
        named_weights = {class_names[cls_idx]: weights_dict[cls_idx] for cls_idx in weights_dict}

        info = {
            "applied": True,
            "reason": f"Imbalance ratio ({imbalance_ratio:.2f}) exceeds threshold ({ratio_threshold:.2f})",
            "imbalance_ratio": round(imbalance_ratio, 3),
            "class_counts": class_counts,
            "weights_by_index": weights_dict,
            "weights_by_name": named_weights,
        }
        print(f"\n[INFO] Class imbalance detected (ratio: {imbalance_ratio:.2f}). Applied class weights:")
        for name, w in named_weights.items():
            print(f"  - {name}: {w:.4f}")
    else:
        weights_dict = None
        info = {
            "applied": False,
            "reason": f"Class distribution is sufficiently balanced (ratio: {imbalance_ratio:.2f} <= {ratio_threshold:.2f})",
            "imbalance_ratio": round(imbalance_ratio, 3),
            "class_counts": class_counts,
            "weights_by_index": None,
            "weights_by_name": None,
        }
        print(f"\n[INFO] Dataset is balanced (ratio: {imbalance_ratio:.2f}). No class weighting required.")

    if logs_dir:
        save_json(info, Path(logs_dir) / "class_weights.json")

    return weights_dict, info


def execute_training(
    config: Dict[str, Any],
    stage: str = "all",
    skip_fine_tune: bool = False,
    stage1_epochs: Optional[int] = None,
    fine_tune_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Execute complete two-stage transfer learning pipeline.

    Guarantees strict isolation: the locked test set is NEVER loaded or evaluated here.
    """
    paths = config["resolved_paths"]
    train_dir = paths["train_dir"]
    val_dir = paths["val_dir"]
    test_dir = paths["locked_test_dir"]
    models_dir = paths["models_dir"]
    logs_dir = paths["logs_dir"]
    reports_dir = paths["reports_dir"]
    test_lock_manifest_path = paths["test_lock_manifest"]

    for d in [models_dir, logs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Strict verification of locked test set
    print("\n" + "=" * 80)
    print("STEP 1: VERIFYING TEST-SET LOCK AND LEAKAGE ISOLATION")
    print("=" * 80)
    is_lock_valid, lock_data = verify_locked_test_set(
        test_lock_manifest_path=test_lock_manifest_path,
        processed_dir=train_dir.parent,
        reports_dir=reports_dir,
    )
    if not is_lock_valid:
        raise RuntimeError(
            f"Training REFUSED: Test lock verification failed.\n"
            f"Errors: {json.dumps(lock_data['errors'], indent=2)}\n"
            f"Examine reports/test_lock_verification.json for details."
        )
    print("[PASS] Test-set lock verified: No test file overlap in train or validation sets.")

    # 2. Print non-negotiable isolation statement
    isolation_notice = (
        "\n" + "*" * 80 + "\n"
        "CONFIRMATION OF ISOLATION:\n"
        "Training will use TRAIN and VALIDATION sets only.\n"
        "The locked test set will not be loaded or touched during this run.\n"
        + "*" * 80 + "\n"
    )
    print(isolation_notice)
    logger.info("Training will use TRAIN and VALIDATION sets only. The locked test set will not be loaded.")

    # 3. Resolve hyperparameters
    t_cfg = config.get("training", {})
    m_cfg = config.get("model", {})
    i_cfg = config.get("image", {})

    b_size = batch_size or t_cfg.get("batch_size", 16)
    s1_epochs = stage1_epochs if stage1_epochs is not None else t_cfg.get("stage1_epochs", 30)
    ft_epochs = fine_tune_epochs if fine_tune_epochs is not None else t_cfg.get("fine_tune_epochs", 15)
    img_size = (i_cfg.get("width", 224), i_cfg.get("height", 224))
    seed = config.get("seed", 42)
    arch = m_cfg.get("architecture", "MobileNetV3Small")
    pretrained_weights = m_cfg.get("pretrained_weights", "imagenet")
    fine_tune_layers = m_cfg.get("fine_tune_last_n_layers", 25)
    s1_lr = t_cfg.get("stage1_learning_rate", 0.001)
    ft_lr = t_cfg.get("fine_tune_learning_rate", 0.00001)
    patience = t_cfg.get("early_stopping_patience", 7)
    cache_ds = t_cfg.get("cache_dataset", False)

    # 4. Load datasets (TRAIN and VAL only)
    class_names_path = models_dir / "class_names.json"
    train_ds, class_names = load_training_dataset(
        train_dir=train_dir,
        image_size=img_size,
        batch_size=b_size,
        seed=seed,
        cache=cache_ds,
        class_names_path=class_names_path,
    )
    val_ds = load_validation_dataset(
        val_dir=val_dir,
        image_size=img_size,
        batch_size=b_size,
        class_names=class_names,
    )

    print(f"[INFO] Discovered class order (TensorFlow loader): {class_names}")

    # 5. Class weights (training data only)
    class_weights, weight_info = determine_class_weights(
        train_dir=train_dir,
        class_names=class_names,
        ratio_threshold=t_cfg.get("class_imbalance_ratio_threshold", 1.2),
        logs_dir=logs_dir,
    )

    # 6. Save training config record
    run_config_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architecture": arch,
        "pretrained_weights": pretrained_weights,
        "image_size": img_size,
        "batch_size": b_size,
        "stage1_epochs": s1_epochs,
        "fine_tune_epochs": ft_epochs,
        "skip_fine_tune": skip_fine_tune,
        "stage": stage,
        "seed": seed,
        "class_names": class_names,
        "class_weights": weight_info,
    }
    save_json(run_config_record, logs_dir / "training_config.json")

    start_time = time.time()
    stage1_model_path = models_dir / "stage1_best_model.keras"
    best_model_path = models_dir / "best_model.keras"
    stage1_csv_path = logs_dir / "training_history_stage1.csv"
    ft_csv_path = logs_dir / "training_history_finetune.csv"

    # =========================================================================
    # STAGE 1: Train Classification Head (Backbone Frozen)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD (BACKBONE FROZEN)")
    print("=" * 80)

    if resume and stage1_model_path.is_file():
        print(f"[RESUME] Loading existing Stage 1 model from {stage1_model_path}...")
        model = tf.keras.models.load_model(stage1_model_path)
    else:
        model = build_classifier_model(
            num_classes=len(class_names),
            architecture=arch,
            pretrained_weights=pretrained_weights,
            input_shape=(img_size[1], img_size[0], 3),
            dropout_1=m_cfg.get("dropout_1", 0.30),
            dropout_2=m_cfg.get("dropout_2", 0.20),
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=s1_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        s1_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(stage1_model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(stage1_csv_path)),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=s1_epochs,
            callbacks=s1_callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    # Evaluate best Stage 1 model on validation data
    if stage1_model_path.is_file():
        model = tf.keras.models.load_model(stage1_model_path)
    s1_val_eval = model.evaluate(val_ds, verbose=0)
    s1_val_loss, s1_val_acc = float(s1_val_eval[0]), float(s1_val_eval[1])
    print(f"\n[STAGE 1 BEST VAL LOSS]: {s1_val_loss:.4f} | VAL ACC: {s1_val_acc:.4f}")

    # =========================================================================
    # STAGE 2: Fine-Tuning (Top Layers Unfrozen, BatchNorm Frozen)
    # =========================================================================
    s2_val_loss, s2_val_acc = s1_val_loss, s1_val_acc

    if stage != "stage1" and not skip_fine_tune and ft_epochs > 0:
        print("\n" + "=" * 80)
        print(f"STAGE 2: FINE-TUNING (UNFREEZING LAST {fine_tune_layers} LAYERS)")
        print("=" * 80)

        model = prepare_model_for_finetuning(
            model=model,
            fine_tune_last_n_layers=fine_tune_layers,
            learning_rate=ft_lr,
        )

        s2_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(ft_csv_path)),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ft_epochs,
            callbacks=s2_callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        if best_model_path.is_file():
            model = tf.keras.models.load_model(best_model_path)
        s2_val_eval = model.evaluate(val_ds, verbose=0)
        s2_val_loss, s2_val_acc = float(s2_val_eval[0]), float(s2_val_eval[1])
        print(f"\n[STAGE 2 BEST VAL LOSS]: {s2_val_loss:.4f} | VAL ACC: {s2_val_acc:.4f}")
    else:
        # If fine-tuning is skipped, stage 1 best model is the primary model
        shutil.copy2(stage1_model_path, best_model_path)
        print("[INFO] Fine-tuning skipped. Selected Stage 1 best model as production model.")

    elapsed_duration = round(time.time() - start_time, 2)

    # 7. Generate plots & validation metrics
    plot_training_history(
        stage1_history_csv=stage1_csv_path,
        finetune_history_csv=ft_csv_path if ft_csv_path.is_file() else None,
        reports_dir=reports_dir,
    )

    # Gather predictions on validation set for comprehensive validation report
    y_true_list = []
    y_pred_list = []
    for imgs, labels in val_ds:
        preds = model.predict(imgs, verbose=0)
        y_true_list.extend(labels.numpy().tolist())
        y_pred_list.extend(np.argmax(preds, axis=1).tolist())

    val_metrics = compute_comprehensive_metrics(
        y_true=np.array(y_true_list),
        y_pred=np.array(y_pred_list),
        class_names=class_names,
        loss_val=s2_val_loss,
    )
    save_json(val_metrics, reports_dir / "validation_metrics.json")

    # 8. Save training summary report
    summary_data = {
        "dataset_counts": {
            "train": {c: len(list((train_dir / c).glob("*"))) for c in class_names},
            "val": {c: len(list((val_dir / c).glob("*"))) for c in class_names},
        },
        "class_order": class_names,
        "architecture": arch,
        "pretrained_backbone": pretrained_weights,
        "stage1_best_val_loss": s1_val_loss,
        "stage1_best_val_accuracy": s1_val_acc,
        "stage2_best_val_loss": s2_val_loss,
        "stage2_best_val_accuracy": s2_val_acc,
        "total_parameters": model.count_params(),
        "model_file_size_mb": round(best_model_path.stat().st_size / (1024 * 1024), 2) if best_model_path.is_file() else 0.0,
        "training_duration_seconds": elapsed_duration,
        "isolation_statement": "No locked test-set metrics were used during model training or selection.",
    }

    md_summary = generate_training_summary_md(summary_data, val_metrics)
    save_markdown(md_summary, reports_dir / "training_summary.md")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE (TRAIN + VAL ONLY)")
    print(f"Best Model Saved: {best_model_path}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f} | Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    print("No locked test set metrics were accessed or used.")
    print("=" * 80 + "\n")

    return summary_data


def generate_training_summary_md(summary: Dict[str, Any], val_metrics: Dict[str, Any]) -> str:
    """Generate Markdown report for training and validation results."""
    md = [
        "# Model Training & Validation Summary",
        "",
        "> [!IMPORTANT]",
        f"> **{summary['isolation_statement']}**",
        "",
        "## Architecture & Training Setup",
        f"- **Model Backbone**: `{summary['architecture']}` (Pretrained weights: `{summary['pretrained_backbone']}`)",
        f"- **Total Model Parameters**: {summary['total_parameters']:,} (~{summary['model_file_size_mb']} MB)",
        f"- **Class Index Order**: `{summary['class_order']}`",
        f"- **Total Training Duration**: {summary['training_duration_seconds']:.1f} seconds",
        "",
        "## Validation Performance Progression",
        f"- **Stage 1 (Head Frozen)**: Loss: `{summary['stage1_best_val_loss']:.4f}`, Accuracy: `{summary['stage1_best_val_accuracy']:.4f}`",
        f"- **Stage 2 (Fine-Tuning)**: Loss: `{summary['stage2_best_val_loss']:.4f}`, Accuracy: `{summary['stage2_best_val_accuracy']:.4f}`",
        f"- **Final Validation Macro F1**: `{val_metrics['macro_f1']:.4f}`",
        f"- **Final Validation Weighted F1**: `{val_metrics['weighted_f1']:.4f}`",
        "",
        "## Training & Validation Dataset Sizes",
        f"- **Train Counts**: `{summary['dataset_counts']['train']}`",
        f"- **Validation Counts**: `{summary['dataset_counts']['val']}`",
        "",
        "## Class-Wise Validation Performance",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| --- | --- | --- | --- | --- |",
    ]

    for c, stats in val_metrics["class_wise"].items():
        md.append(f"| `{c}` | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1_score']:.4f} | {stats['support']} |")

    md.append("")
    md.append("> [!NOTE] Test set evaluation is intentionally NOT run automatically. Test metrics must be computed strictly via `python scripts/evaluate_model.py --split test_locked --final-evaluation`.")
    md.append("")
    return "\n".join(md)
