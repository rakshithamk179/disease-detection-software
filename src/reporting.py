"""Reporting and visualization utilities for training curves, confusion matrices, and metrics."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Headless rendering for server and scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import save_json, save_markdown


def plot_training_history(
    stage1_history_csv: Optional[Path],
    finetune_history_csv: Optional[Path],
    reports_dir: Path,
) -> Tuple[Path, Path]:
    """Plot and save training/validation accuracy and loss curves across training stages."""
    reports_dir = Path(reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    if stage1_history_csv and Path(stage1_history_csv).is_file():
        df1 = pd.read_csv(stage1_history_csv)
        df1["stage"] = "Stage 1 (Head)"
        dfs.append(df1)
    if finetune_history_csv and Path(finetune_history_csv).is_file():
        df2 = pd.read_csv(finetune_history_csv)
        df2["stage"] = "Stage 2 (Fine-tuning)"
        dfs.append(df2)

    if not dfs:
        # Create empty placeholder curves if no history exists
        acc_path = reports_dir / "training_accuracy.png"
        loss_path = reports_dir / "training_loss.png"
        return acc_path, loss_path

    combined = pd.concat(dfs, ignore_index=True)
    epochs = range(1, len(combined) + 1)

    # 1. Plot Accuracy
    plt.figure(figsize=(9, 5))
    if "accuracy" in combined.columns:
        plt.plot(epochs, combined["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in combined.columns:
        plt.plot(epochs, combined["val_accuracy"], "g--s", label="Validation Accuracy", linewidth=2)

    # Mark stage transition if both stages ran
    if len(dfs) > 1:
        stage1_len = len(dfs[0])
        plt.axvline(x=stage1_len + 0.5, color="gray", linestyle=":", label="Stage 2 Unfreeze")

    plt.title("Training and Validation Accuracy (Validation-Driven Selection)", fontsize=13, pad=12)
    plt.xlabel("Cumulative Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    acc_path = reports_dir / "training_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # 2. Plot Loss
    plt.figure(figsize=(9, 5))
    if "loss" in combined.columns:
        plt.plot(epochs, combined["loss"], "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in combined.columns:
        plt.plot(epochs, combined["val_loss"], "m--s", label="Validation Loss", linewidth=2)

    if len(dfs) > 1:
        stage1_len = len(dfs[0])
        plt.axvline(x=stage1_len + 0.5, color="gray", linestyle=":", label="Stage 2 Unfreeze")

    plt.title("Training and Validation Loss", fontsize=13, pad=12)
    plt.xlabel("Cumulative Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    loss_path = reports_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return acc_path, loss_path


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    loss_val: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate accuracy, loss, class-wise precision/recall/F1, and macro/weighted averages."""
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    class_metrics: Dict[str, Any] = {}
    for c in class_names:
        c_stats = report_dict.get(c, {})
        class_metrics[c] = {
            "precision": round(float(c_stats.get("precision", 0.0)), 4),
            "recall": round(float(c_stats.get("recall", 0.0)), 4),
            "f1_score": round(float(c_stats.get("f1-score", 0.0)), 4),
            "support": int(c_stats.get("support", 0)),
        }

    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})

    metrics = {
        "loss": round(float(loss_val), 4) if loss_val is not None else None,
        "accuracy": round(float(report_dict.get("accuracy", 0.0)), 4),
        "macro_precision": round(float(macro.get("precision", 0.0)), 4),
        "macro_recall": round(float(macro.get("recall", 0.0)), 4),
        "macro_f1": round(float(macro.get("f1-score", 0.0)), 4),
        "weighted_precision": round(float(weighted.get("precision", 0.0)), 4),
        "weighted_recall": round(float(weighted.get("recall", 0.0)), 4),
        "weighted_f1": round(float(weighted.get("f1-score", 0.0)), 4),
        "class_wise": class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(4).tolist(),
    }
    return metrics


def plot_confusion_matrices(
    cm: np.ndarray,
    cm_norm: np.ndarray,
    class_names: List[str],
    prefix: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Plot raw and normalized confusion matrices with clean annotations."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(f"{prefix.replace('_', ' ').title()} - Confusion Matrix", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    raw_path = output_dir / f"{prefix}_confusion_matrix.png"
    plt.savefig(raw_path, dpi=300)
    plt.close()

    # 2. Normalized Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(f"{prefix.replace('_', ' ').title()} - Normalized Confusion Matrix", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    norm_path = output_dir / f"{prefix}_confusion_matrix_normalized.png"
    plt.savefig(norm_path, dpi=300)
    plt.close()

    return raw_path, norm_path
