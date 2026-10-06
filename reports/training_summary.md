# Model Training & Validation Summary

> [!IMPORTANT]
> **No locked test-set metrics were used during model training or selection.**

## Architecture & Training Setup
- **Model Backbone**: `MobileNetV3Small` (Pretrained weights: `imagenet`)
- **Total Model Parameters**: 1,013,363 (~8.7 MB)
- **Class Index Order**: `['early_blight', 'healthy', 'late_blight']`
- **Total Training Duration**: 1514.1 seconds

## Validation Performance Progression
- **Stage 1 (Head Frozen)**: Loss: `0.0600`, Accuracy: `0.9814`
- **Stage 2 (Fine-Tuning)**: Loss: `0.0543`, Accuracy: `0.9814`
- **Final Validation Macro F1**: `0.9818`
- **Final Validation Weighted F1**: `0.9813`

## Training & Validation Dataset Sizes
- **Train Counts**: `{'early_blight': 880, 'healthy': 1494, 'late_blight': 1801}`
- **Validation Counts**: `{'early_blight': 60, 'healthy': 49, 'late_blight': 52}`

## Class-Wise Validation Performance
| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| `early_blight` | 1.0000 | 0.9500 | 0.9744 | 60 |
| `healthy` | 0.9800 | 1.0000 | 0.9899 | 49 |
| `late_blight` | 0.9630 | 1.0000 | 0.9811 | 52 |

> [!NOTE] Test set evaluation is intentionally NOT run automatically. Test metrics must be computed strictly via `python scripts/evaluate_model.py --split test_locked --final-evaluation`.
