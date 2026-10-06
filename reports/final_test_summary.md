# Final Locked Test Evaluation Report

> [!IMPORTANT]
> On a locked internal test set containing 164 images across three classes, the model achieved 90.85% accuracy and 0.9117 macro F1-score. The test split was separated using group-aware cryptographic locking. These results estimate internal generalization only and do not by themselves demonstrate field or drone deployment performance.

## Scientific Integrity Statement
The model achieved the reported score on a locked internal test set. This result does not establish field or drone deployment performance unless the test images were independently collected under representative field/drone conditions.

## Overall Summary Metrics
- **Locked Test Accuracy**: `0.9085` (90.85%)
- **Macro Precision**: `0.9116`
- **Macro Recall**: `0.9137`
- **Macro F1-Score**: `0.9117`
- **Weighted F1-Score**: `0.9079`
- **Mean Latency (CPU)**: `195.51 ms` (~`5.1 FPS`)

## Class-Wise Performance
| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| `early_blight` | 0.9091 | 0.8333 | 0.8696 | 60 |
| `healthy` | 0.9592 | 0.9792 | 0.9691 | 48 |
| `late_blight` | 0.8667 | 0.9286 | 0.8966 | 56 |

## Confusion Matrix
| True \ Pred | Pred early_blight | Pred healthy | Pred late_blight |
| --- | --- | --- | --- |
| **True early_blight** | 50 | 2 | 8 |
| **True healthy** | 1 | 47 | 0 |
| **True late_blight** | 4 | 0 | 52 |

## Output Files Generated
- Metrics JSON: `reports/final_test_metrics.json`
- Classification Report JSON: `reports/final_test_classification_report.json`
- Per-Image Predictions CSV: `reports/final_test_predictions.csv`
- Confusion Matrix: `reports/final_test_confusion_matrix.png`
- Normalized Confusion Matrix: `reports/final_test_confusion_matrix_normalized.png`
