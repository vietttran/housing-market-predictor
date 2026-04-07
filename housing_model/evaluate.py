"""
Evaluation metrics for the housing price direction model.

Accuracy alone is insufficient for binary classifiers.  This module
computes a full suite: F1, Matthews Correlation Coefficient, precision,
recall, and a confusion matrix.  MCC is especially informative because
it is robust to any class imbalance and is widely used in ML research.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a full set of binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 = price down, 1 = price up).
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict with keys:
        accuracy, f1, mcc, precision, recall,
        report (str), confusion_matrix (np.ndarray)
    """
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "f1":              f1_score(y_true, y_pred, zero_division=0),
        "mcc":             matthews_corrcoef(y_true, y_pred),
        "precision":       precision_score(y_true, y_pred, zero_division=0),
        "recall":          recall_score(y_true, y_pred, zero_division=0),
        "report":          classification_report(
                               y_true, y_pred,
                               target_names=["Down (0)", "Up (1)"],
                               zero_division=0,
                           ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_comparison_table(
    results: dict[str, dict],
    y_true: np.ndarray,
) -> None:
    """
    Print a side-by-side metric comparison for all models, followed by
    per-model classification reports and confusion matrices.

    Parameters
    ----------
    results : dict
        Output of model.run_comparison().
    y_true : np.ndarray
        Full ground-truth labels starting from BACKTEST_START.
    """
    rows = []
    for name, res in results.items():
        preds = res["preds"]
        n = min(len(preds), len(y_true))
        m = compute_metrics(y_true[:n], preds[:n])
        rows.append({
            "Model":     name,
            "Accuracy":  f"{m['accuracy']:.2%}",
            "F1 (Up)":   f"{m['f1']:.3f}",
            "MCC":       f"{m['mcc']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall":    f"{m['recall']:.3f}",
        })

    table = pd.DataFrame(rows).set_index("Model")

    print()
    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(table.to_string())
    print("=" * 60)

    for name, res in results.items():
        preds = res["preds"]
        n = min(len(preds), len(y_true))
        m = compute_metrics(y_true[:n], preds[:n])
        cm = m["confusion_matrix"]

        print(f"\n--- {name} Classification Report ---")
        print(m["report"])
        print("Confusion Matrix:")
        print(f"                 Pred Down  Pred Up")
        print(f"  Actual Down       {cm[0, 0]:5d}    {cm[0, 1]:5d}")
        print(f"  Actual Up         {cm[1, 0]:5d}    {cm[1, 1]:5d}")
        print()
