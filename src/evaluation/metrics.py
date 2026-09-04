"""
Honest evaluation metrics.

Razorpay's bar: "Honest metrics including false-positive cost."
This module reports everything — including where the model fails.
No cherry-picking.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from loguru import logger


def full_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    save_dir: str = "docs/",
) -> dict:
    """
    Generate comprehensive evaluation report.
    Prints everything, hides nothing.
    """
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, y_scores)

    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc_roc = float("nan")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    report = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Precision':<25} {precision:>10.4f}")
    print(f"{'Recall':<25} {recall:>10.4f}")
    print(f"{'F1 Score':<25} {f1:>10.4f}")
    print(f"{'AUC-PR':<25} {auc_pr:>10.4f}")
    print(f"{'AUC-ROC':<25} {auc_roc:>10.4f}")

    print(f"\n{'Confusion Matrix':}")
    print(f"  True Positives  (caught fraud):    {tp:>6}")
    print(f"  False Positives (blocked legit):   {fp:>6}  ← THIS IS THE COST")
    print(f"  True Negatives  (passed legit):    {tn:>6}")
    print(f"  False Negatives (missed fraud):    {fn:>6}  ← THIS IS THE RISK")

    print(f"\n{'False Positive Rate':<25} {fp/(fp+tn):.4f}")
    print(f"{'False Negative Rate':<25} {fn/(fn+tp) if (fn+tp) > 0 else 0:.4f}")
    print("=" * 60)

    # Full sklearn report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))

    # Save PR curve
    _plot_pr_curve(y_true, y_scores, save_dir)

    return report


def _plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_dir: str,
):
    """Save precision-recall curve — the most honest chart for imbalanced data."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, "b-", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall (fraud caught)", fontsize=12)
    ax.set_ylabel("Precision (alerts that were real)", fontsize=12)
    ax.set_title("Precision-Recall Curve — RiskGuard Ensemble", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    path = f"{save_dir}/pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved to {path}")
