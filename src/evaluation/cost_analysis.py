"""
False-positive cost analysis.

This is what separates a good submission from a great one.
Razorpay explicitly asks for "honest metrics including false-positive cost."

Every false positive = a legitimate customer blocked = revenue lost.
Every false negative = fraud that slipped through = direct loss.
The optimal threshold balances these costs.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from loguru import logger


def compute_daily_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: dict,
) -> dict:
    """
    Compute the daily operational cost of the detector.

    Returns breakdown of costs and net savings vs. no detection.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    # Costs
    fraud_caught_savings = tp * cost_matrix["true_positive_value"]
    false_alarm_cost = fp * cost_matrix["false_positive_cost"]
    missed_fraud_cost = fn * cost_matrix["false_negative_cost"]

    net_savings = fraud_caught_savings - false_alarm_cost - missed_fraud_cost

    # What happens with NO detector (all fraud goes through)
    total_fraud = tp + fn
    no_detector_loss = total_fraud * cost_matrix["false_negative_cost"]

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "fraud_caught_savings": fraud_caught_savings,
        "false_alarm_cost": false_alarm_cost,
        "missed_fraud_cost": missed_fraud_cost,
        "net_savings": net_savings,
        "no_detector_loss": no_detector_loss,
        "detector_improvement": net_savings + no_detector_loss,
    }


def cost_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: dict,
) -> dict:
    """Print a detailed cost analysis."""
    costs = compute_daily_cost(y_true, y_pred, cost_matrix)

    print("\n" + "=" * 60)
    print("COST ANALYSIS (based on test set)")
    print("=" * 60)
    print(f"\nCost Matrix:")
    print(f"  Catching 1 fraud saves:       ₹{cost_matrix['true_positive_value']:,}")
    print(f"  1 false alarm costs:           ₹{cost_matrix['false_positive_cost']:,}")
    print(f"  1 missed fraud costs:          ₹{cost_matrix['false_negative_cost']:,}")

    print(f"\nResults:")
    print(f"  Fraud caught:     {costs['true_positives']:>5} × ₹{cost_matrix['true_positive_value']:,} = ₹{costs['fraud_caught_savings']:>10,}")
    print(f"  False alarms:     {costs['false_positives']:>5} × ₹{cost_matrix['false_positive_cost']:,}   = -₹{costs['false_alarm_cost']:>9,}")
    print(f"  Missed fraud:     {costs['false_negatives']:>5} × ₹{cost_matrix['false_negative_cost']:,} = -₹{costs['missed_fraud_cost']:>9,}")
    print(f"  {'─' * 45}")
    print(f"  Net savings:      ₹{costs['net_savings']:>10,}")
    print(f"  Without detector: -₹{costs['no_detector_loss']:>9,}")
    print(f"  Improvement:      ₹{costs['detector_improvement']:>10,}")
    print("=" * 60)

    return costs


def threshold_optimization(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    cost_matrix: dict,
    save_path: str = "docs/threshold_optimization.png",
) -> float:
    """
    Find the threshold that minimizes total cost.
    Plots cost vs. threshold curve.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        result = compute_daily_cost(y_true, y_pred, cost_matrix)
        total_cost = result["false_alarm_cost"] + result["missed_fraud_cost"]
        costs.append(total_cost)

    costs = np.array(costs)
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, costs, "b-", linewidth=2)
    ax.axvline(x=optimal_threshold, color="r", linestyle="--",
               label=f"Optimal: {optimal_threshold:.3f} (cost: ₹{optimal_cost:,.0f})")
    ax.set_xlabel("Detection Threshold", fontsize=12)
    ax.set_ylabel("Total Cost (₹)", fontsize=12)
    ax.set_title("Threshold vs. Total Cost — Finding the Sweet Spot", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Optimal threshold: {optimal_threshold:.3f} (total cost: ₹{optimal_cost:,.0f})")
    logger.info(f"Threshold optimization plot saved to {save_path}")

    return optimal_threshold
