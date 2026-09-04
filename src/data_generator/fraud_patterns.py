"""
Fraud pattern definitions for synthetic data generation.

Each pattern injects a specific type of anomaly into the transaction stream.
Real-world fraud rarely looks like random noise — it follows recognizable
behavioral patterns. We model five common ones.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class FraudPattern:
    """Defines a single fraud injection pattern."""
    name: str
    description: str
    label: str              # Sub-label for analysis (e.g., "velocity_spike")
    inject_fn: Callable     # Function that generates fraudulent transactions


def velocity_spike(
    merchant_id: str,
    base_time: pd.Timestamp,
    rng: np.random.Generator,
    n_txns: int = 30,
) -> pd.DataFrame:
    """
    Burst of many small transactions in a short window.
    Classic card-testing pattern — attacker validates stolen cards with
    rapid low-value charges.
    """
    timestamps = base_time + pd.to_timedelta(
        rng.uniform(0, 10, n_txns), unit="m"  # 30 txns in 10 minutes
    )
    amounts = rng.uniform(1, 50, n_txns)  # Small test amounts
    cards = [f"card_{rng.integers(90000, 99999)}" for _ in range(n_txns)]

    return pd.DataFrame({
        "merchant_id": merchant_id,
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "card_id": cards,
        "is_fraud": True,
        "fraud_type": "velocity_spike",
    })


def amount_anomaly(
    merchant_id: str,
    base_time: pd.Timestamp,
    rng: np.random.Generator,
    n_txns: int = 5,
    normal_avg: float = 500,
) -> pd.DataFrame:
    """
    Sudden high-value transactions far above merchant's normal average.
    Indicates compromised merchant account or large-scale theft.
    """
    timestamps = base_time + pd.to_timedelta(
        rng.uniform(0, 120, n_txns), unit="m"
    )
    # 10x-50x the normal average
    amounts = rng.uniform(normal_avg * 10, normal_avg * 50, n_txns)
    cards = [f"card_{rng.integers(80000, 89999)}" for _ in range(n_txns)]

    return pd.DataFrame({
        "merchant_id": merchant_id,
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "card_id": cards,
        "is_fraud": True,
        "fraud_type": "amount_anomaly",
    })


def chargeback_burst(
    merchant_id: str,
    base_time: pd.Timestamp,
    rng: np.random.Generator,
    n_txns: int = 15,
) -> pd.DataFrame:
    """
    Cluster of transactions that will result in chargebacks.
    Modeled as transactions with specific amount patterns (round numbers)
    from geographically dispersed cards hitting one merchant.
    """
    timestamps = base_time + pd.to_timedelta(
        rng.uniform(0, 1440, n_txns), unit="m"  # Spread over 24 hours
    )
    # Round-number amounts are a known chargeback signal
    base_amounts = rng.choice([100, 200, 500, 1000, 2000], n_txns)
    amounts = base_amounts.astype(float)
    cards = [f"card_{rng.integers(70000, 79999)}" for _ in range(n_txns)]

    return pd.DataFrame({
        "merchant_id": merchant_id,
        "timestamp": timestamps,
        "amount": amounts,
        "card_id": cards,
        "is_fraud": True,
        "fraud_type": "chargeback_burst",
    })


def abuse_ring(
    merchant_id: str,
    base_time: pd.Timestamp,
    rng: np.random.Generator,
    ring_size: int = 5,
    txns_per_card: int = 3,
) -> pd.DataFrame:
    """
    Small set of cards repeatedly transacting at one merchant.
    Indicates coordinated fraud ring — same group of cards cycling
    through a compromised or colluding merchant.
    """
    ring_cards = [f"card_ring_{rng.integers(10000, 19999)}" for _ in range(ring_size)]
    rows = []
    for card in ring_cards:
        for _ in range(txns_per_card):
            rows.append({
                "merchant_id": merchant_id,
                "timestamp": base_time + pd.Timedelta(minutes=int(rng.uniform(0, 4320))),
                "amount": round(float(rng.uniform(200, 2000)), 2),
                "card_id": card,
                "is_fraud": True,
                "fraud_type": "abuse_ring",
            })

    return pd.DataFrame(rows)


def off_hours_surge(
    merchant_id: str,
    base_time: pd.Timestamp,
    rng: np.random.Generator,
    n_txns: int = 20,
) -> pd.DataFrame:
    """
    Transactions occurring during unusual hours (2 AM - 5 AM IST).
    Most Indian merchants have near-zero activity at these hours.
    """
    # Force timestamps into 2-5 AM window
    day_offset = rng.integers(0, 7, n_txns)
    hour = rng.uniform(2, 5, n_txns)
    timestamps = [
        base_time.normalize() + pd.Timedelta(days=int(d), hours=float(h))
        for d, h in zip(day_offset, hour)
    ]
    amounts = rng.uniform(100, 3000, n_txns)
    cards = [f"card_{rng.integers(60000, 69999)}" for _ in range(n_txns)]

    return pd.DataFrame({
        "merchant_id": merchant_id,
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "card_id": cards,
        "is_fraud": True,
        "fraud_type": "off_hours_surge",
    })


# Registry of all patterns
ALL_PATTERNS = [
    FraudPattern(
        name="Velocity Spike",
        description="Burst of rapid small transactions (card testing)",
        label="velocity_spike",
        inject_fn=velocity_spike,
    ),
    FraudPattern(
        name="Amount Anomaly",
        description="Transactions far above merchant's normal average",
        label="amount_anomaly",
        inject_fn=amount_anomaly,
    ),
    FraudPattern(
        name="Chargeback Burst",
        description="Cluster of round-amount transactions likely to chargeback",
        label="chargeback_burst",
        inject_fn=chargeback_burst,
    ),
    FraudPattern(
        name="Abuse Ring",
        description="Small card group repeatedly hitting one merchant",
        label="abuse_ring",
        inject_fn=abuse_ring,
    ),
    FraudPattern(
        name="Off-Hours Surge",
        description="Unusual transaction volume during 2-5 AM IST",
        label="off_hours_surge",
        inject_fn=off_hours_surge,
    ),
]
