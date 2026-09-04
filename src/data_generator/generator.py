"""
Synthetic transaction data generator.

Generates realistic merchant transaction data with injected fraud patterns.
The generator creates a baseline of normal transactions per merchant,
then injects fraud scenarios at random points.

Usage:
    python -m src.data_generator.generator
"""

import os
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from .fraud_patterns import ALL_PATTERNS


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_merchant_profile(merchant_id: str, rng: np.random.Generator) -> dict:
    """Generate a realistic merchant profile with characteristic transaction patterns."""
    category = rng.choice([
        "electronics", "grocery", "fashion", "food_delivery",
        "saas", "travel", "pharmacy", "education"
    ])

    # Each category has a typical transaction profile
    profiles = {
        "electronics":    {"avg_amount": 3000, "std_amount": 2000, "daily_volume": 50},
        "grocery":        {"avg_amount": 400,  "std_amount": 200,  "daily_volume": 200},
        "fashion":        {"avg_amount": 1200, "std_amount": 800,  "daily_volume": 80},
        "food_delivery":  {"avg_amount": 350,  "std_amount": 150,  "daily_volume": 300},
        "saas":           {"avg_amount": 2000, "std_amount": 1500, "daily_volume": 30},
        "travel":         {"avg_amount": 5000, "std_amount": 3000, "daily_volume": 20},
        "pharmacy":       {"avg_amount": 600,  "std_amount": 400,  "daily_volume": 100},
        "education":      {"avg_amount": 8000, "std_amount": 5000, "daily_volume": 15},
    }

    profile = profiles[category]
    return {
        "merchant_id": merchant_id,
        "category": category,
        "avg_amount": profile["avg_amount"] * rng.uniform(0.7, 1.3),
        "std_amount": profile["std_amount"] * rng.uniform(0.5, 1.5),
        "daily_volume": int(profile["daily_volume"] * rng.uniform(0.5, 2.0)),
    }


def generate_normal_transactions(
    profile: dict,
    start_date: pd.Timestamp,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate normal (legitimate) transactions for a merchant."""
    rows = []
    card_pool = [f"card_{rng.integers(10000, 59999)}" for _ in range(200)]

    for day in range(n_days):
        date = start_date + pd.Timedelta(days=day)

        # Add day-of-week seasonality (weekends = less B2B, more retail)
        weekday = date.weekday()
        volume_multiplier = 0.7 if weekday >= 5 else 1.0
        if profile["category"] in ("food_delivery", "fashion"):
            volume_multiplier = 1.3 if weekday >= 5 else 1.0

        n_txns = max(1, int(
            rng.poisson(profile["daily_volume"] * volume_multiplier)
        ))

        # Transactions clustered during business hours (9 AM - 9 PM IST)
        hours = rng.beta(2, 2, n_txns) * 12 + 9  # Peak around 3 PM
        minutes = rng.uniform(0, 60, n_txns)

        for i in range(n_txns):
            ts = date + pd.Timedelta(hours=float(hours[i]), minutes=float(minutes[i]))
            amount = max(1.0, rng.normal(profile["avg_amount"], profile["std_amount"]))
            card = rng.choice(card_pool)

            rows.append({
                "merchant_id": profile["merchant_id"],
                "timestamp": ts,
                "amount": round(amount, 2),
                "card_id": card,
                "is_fraud": False,
                "fraud_type": "none",
            })

    return pd.DataFrame(rows)


def inject_fraud(
    merchant_profiles: list[dict],
    start_date: pd.Timestamp,
    n_days: int,
    rng: np.random.Generator,
    n_incidents: int = 20,
) -> pd.DataFrame:
    """Inject fraud incidents across random merchants and time points."""
    fraud_dfs = []

    for _ in range(n_incidents):
        pattern = rng.choice(ALL_PATTERNS)
        profile = rng.choice(merchant_profiles)
        inject_day = rng.integers(0, n_days)
        inject_time = start_date + pd.Timedelta(days=int(inject_day))

        fraud_df = pattern.inject_fn(
            merchant_id=profile["merchant_id"],
            base_time=inject_time,
            rng=rng,
        )
        fraud_dfs.append(fraud_df)
        logger.info(
            f"Injected '{pattern.name}' at merchant {profile['merchant_id']} "
            f"on day {inject_day} ({len(fraud_df)} txns)"
        )

    return pd.concat(fraud_dfs, ignore_index=True) if fraud_dfs else pd.DataFrame()


def generate_dataset(config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Generate the full dataset: normal + fraud transactions."""
    cfg = config["data"]
    rng = np.random.default_rng(cfg["seed"])
    start_date = pd.Timestamp("2026-01-01")

    # Step 1: Create merchant profiles
    logger.info(f"Creating {cfg['n_merchants']} merchant profiles...")
    profiles = [
        generate_merchant_profile(f"merchant_{i:03d}", rng)
        for i in range(cfg["n_merchants"])
    ]

    # Step 2: Generate normal transactions
    logger.info("Generating normal transactions...")
    normal_dfs = []
    for profile in profiles:
        df = generate_normal_transactions(
            profile, start_date, cfg["time_span_days"], rng
        )
        normal_dfs.append(df)
    normal_df = pd.concat(normal_dfs, ignore_index=True)
    logger.info(f"Generated {len(normal_df):,} normal transactions")

    # Step 3: Inject fraud
    n_incidents = int(len(normal_df) * cfg["fraud_ratio"] / 15)  # ~15 txns per incident
    logger.info(f"Injecting {n_incidents} fraud incidents...")
    fraud_df = inject_fraud(profiles, start_date, cfg["time_span_days"], rng, n_incidents)
    logger.info(f"Generated {len(fraud_df):,} fraudulent transactions")

    # Step 4: Combine and sort
    full_df = pd.concat([normal_df, fraud_df], ignore_index=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df["txn_id"] = [f"txn_{i:07d}" for i in range(len(full_df))]

    logger.info(
        f"Final dataset: {len(full_df):,} transactions, "
        f"{full_df['is_fraud'].sum():,} fraud ({full_df['is_fraud'].mean():.1%})"
    )

    return full_df, profiles


def main():
    config = load_config()
    os.makedirs(config["data"].get("output_dir", "data/"), exist_ok=True)

    df, profiles = generate_dataset(config)

    # Save
    output_dir = config["data"]["output_dir"]
    df.to_parquet(f"{output_dir}/transactions.parquet", index=False)
    df.to_csv(f"{output_dir}/transactions.csv", index=False)
    pd.DataFrame(profiles).to_csv(f"{output_dir}/merchant_profiles.csv", index=False)

    logger.info(f"Saved to {output_dir}/")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total transactions:  {len(df):,}")
    print(f"Fraudulent:          {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")
    print(f"Merchants:           {df['merchant_id'].nunique()}")
    print(f"Date range:          {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nFraud breakdown:")
    for ftype, count in df[df["is_fraud"]].groupby("fraud_type").size().items():
        print(f"  {ftype:20s}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
