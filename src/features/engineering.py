"""
Feature engineering for fraud detection.

Converts raw transactions into ML-ready features using rolling windows,
velocity metrics, and merchant-level aggregations. Mirrors the approach
used in SCADA anomaly detection — compute statistical features over
sliding time windows to capture behavioral shifts.
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_off_hours"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    return df


def add_velocity_features(
    df: pd.DataFrame,
    windows_minutes: list[int] = [5, 15, 60, 1440],
) -> pd.DataFrame:
    """
    Compute transaction velocity features per merchant over rolling windows.

    For each window size, calculates:
    - Transaction count
    - Total amount
    - Unique cards
    - Mean amount
    - Std of amounts
    """
    df = df.sort_values(["merchant_id", "timestamp"]).copy()

    for window in windows_minutes:
        w = f"{window}min"
        prefix = f"w{window}"
        logger.info(f"Computing velocity features for {w} window...")

        grouped = df.set_index("timestamp").groupby("merchant_id")

        # Transaction count in window
        df[f"{prefix}_txn_count"] = (
            grouped["amount"]
            .rolling(w, min_periods=1)
            .count()
            .reset_index(level=0, drop=True)
            .values
        )

        # Sum of amounts in window
        df[f"{prefix}_amount_sum"] = (
            grouped["amount"]
            .rolling(w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .values
        )

        # Mean amount in window
        df[f"{prefix}_amount_mean"] = (
            grouped["amount"]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .values
        )

        # Std of amounts in window
        df[f"{prefix}_amount_std"] = (
            grouped["amount"]
            .rolling(w, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(level=0, drop=True)
            .values
        )

    return df


def add_card_diversity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute card diversity — how many unique cards per merchant in recent history.
    High card diversity in short windows = potential card testing.
    """
    df = df.sort_values(["merchant_id", "timestamp"]).copy()

    # Unique cards in last N transactions (approximation)
    for lookback in [10, 50]:
        col = f"unique_cards_last{lookback}"
        df[col] = (
            df.groupby("merchant_id")["card_id"]
            .rolling(lookback, min_periods=1)
            .apply(lambda x: x.nunique(), raw=False)
            .reset_index(level=0, drop=True)
            .values
        )

    return df


def add_merchant_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute merchant-level baseline stats (long-term averages).
    Deviation from baseline is a key fraud signal.
    """
    df = df.copy()

    merchant_stats = df.groupby("merchant_id")["amount"].agg(
        merchant_avg_amount="mean",
        merchant_std_amount="std",
        merchant_median_amount="median",
    ).fillna(0)

    df = df.merge(merchant_stats, on="merchant_id", how="left")

    # Z-score of current amount vs merchant baseline
    df["amount_zscore"] = (
        (df["amount"] - df["merchant_avg_amount"]) / df["merchant_std_amount"].clip(lower=1)
    )

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    windows_minutes: list[int] = [5, 15, 60, 1440],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full feature engineering pipeline.

    Returns:
        df: DataFrame with all features added
        feature_cols: List of column names to use as model input
    """
    logger.info("Starting feature engineering...")

    df = add_time_features(df)
    df = add_velocity_features(df, windows_minutes)
    df = add_merchant_baseline_features(df)

    # Define feature columns (everything the model sees)
    feature_cols = [
        "amount", "hour", "day_of_week", "is_weekend", "is_off_hours",
        "amount_zscore",
    ]

    # Add velocity features
    for w in windows_minutes:
        prefix = f"w{w}"
        feature_cols.extend([
            f"{prefix}_txn_count",
            f"{prefix}_amount_sum",
            f"{prefix}_amount_mean",
            f"{prefix}_amount_std",
        ])

    # Fill NaN and inf
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Built {len(feature_cols)} features for {len(df):,} transactions")
    return df, feature_cols
