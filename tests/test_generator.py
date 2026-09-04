"""Tests for synthetic data generator."""

import numpy as np
import pandas as pd
from src.data_generator.generator import generate_dataset, load_config


def test_dataset_generation():
    """Test that generated data has correct shape and labels."""
    config = load_config()
    config["data"]["n_merchants"] = 5
    config["data"]["time_span_days"] = 10

    df, profiles = generate_dataset(config)

    assert len(df) > 0, "Dataset should not be empty"
    assert "is_fraud" in df.columns, "Missing fraud label column"
    assert "txn_id" in df.columns, "Missing transaction ID"
    assert df["is_fraud"].sum() > 0, "Should contain fraud transactions"
    assert df["is_fraud"].sum() < len(df), "Should contain normal transactions"
    assert df["timestamp"].is_monotonic_increasing, "Should be sorted by time"


def test_fraud_types_present():
    """Verify multiple fraud types are injected."""
    config = load_config()
    config["data"]["n_merchants"] = 10
    config["data"]["time_span_days"] = 30

    df, _ = generate_dataset(config)
    fraud_types = df[df["is_fraud"]]["fraud_type"].unique()

    assert len(fraud_types) >= 2, f"Expected multiple fraud types, got {fraud_types}"


def test_merchant_profiles():
    """Test merchant profile generation."""
    config = load_config()
    config["data"]["n_merchants"] = 5
    config["data"]["time_span_days"] = 5

    _, profiles = generate_dataset(config)

    assert len(profiles) == 5
    for p in profiles:
        assert p["avg_amount"] > 0
        assert p["daily_volume"] > 0
