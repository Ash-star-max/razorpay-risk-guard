"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
from src.features.engineering import build_feature_matrix, add_time_features


def _make_sample_df(n=100):
    """Create minimal transaction DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "merchant_id": [f"m_{i % 3}" for i in range(n)],
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "amount": rng.uniform(100, 5000, n),
        "card_id": [f"card_{i % 20}" for i in range(n)],
        "is_fraud": [False] * n,
        "fraud_type": ["none"] * n,
    })


def test_time_features():
    df = _make_sample_df()
    result = add_time_features(df)
    assert "hour" in result.columns
    assert "is_weekend" in result.columns
    assert result["hour"].between(0, 23).all()


def test_feature_matrix_shape():
    df = _make_sample_df(200)
    result, feature_cols = build_feature_matrix(df, windows_minutes=[5, 60])
    assert len(feature_cols) > 0
    assert not result[feature_cols].isnull().any().any(), "No NaN in features"
    assert not np.isinf(result[feature_cols].values).any(), "No inf in features"
