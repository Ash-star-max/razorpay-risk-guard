"""Tests for detection models."""

import numpy as np
from src.models.baseline import ZScoreDetector, BaselineConfig
from src.models.isolation_forest import IsolationForestDetector


def _make_features(n_normal=500, n_anomaly=20, n_features=10):
    """Create synthetic feature data with clear anomalies."""
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (n_normal, n_features))
    anomaly = rng.normal(5, 1, (n_anomaly, n_features))  # Shifted mean
    X = np.vstack([normal, anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)
    return X, y


def test_zscore_detector():
    X, y = _make_features()
    names = [f"f{i}" for i in range(X.shape[1])]
    detector = ZScoreDetector(BaselineConfig(z_threshold=3.0))
    detector.fit(X, names)
    scores = detector.predict_scores(X)

    assert scores.shape == (len(X),)
    assert scores.min() >= 0
    # Anomalies should generally have higher scores
    assert scores[-20:].mean() > scores[:500].mean()


def test_isolation_forest():
    X, y = _make_features()
    detector = IsolationForestDetector(n_estimators=50, contamination=0.05)
    detector.fit(X)
    scores = detector.predict_scores(X)

    assert scores.shape == (len(X),)
    assert 0 <= scores.min() and scores.max() <= 1
    assert scores[-20:].mean() > scores[:500].mean()
