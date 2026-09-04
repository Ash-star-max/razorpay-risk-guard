"""
FastAPI server for real-time transaction risk scoring.

Endpoints:
    POST /score       — Score a single transaction
    POST /score/batch — Score a batch of transactions
    GET  /health      — Health check
    GET  /audit/{txn_id} — Retrieve audit trail for a scored transaction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import yaml

from src.models.ensemble import EnsembleDetector
from src.features.engineering import build_feature_matrix

app = FastAPI(
    title="RiskGuard API",
    description="Real-time fraud-spike detection for Razorpay merchants",
    version="0.1.0",
)

# Load model on startup
detector: EnsembleDetector | None = None
audit_store: dict = {}  # In-memory audit trail (use Redis in production)


@app.on_event("startup")
async def load_model():
    global detector
    try:
        detector = EnsembleDetector.load()
    except FileNotFoundError:
        raise RuntimeError("Model not found. Run: python -m src.models.ensemble --train")


# --- Request/Response schemas ---

class Transaction(BaseModel):
    txn_id: str = Field(..., example="txn_0000001")
    merchant_id: str = Field(..., example="merchant_001")
    amount: float = Field(..., gt=0, example=1500.0)
    card_id: str = Field(..., example="card_12345")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskResponse(BaseModel):
    txn_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str
    is_flagged: bool
    audit: dict


class BatchRequest(BaseModel):
    transactions: list[Transaction]


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": detector is not None}


@app.post("/score", response_model=RiskResponse)
async def score_transaction(txn: Transaction):
    """Score a single transaction and return risk assessment with audit trail."""
    if detector is None:
        raise HTTPException(503, "Model not loaded")

    # Build minimal feature vector
    # In production, this would pull merchant history from a feature store
    features = _extract_features(txn)
    scores = detector.predict_scores(features)

    ensemble_score = float(scores["ensemble_score"][0])
    risk_level = _score_to_level(ensemble_score)
    is_flagged = ensemble_score > detector.threshold

    audit = {
        "baseline_score": float(scores["baseline_score"][0]),
        "isolation_forest_score": float(scores["isolation_forest_score"][0]),
        "autoencoder_score": float(scores["autoencoder_score"][0]),
        "ensemble_score": ensemble_score,
        "threshold": detector.threshold,
        "decision_reason": _explain_decision(scores, detector.threshold),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Store audit trail
    audit_store[txn.txn_id] = audit

    return RiskResponse(
        txn_id=txn.txn_id,
        risk_score=ensemble_score,
        risk_level=risk_level,
        is_flagged=is_flagged,
        audit=audit,
    )


@app.get("/audit/{txn_id}")
async def get_audit(txn_id: str):
    """Retrieve the full audit trail for a previously scored transaction."""
    if txn_id not in audit_store:
        raise HTTPException(404, f"No audit record for {txn_id}")
    return audit_store[txn_id]


# --- Helpers ---

def _extract_features(txn: Transaction) -> np.ndarray:
    """Extract feature vector from a single transaction."""
    hour = txn.timestamp.hour
    dow = txn.timestamp.weekday()
    # Simplified features for real-time scoring
    # Full pipeline would pull rolling stats from a feature store
    features = np.array([[
        txn.amount,
        hour,
        dow,
        1 if dow >= 5 else 0,       # is_weekend
        1 if hour < 6 else 0,        # is_off_hours
        0,                            # amount_zscore (needs merchant history)
        1, txn.amount, txn.amount, 0, # w5 placeholders
        1, txn.amount, txn.amount, 0, # w15 placeholders
        1, txn.amount, txn.amount, 0, # w60 placeholders
        1, txn.amount, txn.amount, 0, # w1440 placeholders
    ]])
    return features


def _score_to_level(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.5:
        return "medium"
    elif score < 0.7:
        return "high"
    return "critical"


def _explain_decision(scores: dict, threshold: float) -> str:
    """Generate human-readable explanation of the decision."""
    ensemble = float(scores["ensemble_score"][0])
    parts = []

    if scores["baseline_score"][0] > 0.5:
        parts.append("statistical deviation from merchant baseline")
    if scores["isolation_forest_score"][0] > 0.5:
        parts.append("isolated behavior pattern detected")
    if scores["autoencoder_score"][0] > 0.5:
        parts.append("unusual reconstruction signature")

    if ensemble > threshold:
        reason = f"FLAGGED (score {ensemble:.3f} > threshold {threshold:.3f})"
        if parts:
            reason += f" — triggers: {', '.join(parts)}"
    else:
        reason = f"PASSED (score {ensemble:.3f} ≤ threshold {threshold:.3f})"

    return reason
