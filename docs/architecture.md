# Architecture — RiskGuard Fraud Spike Detector

## System Overview

RiskGuard is a real-time fraud-spike detection system for Razorpay merchants.
It monitors transaction streams, detects anomalous patterns using a three-model
ensemble, and produces explainable risk scores with full audit trails.

## Design Principles

1. **Defense-only**: No offensive capabilities. Detects and flags — never acts autonomously.
2. **Explainable**: Every flag comes with an audit trail showing which models fired and why.
3. **Honest evaluation**: We report precision, recall, false-positive cost, and failure cases.
4. **Graceful degradation**: If one model fails, the ensemble still functions with reduced confidence.

## Data Pipeline

```
Raw Transactions
       │
       ▼
┌──────────────┐
│ Time Features │  hour, day_of_week, is_weekend, is_off_hours
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Velocity Features │  txn_count, amount_sum, amount_mean, amount_std
│ (4 time windows)  │  per 5min, 15min, 1hr, 1day windows
└──────┬───────────┘
       │
       ▼
┌───────────────────┐
│ Merchant Baseline  │  amount_zscore vs merchant's historical average
└──────┬────────────┘
       │
       ▼
   Feature Matrix (18 features per transaction)
```

## Detection Models

### 1. Z-Score Baseline (weight: 0.2)
- Statistical approach: flags transactions whose features exceed 3σ from the mean
- Fast, interpretable, no training required beyond computing statistics
- Weakness: assumes normal distribution, misses complex patterns

### 2. Isolation Forest (weight: 0.4)
- Ensemble of random trees that isolates anomalies
- No distributional assumptions — works on any feature space
- Good at detecting multivariate anomalies (combinations of unusual features)
- Weakness: static model, doesn't adapt to concept drift

### 3. Autoencoder (weight: 0.4)
- Neural network trained to reconstruct normal transactions
- High reconstruction error = never-seen-before pattern = anomaly
- Best at detecting novel fraud types not in training data
- Weakness: needs sufficient normal data, training is slower

### Ensemble Strategy
- Weighted average of normalized scores from all three models
- Final threshold optimized using cost-sensitive analysis (see below)
- Any single model exceeding 0.7 triggers a "high confidence" flag regardless of ensemble

## Cost-Sensitive Evaluation

The key insight: not all errors are equal.

| Error Type      | Real-World Impact                        | Cost (₹)  |
|-----------------|------------------------------------------|-----------|
| False Positive  | Legitimate customer blocked, revenue lost | 800       |
| False Negative  | Fraud slips through, direct loss          | 5,000     |
| True Positive   | Fraud caught, loss prevented              | +5,000    |

We optimize the detection threshold to minimize **total cost**, not just maximize F1.

## API Design

```
POST /score
  Input:  { txn_id, merchant_id, amount, card_id, timestamp }
  Output: { risk_score, risk_level, is_flagged, audit: {...} }

GET /audit/{txn_id}
  Output: Full decision audit trail with per-model scores

GET /health
  Output: System health status
```

## Failure Modes (Documented Honestly)

1. **Cold-start merchants**: New merchants have no baseline → high false-positive rate initially
2. **Concept drift**: Fraud patterns evolve; model needs periodic retraining
3. **Feature store latency**: Real-time velocity features depend on recent history availability
4. **Adversarial adaptation**: Sophisticated attackers can learn to stay below thresholds

## What I'd Build Next (Given More Time)

- Online learning for concept drift adaptation
- Graph-based features (card-merchant network analysis) for abuse ring detection
- A/B testing framework for threshold optimization in production
- Integration with Razorpay's actual webhook/event system
