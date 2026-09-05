# 🛡️ RiskGuard — AI Fraud-Spike Detector for Razorpay Merchants

> Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager

A real-time fraud-spike detection system that monitors merchant transaction streams,
flags anomalous bursts (sudden spikes in chargebacks, unusual velocity patterns,
coordinated fraud rings), and reports honest precision/recall with false-positive
cost analysis.

## 🎯 Problem Statement

Indian BFSI is seeing a rise in AI-enabled fraud while returns and chargebacks
quietly eat into merchant margins. Existing rule-based systems are brittle —
they either miss novel patterns or flood ops teams with false alarms.

RiskGuard uses statistical baselines + ML anomaly detection to catch fraud spikes
**early**, with a focus on minimizing false-positive cost (every false alert =
a legitimate transaction blocked = revenue lost).

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Transaction     │────▶│  Feature     │────▶│  Detection      │
│  Stream (Synth)  │     │  Engineering │     │  Pipeline       │
└─────────────────┘     └──────────────┘     │  ┌───────────┐  │
                                              │  │ Baseline  │  │
                                              │  │ (Z-Score) │  │
                                              │  ├───────────┤  │
                                              │  │ Isolation │  │
                                              │  │ Forest    │  │
                                              │  ├───────────┤  │
                                              │  │ Autoenc.  │  │
                                              │  └───────────┘  │
                                              └────────┬────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │  Risk Scoring   │
                                              │  + Audit Trail  │
                                              └────────┬────────┘
                                                       │
                                    ┌──────────────────┼──────────────────┐
                                    ▼                  ▼                  ▼
                             ┌────────────┐   ┌──────────────┐   ┌────────────┐
                             │ Alert API  │   │  Dashboard   │   │ Evaluation │
                             │ (FastAPI)  │   │ (Streamlit)  │   │ Reports    │
                             └────────────┘   └──────────────┘   └────────────┘
```

## 📂 Project Structure

```
razorpay-risk-guard/
├── config/
│   └── settings.yaml            # Thresholds, model params, cost matrix
├── src/
│   ├── data_generator/          # Synthetic transaction data with injected fraud
│   │   ├── generator.py         # Normal + fraud pattern generation
│   │   └── fraud_patterns.py    # 5 fraud scenario definitions
│   ├── features/                # Feature engineering pipeline
│   │   └── engineering.py       # Velocity, aggregation, time-window features
│   ├── models/                  # Detection models
│   │   ├── baseline.py          # Statistical Z-score baseline
│   │   ├── isolation_forest.py  # Isolation Forest detector
│   │   ├── autoencoder.py       # Autoencoder anomaly detector
│   │   └── ensemble.py          # Weighted ensemble scorer
│   ├── evaluation/              # Honest metrics & cost analysis
│   │   ├── metrics.py           # Precision, recall, F1, PR curves
│   │   └── cost_analysis.py     # False-positive cost modeling
│   ├── api/                     # REST API for real-time scoring
│   │   └── server.py            # FastAPI endpoints with audit trail
│   └── dashboard/               # Monitoring UI
│       └── app.py               # Streamlit dashboard
├── tests/
├── docs/
├── requirements.txt
├── Dockerfile
└── Makefile
```

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Ash-star-max/razorpay-risk-guard.git
cd razorpay-risk-guard
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Generate synthetic data
python -m src.data_generator.generator

# 3. Train models and evaluate
python -m src.models.ensemble --train --evaluate

# 4. Run the API (Terminal 1)
uvicorn src.api.server:app --reload

# 5. Launch dashboard (Terminal 2)
streamlit run src/dashboard/app.py
```

---

## 📊 Results (Held-out Test Set — 115,573 transactions)

| Metric             | Value      |
|--------------------|------------|
| Precision          | 0.4023     |
| Recall             | 0.6805     |
| F1 Score           | 0.5057     |
| AUC-PR             | 0.3861     |
| AUC-ROC            | **0.9291** |
| False Positive Rate| 0.0334     |
| False Negative Rate| 0.3195     |

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 108,141 ✅ | 3,736 ❌ (cost) |
| **Actual Fraud**  | 1,181 ❌ (risk) | 2,515 ✅ |

### Per Fraud-Type Detection Rate

| Fraud Type       | Detection Rate | Samples |
|------------------|---------------|---------|
| velocity_spike   | **100.0%**    | 1,110   |
| amount_anomaly   | **100.0%**    | 205     |
| off_hours_surge  | 85.7%         | 1,020   |
| chargeback_burst | 26.1%         | 708     |
| abuse_ring       | 21.6%         | 653     |

---

## 💰 Cost Analysis

| Item                          | Value          |
|-------------------------------|----------------|
| Fraud caught (2,515 × ₹5,000) | +₹1,25,75,000  |
| False alarms (3,736 × ₹800)   | -₹29,88,800    |
| Missed fraud (1,181 × ₹5,000) | -₹59,05,000    |
| **Net savings**               | **₹36,81,200** |
| Without detector              | -₹1,84,80,000  |
| **Total improvement**         | **₹2,21,61,200** |

> RiskGuard saves ₹2.2 crore compared to no detection on this test period.

---

## 🔍 Honest Failure Analysis

**What works well:**
- Velocity spikes and amount anomalies: 100% detection — these have clear statistical signatures
- Off-hours surge: 85.7% — temporal features catch most of these

**Where it fails:**
- Abuse rings: only 21.6% detected — coordinated fraud using many different cards over days looks like normal traffic. Needs graph-based features (card-merchant network analysis) to catch properly.
- Chargeback bursts: 26.1% — round-number amounts aren't distinctive enough in isolation. Needs post-transaction chargeback labels to train on.
- False positive rate of 3.34% is high — 3,736 legitimate transactions blocked per test period. The threshold can be tuned to trade recall for precision depending on merchant risk appetite.

**What I'd build next:**
- Graph neural network for abuse ring detection
- Online learning for concept drift adaptation
- Per-merchant threshold calibration
- Integration with actual Razorpay webhook events

---

## 🔌 API

**Score a transaction:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "txn_id": "txn_test_001",
    "merchant_id": "merchant_001",
    "amount": 50000.00,
    "card_id": "card_99999",
    "timestamp": "2026-09-04T03:30:00"
  }'
```

**Response includes full audit trail:**
```json
{
  "txn_id": "txn_test_001",
  "risk_score": 0.87,
  "risk_level": "critical",
  "is_flagged": true,
  "audit": {
    "baseline_score": 0.91,
    "isolation_forest_score": 0.83,
    "autoencoder_score": 0.85,
    "ensemble_score": 0.87,
    "threshold": 0.5,
    "decision_reason": "FLAGGED — triggers: statistical deviation, isolated behavior pattern, unusual reconstruction signature"
  }
}
```

**Interactive API docs:** `http://localhost:8000/docs`

---

## 🛡️ Design Principles

1. **Every flag is explainable** — audit trail shows which models fired and why
2. **Bounded actions** — detector flags only, never acts autonomously
3. **Honest metrics** — we report false positives, failures, and limitations
4. **Graceful degradation** — if one model fails, ensemble continues with reduced confidence

---

## 🧑‍💻 Author

**Ashish Pal**
B.Tech CSE (AI & ML) — NIET Greater Noida | CGPA 8.47
Intern, Asset Performance — Hero Future Energies
[LinkedIn](https://linkedin.com/in/ashishpa)

---

*Built for Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager*
