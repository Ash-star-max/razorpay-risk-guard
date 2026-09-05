# 🛡️ RiskGuard — AI Fraud-Spike Detector for Razorpay Merchants

> Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager

A real-time fraud-spike detection system that monitors merchant transaction streams,
flags anomalous bursts (velocity spikes, coordinated fraud rings, off-hours surges),
and reports honest precision/recall with false-positive cost analysis.

## 🎯 Problem Statement

Indian merchants lose crores every year to payment fraud. Existing rule-based systems
are brittle — they either miss novel patterns or flood ops teams with false alarms,
blocking legitimate customers and losing revenue. RiskGuard uses a 3-model ML ensemble
to catch fraud spikes early, with a focus on minimizing false-positive cost.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Transaction     │────▶│  Feature     │────▶│  Detection      │
│  Stream (Synth)  │     │  Engineering │     │  Pipeline       │
└─────────────────┘     └──────────────┘     │  ┌───────────┐  │
                                              │  │ Z-Score   │  │
                                              │  │ Baseline  │  │
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
├── config/settings.yaml         # Thresholds, model params, cost matrix
├── src/
│   ├── data_generator/          # Synthetic transactions with injected fraud
│   ├── features/                # Velocity, aggregation, time-window features
│   ├── models/                  # Z-Score, Isolation Forest, Autoencoder, Ensemble
│   ├── evaluation/              # Honest metrics + false-positive cost analysis
│   ├── api/                     # FastAPI real-time scoring with audit trail
│   └── dashboard/               # Streamlit monitoring dashboard
├── tests/
├── docs/
├── requirements.txt
└── Dockerfile
```

## 🚀 Quick Start

```bash
git clone https://github.com/Ash-star-max/razorpay-risk-guard.git
cd razorpay-risk-guard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Generate data
python -m src.data_generator.generator

# Train + evaluate
python -m src.models.ensemble --train --evaluate

# Terminal 1 — API
uvicorn src.api.server:app --reload

# Terminal 2 — Dashboard
streamlit run src/dashboard/app.py
```

---

## 📊 Results (Held-out Test Set — 115,573 transactions)

| Metric              | Value      |
|---------------------|------------|
| Precision           | 0.3762     |
| Recall              | 0.6161     |
| F1 Score            | 0.4672     |
| AUC-PR              | 0.3643     |
| AUC-ROC             | **0.9227** |
| False Positive Rate | 0.0337     |
| False Negative Rate | 0.3839     |

### Confusion Matrix

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 108,102 ✅ | 3,775 ❌ (cost) |
| **Actual Fraud**  | 1,419 ❌ (risk) | 2,277 ✅ |

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

| Item                           | Value           |
|--------------------------------|-----------------|
| Fraud caught (2,277 × ₹5,000)  | +₹1,13,85,000   |
| False alarms (3,775 × ₹800)    | -₹30,20,000     |
| Missed fraud (1,419 × ₹5,000)  | -₹70,95,000     |
| **Net savings**                | **₹12,70,000**  |
| Without detector               | -₹1,84,80,000   |
| **Total improvement**          | **₹1,97,50,000** |

> RiskGuard saves ₹1.97 crore compared to no detection on this test period.

---

## 🔍 Honest Failure Analysis

**What works well:**
- Velocity spikes and amount anomalies — **100% detection**. Clear statistical signatures.
- Off-hours surge — **85.7%**. Temporal features catch most of these.

**Where it fails:**
- Abuse rings — **21.6% only**. Coordinated fraud spread across many cards over days looks statistically normal. Needs graph-based card-merchant network analysis to solve properly.
- Chargeback bursts — **26.1%**. Round-number amounts alone aren't distinctive enough without post-transaction chargeback labels.
- False positive rate **3.37%** — 3,775 legitimate transactions blocked. Threshold is tunable per merchant risk appetite but this is the current tradeoff.

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
    "timestamp": "2026-09-05T03:30:00"
  }'
```

**Response with full audit trail:**
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

**Interactive docs:** `http://localhost:8000/docs`

---

## 🛡️ Design Principles

1. **Every flag is explainable** — audit trail shows which models fired and why
2. **Bounded actions** — detector flags only, never acts autonomously
3. **Honest metrics** — we report false positives, failures, and limitations openly
4. **Graceful degradation** — if one model fails, ensemble continues with reduced confidence

---

## 🧑‍💻 Author

**Ashish Pal**
B.Tech CSE (AI & ML) — NIET Greater Noida | CGPA 8.47
Intern, Asset Performance — Hero Future Energies
[LinkedIn](https://linkedin.com/in/ashishpa) | [GitHub](https://github.com/Ash-star-max)

---

*Built for Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager*
