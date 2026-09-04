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
│   │   ├── __init__.py
│   │   ├── generator.py         # Normal + fraud pattern generation
│   │   └── fraud_patterns.py    # Fraud scenario definitions
│   ├── features/                # Feature engineering pipeline
│   │   ├── __init__.py
│   │   └── engineering.py       # Velocity, aggregation, time-window features
│   ├── models/                  # Detection models
│   │   ├── __init__.py
│   │   ├── baseline.py          # Statistical Z-score baseline
│   │   ├── isolation_forest.py  # Isolation Forest detector
│   │   ├── autoencoder.py       # Autoencoder anomaly detector
│   │   └── ensemble.py          # Ensemble scorer combining all models
│   ├── evaluation/              # Honest metrics & cost analysis
│   │   ├── __init__.py
│   │   ├── metrics.py           # Precision, recall, F1, PR curves
│   │   └── cost_analysis.py     # False-positive cost modeling
│   ├── api/                     # REST API for real-time scoring
│   │   ├── __init__.py
│   │   └── server.py            # FastAPI endpoints
│   └── dashboard/               # Monitoring UI
│       └── app.py               # Streamlit dashboard
├── tests/
│   ├── test_generator.py
│   ├── test_features.py
│   └── test_models.py
├── notebooks/
│   └── exploration.ipynb        # EDA and model experimentation
├── docs/
│   └── architecture.md          # Detailed architecture document
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── Makefile
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/razorpay-risk-guard.git
cd razorpay-risk-guard
pip install -r requirements.txt

# 2. Generate synthetic data
python -m src.data_generator.generator

# 3. Train models and evaluate
python -m src.models.ensemble --train --evaluate

# 4. Run the API
uvicorn src.api.server:app --reload

# 5. Launch dashboard
streamlit run src/dashboard/app.py
```

## 📊 Results

| Model            | Precision | Recall | F1    | FP Cost (₹/day) |
|------------------|-----------|--------|-------|------------------|
| Z-Score Baseline | —         | —      | —     | —                |
| Isolation Forest | —         | —      | —     | —                |
| Autoencoder      | —         | —      | —     | —                |
| **Ensemble**     | **—**     | **—**  | **—** | **—**            |

> *Fill after training. Be honest — report where it fails.*

## 🔍 Failure Handling

- **False positive example**: [Document a real case from test set]
- **Missed fraud example**: [Document a real case from test set]
- **Why**: [Analysis of why the model failed]
- **Mitigation**: [What you'd do with more time]

## 🧑‍💻 Author

**Ashish Pal**
B.Tech CSE (AI & ML) — NIET Greater Noida
[LinkedIn](https://linkedin.com/in/ashishpa)

---

*Built for Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager*
