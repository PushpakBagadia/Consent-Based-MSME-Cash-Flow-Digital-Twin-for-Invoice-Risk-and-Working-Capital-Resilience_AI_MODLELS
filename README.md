# AI Models & Intelligence Layer  
## MSME Cash-Flow Digital Twin

**Consent-Based Invoice Risk, Liquidity Forecasting & Working-Capital Resilience Platform**

This repository contains the complete specification and implementation of the **AI/ML Intelligence Layer** for the MSME Cash-Flow Digital Twin platform, built for the **Smart India Hackathon**.

The financial engine (cash balances, due dates, scenario math) remains fully deterministic and auditable.  
AI is layered on top **only** for:

- **Prediction** — When will this invoice actually be paid?
- **Detection** — What looks unusual?
- **Explanation** — Why is the model saying this?
- **Narration** — Turning numbers into plain language

AI never replaces core accounting logic. This clean separation is a deliberate architectural choice for explainability and audit-safety.

---

## Architecture Overview

```
Raw data (invoices, receivables, expenses, payment history)
        ↓
OCR + Validation                          (Model 4)
        ↓
Payment Behaviour Prediction              (Model 1)
        ↓
Probabilistic Cash-Flow Simulation        (Model 2)
        ↓
Risk & Anomaly Detection                  (Model 3)
        ↓
SHAP Explainability                       (Model 5)
        ↓
LLM Narration & Conversational Layer      (Model 6)
        ↓
Non-Debt-First Recommendation Ranker      (Model 7)
        ↓
Causal Risk Graph Builder                 (Model 8)
        ↓
Dashboard
```

| AI Layer                        | Responsibility                                      | Must NOT Do                     |
|--------------------------------|-----------------------------------------------------|---------------------------------|
| Payment Prediction (XGBoost)   | Predict payment date distribution per invoice      | Decide loan approval            |
| Monte Carlo Simulation         | Turn distributions into cash-balance forecast      | Replace core cash formula       |
| Anomaly Detection              | Flag unusual expense/payment spikes                | Auto-block transactions         |
| OCR + Confidence Scoring       | Extract invoice fields + flag low-confidence       | Auto-correct without review     |
| SHAP Explainability            | Show which features drove a risk prediction        | Generate the prediction itself  |
| LLM (Groq / Llama 3.3 70B)     | Narrate computed numbers + answer Q&A              | Perform financial arithmetic    |
| Recommendation Ranker          | Transparently rank recovery options                | Opaque / autonomous decisioning |

---

## Models

### Model 1 — Payment Behaviour Prediction Engine
**Foundational model.** Every downstream module depends on it.

Predicts *when* each invoice will actually be paid as a probability distribution (P10 / P50 / P90) instead of trusting the stated “Net 30”.

- **Type**: XGBoost / LightGBM regressor with quantile objective (pinball loss)
- **Key features**: Historical delay & variance, invoice amount, seasonality, sector, recent payment trend, contractual term vs actual deviation
- **Cold-start**: Falls back to sector-average priors + low-confidence flag
- **Output**: Per-invoice payment date range (P10, P50, P90) + confidence flag

### Model 2 — Probabilistic Cash-Flow Simulation (Monte Carlo)
Converts per-invoice distributions into a full optimistic / expected / pessimistic cash-balance forecast and probability-based “Days to Liquidity Breach”.

- 2,000–5,000 vectorized draws
- Same engine powers baseline forecast **and** every what-if scenario
- Extremely lightweight (runs in well under a second)

### Model 3 — Anomaly & Volatility Detection
Flags unusual expenses, payments, or sudden behaviour shifts that a static dashboard would miss.

- Isolation Forest (unsupervised) + rolling z-score baseline
- Trained per-business on its own history (no labels required)

### Model 4 — OCR Extraction & Confidence Scoring
Extracts structured fields from uploaded PDFs / images and surfaces low-confidence values for human review.

- PaddleOCR (preferred) or Tesseract + PyMuPDF for native PDFs
- Per-field confidence score → “needs verification” workflow
- Enables the “correctable financial AI” USP

### Model 5 — Explainability Engine (SHAP)
Produces ranked feature contributions for every risk prediction.

- SHAP TreeExplainer on the XGBoost model
- Structured output is fed to the LLM — the LLM never invents attributions

### Model 6 — LLM Narration & Conversational Layer
Turns structured model outputs into plain-language explanations and answers owner questions.

- **Model**: Llama 3.3 70B via Groq API
- **Strict role boundary**: LLM only narrates pre-computed numbers. It never performs arithmetic.
- System prompt explicitly forbids generating or altering numbers

### Model 7 — Non-Debt-First Recommendation Ranker
Transparently ranks recovery options (supplier extension, early payment, invoice financing, etc.) by cost, recovery time, and liquidity impact.

- Pure rule-based weighted multi-criteria scoring (no ML)
- Fully explainable by design — every ranked option shows its inputs in the UI

### Model 8 — Causal Risk Graph Builder
Builds a directed causal graph:  
`Customer → Delayed Payment → Cash Buffer Breach → Downstream Obligation at Risk`

- Deterministic graph construction (networkx optional on backend)
- Rendered with React Flow / D3 / Recharts on the frontend
- Most memorable visual for jury demos

---

## Tech Stack

| Layer              | Technology                          | Used By                  |
|--------------------|-------------------------------------|--------------------------|
| Core ML            | XGBoost / LightGBM (quantile)      | Model 1                  |
| Simulation         | NumPy vectorized Monte Carlo       | Model 2                  |
| Anomaly Detection  | scikit-learn IsolationForest       | Model 3                  |
| OCR                | PaddleOCR / Tesseract + PyMuPDF    | Model 4                  |
| Explainability     | SHAP TreeExplainer                 | Model 5                  |
| LLM                | Groq API – Llama 3.3 70B           | Model 6                  |
| Ranking            | Plain Python / pandas              | Model 7                  |
| Graph              | networkx + React Flow / D3         | Model 8                  |
| Serving            | FastAPI + Pydantic                 | All models               |
| Storage            | PostgreSQL                         | Training data, audit log |
| Async (optional)   | Redis + Celery                     | Background recompute     |

**Environment**
- Python 3.10+
- No GPU required (tree-based models + API-hosted LLM)
- Groq free tier is sufficient for hackathon demo volume

```bash
pip install xgboost lightgbm scikit-learn shap numpy pandas \
            paddleocr pymupdf groq fastapi pydantic networkx
```

---

##Build Order (Hackathon Timeline)

| Priority       | Model                              | Reason                                      |
|----------------|------------------------------------|---------------------------------------------|
| **P0 **   | Sample dataset + Model 1          | Everything downstream depends on this       |
| **P0 **   | Model 2 (Monte Carlo)             | Core forecast & scenario engine             |
| **P1 **   | Model 7 (Recommendation Ranker)   | Simple, high demo value                     |
| **P1**    | Model 5 (SHAP)                    | Near-instant once Model 1 exists            |
| **P1**    | Model 6 (LLM Narration)           | Wraps SHAP + forecast into plain language   |
| **P2**    | Model 4 (OCR + Confidence)        | Needed for the “live correction” demo moment|
| **P2**    | Model 8 (Causal Risk Graph)       | Visualization layer                         |
| **P3**    | Model 3 (Anomaly Detection)       | Nice-to-have, not essential for core demo   |

> Generate a synthetic dataset of 200–500 invoices across 10–20 customers **first** — every other model depends on it.

---

## AI Unique Selling Points (USPs)

1. **Probabilistic forecasting, not a single guessed number**  
   Optimistic / expected / pessimistic bands come from a real model distribution, not a hardcoded ±20% rule.

2. **Causal risk graph instead of a risk score**  
   Judges remember a visual causal chain far longer than “82/100”.

3. **SHAP-linked explanations, not templated text**  
   Every explanation is traceable to real feature contributions.

4. **Live correction → live recompute**  
   Edit one OCR-imported value on stage and watch forecast, risk graph, and recommendations update in real time.

5. **Recovery ranked by time-to-safe-cash, not just cost**  
   Time-to-safe-position is a first-class metric.

6. **Confidence-aware AI, not false certainty**  
   Low-confidence OCR fields and cold-start predictions are explicitly flagged.

7. **Clean separation of arithmetic vs. AI**  
   Financial engine does the math. ML predicts & detects. LLM only narrates.  
   This architecture pre-empts the “is this just an LLM wrapper?” question.

---

## Suggested Project Structure

```
ai-intelligence-layer/
├── models/
│   ├── payment_prediction/      # Model 1
│   ├── monte_carlo/             # Model 2
│   ├── anomaly_detection/       # Model 3
│   ├── ocr/                     # Model 4
│   ├── shap_explainer/          # Model 5
│   ├── llm_narration/           # Model 6
│   ├── recommendation_ranker/   # Model 7
│   └── causal_risk_graph/       # Model 8
├── data/
│   └── synthetic/               
├── api/                         # FastAPI endpoints
├── notebooks/                   # Exploration & training notebooks
├── requirements.txt
└── README.md
```

---

## License

This project is developed for the **Smart India Hackathon**.  
All rights reserved by the team.

---

**Prepared for**: AI/ML Development Track  
**Role**: AI Developer
```

The `README.md` has been created at:
