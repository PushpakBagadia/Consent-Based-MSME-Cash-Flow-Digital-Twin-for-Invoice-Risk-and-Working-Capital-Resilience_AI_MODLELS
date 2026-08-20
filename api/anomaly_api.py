"""
Model 3 - Anomaly & Volatility Detection API.

Two endpoints, since closed and open invoices use different feature sets
(features_closed.py vs features_open.py - see those files for why).

Run as its own server, its own port:
    python -m uvicorn api.anomaly_api:app --port 8002

The open-invoices endpoint calls Model 1's live API internally (same as
Model 2 does), so Model 1's server (port 8000) must be running too.

Invoke-RestMethod -Uri "http://127.0.0.1:8002/detect-anomalies/closed" -Method Get
Invoke-RestMethod -Uri "http://127.0.0.1:8002/detect-anomalies/open" -Method Get
"""
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from anomaly.features_closed import build_closed_invoice_features, FEATURE_COLUMNS as CLOSED_COLS
from anomaly.features_open import build_open_invoice_features, FEATURE_COLUMNS as OPEN_COLS
from anomaly.isolation_forest_detector import detect_anomalies
from anomaly.anomaly_explainer import add_anomaly_types, CLOSED_INVOICE_RULES, OPEN_INVOICE_RULES
from simulation.model1_client import load_model1_predictions

RAW_INVOICES_PATH = "data/raw/invoices.csv"

app = FastAPI(title="Model 3 - Anomaly Detection API")


class AnomalyResult(BaseModel):
    invoice_id: str
    anomaly_score: float
    anomaly_type: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/detect-anomalies/closed", response_model=list[AnomalyResult])
def detect_closed_anomalies(contamination: float = Query(0.05, ge=0.01, le=0.5)):
    """
    Flags unusual CLOSED invoices - real, already-observed payment behaviour
    (unusual amounts, unusual delay vs. that customer's own history).
    Returns only the flagged invoices, sorted most-anomalous first.
    """
    try:
        raw = pd.read_csv(RAW_INVOICES_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"{RAW_INVOICES_PATH} not found")

    features = build_closed_invoice_features(raw)
    result = detect_anomalies(features, CLOSED_COLS, contamination=contamination)
    result = add_anomaly_types(result, CLOSED_INVOICE_RULES)

    flagged = result[result["anomaly_flag"]]
    return [
        AnomalyResult(invoice_id=row.invoice_id, anomaly_score=row.anomaly_score, anomaly_type=row.anomaly_type)
        for row in flagged.itertuples(index=False)
    ]


@app.get("/detect-anomalies/open", response_model=list[AnomalyResult])
def detect_open_anomalies(
    contamination: float = Query(0.05, ge=0.01, le=0.5),
    model1_api_url: str = "http://127.0.0.1:8000/predict/open-invoices",
):
    """
    Flags unusual OPEN invoices - unusual amounts, or already past their
    own Model 1 P90 prediction (reuses the same signal Model 2's
    overdue-invoice handling uses).
    Returns only the flagged invoices, sorted most-anomalous first.
    """
    try:
        predictions = load_model1_predictions(api_url=model1_api_url)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not get predictions from Model 1's API: {e}",
        )

    try:
        raw = pd.read_csv(RAW_INVOICES_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"{RAW_INVOICES_PATH} not found")

    features = build_open_invoice_features(predictions, raw)
    result = detect_anomalies(features, OPEN_COLS, contamination=contamination)
    result = add_anomaly_types(result, OPEN_INVOICE_RULES)

    flagged = result[result["anomaly_flag"]]
    return [
        AnomalyResult(invoice_id=row.invoice_id, anomaly_score=row.anomaly_score, anomaly_type=row.anomaly_type)
        for row in flagged.itertuples(index=False)
    ]