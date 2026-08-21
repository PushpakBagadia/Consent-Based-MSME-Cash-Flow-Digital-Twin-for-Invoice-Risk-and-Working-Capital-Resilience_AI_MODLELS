"""
FastAPI serving layer for Model 1 (Payment Behaviour Prediction Engine).

Also exposes:

    Model 4:
        POST /extract/invoice

    Model 5:
        POST /explain/invoices

Run from the app/ directory:

    uv run uvicorn main:app --reload --port 8000

Or:

    uvicorn main:app --reload --port 8000
"""

import os
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from model1_inference import (
    Model1Artifacts,
    predict_payment_window,
)

from model5_shap import (
    Model5Artifacts,
    explain_invoice,
)
from model6_explanation import narrate_invoice

from ocr_extraction import extract_invoice
from api.cashflow_api import router as cashflow_router
from api.anomaly_api import router as anomaly_router
from api.risk_graph_api import router as risk_graph_router
from api.recommendation_ranker_api import router as recommendation_ranker_router

# ============================================================
# PATHS
# ============================================================

# Resolve paths relative to THIS file.
#
# This means the API behaves consistently regardless of where
# uvicorn is launched from.
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(
    os.environ.get(
        "MODEL1_MODEL_DIR",
        BASE_DIR / "models",
    )
)

RAW_INVOICES_PATH = Path(
    os.environ.get(
        "MODEL1_DATA_PATH",
        BASE_DIR / "data" / "raw" / "invoices.csv",
    )
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="MSME Cash-Flow AI - Model 1 API",
)

app.include_router(cashflow_router)
app.include_router(anomaly_router)
app.include_router(risk_graph_router)
app.include_router(recommendation_ranker_router)
# ============================================================
# GLOBAL MODEL ARTIFACTS
# ============================================================

# Model 1 artifacts.
#
# Loaded ONCE during startup.
artifacts: Model1Artifacts | None = None


# Model 5 SHAP artifacts.
#
# Uses the EXACT SAME Model1Artifacts instance.
shap_artifacts: Model5Artifacts | None = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def load_models():

    global artifacts
    global shap_artifacts

    # --------------------------------------------------------
    # Load Model 1
    # --------------------------------------------------------

    artifacts = Model1Artifacts(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # Load historical invoice data
    # --------------------------------------------------------

    raw = pd.read_csv(
        RAW_INVOICES_PATH
    )

    raw["issue_date"] = pd.to_datetime(
        raw["issue_date"]
    )

    # --------------------------------------------------------
    # Build customer history
    # --------------------------------------------------------

    closed_history = raw[
        raw["status"] == "closed"
    ].copy()

    artifacts.refresh_customer_stats(
        closed_history
    )

    # --------------------------------------------------------
    # Initialize Model 5
    #
    # IMPORTANT:
    #
    # We pass the SAME Model1Artifacts object.
    #
    # Therefore Model 5 uses:
    #   - same preprocessor
    #   - same RandomForest
    #   - same feature configuration
    #   - same customer statistics
    #
    # Nothing is retrained or reloaded.
    # --------------------------------------------------------

    shap_artifacts = Model5Artifacts(
        artifacts
    )


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class InvoiceInput(BaseModel):
    invoice_id: str
    cust_number: str
    sector: str
    invoice_amount: float
    payment_term_days: int
    issue_date: date


class PredictionOutput(BaseModel):
    invoice_id: str
    customer_id: str
    p10_payment_days: int
    p50_payment_days: int
    p90_payment_days: int
    confidence: Literal[
        "normal",
        "low",
    ]

class NarrationOutput(BaseModel):
    invoice_id: str
    text: str
    source: Literal["llm", "fallback"]


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": artifacts is not None,
        "shap_loaded": shap_artifacts is not None,
    }


# ============================================================
# MODEL 1
# ============================================================

@app.post(
    "/predict/invoices",
    response_model=list[PredictionOutput],
)
def predict_invoices(
    invoices: list[InvoiceInput],
):
    """
    Predict P10/P50/P90 days-to-payment for arbitrary invoices.

    This endpoint can be used for:
        - new invoices
        - corrected OCR invoices
        - batch invoices
        - live correction demo
    """

    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded yet",
        )

    df = pd.DataFrame(
        [
            inv.model_dump()
            for inv in invoices
        ]
    )

    df["issue_date"] = pd.to_datetime(
        df["issue_date"]
    )

    result = predict_payment_window(
        df,
        artifacts,
    )

    return [
        PredictionOutput(
            invoice_id=row.invoice_id,
            customer_id=row.cust_number,
            p10_payment_days=row.predicted_days_p10,
            p50_payment_days=row.predicted_days_p50,
            p90_payment_days=row.predicted_days_p90,
            confidence=row.confidence,
        )
        for row in result.itertuples()
    ]


# ============================================================
# MODEL 1 - OPEN INVOICES
# ============================================================

@app.get(
    "/predict/open-invoices",
    response_model=list[PredictionOutput],
)
def predict_open_invoices():
    """
    Predict every currently open/disputed_open invoice.

    Model 2 can call this endpoint to obtain the current
    outstanding receivables forecast.
    """

    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded yet",
        )

    raw = pd.read_csv(
        RAW_INVOICES_PATH
    )

    raw["issue_date"] = pd.to_datetime(
        raw["issue_date"]
    )

    open_invoices = raw[
        raw["status"].isin(
            [
                "open",
                "disputed_open",
            ]
        )
    ].copy()

    result = predict_payment_window(
        open_invoices,
        artifacts,
    )

    return [
        PredictionOutput(
            invoice_id=row.invoice_id,
            customer_id=row.cust_number,
            p10_payment_days=row.predicted_days_p10,
            p50_payment_days=row.predicted_days_p50,
            p90_payment_days=row.predicted_days_p90,
            confidence=row.confidence,
        )
        for row in result.itertuples()
    ]


# ============================================================
# REFRESH CUSTOMER HISTORY
# ============================================================

@app.post(
    "/admin/refresh-customer-stats"
)
def refresh_customer_stats():
    """
    Rebuild customer payment history after invoices are marked
    paid/closed.

    Model 5 automatically uses the refreshed Model1Artifacts
    because it references the same object.
    """

    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded yet",
        )

    raw = pd.read_csv(
        RAW_INVOICES_PATH
    )

    raw["issue_date"] = pd.to_datetime(
        raw["issue_date"]
    )

    closed_history = raw[
        raw["status"] == "closed"
    ].copy()

    artifacts.refresh_customer_stats(
        closed_history
    )

    return {
        "status": "refreshed",
        "customers": len(
            artifacts.customer_stats
        ),
    }


# ============================================================
# MODEL 5 - SHAP EXPLANATION
# ============================================================

@app.post(
    "/explain/invoices"
)
def explain_invoices(
    invoices: list[InvoiceInput],
):
    """
    Generate a complete numerical SHAP explanation for
    each invoice.

    The endpoint returns:

        invoice_id
        base_value
        predicted_value
        contributions[]

    Every contribution contains:

        feature
        value
        shap_value
        direction

    No top-N truncation is performed.
    """

    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded yet",
        )

    if shap_artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer not loaded yet",
        )

    # Convert Pydantic objects to DataFrame.
    df = pd.DataFrame(
        [
            inv.model_dump()
            for inv in invoices
        ]
    )

    df["issue_date"] = pd.to_datetime(
        df["issue_date"]
    )

    # Model 5 uses the SAME Model 1 artifacts.
    explanations = explain_invoice(
        df,
        shap_artifacts,
    )

    return explanations


# ============================================================
# MODEL 6 - LLM NARRATION OF SHAP EXPLANATIONS
# ============================================================

@app.post(
    "/narrate/invoices",
    response_model=list[NarrationOutput],
)
def narrate_invoices(
    invoices: list[InvoiceInput],
):
    """
    Runs Model 1 (confidence) + Model 5 (SHAP explanation) internally,
    then narrates each invoice's result in plain language via Model 6.

    Model 6 itself never computes anything - it only narrates the
    explanation dict it's handed here. This endpoint exists to wire
    that narrow scope up to real data in one call, rather than
    requiring the caller to hit /predict/invoices and /explain/invoices
    separately and stitch the results together themselves.

    Note: narrate_invoice() makes a real LLM call per invoice (with a
    timeout + one retry + deterministic fallback baked in). This is
    fine for the per-invoice/on-demand usage it's designed for, but
    isn't meant for large batch requests - keep invoice lists small.
    """

    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded yet",
        )

    if shap_artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer not loaded yet",
        )

    df = pd.DataFrame(
        [
            inv.model_dump()
            for inv in invoices
        ]
    )

    df["issue_date"] = pd.to_datetime(
        df["issue_date"]
    )

    # Model 1 - needed only for the confidence flag Model 6 uses to
    # soften language on low-confidence (new-customer) predictions.
    predictions = predict_payment_window(
        df,
        artifacts,
    )

    confidence_by_invoice = {
        row.invoice_id: row.confidence
        for row in predictions.itertuples()
    }

    # Model 5 - the actual explanation Model 6 narrates. Same call
    # /explain/invoices makes, same output shape (invoice_id,
    # base_value, predicted_value, contributions[]).
    explanations = explain_invoice(
        df,
        shap_artifacts,
    )

    results = []

    for explanation in explanations:

        confidence = confidence_by_invoice.get(
            explanation["invoice_id"]
        )

        narration = narrate_invoice(
            explanation,
            confidence,
        )

        results.append(
            NarrationOutput(
                invoice_id=explanation["invoice_id"],
                text=narration["text"],
                source=narration["source"],
            )
        )

    return results

# ============================================================
# MODEL 4 - OCR EXTRACTION
# ============================================================

@app.post(
    "/extract/invoice"
)
async def extract_invoice_endpoint(
    file: UploadFile,
):
    """
    Upload a PDF or image invoice.

    Returns extracted invoice fields and confidence scores.

    The frontend can allow the user to correct the extracted
    values and then send the corrected invoice to:

        /predict/invoices

    and/or:

        /explain/invoices
    """

    allowed_ext = {
        "pdf",
        "png",
        "jpg",
        "jpeg",
    }

    ext = (
        (file.filename or "")
        .lower()
        .rsplit(".", 1)[-1]
    )

    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: .{ext}",
        )

    file_bytes = await file.read()

    try:

        result = extract_invoice(
            file_bytes,
            file.filename,
        )

    except Exception as e:

        raise HTTPException(
            status_code=422,
            detail=f"extraction failed: {e}",
        )

    return result