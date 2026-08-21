"""
FastAPI router for Model 3 (Anomaly & Volatility Detection).

Mounted into main.py the same way api/cashflow_api.py is:

    from api.anomaly_api import router as anomaly_router
    app.include_router(anomaly_router)

Design note: tune_parameters() fits 16 Isolation Forests (4 z_thresholds
x 4 contaminations) against the full invoice history. That's fine to do
once, but far too slow to repeat on every request - so it runs on the
router's own startup event (merged into main.py's app startup when
included) and the fitted result is cached in-process. Hit
POST /admin/refresh-anomaly-model to recompute after the CSV changes,
same pattern as /admin/refresh-customer-stats does for Model 1.
"""

import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from model3_real_data import (
    load_invoices_csv,
    compute_zscore_features,
    run_isolation_forest,
    combine_flags,
    label_anomaly,
    build_output_json,
    evaluate_against_ground_truth,
    tune_parameters,
)


# ============================================================
# PATHS
# ============================================================

# api/anomaly_api.py -> parent is api/, parent.parent is the app/ root,
# same root main.py resolves BASE_DIR from.
BASE_DIR = Path(__file__).resolve().parent

# Falls back to MODEL1_DATA_PATH so this points at the same invoices.csv
# as Model 1 by default, without requiring a second env var to be set.
RAW_INVOICES_PATH = Path(
    os.environ.get(
        "MODEL3_DATA_PATH",
        os.environ.get(
            "MODEL1_DATA_PATH",
            BASE_DIR / "data" / "raw" / "invoices.csv",
        ),
    )
)

Z_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]
CONTAMINATIONS = [0.02, 0.03, 0.05, 0.08]


router = APIRouter()


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class AnomalyOutput(BaseModel):
    transaction_id: str
    business_id: str
    date: str
    amount: float
    category: str
    counterparty: str
    z_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    flag_level: Literal[
        "normal",
        "possible_anomaly",
        "high_confidence_anomaly",
    ]
    label: str


class AnomalyMetrics(BaseModel):
    z_threshold: float
    contamination: float
    precision: float
    recall: float
    f1: float
    flagged_count: int
    ground_truth_count: int


class RefreshResponse(BaseModel):
    status: str
    z_threshold: float
    contamination: float
    metrics: AnomalyMetrics
    n_records: int


# ============================================================
# CACHED MODEL STATE
#
# Populated once on router startup. Replaced wholesale by
# /admin/refresh-anomaly-model. Never mutated in place, so a request
# reading _cache mid-refresh always sees one consistent snapshot.
# ============================================================

_cache: dict = {
    "df": None,
    "best_z": None,
    "best_cont": None,
    "metrics": None,
}


def _fit_anomaly_model(data_path: Path):
    """
    Loads invoices, tunes (z_threshold, contamination) against the real
    is_big_ticket_spike ground truth, then fits the final pipeline with
    the best combo. Same sequence as model3_real_data.run_pipeline(),
    minus the console printing.
    """

    df_base = load_invoices_csv(str(data_path))

    if df_base.empty:
        raise ValueError(f"No invoices found at {data_path}")

    tuning_results = tune_parameters(
        df_base,
        Z_THRESHOLDS,
        CONTAMINATIONS,
    )

    best = tuning_results.iloc[0]
    best_z = float(best["z_threshold"])
    best_cont = float(best["contamination"])

    df = compute_zscore_features(df_base, z_threshold=best_z)
    df = run_isolation_forest(df, contamination=best_cont)
    df["flag_level"] = df.apply(combine_flags, axis=1)
    df["label"] = df.apply(label_anomaly, axis=1)

    metrics = evaluate_against_ground_truth(df)
    metrics["z_threshold"] = best_z
    metrics["contamination"] = best_cont

    return df, best_z, best_cont, metrics


# ============================================================
# STARTUP
#
# FastAPI merges a router's on_startup handlers into the parent app's
# startup sequence when app.include_router(router) is called, so this
# runs automatically alongside main.py's load_models().
# ============================================================

@router.on_event("startup")
def load_anomaly_model():

    global _cache

    try:
        df, best_z, best_cont, metrics = _fit_anomaly_model(
            RAW_INVOICES_PATH
        )
    except FileNotFoundError:
        # Don't crash app startup if the CSV isn't in place yet -
        # endpoints below 503 with a clear message until a manual
        # refresh succeeds.
        return

    _cache = {
        "df": df,
        "best_z": best_z,
        "best_cont": best_cont,
        "metrics": metrics,
    }


def _require_cache():

    if _cache["df"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model 3 hasn't loaded data yet (checked "
                f"{RAW_INVOICES_PATH}). Confirm the CSV path and call "
                f"POST /admin/refresh-anomaly-model."
            ),
        )

    return _cache["df"]


# ============================================================
# MODEL 3 - ANOMALY DETECTION
# ============================================================

@router.get(
    "/predict/anomalies",
    response_model=list[AnomalyOutput],
)
def predict_anomalies(
    flagged_only: bool = Query(
        False,
        description="If true, only return non-normal rows",
    ),
):
    """
    Return every invoice annotated with its z-score / Isolation Forest
    flag_level, using the tuned parameters cached at startup.
    """

    df = _require_cache()
    out = build_output_json(df)

    if flagged_only:
        out = [r for r in out if r["flag_level"] != "normal"]

    return out


@router.get(
    "/predict/anomalies/flagged",
    response_model=list[AnomalyOutput],
)
def predict_flagged_anomalies():
    return predict_anomalies(flagged_only=True)


@router.get(
    "/predict/anomalies/metrics",
    response_model=AnomalyMetrics,
)
def predict_anomaly_metrics():
    """
    Precision/recall/f1 of the current cached model against the real
    is_big_ticket_spike ground-truth column.
    """

    _require_cache()
    return _cache["metrics"]


# ============================================================
# REFRESH ANOMALY MODEL
# ============================================================

@router.post(
    "/admin/refresh-anomaly-model",
    response_model=RefreshResponse,
)
def refresh_anomaly_model(
    data_path: Optional[str] = Query(
        None,
        description="Override CSV path for this refresh only",
    ),
):
    """
    Re-run the tuning grid search and refit against the current CSV.

    Call this after invoices are added/updated, same way
    /admin/refresh-customer-stats refreshes Model 1's customer history.
    """

    global _cache

    path = Path(data_path) if data_path else RAW_INVOICES_PATH

    try:
        df, best_z, best_cont, metrics = _fit_anomaly_model(path)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"could not find {path}: {e}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    _cache = {
        "df": df,
        "best_z": best_z,
        "best_cont": best_cont,
        "metrics": metrics,
    }

    return {
        "status": "refreshed",
        "z_threshold": best_z,
        "contamination": best_cont,
        "metrics": metrics,
        "n_records": len(df),
    }