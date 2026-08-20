"""
Calls Model 3's REAL, running FastAPI server (api/anomaly_api.py, started
with `python -m uvicorn api.anomaly_api:app --port 8002`) and returns its
open-invoice anomaly flags as a DataFrame Model 8 can join onto its graph.

Mirrors simulation/model1_client.py's shape: a thin bridge whose only job
is "call the real endpoint, hand back a clean DataFrame with the columns
the caller needs" - `build_risk_graph()` should never have to know this is
an HTTP call underneath.

`GET /detect-anomalies/open` (see api/anomaly_api.py's AnomalyResult model)
returns ONLY the flagged invoices - invoice_id, anomaly_score, anomaly_type.
Un-flagged invoices simply aren't in the response. Anything that later left-
joins this onto a full invoice list will get NaN / "not_flagged" for those,
which is the correct behaviour, not missing data.
"""
import pandas as pd
import requests

NOT_FLAGGED_LABEL = "not_flagged"


def load_anomaly_flags(
    api_url="http://127.0.0.1:8002/detect-anomalies/open",
    contamination=0.05,
    timeout=30,
):
    """
    Returns a DataFrame with columns: invoice_id, anomaly_score, anomaly_type
    - one row per invoice Model 3 flagged as anomalous. Empty (but correctly
    shaped) DataFrame if nothing was flagged, so callers can always safely
    .merge() on it without a None check.
    """
    response = requests.get(api_url, params={"contamination": contamination}, timeout=timeout)
    response.raise_for_status()  # fail loudly if the server errored, not silently
    flags = pd.DataFrame(response.json())

    if flags.empty:
        return pd.DataFrame(columns=["invoice_id", "anomaly_score", "anomaly_type"])

    return flags[["invoice_id", "anomaly_score", "anomaly_type"]]


def attach_anomaly_flags(invoice_df, anomaly_flags_df, id_col="invoice_id"):
    """
    Left-joins anomaly_score / anomaly_type onto invoice_df. Invoices Model 3
    didn't flag get anomaly_type=NOT_FLAGGED_LABEL and anomaly_score=NaN -
    NaN (not 0.0) because Isolation Forest scores aren't on a "0 = definitely
    fine" scale; we simply don't have a score for something that wasn't run
    through the flagged-path logic in anomaly_explainer.py.
    """
    merged = invoice_df.merge(anomaly_flags_df, on=id_col, how="left")
    merged["anomaly_type"] = merged["anomaly_type"].fillna(NOT_FLAGGED_LABEL)
    return merged