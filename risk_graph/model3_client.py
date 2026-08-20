"""
Client for Model 3's anomaly detection API - used by Model 8 to annotate
risk-graph invoice nodes with their anomaly_type, when one exists.

Only calls the OPEN-invoices endpoint (GET /detect-anomalies/open), since
Model 8's graph is built from open invoices (Model 1's predictions) - the
closed-invoice check isn't relevant to a forward-looking risk graph.
"""
import pandas as pd
import requests


def load_anomaly_flags(api_url="http://127.0.0.1:8002/detect-anomalies/open", timeout=30) -> pd.DataFrame:
    """
    Returns a DataFrame of invoice_id + anomaly_type, ONE ROW PER FLAGGED
    INVOICE ONLY - Model 3's endpoint only returns flagged invoices (see
    anomaly_api.py: `flagged = result[result["anomaly_flag"]]`), so an
    invoice's absence here means "not flagged", not "unknown".
    """
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not data:
        return pd.DataFrame(columns=["invoice_id", "anomaly_type"])
    df = pd.DataFrame(data)
    return df[["invoice_id", "anomaly_type"]]


def attach_anomaly_flags(invoice_ids_df: pd.DataFrame, anomaly_flags_df: pd.DataFrame) -> pd.DataFrame:
    """
    invoice_ids_df: DataFrame with at least an invoice_id column (the
    invoices currently in the risk graph's focus set).
    anomaly_flags_df: output of load_anomaly_flags().

    Returns invoice_ids_df with an anomaly_type column added - "normal"
    for any invoice Model 3 did NOT flag, since absence from that endpoint
    means "checked and not anomalous", not "not checked".
    """
    merged = invoice_ids_df.merge(anomaly_flags_df, on="invoice_id", how="left")
    merged["anomaly_type"] = merged["anomaly_type"].fillna("normal")
    return merged