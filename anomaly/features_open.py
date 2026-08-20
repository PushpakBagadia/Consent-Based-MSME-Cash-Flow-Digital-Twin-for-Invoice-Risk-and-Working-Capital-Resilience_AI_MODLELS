"""
Builds anomaly-detection features for OPEN invoices - these don't have an
actual delay_vs_due_date yet (that's the whole point, they haven't been
paid), so we can't reuse features_closed.py's approach directly.

Instead we reuse the exact same signal Model 2's overdue-invoice handling
already computes: how far past Model 1's OWN p90 prediction this invoice
already is. An invoice sitting well past its own pessimistic forecast is
inherently anomalous - Model 1 didn't expect this, so it's worth a second
look regardless of amount.
"""
import pandas as pd


FEATURE_COLUMNS = [
    "amount_zscore_vs_sector",
    "days_past_own_p90",       # 0 or negative = still within Model 1's expected window
    "days_since_issue",
    "is_cold_start",           # confidence == "low" from Model 1
]


def build_open_invoice_features(predictions: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    predictions: output of simulation/model1_client.py's load_model1_predictions()
        (invoice_id, invoice_amount, days_since_issue, p10/p50/p90_payment_days)
    raw: the full invoices dataframe, used only for sector lookup
        (predictions doesn't carry sector - not needed by Model 1 or Model 2 so far)
    """
    df = predictions.merge(raw[["invoice_id", "sector"]], on="invoice_id", how="left")

    sector_amount_stats = raw[raw["status"] == "closed"].groupby("sector")["invoice_amount"].agg(["mean", "std"])
    df = df.merge(
        sector_amount_stats.rename(columns={"mean": "sector_amt_mean", "std": "sector_amt_std"}),
        on="sector", how="left",
    )
    std_safe = df["sector_amt_std"].replace(0, 1).fillna(1)
    df["amount_zscore_vs_sector"] = ((df["invoice_amount"] - df["sector_amt_mean"]) / std_safe).fillna(0)

    df["days_past_own_p90"] = df["days_since_issue"] - df["p90_payment_days"]

    # confidence isn't in the DataFrame model1_client.py returns (it drops it
    # to match simulate_cashflow's exact expected columns) - if it's missing,
    # default to "not cold start" (0) rather than crash
    if "confidence" in df.columns:
        df["is_cold_start"] = (df["confidence"] == "low").astype(int)
    else:
        df["is_cold_start"] = 0

    return df[["invoice_id", "cust_number"] + FEATURE_COLUMNS] if "cust_number" in df.columns \
        else df[["invoice_id"] + FEATURE_COLUMNS]