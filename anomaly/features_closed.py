"""
Builds anomaly-detection features for CLOSED invoices (the ones we have a
real, actual outcome for: days_to_payment, delay_vs_due_date).

Design choice: every feature is a DEVIATION (from the customer's own
history, or from the sector average), not a raw value. A raw invoice_amount
of 5,00,000 might be completely normal for one customer and wildly unusual
for another - what we actually want to flag is BEHAVIOR that's unusual,
not customers who happen to be big. Same reasoning the audit already
established for Model 1's features.
"""
import pandas as pd


FEATURE_COLUMNS = [
    "amount_zscore_vs_customer",
    "amount_zscore_vs_sector",
    "delay_zscore_vs_customer",
    "delay_days",
    "had_partial_payment_flag",
    "is_big_ticket_spike",
]


def build_closed_invoice_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    raw: the full invoices dataframe (data/raw/invoices.csv).
    Returns a DataFrame with invoice_id + the anomaly-detection features,
    for status == 'closed' invoices only.
    """
    closed = raw[raw["status"] == "closed"].copy()

    # --- customer-level baselines (mean/std of THIS customer's own history) ---
    cust_amount_stats = closed.groupby("cust_number")["invoice_amount"].agg(["mean", "std"])
    cust_delay_stats = closed.groupby("cust_number")["delay_vs_due_date"].agg(["mean", "std"])

    closed = closed.merge(
        cust_amount_stats.rename(columns={"mean": "cust_amt_mean", "std": "cust_amt_std"}),
        on="cust_number", how="left",
    )
    closed = closed.merge(
        cust_delay_stats.rename(columns={"mean": "cust_delay_mean", "std": "cust_delay_std"}),
        on="cust_number", how="left",
    )

    # --- sector-level baselines ---
    sector_amount_stats = closed.groupby("sector")["invoice_amount"].agg(["mean", "std"])
    closed = closed.merge(
        sector_amount_stats.rename(columns={"mean": "sector_amt_mean", "std": "sector_amt_std"}),
        on="sector", how="left",
    )

    # --- z-scores: (value - baseline_mean) / baseline_std ---
    # std of 0 or NaN (customer/sector has only 1 invoice, or all identical
    # amounts) would divide-by-zero - fall back to 0 (not anomalous, since
    # we have no variation to compare against) instead of crashing/inf.
    def safe_zscore(value, mean, std):
        std_safe = std.replace(0, pd.NA).fillna(value.std() if value.std() > 0 else 1)
        z = (value - mean) / std_safe
        return z.fillna(0)

    closed["amount_zscore_vs_customer"] = safe_zscore(
        closed["invoice_amount"], closed["cust_amt_mean"], closed["cust_amt_std"]
    )
    closed["amount_zscore_vs_sector"] = safe_zscore(
        closed["invoice_amount"], closed["sector_amt_mean"], closed["sector_amt_std"]
    )
    closed["delay_zscore_vs_customer"] = safe_zscore(
        closed["delay_vs_due_date"], closed["cust_delay_mean"], closed["cust_delay_std"]
    )
    closed["delay_days"] = closed["delay_vs_due_date"]

    return closed[["invoice_id", "cust_number"] + FEATURE_COLUMNS]