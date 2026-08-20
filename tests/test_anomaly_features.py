"""
Unit tests for anomaly/features_closed.py and anomaly/features_open.py.

Run: python -m tests.test_anomaly_features
"""
import pandas as pd

from anomaly.features_closed import build_closed_invoice_features
from anomaly.features_open import build_open_invoice_features


def test_obvious_amount_outlier_gets_high_zscore():
    """A customer whose invoices are normally ~10,000 but has one invoice
    of 500,000 should get a large positive amount_zscore_vs_customer for
    that one invoice, and near-zero for the normal ones."""
    raw = pd.DataFrame({
        "invoice_id": ["A1", "A2", "A3", "A4"],
        "cust_number": ["C1", "C1", "C1", "C1"],
        "sector": ["retail", "retail", "retail", "retail"],
        "invoice_amount": [10000, 11000, 9500, 500000],  # last one is the outlier
        "delay_vs_due_date": [2, -1, 3, 5],
        "had_partial_payment_flag": [False, False, False, False],
        "is_big_ticket_spike": [False, False, False, True],
        "status": ["closed"] * 4,
    })
    features = build_closed_invoice_features(raw)
    outlier_row = features[features["invoice_id"] == "A4"].iloc[0]
    normal_row = features[features["invoice_id"] == "A1"].iloc[0]

    assert outlier_row["amount_zscore_vs_customer"] > 1.4, (
        f"expected outlier to have high z-score, got {outlier_row['amount_zscore_vs_customer']}"
    )
    assert abs(normal_row["amount_zscore_vs_customer"]) < outlier_row["amount_zscore_vs_customer"], (
        "normal invoice should have a much lower z-score than the outlier"
    )
    print("PASSED: test_obvious_amount_outlier_gets_high_zscore")


def test_single_invoice_customer_does_not_crash():
    """A customer with only ONE invoice has zero standard deviation in
    their own history - this should fall back safely to 0, not raise a
    divide-by-zero error or produce inf/NaN."""
    raw = pd.DataFrame({
        "invoice_id": ["B1"],
        "cust_number": ["C_ONLY_ONE"],
        "sector": ["retail"],
        "invoice_amount": [50000],
        "delay_vs_due_date": [3],
        "had_partial_payment_flag": [False],
        "is_big_ticket_spike": [False],
        "status": ["closed"],
    })
    features = build_closed_invoice_features(raw)
    row = features.iloc[0]
    assert row["amount_zscore_vs_customer"] == 0, (
        f"single-invoice customer should default to z-score 0, got {row['amount_zscore_vs_customer']}"
    )
    assert not pd.isna(row["amount_zscore_vs_customer"]), "should never be NaN"
    print("PASSED: test_single_invoice_customer_does_not_crash")


def test_days_past_own_p90_sign_is_correct():
    """An invoice issued 100 days ago with p90=60 should show
    days_past_own_p90 = 40 (positive = already past the worst case).
    An invoice issued 10 days ago with p90=60 should show a NEGATIVE
    value (still well within the expected window)."""
    predictions = pd.DataFrame({
        "invoice_id": ["X1", "X2"],
        "invoice_amount": [20000, 30000],
        "days_since_issue": [100, 10],
        "p10_payment_days": [30, 30],
        "p50_payment_days": [45, 45],
        "p90_payment_days": [60, 60],
    })
    raw = pd.DataFrame({
        "invoice_id": ["X1", "X2"],
        "sector": ["retail", "retail"],
        "invoice_amount": [20000, 30000],
        "status": ["open", "open"],
    })
    features = build_open_invoice_features(predictions, raw)
    overdue_row = features[features["invoice_id"] == "X1"].iloc[0]
    ontrack_row = features[features["invoice_id"] == "X2"].iloc[0]

    assert overdue_row["days_past_own_p90"] == 40, (
        f"expected 100-60=40, got {overdue_row['days_past_own_p90']}"
    )
    assert ontrack_row["days_past_own_p90"] == -50, (
        f"expected 10-60=-50, got {ontrack_row['days_past_own_p90']}"
    )
    print("PASSED: test_days_past_own_p90_sign_is_correct")


if __name__ == "__main__":
    test_obvious_amount_outlier_gets_high_zscore()
    test_single_invoice_customer_does_not_crash()
    test_days_past_own_p90_sign_is_correct()
    print("\nAll anomaly feature tests passed.")