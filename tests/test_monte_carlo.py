"""
Unit tests for simulation/monte_carlo.py (v4 - backlog split).

Run: python -m tests.test_monte_carlo
"""
import pandas as pd

from Monte_Carlo.monte_carlo import simulate_cashflow, split_backlog


def test_single_invoice_pays_within_its_window():
    """A single FORWARD-looking invoice with p10=44/p50=52/p90=68 should
    show ~0% paid before day 44, and fully paid by day 80."""
    invoice = pd.DataFrame({
        "invoice_id": ["INV1"], "invoice_amount": [100000.0],
        "days_since_issue": [0],
        "p10_payment_days": [44], "p50_payment_days": [52], "p90_payment_days": [68],
    })
    forecast, _ = simulate_cashflow(
        invoice, opening_cash=0, daily_expense=0, horizon_days=90, n_sims=20000
    )
    cash_at_day = lambda d: forecast.loc[forecast["day"] == d, "cash_p50"].iloc[0]

    assert cash_at_day(30) == 0, "well before p10, median cash should still be 0"
    assert cash_at_day(80) == 100000, "well after p90, invoice should be fully paid"
    print("PASSED: test_single_invoice_pays_within_its_window")


def test_expenses_reduce_cash_with_no_inflow():
    """With zero invoices, cash should just drain by daily_expense each day."""
    empty = pd.DataFrame({
        "invoice_id": [], "invoice_amount": [], "days_since_issue": [],
        "p10_payment_days": [], "p50_payment_days": [], "p90_payment_days": [],
    })
    forecast, _ = simulate_cashflow(
        empty, opening_cash=100000, daily_expense=1000, horizon_days=10, n_sims=100
    )
    expected_day5 = 100000 - 1000 * 5
    actual_day5 = forecast.loc[forecast["day"] == 5, "cash_p50"].iloc[0]
    assert actual_day5 == expected_day5, f"expected {expected_day5}, got {actual_day5}"
    print("PASSED: test_expenses_reduce_cash_with_no_inflow")


def test_split_backlog_uses_p50_threshold():
    """An invoice past its own P50 (median) goes to backlog, even if it
    hasn't reached P90 yet. An invoice before its P50 stays forward."""
    predictions = pd.DataFrame({
        "invoice_id": ["PAST_P50", "BEFORE_P50"],
        "invoice_amount": [100000.0, 50000.0],
        "days_since_issue": [80, 40],  # PAST_P50: past p50=70 but not p90=90 | BEFORE_P50: before p50=70
        "p10_payment_days": [50, 50], "p50_payment_days": [70, 70], "p90_payment_days": [90, 90],
    })
    backlog, forward = split_backlog(predictions)

    assert list(backlog["invoice_id"]) == ["PAST_P50"], f"got {list(backlog['invoice_id'])}"
    assert list(forward["invoice_id"]) == ["BEFORE_P50"], f"got {list(forward['invoice_id'])}"
    print("PASSED: test_split_backlog_uses_p50_threshold")


def test_backlog_invoice_excluded_from_forecast():
    """A backlog invoice should contribute NOTHING to the day-by-day
    forecast - this is the actual bug fix: previously, an invoice like
    this would get clipped onto day 0 and inflate cash unrealistically."""
    backlog_invoice = pd.DataFrame({
        "invoice_id": ["OLD1"], "invoice_amount": [100000.0],
        "days_since_issue": [80],  # past p50=70, still under p90=90
        "p10_payment_days": [50], "p50_payment_days": [70], "p90_payment_days": [90],
    })
    forecast, summary = simulate_cashflow(
        backlog_invoice, opening_cash=5000, daily_expense=0, horizon_days=90, n_sims=5000
    )
    day0_cash = forecast.loc[forecast["day"] == 0, "cash_p50"].iloc[0]
    day90_cash = forecast.loc[forecast["day"] == 90, "cash_p50"].iloc[0]

    assert day0_cash == 5000, f"backlog invoice should add nothing at day 0, got {day0_cash}"
    assert day90_cash == 5000, f"backlog invoice should add nothing anywhere in the forecast, got {day90_cash}"
    assert summary["backlog_invoice_count"] == 1
    assert summary["backlog_invoice_value"] == 100000.0
    assert summary["forward_invoice_count"] == 0
    print("PASSED: test_backlog_invoice_excluded_from_forecast")


def test_forward_invoice_still_simulated_normally():
    """A genuinely forward-looking invoice (not past its own P50) should
    still be simulated into the forecast exactly as before."""
    forward_invoice = pd.DataFrame({
        "invoice_id": ["NEW1"], "invoice_amount": [100000.0],
        "days_since_issue": [0],
        "p10_payment_days": [44], "p50_payment_days": [52], "p90_payment_days": [68],
    })
    forecast, summary = simulate_cashflow(
        forward_invoice, opening_cash=0, daily_expense=0, horizon_days=90, n_sims=20000
    )
    cash_at_80 = forecast.loc[forecast["day"] == 80, "cash_p50"].iloc[0]
    assert cash_at_80 == 100000, "forward invoice should still pay out normally"
    assert summary["forward_invoice_count"] == 1
    assert summary["backlog_invoice_count"] == 0
    print("PASSED: test_forward_invoice_still_simulated_normally")


def test_breach_probability_detected():
    """If opening cash is already below the buffer, breach probability at
    day 0 should be 100% (1.0), since there's no invoice to save it yet."""
    invoice = pd.DataFrame({
        "invoice_id": ["INV1"], "invoice_amount": [100000.0],
        "days_since_issue": [0],
        "p10_payment_days": [80], "p50_payment_days": [85], "p90_payment_days": [89],
    })
    forecast, _ = simulate_cashflow(
        invoice, opening_cash=500, daily_expense=0, horizon_days=10,
        n_sims=1000, min_buffer=1000,
    )
    day0_breach = forecast.loc[forecast["day"] == 0, "prob_breach"].iloc[0]
    assert day0_breach == 1.0, f"expected certain breach at day 0, got {day0_breach}"
    print("PASSED: test_breach_probability_detected")


if __name__ == "__main__":
    test_single_invoice_pays_within_its_window()
    test_expenses_reduce_cash_with_no_inflow()
    test_split_backlog_uses_p50_threshold()
    test_backlog_invoice_excluded_from_forecast()
    test_forward_invoice_still_simulated_normally()
    test_breach_probability_detected()
    print("\nAll monte_carlo tests passed.")