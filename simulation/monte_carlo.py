"""
Monte Carlo cash-flow simulation engine (v4).

Consumes Model 1's real output contract (invoice_id, invoice_amount,
days_since_issue, p10/p50/p90_payment_days) and produces a daily
P10/P50/P90 cash forecast.

v4 change: BACKLOG SPLIT. Invoices already past their own P50 (median)
prediction - i.e. the model's own "typical case" already didn't happen -
are no longer fed into the day-by-day forecast at all. The previous
approach (v3's overdue_cap) only special-cased invoices past their P90,
but invoices between P50 and P90 were still simulated normally, and any
of their sampled "days from today" that came out negative got clipped to
0 - piling probability mass onto day 0 and producing an unrealistically
large day-0 cash jump (confirmed on real data: 100 invoices, ~1.65 crore,
were already past their own P50, vs. only 78 past P90).

Backlog invoices are now reported separately (count + value) instead of
folded into the forecast curve - the day-by-day forecast only ever
contains invoices whose predicted payment window is still genuinely ahead
of today. This is an honest separation of two different questions: "what
does our near-term cash trajectory look like" (forward) vs. "how much are
we owed that's already overdue and may or may not ever be collected"
(backlog) - rather than blending the second into the first.
"""
import numpy as np
import pandas as pd

from simulation.quantile_sampler import sample_payment_days


def split_backlog(predictions: pd.DataFrame):
    """
    Splits predictions into (backlog_df, forward_df).

    backlog: days_since_issue > p50_payment_days - the model's OWN median
    prediction has already passed, so this invoice is now more likely late
    than on-time. Excluded from the day-by-day forecast.

    forward: everything else - genuinely still-plausible future payments,
    safe to simulate day-by-day.
    """
    is_backlog = predictions["days_since_issue"] > predictions["p50_payment_days"]
    return predictions[is_backlog].copy(), predictions[~is_backlog].copy()


def simulate_cashflow(
    predictions,        # DataFrame: invoice_id, invoice_amount, days_since_issue,
                         #            p10_payment_days, p50_payment_days, p90_payment_days
    opening_cash,        # float: cash on hand today
    daily_expense,        # float: flat expected daily outflow
    horizon_days=90,       # how many days ahead to forecast
    n_sims=3000,
    min_buffer=0,          # cash level considered a "liquidity breach"
    seed=42,
):
    backlog, forward = split_backlog(predictions)

    rng = np.random.default_rng(seed)
    days = np.arange(0, horizon_days + 1)
    cumulative_inflow = np.zeros((n_sims, len(days)))

    for row in forward.itertuples(index=False):
        payment_days_from_issue = sample_payment_days(
            row.p10_payment_days, row.p50_payment_days, row.p90_payment_days,
            n_sims, rng
        )
        payment_day_from_today = payment_days_from_issue - row.days_since_issue
        # A forward invoice is, by definition, not past its own median yet -
        # so only a small tail of samples can still land before today. That
        # residual is still clipped to 0 rather than allowed negative, but
        # it's now a minor edge effect, not the dominant driver it was before.
        payment_day_from_today = np.clip(payment_day_from_today, 0, None)

        paid_on_or_before = (payment_day_from_today[:, None] <= days[None, :])
        cumulative_inflow += paid_on_or_before * row.invoice_amount

    outflow = daily_expense * days[None, :]
    cash_balance = opening_cash - outflow + cumulative_inflow

    p10 = np.percentile(cash_balance, 10, axis=0)
    p50 = np.percentile(cash_balance, 50, axis=0)
    p90 = np.percentile(cash_balance, 90, axis=0)
    breach_prob = (cash_balance < min_buffer).mean(axis=0)

    forecast = pd.DataFrame({
        "day": days,
        "cash_p10": p10,
        "cash_p50": p50,
        "cash_p90": p90,
        "prob_breach": breach_prob,
    })

    expected_min_cash = cash_balance.min(axis=1).mean()
    breach_days = forecast.index[forecast["prob_breach"] > 0.5]
    days_to_likely_breach = int(forecast.loc[breach_days[0], "day"]) if len(breach_days) > 0 else None

    summary = {
        "expected_min_cash": float(expected_min_cash),
        "days_to_likely_breach": days_to_likely_breach,
        "forward_invoice_count": len(forward),
        "forward_invoice_value": float(forward["invoice_amount"].sum()) if len(forward) else 0.0,
        "backlog_invoice_count": len(backlog),
        "backlog_invoice_value": float(backlog["invoice_amount"].sum()) if len(backlog) else 0.0,
    }

    return forecast, summary