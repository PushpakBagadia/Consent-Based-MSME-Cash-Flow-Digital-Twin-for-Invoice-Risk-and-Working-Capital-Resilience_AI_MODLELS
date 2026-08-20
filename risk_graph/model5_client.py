"""
Client for Model 5's SHAP explanation API - used by Model 8 to weight
`delays` edges (customer -> invoice) by which single feature drove that
invoice's prediction the most, instead of only a flat "days overdue" number.

DESIGN DECISION (explicit, confirmed - not assumed): the edge weight is
whichever feature has the single LARGEST absolute SHAP value for that
invoice, regardless of which feature it is - it could be a customer
behaviour feature, invoice_amount, payment_term_days, or sector. This is
more general than restricting to customer-behaviour-only features: it lets
the graph honestly show "this invoice's biggest driver was actually its
amount" when that's genuinely what the model found most influential,
rather than forcing every edge to describe customer behaviour specifically
even when that isn't the dominant factor.

Model 5's own explain_invoice() already sorts each invoice's contributions
by absolute SHAP value, most-influential first (see model5_shap.py's
`sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)`) - so
"top contribution" is simply contributions[0], no re-sorting needed here.
"""
import pandas as pd
import requests

REQUIRED_COLUMNS = ["invoice_id", "cust_number", "sector", "invoice_amount", "payment_term_days", "issue_date"]


def build_explain_payload(focus: pd.DataFrame) -> list:
    """
    focus: DataFrame that MUST already include, per invoice, all of
    REQUIRED_COLUMNS. Model 1's predictions alone don't carry sector or
    payment_term_days (Model 2 never needed them) - build_risk_graph.py
    merges these in from raw invoices.csv before calling this function.

    Returns a list of dicts matching main.py's InvoiceInput schema, ready
    to POST to /explain/invoices.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in focus.columns]
    if missing:
        raise ValueError(
            f"build_explain_payload is missing required columns: {missing} - "
            f"merge these in from raw invoices.csv before calling this"
        )

    payload = []
    for row in focus.itertuples(index=False):
        issue_date = row.issue_date
        if hasattr(issue_date, "strftime"):
            issue_date = issue_date.strftime("%Y-%m-%d")
        payload.append({
            "invoice_id": row.invoice_id,
            "cust_number": row.cust_number,
            "sector": row.sector,
            "invoice_amount": float(row.invoice_amount),
            "payment_term_days": int(row.payment_term_days),
            "issue_date": issue_date,
        })
    return payload


def load_shap_edge_weights(focus: pd.DataFrame, api_url="http://127.0.0.1:8000/explain/invoices", timeout=30) -> dict:
    """
    Returns {invoice_id: {"top_feature": str, "shap_value": float (signed),
                           "direction": "increases"|"decreases"}}

    shap_value is SIGNED - positive means this feature pushed the
    prediction toward MORE delay, negative means toward LESS delay.
    Magnitude (not sign) is what should drive edge thickness; direction is
    separate information for color/labeling on the frontend.

    Any invoice Model 5 doesn't return a result for (or returns an empty
    contributions list for) simply won't appear in the dict - callers
    should treat a missing key as "no SHAP weight available" (fall back to
    days-overdue only), never as zero - zero would falsely claim "confirmed
    no effect" rather than "we don't know."
    """
    payload = build_explain_payload(focus)
    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    explanations = response.json()

    weights = {}
    for exp in explanations:
        contributions = exp.get("contributions", [])
        if not contributions:
            continue
        top = contributions[0]  # already sorted by abs(shap_value) descending
        weights[exp["invoice_id"]] = {
            "top_feature": top["feature"],
            "shap_value": round(top["shap_value"], 4),
            "direction": top["direction"],
        }
    return weights