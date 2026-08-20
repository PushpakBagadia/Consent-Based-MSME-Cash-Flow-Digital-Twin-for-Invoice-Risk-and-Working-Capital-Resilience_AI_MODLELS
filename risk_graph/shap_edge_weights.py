"""
Wires Model 5's SHAP output into Model 8's graph as edge weights, per the
spec note: "SHAP rankings can annotate edge weights - e.g. a thicker edge
for a customer whose delay contributes more to the breach."

ASSUMPTION FLAGGED FOR REVIEW
------------------------------
I do not have model5_shap.py's exact response schema for
`POST /explain/invoices` - only the brief's description ("returns a full
ranked list of feature contributions per invoice - no top-N truncation, no
canned text, numbers only"). `_extract_contributions()` below tries several
likely key names so it won't hard-crash on a reasonable shape, but you
should open one real response from `/explain/invoices` and confirm/adjust
the key names marked with "# ADJUST IF NEEDED" - that's the one place this
file might not match Dev1's actual output.

DESIGN DECISION (per the brief: "worth deciding deliberately, not picking
the first contribution in the list")
------------------------------------
The edge this annotates is `customer --delays--> invoice`, i.e. "how much
did this customer's own payment behaviour drive the model's prediction for
this invoice." Of Model 1's features, the two most directly about customer
behaviour (not invoice-intrinsic like amount, or generic like sector) are:

    1. customer_avg_payment_days   (their historical average - primary)
    2. payment_behavior_trend      (fallback, if the model didn't use #1
                                     e.g. a cold-start customer)

SHAP_FEATURE_PRIORITY below is that ordered list. If neither feature is in
a given invoice's contribution list (e.g. a true cold-start invoice using
only sector-level priors), the edge falls back to the days-overdue weight
`build_risk_graph.py` already computes, so the graph never has a "hole" -
we just lose the SHAP-specific annotation for that one edge.
"""
import pandas as pd
import requests

SHAP_FEATURE_PRIORITY = ["customer_avg_payment_days", "payment_behavior_trend"]


def _build_explain_payload(focus_df):
    """
    /explain/invoices takes the same input shape as /predict/invoices:
    invoice_id, cust_number, sector, invoice_amount, payment_term_days,
    issue_date. focus_df must already have sector and payment_term_days
    joined in (build_risk_graph.py does this from raw_invoices_path,
    same file model1_client.py reads).
    """
    required = ["invoice_id", "cust_number", "sector", "invoice_amount", "payment_term_days", "issue_date"]
    missing = [c for c in required if c not in focus_df.columns]
    if missing:
        raise ValueError(
            f"shap_edge_weights needs {missing} on the focus DataFrame - "
            f"these come from data/raw/invoices.csv, same as model1_client.py's merge."
        )

    payload = focus_df[required].copy()
    payload["issue_date"] = pd.to_datetime(payload["issue_date"]).dt.strftime("%Y-%m-%d")  # ADJUST IF NEEDED
    return payload.to_dict(orient="records")


def _extract_contributions(item):
    """Tries the likely key names for the per-feature contribution list and each entry's feature/value keys."""
    for list_key in ("contributions", "shap_values", "explanations", "feature_contributions"):
        if list_key in item:
            raw_list = item[list_key]
            break
    else:  # ADJUST IF NEEDED - none of the guessed keys matched
        raise KeyError(
            f"Could not find a contributions list on an /explain/invoices response item. "
            f"Keys present: {list(item.keys())}. Update the list_key options above to match."
        )

    out = {}
    for entry in raw_list:
        feature_name = entry.get("feature") or entry.get("feature_name") or entry.get("name")  # ADJUST IF NEEDED
        shap_val = entry.get("shap_value") or entry.get("value") or entry.get("contribution")  # ADJUST IF NEEDED
        if feature_name is not None and shap_val is not None:
            out[feature_name] = float(shap_val)
    return out


def load_shap_edge_weights(
    focus_df,
    api_url="http://127.0.0.1:8000/explain/invoices",
    feature_priority=SHAP_FEATURE_PRIORITY,
    timeout=30,
):
    """
    Calls Model 5's /explain/invoices for every invoice in focus_df (the
    same overdue-scoped set build_risk_graph.py already narrowed down to -
    NOT all 347 open invoices, keeping this call small).

    Returns a dict {invoice_id: {"shap_weight": float, "shap_feature": str}}
    for invoices where one of feature_priority was found in the response.
    Invoices not in the returned dict should keep their existing
    days-overdue edge weight in build_risk_graph.py - this function never
    raises just because one feature was missing for one invoice.
    """
    if focus_df.empty:
        return {}

    payload = _build_explain_payload(focus_df)
    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    results = response.json()

    weights = {}
    for item in results:
        invoice_id = item.get("invoice_id")
        if invoice_id is None:
            continue
        try:
            contributions = _extract_contributions(item)
        except KeyError:
            continue  # this invoice's response didn't parse; leave it out, don't fail the whole batch

        for feature in feature_priority:
            if feature in contributions:
                weights[invoice_id] = {"shap_weight": contributions[feature], "shap_feature": feature}
                break

    return weights