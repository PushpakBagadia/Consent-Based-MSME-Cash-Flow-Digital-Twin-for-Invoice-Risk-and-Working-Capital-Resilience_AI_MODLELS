"""
Model 8 - Causal Risk Graph Builder.

NOT a trained model - deterministic graph construction from what Models 1,
2, 3, and 5 already computed:

    customer --delays--> invoice --contributes_to--> buffer --breaches--> obligation

Node types: customer, invoice, buffer, obligation
Edge types: delays, contributes_to, breaches

Enrichment (both OPTIONAL and FAIL-SOFT - the graph must always build even
if Model 3 or Model 5 are down, since a live demo can't depend on every
other service being up):
  - invoice nodes get an anomaly_type from Model 3 (default "normal" if
    Model 3 is unreachable or an invoice wasn't flagged)
  - delays edges get an additional shap_top_feature / shap_value /
    shap_direction from Model 5 (whichever single feature had the largest
    SHAP impact for that invoice - see model5_client.py), alongside their
    always-available days_overdue weight

Scope defaults to "overdue" - only invoices already past their own Model 1
P90 prediction (a narrower, more severe slice than Model 2's P50-based
backlog split - intentional, since the graph is meant to highlight the
clearest, most demo-legible risk cases, not the full backlog) - keeps the
graph demo-readable instead of showing all 347 open invoices at once.
"""
import pandas as pd

from simulation.model1_client import load_model1_predictions
from simulation.monte_carlo import simulate_cashflow
from risk_graph.model3_client import load_anomaly_flags, attach_anomaly_flags
from risk_graph.model5_client import load_shap_edge_weights

OBLIGATION_NODE_ID = "obligation:working_capital"
BUFFER_NODE_ID = "buffer:cash_position"


def build_risk_graph(
    opening_cash,
    daily_expense,
    raw_invoices_path="data/raw/invoices.csv",
    model1_api_url="http://127.0.0.1:8000/predict/open-invoices",
    model3_api_url="http://127.0.0.1:8002/detect-anomalies/open",
    model5_api_url="http://127.0.0.1:8000/explain/invoices",
    min_buffer=0,
    scope="overdue",  # "overdue" = only invoices past their own P90 | "all" = every open invoice
    include_anomalies=True,
    include_shap=True,
):
    predictions = load_model1_predictions(api_url=model1_api_url, raw_invoices_path=raw_invoices_path)
    raw = pd.read_csv(raw_invoices_path)

    forecast, summary = simulate_cashflow(
        predictions, opening_cash=opening_cash, daily_expense=daily_expense, min_buffer=min_buffer,
    )
    max_breach_prob = float(forecast["prob_breach"].max())

    if scope == "overdue":
        focus = predictions[predictions["days_since_issue"] > predictions["p90_payment_days"]].copy()
    else:
        focus = predictions.copy()

    # Single merge pulling everything downstream needs: display info
    # (customer_name) AND the fields Model 5's /explain/invoices requires
    # (sector, payment_term_days, issue_date) that Model 1's predictions
    # alone don't carry.
    focus = focus.merge(
        raw[["invoice_id", "cust_number", "customer_name", "sector", "payment_term_days", "issue_date"]],
        on="invoice_id", how="left",
    )

    # ---------------------------------------------------------
    # Model 3 enrichment - fail-soft
    # ---------------------------------------------------------
    anomaly_by_invoice = {}
    anomaly_status = "disabled"
    if include_anomalies and not focus.empty:
        try:
            anomaly_flags = load_anomaly_flags(api_url=model3_api_url)
            annotated = attach_anomaly_flags(focus[["invoice_id"]], anomaly_flags)
            anomaly_by_invoice = dict(zip(annotated["invoice_id"], annotated["anomaly_type"]))
            anomaly_status = "ok"
        except Exception as e:
            anomaly_status = f"skipped: {e}"
            print(f"[build_risk_graph] Model 3 anomaly join skipped: {e}")

    # ---------------------------------------------------------
    # Model 5 enrichment - fail-soft
    # ---------------------------------------------------------
    shap_by_invoice = {}
    shap_status = "disabled"
    if include_shap and not focus.empty:
        try:
            shap_by_invoice = load_shap_edge_weights(focus, api_url=model5_api_url)
            shap_status = "ok"
        except Exception as e:
            shap_status = f"skipped: {e}"
            print(f"[build_risk_graph] Model 5 SHAP edge-weighting skipped: {e}")

    # ---------------------------------------------------------
    # Build nodes and edges
    # ---------------------------------------------------------
    nodes = [
        {
            "id": BUFFER_NODE_ID, "type": "buffer", "label": "Cash Position",
            "min_buffer": min_buffer, "max_breach_probability": round(max_breach_prob, 4),
        },
        {
            "id": OBLIGATION_NODE_ID, "type": "obligation", "label": "Working Capital Obligations",
            "note": (
                "Illustrative - represents downstream obligations (payroll, "
                "suppliers, EMIs) dependent on cash position; no per-obligation "
                "dataset exists to break this down further"
            ),
        },
    ]
    edges = []
    seen_customers = set()

    for row in focus.itertuples(index=False):
        cust_id = f"customer:{row.cust_number}"
        if row.cust_number not in seen_customers:
            nodes.append({
                "id": cust_id, "type": "customer",
                "label": row.customer_name if pd.notna(row.customer_name) else row.cust_number,
                "cust_number": row.cust_number,
            })
            seen_customers.add(row.cust_number)

        inv_id = f"invoice:{row.invoice_id}"
        days_overdue = int(row.days_since_issue - row.p90_payment_days)
        nodes.append({
            "id": inv_id, "type": "invoice", "label": row.invoice_id,
            "invoice_amount": float(row.invoice_amount),
            "days_past_own_p90": days_overdue,
            "anomaly_type": anomaly_by_invoice.get(row.invoice_id, "normal"),
        })

        shap_info = shap_by_invoice.get(row.invoice_id)  # None if unavailable
        edges.append({
            "source": cust_id, "target": inv_id, "type": "delays",
            "weight": days_overdue,
            "shap_top_feature": shap_info["top_feature"] if shap_info else None,
            "shap_value": shap_info["shap_value"] if shap_info else None,
            "shap_direction": shap_info["direction"] if shap_info else None,
        })
        edges.append({
            "source": inv_id, "target": BUFFER_NODE_ID, "type": "contributes_to",
            "weight": float(row.invoice_amount),
        })

    edges.append({
        "source": BUFFER_NODE_ID, "target": OBLIGATION_NODE_ID, "type": "breaches",
        "weight": round(max_breach_prob, 4),
    })

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "focus_invoice_count": len(focus),
            "focus_invoice_value": float(focus["invoice_amount"].sum()) if len(focus) else 0.0,
            "max_breach_probability": round(max_breach_prob, 4),
            "expected_min_cash": summary["expected_min_cash"],
            "anomaly_enrichment": anomaly_status,
            "shap_enrichment": shap_status,
        },
    }