"""
Model 8 - Causal Risk Graph Builder.

NOT a trained model - deterministic graph construction from what Models 1
and 2 already computed. Follows the project spec's structure:

    customer --delays--> invoice --contributes_to--> buffer --breaches--> obligation

Node types: customer, invoice, buffer, obligation
Edge types: delays, contributes_to, breaches

Edge weights currently come from invoice_amount / days-overdue / breach
probability. The spec notes SHAP (Model 5) could later annotate edge
weights (e.g. thicker edge for a customer whose delay contributes more to
the breach) - Model 5 isn't built yet, so this is a clear future swap-in
point, not implemented now.

Scope defaults to "overdue" - only invoices already past their own Model 1
P90 prediction, same definition Model 3's severely_overdue tag uses (note:
Model 2's backlog split now uses a different, broader threshold - P50, not
P90 - see simulation/monte_carlo.py's split_backlog()) - keeps the graph
demo-readable instead of showing all 347 open invoices at once.
"""
import pandas as pd

from simulation.model1_client import load_model1_predictions
from simulation.monte_carlo import simulate_cashflow

OBLIGATION_NODE_ID = "obligation:working_capital"
BUFFER_NODE_ID = "buffer:cash_position"


def build_risk_graph(
    opening_cash,
    daily_expense,
    raw_invoices_path="data/raw/invoices.csv",
    model1_api_url="http://127.0.0.1:8000/predict/open-invoices",
    min_buffer=0,
    scope="overdue",  # "overdue" = only invoices past their own P90 | "all" = every open invoice
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

    focus = focus.merge(raw[["invoice_id", "cust_number", "customer_name"]], on="invoice_id", how="left")

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
        })

        edges.append({"source": cust_id, "target": inv_id, "type": "delays", "weight": days_overdue})
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
        },
    }