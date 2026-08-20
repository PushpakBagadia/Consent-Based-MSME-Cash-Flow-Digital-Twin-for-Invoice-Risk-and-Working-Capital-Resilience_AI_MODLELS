"""
Model 8 - Causal Risk Graph API.

Wraps build_risk_graph() (which internally calls Models 1, 2, 3, and 5's
live APIs) behind a single endpoint - same pattern as Models 2 and 3.

Run as its own server, its own port:
    python -m uvicorn api.risk_graph_api:app --port 8003

Models 1 (port 8000) and, if you want the enrichment, 3 (port 8002) should
be running too. Model 5's endpoint lives on Model 1's server (port 8000),
so no separate server is needed for that part. If Model 3 or Model 5 are
down, the graph still builds (fail-soft) - just without anomaly_type /
SHAP annotations, reported plainly in the response's summary.
"""
from fastapi import FastAPI, HTTPException, Query

from risk_graph.build_risk_graph import build_risk_graph

app = FastAPI(title="Model 8 - Causal Risk Graph API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/risk-graph")
def risk_graph(
    opening_cash: float = Query(..., description="Cash on hand today"),
    daily_expense: float = Query(..., description="Flat expected daily outflow"),
    min_buffer: float = Query(0, description="Cash level considered a liquidity breach"),
    scope: str = Query("overdue", description="'overdue' (default, past-P90 invoices only) or 'all'"),
    include_anomalies: bool = Query(True, description="Join Model 3's anomaly_type per invoice"),
    include_shap: bool = Query(True, description="Weight delays edges with Model 5's top SHAP contribution"),
):
    """
    Returns the causal risk graph: customer -> invoice -> buffer -> obligation,
    with delays / contributes_to / breaches edges. See risk_graph/build_risk_graph.py
    for the full node/edge schema and enrichment details.
    """
    try:
        return build_risk_graph(
            opening_cash=opening_cash,
            daily_expense=daily_expense,
            min_buffer=min_buffer,
            scope=scope,
            include_anomalies=include_anomalies,
            include_shap=include_shap,
        )
    except Exception as e:
        # Model 1 (required, not fail-soft) being down is the main way this
        # can fail outright - Model 3/5 failures are already handled
        # gracefully inside build_risk_graph() itself.
        raise HTTPException(status_code=502, detail=f"Could not build risk graph: {e}")