"""
Model 8 - Causal Risk Graph API.

Wraps risk_graph/build_risk_graph.py, following the project's established
one-port-per-model pattern (Model 1: 8000, Model 2: 8001, Model 3: 8002).

Run as its own server:
    python -m uvicorn api.risk_graph_api:app --port 8003

Depends on Model 1's server (8000) always - build_risk_graph() calls it
directly for predictions, and again for /explain/invoices if include_shap
is true. Model 3's server (8002) is only needed if include_anomalies is
true. Both Model 3 and Model 5 calls fail SOFT (see build_risk_graph.py's
docstring) - if either server is down, /risk-graph still returns a graph,
just without that one enrichment.

Invoke-RestMethod -Uri "http://127.0.0.1:8003/risk-graph?opening_cash=500000&daily_expense=15000" -Method Get
"""
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query

from risk_graph.build_risk_graph import build_risk_graph

app = FastAPI(title="Model 8 - Causal Risk Graph API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/risk-graph")
def get_risk_graph(
    opening_cash: float = Query(..., description="Current cash position to seed the simulation"),
    daily_expense: float = Query(..., description="Assumed daily operating expense"),
    min_buffer: float = Query(0, description="Minimum cash buffer before a 'breach' is counted"),
    scope: str = Query("overdue", pattern="^(overdue|all)$"),
    include_anomalies: bool = Query(True, description="Join Model 3's anomaly_type onto invoice nodes"),
    include_shap: bool = Query(True, description="Annotate delays edges with Model 5's SHAP contribution"),
) -> Dict[str, Any]:
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
        # Model 1 (always required) unreachable, or invoices.csv missing - this one's fatal,
        # unlike the Model 3 / Model 5 fail-soft paths inside build_risk_graph() itself.
        raise HTTPException(status_code=502, detail=f"Could not build risk graph: {e}")