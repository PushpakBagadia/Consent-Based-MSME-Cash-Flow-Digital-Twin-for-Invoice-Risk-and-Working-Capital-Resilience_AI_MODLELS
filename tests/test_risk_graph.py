"""
Unit test for risk_graph/build_risk_graph.py.

Run: python -m tests.test_risk_graph
Uses a tiny hand-built scenario (2 invoices: one overdue, one not) so the
expected node/edge counts can be reasoned about exactly.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

from risk_graph.build_risk_graph import build_risk_graph


def test_scope_overdue_only_includes_past_p90_invoices():
    # invoice A is 50 days since issue, p90=30 -> 20 days overdue (in scope)
    # invoice B is 10 days since issue, p90=30 -> still on track (NOT in scope)
    fake_predictions = [
        {"invoice_id": "A1", "customer_id": "C1", "p10_payment_days": 20,
         "p50_payment_days": 25, "p90_payment_days": 30},
        {"invoice_id": "B1", "customer_id": "C2", "p10_payment_days": 20,
         "p50_payment_days": 25, "p90_payment_days": 30},
    ]
    raw_df = pd.DataFrame({
        "invoice_id": ["A1", "B1"],
        "cust_number": ["C1", "C2"],
        "customer_name": ["Alpha Textiles", "Beta Traders"],
        "invoice_amount": [100000.0, 50000.0],
        "issue_date": [
            (pd.Timestamp.now() - pd.Timedelta(days=50)).strftime("%Y-%m-%d"),
            (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        ],
    })

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "invoices.csv"
        raw_df.to_csv(raw_path, index=False)

        class FakeResponse:
            def raise_for_status(self): pass
            def json(self): return fake_predictions

        original_get = requests.get
        requests.get = lambda url, timeout=30: FakeResponse()
        try:
            graph = build_risk_graph(
                opening_cash=100000, daily_expense=0,
                raw_invoices_path=str(raw_path), scope="overdue",
            )
        finally:
            requests.get = original_get

    customer_nodes = [n for n in graph["nodes"] if n["type"] == "customer"]
    invoice_nodes = [n for n in graph["nodes"] if n["type"] == "invoice"]

    assert len(invoice_nodes) == 1, f"expected only A1 (overdue) in scope, got {len(invoice_nodes)}"
    assert invoice_nodes[0]["label"] == "A1"
    assert len(customer_nodes) == 1, "only C1 (A1's customer) should appear, not C2"
    assert customer_nodes[0]["label"] == "Alpha Textiles"

    delays_edges = [e for e in graph["edges"] if e["type"] == "delays"]
    assert len(delays_edges) == 1
    assert delays_edges[0]["weight"] == 20, f"expected 50-30=20 days overdue, got {delays_edges[0]['weight']}"

    buffer_nodes = [n for n in graph["nodes"] if n["type"] == "buffer"]
    obligation_nodes = [n for n in graph["nodes"] if n["type"] == "obligation"]
    assert len(buffer_nodes) == 1
    assert len(obligation_nodes) == 1

    breaches_edges = [e for e in graph["edges"] if e["type"] == "breaches"]
    assert len(breaches_edges) == 1
    assert breaches_edges[0]["source"] == "buffer:cash_position"
    assert breaches_edges[0]["target"] == "obligation:working_capital"

    print("PASSED: test_scope_overdue_only_includes_past_p90_invoices")


if __name__ == "__main__":
    test_scope_overdue_only_includes_past_p90_invoices()
    print("\nAll risk_graph tests passed.")