"""
Integration tests for risk_graph/build_risk_graph.py.

Lower-level client tests live separately in test_risk_graph_model3_client.py
and test_risk_graph_model5_client.py - this file only tests build_risk_graph()
itself: scoping, full successful enrichment, and fail-soft behavior when
Model 3 / Model 5 are unreachable.

Run: python -m tests.test_risk_graph
"""
import tempfile
from pathlib import Path

import pandas as pd
import requests

from risk_graph.build_risk_graph import build_risk_graph


def _make_raw_df():
    return pd.DataFrame({
        "invoice_id": ["A1", "B1"],
        "cust_number": ["C1", "C2"],
        "customer_name": ["Alpha Textiles", "Beta Traders"],
        "sector": ["textiles", "retail"],
        "payment_term_days": [30, 30],
        "invoice_amount": [100000.0, 50000.0],
        "issue_date": [
            (pd.Timestamp.now() - pd.Timedelta(days=50)).strftime("%Y-%m-%d"),
            (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        ],
    })


FAKE_PREDICTIONS = [
    {"invoice_id": "A1", "customer_id": "C1", "p10_payment_days": 20, "p50_payment_days": 25, "p90_payment_days": 30},
    {"invoice_id": "B1", "customer_id": "C2", "p10_payment_days": 20, "p50_payment_days": 25, "p90_payment_days": 30},
]


class FakeGetResponse:
    def __init__(self, data): self._data = data
    def raise_for_status(self): pass
    def json(self): return self._data


class FakePostResponse:
    def __init__(self, data): self._data = data
    def raise_for_status(self): pass
    def json(self): return self._data


def _mock_requests(get_routes=None, post_routes=None):
    get_routes = get_routes or {}
    post_routes = post_routes or {}

    def mock_get(url, timeout=30):
        for substring, data in get_routes.items():
            if substring in url:
                return FakeGetResponse(data)
        raise requests.exceptions.ConnectionError(f"no mock route for GET {url}")

    def mock_post(url, json=None, timeout=30):
        for substring, data in post_routes.items():
            if substring in url:
                return FakePostResponse(data)
        raise requests.exceptions.ConnectionError(f"no mock route for POST {url}")

    return mock_get, mock_post


def test_scope_overdue_only_includes_past_p90_invoices():
    """A1 (50 days since issue, p90=30) is overdue and in scope.
    B1 (10 days since issue, p90=30) is not, and should be excluded.
    Enrichment disabled here to isolate just the scoping logic."""
    raw_df = _make_raw_df()
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "invoices.csv"
        raw_df.to_csv(raw_path, index=False)

        mock_get, _ = _mock_requests(get_routes={"predict/open-invoices": FAKE_PREDICTIONS})
        original_get = requests.get
        requests.get = mock_get
        try:
            graph = build_risk_graph(
                opening_cash=100000, daily_expense=0,
                raw_invoices_path=str(raw_path), scope="overdue",
                include_anomalies=False, include_shap=False,
            )
        finally:
            requests.get = original_get

    invoice_nodes = [n for n in graph["nodes"] if n["type"] == "invoice"]
    delays_edges = [e for e in graph["edges"] if e["type"] == "delays"]

    assert len(invoice_nodes) == 1 and invoice_nodes[0]["label"] == "A1", (
        f"expected only A1 (overdue) in scope, got {[n['label'] for n in invoice_nodes]}"
    )
    assert delays_edges[0]["weight"] == 20, f"expected 50-30=20 days overdue, got {delays_edges[0]['weight']}"
    assert delays_edges[0]["shap_top_feature"] is None, "shap disabled - should be None"
    assert invoice_nodes[0]["anomaly_type"] == "normal", "anomaly disabled - should default to normal"
    print("PASSED: test_scope_overdue_only_includes_past_p90_invoices")


def test_full_enrichment_end_to_end():
    """Both Model 3 and Model 5 succeed - the graph should carry real
    anomaly_type and real SHAP top-feature info, not defaults."""
    raw_df = _make_raw_df()
    fake_anomalies = [{"invoice_id": "A1", "anomaly_score": -0.7, "anomaly_type": "severely_overdue"}]
    fake_explanation = [
        {
            "invoice_id": "A1", "base_value": 40.0, "predicted_value": 55.0,
            "contributions": [
                {"feature": "customer_avg_payment_days", "value": 45, "shap_value": 8.0, "direction": "increases"},
                {"feature": "invoice_amount", "value": 100000, "shap_value": -2.0, "direction": "decreases"},
            ],
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "invoices.csv"
        raw_df.to_csv(raw_path, index=False)

        mock_get, mock_post = _mock_requests(
            get_routes={
                "predict/open-invoices": FAKE_PREDICTIONS,
                "detect-anomalies/open": fake_anomalies,
            },
            post_routes={"explain/invoices": fake_explanation},
        )
        original_get, original_post = requests.get, requests.post
        requests.get, requests.post = mock_get, mock_post
        try:
            graph = build_risk_graph(
                opening_cash=100000, daily_expense=0,
                raw_invoices_path=str(raw_path), scope="overdue",
            )
        finally:
            requests.get, requests.post = original_get, original_post

    invoice_node = [n for n in graph["nodes"] if n["type"] == "invoice"][0]
    delays_edge = [e for e in graph["edges"] if e["type"] == "delays"][0]

    assert invoice_node["anomaly_type"] == "severely_overdue"
    # customer_avg_payment_days (8.0) has larger magnitude than invoice_amount (-2.0) - should win
    assert delays_edge["shap_top_feature"] == "customer_avg_payment_days"
    assert delays_edge["shap_value"] == 8.0
    assert delays_edge["shap_direction"] == "increases"
    assert graph["summary"]["anomaly_enrichment"] == "ok"
    assert graph["summary"]["shap_enrichment"] == "ok"
    print("PASSED: test_full_enrichment_end_to_end")


def test_fails_soft_when_model3_and_model5_unreachable():
    """The exact bug this whole round of changes was about: if Model 3 or
    Model 5 are unreachable, the graph must still build - no NameError,
    no crash - with clean defaults."""
    raw_df = _make_raw_df()
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "invoices.csv"
        raw_df.to_csv(raw_path, index=False)

        # Only Model 1's route is mocked - Model 3/5 calls hit the
        # "no mock route" ConnectionError branch, simulating them being down.
        mock_get, mock_post = _mock_requests(
            get_routes={"predict/open-invoices": FAKE_PREDICTIONS},
        )
        original_get, original_post = requests.get, requests.post
        requests.get, requests.post = mock_get, mock_post
        try:
            graph = build_risk_graph(
                opening_cash=100000, daily_expense=0,
                raw_invoices_path=str(raw_path), scope="overdue",
            )
        finally:
            requests.get, requests.post = original_get, original_post

    invoice_node = [n for n in graph["nodes"] if n["type"] == "invoice"][0]
    delays_edge = [e for e in graph["edges"] if e["type"] == "delays"][0]

    assert invoice_node["anomaly_type"] == "normal", "should default to normal, not crash"
    assert delays_edge["shap_top_feature"] is None, "should be None, not crash"
    assert delays_edge["shap_value"] is None
    assert delays_edge["shap_direction"] is None
    assert delays_edge["weight"] == 20, "days-overdue weight must still be present regardless"
    assert graph["summary"]["anomaly_enrichment"].startswith("skipped:")
    assert graph["summary"]["shap_enrichment"].startswith("skipped:")
    print("PASSED: test_fails_soft_when_model3_and_model5_unreachable")


def test_expense_node_and_reduces_edge():
    """The spec explicitly requires an expense node - this proves it's
    present with the right values, and the reduces edge points at buffer."""
    raw_df = _make_raw_df()
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "invoices.csv"
        raw_df.to_csv(raw_path, index=False)

        mock_get, _ = _mock_requests(get_routes={"predict/open-invoices": FAKE_PREDICTIONS})
        original_get = requests.get
        requests.get = mock_get
        try:
            graph = build_risk_graph(
                opening_cash=100000, daily_expense=2500, horizon_days=30,
                raw_invoices_path=str(raw_path), scope="overdue",
                include_anomalies=False, include_shap=False,
            )
        finally:
            requests.get = original_get

    expense_nodes = [n for n in graph["nodes"] if n["type"] == "expense"]
    reduces_edges = [e for e in graph["edges"] if e["type"] == "reduces"]

    assert len(expense_nodes) == 1
    assert expense_nodes[0]["daily_expense"] == 2500.0
    assert expense_nodes[0]["horizon_days"] == 30
    assert expense_nodes[0]["total_expense_over_horizon"] == 75000.0, "expected 2500*30=75000"

    assert len(reduces_edges) == 1
    assert reduces_edges[0]["source"] == "expense:daily_operating"
    assert reduces_edges[0]["target"] == "buffer:cash_position"
    assert reduces_edges[0]["weight"] == 75000.0, (
        "reduces edge weight should be the TOTAL projected expense over the "
        "horizon (2500*30=75000), consistent with contributes_to edges using "
        "full invoice_amount rather than a per-day figure - not just daily_expense"
    )
    print("PASSED: test_expense_node_and_reduces_edge")


if __name__ == "__main__":
    test_scope_overdue_only_includes_past_p90_invoices()
    test_full_enrichment_end_to_end()
    test_fails_soft_when_model3_and_model5_unreachable()
    test_expense_node_and_reduces_edge()
    print("\nAll risk_graph integration tests passed.")