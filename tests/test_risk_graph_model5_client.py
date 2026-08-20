"""
Unit tests for risk_graph/model5_client.py.
Run: python -m tests.test_risk_graph_model5_client
"""
import pandas as pd

from risk_graph.model5_client import build_explain_payload, load_shap_edge_weights


def test_build_explain_payload_has_required_fields():
    focus = pd.DataFrame({
        "invoice_id": ["A1"], "cust_number": ["C1"], "sector": ["retail"],
        "invoice_amount": [50000.0], "payment_term_days": [30],
        "issue_date": pd.to_datetime(["2026-01-01"]),
    })
    payload = build_explain_payload(focus)
    assert payload[0]["invoice_id"] == "A1"
    assert payload[0]["issue_date"] == "2026-01-01", f"expected ISO date string, got {payload[0]['issue_date']}"
    print("PASSED: test_build_explain_payload_has_required_fields")


def test_build_explain_payload_raises_on_missing_columns():
    focus = pd.DataFrame({"invoice_id": ["A1"]})  # missing everything else
    try:
        build_explain_payload(focus)
        assert False, "expected a ValueError for missing columns"
    except ValueError as e:
        assert "missing required columns" in str(e)
    print("PASSED: test_build_explain_payload_raises_on_missing_columns")


def test_top_contribution_picked_regardless_of_which_feature():
    """The approved design: whichever feature has the largest ABSOLUTE
    shap_value wins, even if it's not a 'customer behaviour' feature."""
    focus = pd.DataFrame({
        "invoice_id": ["A1"], "cust_number": ["C1"], "sector": ["retail"],
        "invoice_amount": [50000.0], "payment_term_days": [30],
        "issue_date": pd.to_datetime(["2026-01-01"]),
    })

    import risk_graph.model5_client as mod

    def fake_post(url, json, timeout=30):
        class FakeResponse:
            def raise_for_status(self): pass
            def json(self):
                return [{
                    "invoice_id": "A1",
                    "contributions": [
                        # invoice_amount has the LARGEST magnitude (-9.5),
                        # even though it's not a customer-behaviour feature -
                        # it should still win.
                        {"feature": "invoice_amount", "value": 50000.0, "shap_value": -9.5, "direction": "decreases"},
                        {"feature": "customer_avg_payment_days", "value": 40, "shap_value": 6.2, "direction": "increases"},
                    ],
                }]
        return FakeResponse()

    original_post = mod.requests.post
    mod.requests.post = fake_post
    try:
        weights = load_shap_edge_weights(focus)
    finally:
        mod.requests.post = original_post

    assert weights["A1"]["top_feature"] == "invoice_amount", (
        f"expected invoice_amount (largest magnitude) to win, got {weights['A1']['top_feature']}"
    )
    assert weights["A1"]["shap_value"] == -9.5
    assert weights["A1"]["direction"] == "decreases"
    print("PASSED: test_top_contribution_picked_regardless_of_which_feature")


def test_missing_invoice_not_in_result():
    """If Model 5 returns an empty contributions list for an invoice, it
    should NOT appear in the result dict at all (not zero, not crash)."""
    focus = pd.DataFrame({
        "invoice_id": ["A1"], "cust_number": ["C1"], "sector": ["retail"],
        "invoice_amount": [50000.0], "payment_term_days": [30],
        "issue_date": pd.to_datetime(["2026-01-01"]),
    })

    import risk_graph.model5_client as mod

    def fake_post(url, json, timeout=30):
        class FakeResponse:
            def raise_for_status(self): pass
            def json(self):
                return [{"invoice_id": "A1", "contributions": []}]
        return FakeResponse()

    original_post = mod.requests.post
    mod.requests.post = fake_post
    try:
        weights = load_shap_edge_weights(focus)
    finally:
        mod.requests.post = original_post

    assert "A1" not in weights, "invoice with no contributions should be absent, not zero"
    print("PASSED: test_missing_invoice_not_in_result")


if __name__ == "__main__":
    test_build_explain_payload_has_required_fields()
    test_build_explain_payload_raises_on_missing_columns()
    test_top_contribution_picked_regardless_of_which_feature()
    test_missing_invoice_not_in_result()
    print("\nAll model5_client tests passed.")