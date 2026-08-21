"""
Synthetic tests for the two Model 8 enrichment threads (Model 3 anomaly
join, Model 5 SHAP edge weights). Same style as test_risk_graph.py: small
hand-built cases, plain asserts, no live servers needed - HTTP calls are
mocked so these run offline and fast.

Run: python tests/test_risk_graph_enrichment.py
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from Monte_Carlo.anomaly_client import attach_anomaly_flags, NOT_FLAGGED_LABEL
from risk_graph.shap_edge_weights import _extract_contributions, _build_explain_payload, load_shap_edge_weights


def _fake_response(json_body, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock() if status_ok else MagicMock(side_effect=RuntimeError("bad status"))
    return resp


# ---------- Model 3 anomaly join ----------

def test_attach_anomaly_flags_joins_and_fills_not_flagged():
    invoices = pd.DataFrame({"invoice_id": ["INV1", "INV2", "INV3"]})
    flags = pd.DataFrame({
        "invoice_id": ["INV2"],
        "anomaly_score": [0.83],
        "anomaly_type": ["severely_overdue"],
    })

    merged = attach_anomaly_flags(invoices, flags)

    assert len(merged) == 3
    row2 = merged[merged["invoice_id"] == "INV2"].iloc[0]
    assert row2["anomaly_type"] == "severely_overdue"
    assert row2["anomaly_score"] == 0.83

    row1 = merged[merged["invoice_id"] == "INV1"].iloc[0]
    assert row1["anomaly_type"] == NOT_FLAGGED_LABEL
    assert pd.isna(row1["anomaly_score"])
    print("test_attach_anomaly_flags_joins_and_fills_not_flagged: PASS")


def test_attach_anomaly_flags_empty_flags_still_fills_not_flagged():
    invoices = pd.DataFrame({"invoice_id": ["INV1"]})
    flags = pd.DataFrame(columns=["invoice_id", "anomaly_score", "anomaly_type"])

    merged = attach_anomaly_flags(invoices, flags)

    assert merged.iloc[0]["anomaly_type"] == NOT_FLAGGED_LABEL
    print("test_attach_anomaly_flags_empty_flags_still_fills_not_flagged: PASS")


# ---------- Model 5 SHAP edge weights ----------

def test_extract_contributions_handles_documented_shape():
    item = {
        "invoice_id": "INV1",
        "base_value": 12.0,
        "prediction": 30.0,
        "contributions": [
            {"feature": "customer_avg_payment_days", "shap_value": 8.4},
            {"feature": "invoice_amount", "shap_value": -1.2},
        ],
    }
    contributions = _extract_contributions(item)
    assert contributions["customer_avg_payment_days"] == 8.4
    assert contributions["invoice_amount"] == -1.2
    print("test_extract_contributions_handles_documented_shape: PASS")


def test_extract_contributions_handles_alternate_key_names():
    # covers the "ADJUST IF NEEDED" fallback keys, in case Model 5's real
    # response uses shap_values / feature_name / value instead
    item = {
        "invoice_id": "INV1",
        "shap_values": [
            {"feature_name": "payment_behavior_trend", "value": 3.1},
        ],
    }
    contributions = _extract_contributions(item)
    assert contributions["payment_behavior_trend"] == 3.1
    print("test_extract_contributions_handles_alternate_key_names: PASS")


def test_extract_contributions_raises_clearly_on_unknown_shape():
    item = {"invoice_id": "INV1", "totally_unexpected_key": []}
    try:
        _extract_contributions(item)
        raise AssertionError("expected KeyError for unrecognised response shape")
    except KeyError:
        pass
    print("test_extract_contributions_raises_clearly_on_unknown_shape: PASS")


def test_build_explain_payload_shape_and_date_format():
    focus = pd.DataFrame({
        "invoice_id": ["INV1"],
        "cust_number": ["C001"],
        "sector": ["Textiles"],
        "invoice_amount": [50000.0],
        "payment_term_days": [30],
        "issue_date": ["2026-07-01"],
    })
    payload = _build_explain_payload(focus)
    assert payload == [{
        "invoice_id": "INV1", "cust_number": "C001", "sector": "Textiles",
        "invoice_amount": 50000.0, "payment_term_days": 30, "issue_date": "2026-07-01",
    }]
    print("test_build_explain_payload_shape_and_date_format: PASS")


def test_build_explain_payload_missing_column_raises():
    focus = pd.DataFrame({"invoice_id": ["INV1"], "cust_number": ["C001"]})
    try:
        _build_explain_payload(focus)
        raise AssertionError("expected ValueError for missing columns")
    except ValueError:
        pass
    print("test_build_explain_payload_missing_column_raises: PASS")


def test_load_shap_edge_weights_priority_and_fallback():
    """
    INV1: has both priority features -> should pick customer_avg_payment_days (priority 1)
    INV2: only has the fallback feature -> should pick payment_behavior_trend
    INV3: has neither -> should be absent from the result entirely (build_risk_graph.py's
          job to fall back to days-overdue weight for this one, not this function's)
    """
    focus = pd.DataFrame({
        "invoice_id": ["INV1", "INV2", "INV3"],
        "cust_number": ["C001", "C002", "C003"],
        "sector": ["Textiles", "Textiles", "Retail"],
        "invoice_amount": [50000.0, 20000.0, 10000.0],
        "payment_term_days": [30, 30, 15],
        "issue_date": ["2026-07-01", "2026-07-05", "2026-07-10"],
    })
    fake_json = [
        {"invoice_id": "INV1", "contributions": [
            {"feature": "customer_avg_payment_days", "shap_value": 5.0},
            {"feature": "payment_behavior_trend", "shap_value": 1.0},
        ]},
        {"invoice_id": "INV2", "contributions": [
            {"feature": "payment_behavior_trend", "shap_value": 2.5},
        ]},
        {"invoice_id": "INV3", "contributions": [
            {"feature": "invoice_amount", "shap_value": -0.3},
        ]},
    ]

    with patch("risk_graph.shap_edge_weights.requests.post", return_value=_fake_response(fake_json)):
        weights = load_shap_edge_weights(focus)

    assert weights["INV1"] == {"shap_weight": 5.0, "shap_feature": "customer_avg_payment_days"}
    assert weights["INV2"] == {"shap_weight": 2.5, "shap_feature": "payment_behavior_trend"}
    assert "INV3" not in weights
    print("test_load_shap_edge_weights_priority_and_fallback: PASS")


def test_load_shap_edge_weights_empty_focus_short_circuits():
    empty = pd.DataFrame(columns=["invoice_id", "cust_number", "sector", "invoice_amount", "payment_term_days", "issue_date"])
    with patch("risk_graph.shap_edge_weights.requests.post") as mock_post:
        weights = load_shap_edge_weights(empty)
    assert weights == {}
    mock_post.assert_not_called()
    print("test_load_shap_edge_weights_empty_focus_short_circuits: PASS")


if __name__ == "__main__":
    test_attach_anomaly_flags_joins_and_fills_not_flagged()
    test_attach_anomaly_flags_empty_flags_still_fills_not_flagged()
    test_extract_contributions_handles_documented_shape()
    test_extract_contributions_handles_alternate_key_names()
    test_extract_contributions_raises_clearly_on_unknown_shape()
    test_build_explain_payload_shape_and_date_format()
    test_build_explain_payload_missing_column_raises()
    test_load_shap_edge_weights_priority_and_fallback()
    test_load_shap_edge_weights_empty_focus_short_circuits()
    print("\nAll Model 8 enrichment tests passed.")