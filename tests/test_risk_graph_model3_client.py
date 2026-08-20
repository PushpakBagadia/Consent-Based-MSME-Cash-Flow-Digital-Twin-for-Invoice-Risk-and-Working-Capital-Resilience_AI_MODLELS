"""
Unit tests for risk_graph/model3_client.py.
Run: python -m tests.test_risk_graph_model3_client
"""
import pandas as pd

from risk_graph.model3_client import attach_anomaly_flags


def test_flagged_invoice_gets_its_type():
    invoice_ids = pd.DataFrame({"invoice_id": ["A1", "A2", "A3"]})
    anomaly_flags = pd.DataFrame({
        "invoice_id": ["A2"], "anomaly_type": ["severely_overdue"],
    })
    result = attach_anomaly_flags(invoice_ids, anomaly_flags)

    assert result.loc[result["invoice_id"] == "A2", "anomaly_type"].iloc[0] == "severely_overdue"
    print("PASSED: test_flagged_invoice_gets_its_type")


def test_unflagged_invoice_defaults_to_normal():
    invoice_ids = pd.DataFrame({"invoice_id": ["A1", "A2"]})
    anomaly_flags = pd.DataFrame({"invoice_id": ["A2"], "anomaly_type": ["large_amount"]})
    result = attach_anomaly_flags(invoice_ids, anomaly_flags)

    assert result.loc[result["invoice_id"] == "A1", "anomaly_type"].iloc[0] == "normal"
    print("PASSED: test_unflagged_invoice_defaults_to_normal")


def test_empty_anomaly_flags_all_default_normal():
    """If Model 3 flagged nothing at all (empty response), every invoice
    should default to normal - not crash, not leave NaN."""
    invoice_ids = pd.DataFrame({"invoice_id": ["A1", "A2"]})
    empty_flags = pd.DataFrame(columns=["invoice_id", "anomaly_type"])
    result = attach_anomaly_flags(invoice_ids, empty_flags)

    assert (result["anomaly_type"] == "normal").all()
    print("PASSED: test_empty_anomaly_flags_all_default_normal")


if __name__ == "__main__":
    test_flagged_invoice_gets_its_type()
    test_unflagged_invoice_defaults_to_normal()
    test_empty_anomaly_flags_all_default_normal()
    print("\nAll model3_client tests passed.")