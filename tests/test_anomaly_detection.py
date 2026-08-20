"""
Unit tests for anomaly/isolation_forest_detector.py and anomaly/anomaly_explainer.py.

Run: python -m tests.test_anomaly_detection
"""
import numpy as np
import pandas as pd

from anomaly.isolation_forest_detector import detect_anomalies
from anomaly.anomaly_explainer import label_anomaly_type, add_anomaly_types


def test_obvious_outlier_gets_flagged():
    """20 normal points clustered near (0,0) plus one point way out at
    (50,50) - the detector should flag that one point and nothing else."""
    rng = np.random.default_rng(1)
    normal_points = pd.DataFrame({
        "id": [f"n{i}" for i in range(20)],
        "feature_a": rng.normal(0, 1, 20),
        "feature_b": rng.normal(0, 1, 20),
    })
    outlier = pd.DataFrame({"id": ["outlier"], "feature_a": [50.0], "feature_b": [50.0]})
    data = pd.concat([normal_points, outlier], ignore_index=True)

    result = detect_anomalies(data, ["feature_a", "feature_b"], contamination=0.05)
    outlier_result = result[result["id"] == "outlier"].iloc[0]

    assert outlier_result["anomaly_flag"] == True, "the obvious outlier should be flagged"
    print("PASSED: test_obvious_outlier_gets_flagged")


def test_contamination_controls_flagged_fraction():
    """With contamination=0.1 on 100 points, roughly 10 should be flagged
    (Isolation Forest doesn't guarantee exactly 10, but it should be close)."""
    rng = np.random.default_rng(2)
    data = pd.DataFrame({
        "id": [f"p{i}" for i in range(100)],
        "feature_a": rng.normal(0, 1, 100),
    })
    result = detect_anomalies(data, ["feature_a"], contamination=0.1)
    flagged_count = result["anomaly_flag"].sum()
    assert 5 <= flagged_count <= 15, f"expected ~10 flagged, got {flagged_count}"
    print("PASSED: test_contamination_controls_flagged_fraction")


def test_label_picks_largest_margin_rule():
    """If two rules both trigger, the one with the larger (value/threshold)
    margin should win, not just whichever rule appears first in the list."""
    rules = [
        {"column": "a", "threshold": 2.0, "direction": "high", "tag": "tag_a"},
        {"column": "b", "threshold": 2.0, "direction": "high", "tag": "tag_b"},
    ]
    # a triggers at 2x its threshold, b triggers at only 1.1x its threshold
    row = pd.Series({"a": 4.0, "b": 2.2})
    tag = label_anomaly_type(row, rules)
    assert tag == "tag_a", f"expected tag_a (larger margin), got {tag}"
    print("PASSED: test_label_picks_largest_margin_rule")


def test_label_falls_back_to_unusual_combination():
    """If no individual rule triggers, the label should be
    'unusual_combination', not None or a crash."""
    rules = [{"column": "a", "threshold": 10.0, "direction": "high", "tag": "tag_a"}]
    row = pd.Series({"a": 1.0})  # well below threshold
    tag = label_anomaly_type(row, rules)
    assert tag == "unusual_combination", f"expected fallback tag, got {tag}"
    print("PASSED: test_label_falls_back_to_unusual_combination")


def test_add_anomaly_types_only_labels_flagged_rows():
    """Non-flagged rows should be labeled 'normal', not run through the
    rule logic at all."""
    df = pd.DataFrame({
        "id": ["r1", "r2"],
        "a": [100.0, 0.5],
        "anomaly_flag": [True, False],
    })
    rules = [{"column": "a", "threshold": 2.0, "direction": "high", "tag": "tag_a"}]
    result = add_anomaly_types(df, rules)

    assert result.loc[result["id"] == "r1", "anomaly_type"].iloc[0] == "tag_a"
    assert result.loc[result["id"] == "r2", "anomaly_type"].iloc[0] == "normal"
    print("PASSED: test_add_anomaly_types_only_labels_flagged_rows")


if __name__ == "__main__":
    test_obvious_outlier_gets_flagged()
    test_contamination_controls_flagged_fraction()
    test_label_picks_largest_margin_rule()
    test_label_falls_back_to_unusual_combination()
    test_add_anomaly_types_only_labels_flagged_rows()
    print("\nAll anomaly detection tests passed.")