"""
Rule-based anomaly_type labeling.

Isolation Forest is good at DETECTING that something is unusual, but it
doesn't naturally explain WHY in human terms. Rather than reach for another
model to explain the first model (which would violate the project's own
"clean separation of arithmetic vs. AI" principle), this uses simple,
transparent threshold rules to name the most likely reason - consistent
with how the Recommendation Ranker and Risk Graph are also kept as plain
logic, not ML.

For each flagged invoice: check each rule's column against its threshold,
and pick whichever rule is triggered by the LARGEST margin (value / threshold
ratio) - i.e. "most clearly the driving factor." If no single rule is
individually triggered but Isolation Forest still flagged the row, it's
labeled "unusual_combination" - an honest way of saying "no single feature
explains this, but the overall pattern was unusual."
"""
import pandas as pd


def label_anomaly_type(row, rules):
    """
    rules: list of dicts, each:
        {"column": str, "threshold": float, "direction": "high"|"low", "tag": str}
        "high" triggers if row[column] > threshold
        "low"  triggers if row[column] < -threshold

    Returns the tag of whichever triggered rule has the largest margin
    (row_value / threshold), or "unusual_combination" if none triggered.
    """
    best_tag = None
    best_margin = 0
    for rule in rules:
        value = row[rule["column"]]
        threshold = rule["threshold"]
        if threshold == 0:
            continue
        if rule["direction"] == "high" and value > threshold:
            margin = value / threshold
        elif rule["direction"] == "low" and value < -threshold:
            margin = -value / threshold
        else:
            continue
        if margin > best_margin:
            best_margin = margin
            best_tag = rule["tag"]
    return best_tag if best_tag else "unusual_combination"


CLOSED_INVOICE_RULES = [
    {"column": "amount_zscore_vs_sector", "threshold": 2.5, "direction": "high", "tag": "large_amount"},
    {"column": "amount_zscore_vs_customer", "threshold": 2.5, "direction": "high", "tag": "large_amount_for_customer"},
    {"column": "delay_zscore_vs_customer", "threshold": 2.0, "direction": "high", "tag": "slower_than_usual"},
    {"column": "delay_zscore_vs_customer", "threshold": 2.0, "direction": "low", "tag": "faster_than_usual"},
]

OPEN_INVOICE_RULES = [
    {"column": "amount_zscore_vs_sector", "threshold": 2.5, "direction": "high", "tag": "large_amount"},
    {"column": "days_past_own_p90", "threshold": 30, "direction": "high", "tag": "severely_overdue"},
]


def add_anomaly_types(result_df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Adds an anomaly_type column: the rule-based tag for flagged rows,
    "normal" for everything else.
    """
    result_df = result_df.copy()
    result_df["anomaly_type"] = "normal"
    flagged = result_df["anomaly_flag"]
    result_df.loc[flagged, "anomaly_type"] = result_df.loc[flagged].apply(
        lambda row: label_anomaly_type(row, rules), axis=1
    )
    return result_df