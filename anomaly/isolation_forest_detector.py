"""
Shared Isolation Forest wrapper for anomaly detection.

Isolation Forest works by randomly splitting the data repeatedly - an
outlier gets separated ("isolated") from the rest of the data in very few
splits, since it's already far from everything else. A normal point takes
many splits to isolate, since it's surrounded by similar points. The model
scores every point by how few splits it took - fewer splits = more
anomalous.

No labels needed (this is unsupervised) - it just learns what "normal"
looks like from the data's own structure.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(features_df: pd.DataFrame, feature_columns: list, contamination=0.05, seed=42):
    """
    features_df: DataFrame with an id column + the feature columns.
    feature_columns: which columns to actually feed the model.
    contamination: expected proportion of anomalies (0.05 = flag ~top 5%
        most unusual invoices). This is a judgment call, not derived from
        data - same honest-about-being-a-parameter approach as Model 2's
        overdue_cap.

    Returns features_df with two new columns added:
        anomaly_score - lower (more negative) = more anomalous
        anomaly_flag  - True/False, top `contamination` fraction flagged
    """
    X = features_df[feature_columns].fillna(0)

    model = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
    )
    predictions = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = model.score_samples(X)      # lower = more anomalous

    result = features_df.copy()
    result["anomaly_score"] = scores
    result["anomaly_flag"] = predictions == -1
    return result.sort_values("anomaly_score")