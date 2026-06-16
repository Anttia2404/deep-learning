from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_classification_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> dict:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr > threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred) * 100.0),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred).tolist(),
    }
    if len(np.unique(y_true_arr)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        metrics["auc_roc"] = float("nan")
    return metrics
