"""Benchmarking metrics computation routines."""
import time
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_detection_performance(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, latency_sec: float = 0.0, num_samples: int = 1
) -> Dict[str, float]:
    """Calculates standardized classification metrics and latency per sample."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "latency_ms_per_sample": float((latency_sec / max(num_samples, 1)) * 1000.0)
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics
