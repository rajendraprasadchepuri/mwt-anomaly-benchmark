"""Isolation Forest baseline detector."""
import time
from typing import Dict
import numpy as np
from sklearn.ensemble import IsolationForest
from mwt_benchmark.utils.metrics import evaluate_detection_performance


class IsolationForestDetector:
    """Unsupervised tree-based anomaly isolation."""

    def __init__(self, contamination: float = 0.03, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray) -> None:
        self.model.fit(X_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        start_time = time.perf_counter()
        raw_preds = self.model.predict(X_test)
        inference_time = time.perf_counter() - start_time

        y_pred = np.where(raw_preds == -1, 1, 0)
        scores = -self.model.score_samples(X_test)

        return evaluate_detection_performance(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=scores,
            latency_sec=inference_time,
            num_samples=len(X_test)
        )
