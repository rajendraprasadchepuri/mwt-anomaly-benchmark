"""Supervised XGBoost baseline classifier."""
import time
from typing import Dict
import numpy as np
import xgboost as xgb
from mwt_benchmark.utils.metrics import evaluate_detection_performance


class XGBoostAnomalyDetector:
    """Gradient boosted tree classifier for supervised fault detection."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 5, learning_rate: float = 0.05, seed: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        start_time = time.perf_counter()
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        inference_time = time.perf_counter() - start_time

        return evaluate_detection_performance(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            latency_sec=inference_time,
            num_samples=len(X_test)
        )
