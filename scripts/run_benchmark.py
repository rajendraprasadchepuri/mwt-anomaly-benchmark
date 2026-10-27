#!/usr/bin/env python3
"""Executes end-to-end benchmark comparison across Isolation Forest, Autoencoder, and XGBoost."""
import os
import argparse
import json
import yaml
import numpy as np
from mwt_benchmark.data.loader import TelemetryDataLoader
from mwt_benchmark.models.isolation_forest import IsolationForestDetector
from mwt_benchmark.models.xgboost_model import XGBoostAnomalyDetector
from mwt_benchmark.models.autoencoder import PyTorchAutoencoderDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parquet_file = os.path.join(cfg["dataset"]["output_dir"], cfg["dataset"]["parquet_filename"])
    loader = TelemetryDataLoader(
        file_path=parquet_file,
        feature_cols=cfg["features"]["numerical_cols"],
        target_col=cfg["features"]["target_col"]
    )

    print("Loading temporal splits...")
    train_df, test_df = loader.get_temporal_splits(
        train_ratio=cfg["training"]["train_split"],
        test_ratio=cfg["training"]["test_split"]
    )
    train_scaled, test_scaled = loader.scale_features(train_df, test_df)

    feature_cols = cfg["features"]["numerical_cols"]
    X_train = train_scaled[feature_cols].values
    y_train = train_scaled[cfg["features"]["target_col"]].values
    X_test = test_scaled[feature_cols].values
    y_test = test_scaled[cfg["features"]["target_col"]].values

    normal_mask_train = (y_train == 0)
    X_train_normal = X_train[normal_mask_train]

    print("\n--- Running Isolation Forest ---")
    iforest = IsolationForestDetector()
    iforest.fit(X_train)
    res_iforest = iforest.evaluate(X_test, y_test)

    print("\n--- Running PyTorch Autoencoder ---")
    ae = PyTorchAutoencoderDetector(input_dim=len(feature_cols))
    ae.fit(X_train_normal, epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"])
    res_ae = ae.evaluate(X_test, y_test)

    print("\n--- Running Supervised XGBoost ---")
    xgb = XGBoostAnomalyDetector()
    xgb.fit(X_train, y_train)
    res_xgb = xgb.evaluate(X_test, y_test)

    results = {
        "Isolation Forest": res_iforest,
        "Autoencoder": res_ae,
        "XGBoost": res_xgb
    }

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS (Matching Paper Table 4.2)")
    print("=" * 50)
    print(f"{'Model':<20} | {'Precision':<8} | {'Recall':<8} | {'F1':<8} | {'Latency (ms)':<12}")
    print("-" * 62)
    for model_name, m in results.items():
        print(f"{model_name:<20} | {m['precision']:<8.3f} | {m['recall']:<8.3f} | {m['f1']:<8.3f} | {m['latency_ms_per_sample']:<12.2f}")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\nSaved output metrics to benchmark_results.json")


if __name__ == "__main__":
    main()
