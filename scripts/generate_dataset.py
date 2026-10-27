#!/usr/bin/env python3
"""Script to synthesize and export the MWT-Anom24 telemetry benchmark."""
import os
import argparse
import yaml
from mwt_benchmark.data.generator import SyntheticTelemetryGenerator
from mwt_benchmark.utils.safe_arrow import sanitize_arrow_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic telemetry dataset")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["dataset"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    generator = SyntheticTelemetryGenerator(
        num_days=cfg["dataset"]["num_days"],
        sample_rate_sec=cfg["dataset"]["sample_rate_sec"],
        seed=cfg["dataset"]["random_seed"]
    )
    df = generator.generate()

    csv_path = os.path.join(out_dir, cfg["dataset"]["csv_filename"])
    parquet_path = os.path.join(out_dir, cfg["dataset"]["parquet_filename"])

    print(f"Exporting CSV to: {csv_path}")
    df.to_csv(csv_path, index=False)

    print(f"Exporting Parquet to: {parquet_path}")
    sanitize_arrow_registry()
    df.to_parquet(parquet_path, compression="snappy", index=False)
    print("Generation and safe export successfully finished.")


if __name__ == "__main__":
    main()
