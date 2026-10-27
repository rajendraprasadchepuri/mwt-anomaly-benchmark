# Municipal Water Treatment Telemetry Benchmark (MWT-Anom24)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official open-source benchmark repository for the paper:  
**"An Annotated High-Frequency Telemetry Dataset for Industrial Anomaly Detection and Predictive Maintenance in Municipal Water Infrastructure"** (Target: *MDPI Data / MDPI Eng*).

---

## Overview
`MWT-Anom24` provides continuous 1 Hz multi-sensor telemetry capturing electrical, hydraulic, thermal, and vibration signals from high-capacity municipal centrifugal pumps. It contains ground-truth physical annotations for cavitation, bearing wear, and nominal operations.

## Quickstart

### 1. Installation
Clone the repository and install dependencies in editable mode:
```bash
git clone https://github.com/your-username/mwt-anomaly-benchmark.git
cd mwt-anomaly-benchmark
pip install -e .
```

### 2. Generate Dataset
Synthesize the 14-day 1 Hz benchmark (`1,209,600` records):
```bash
python scripts/generate_dataset.py
```
This generates:
* `data/mwt_pump_telemetry_benchmark.parquet` (Compressed Apache Parquet format)
* `data/mwt_pump_telemetry_benchmark.csv` (Standard delimited format)

### 3. Run Baseline Benchmark
Train and evaluate Isolation Forest, Autoencoder, and XGBoost baselines:
```bash
python scripts/run_benchmark.py
```

---

## Dataset Schema

| Sensor / Metric | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime | ISO 8601 | UTC 1 Hz discrete sample |
| `vibration_x` | Float32 | mm/s (RMS) | Radial Horizontal Vibration |
| `vibration_y` | Float32 | mm/s (RMS) | Radial Vertical Vibration |
| `vibration_z` | Float32 | mm/s (RMS) | Axial Thrust Vibration |
| `press_suction`| Float32 | bar | Intake Suction Pressure |
| `press_disch`  | Float32 | bar | Discharge Pumping Head |
| `flow_rate`    | Float32 | m³/h | Volumetric Water Flow |
| `temp_bearing` | Float32 | °C | Drive-End Bearing Shell Temp |
| `temp_stator`  | Float32 | °C | Motor Stator Winding Temp |
| `current_L1`   | Float32 | A | Phase 1 Electrical Current |
| `vfd_frequency`| Float32 | Hz | Operating Frequency (25–50 Hz) |
| `is_anomaly`   | Int8 | {0, 1} | Binary Anomaly Label |
| `fault_type`   | String | Categorical | Fault Mode Taxonomy |

---

## Citation
If you use this benchmark dataset or code in your research, please cite:
```bibtex
@article{chepuri2026mwt,
  title={An Annotated High-Frequency Telemetry Dataset for Industrial Anomaly Detection and Predictive Maintenance in Municipal Water Infrastructure},
  author={Chepuri, Rajendra Prasad},
  journal={MDPI Data},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX}
}
```
