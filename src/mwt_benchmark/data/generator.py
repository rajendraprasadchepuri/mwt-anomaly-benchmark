"""Physically-consistent municipal pump telemetry data generator."""
from datetime import datetime, timedelta
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticTelemetryGenerator:
    """Generates continuous multivariate telemetry for centrifugal pumps with industrial fault states."""

    def __init__(self, num_days: int = 14, sample_rate_sec: int = 1, seed: int = 42) -> None:
        self.num_days = num_days
        self.sample_rate_sec = sample_rate_sec
        self.seed = seed
        self.total_samples = (self.num_days * 24 * 3600) // self.sample_rate_sec

    def generate(self) -> pd.DataFrame:
        """Executes differential simulation and generates the annotated DataFrame."""
        np.random.seed(self.seed)
        logger.info("Synthesizing %d records (%d days)...", self.total_samples, self.num_days)

        start_dt = datetime(2026, 1, 1, 0, 0, 0)
        time_index = [start_dt + timedelta(seconds=i * self.sample_rate_sec) for i in range(self.total_samples)]

        # Diurnal flow demand signal (peak hours 8am, 7pm)
        hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in time_index])
        diurnal_flow = (
            250.0
            + 60.0 * np.sin(2 * np.pi * (hours - 4) / 24)
            + 30.0 * np.sin(4 * np.pi * (hours - 8) / 24)
        )

        # VFD frequency control [25Hz to 50Hz]
        vfd_freq = 30.0 + (diurnal_flow / 350.0) * 20.0 + np.random.normal(0, 0.2, self.total_samples)
        vfd_freq = np.clip(vfd_freq, 25.0, 50.0)

        # Flow Rate (Affinity Law Q ~ N)
        flow_rate = diurnal_flow + np.random.normal(0, 2.5, self.total_samples)
        flow_rate = np.clip(flow_rate, 0.0, 420.0)

        # Pressure Head (Affinity Law H ~ N^2)
        press_suction = 1.8 + np.random.normal(0, 0.05, self.total_samples)
        press_disch = press_suction + 4.5 * (vfd_freq / 50.0) ** 2 + np.random.normal(0, 0.12, self.total_samples)

        # Motor Current
        motor_current = 28.0 + 85.0 * ((flow_rate * press_disch) / (400.0 * 8.5)) + np.random.normal(0, 0.8, self.total_samples)

        # Bearing & Stator Temperatures (°C)
        temp_bearing = 42.0 + 15.0 * (motor_current / 115.0) + np.random.normal(0, 0.3, self.total_samples)
        temp_stator = 50.0 + 25.0 * (motor_current / 115.0) + np.random.normal(0, 0.4, self.total_samples)

        # Vibration RMS (ISO 10816 standards)
        vib_x = 1.2 + 0.8 * (vfd_freq / 50.0) + np.random.normal(0, 0.08, self.total_samples)
        vib_y = 1.0 + 0.7 * (vfd_freq / 50.0) + np.random.normal(0, 0.06, self.total_samples)
        vib_z = 0.8 + 0.5 * (vfd_freq / 50.0) + np.random.normal(0, 0.05, self.total_samples)

        # Ground truth labels
        is_anomaly = np.zeros(self.total_samples, dtype=np.int8)
        fault_type = ["Normal"] * self.total_samples

        # Injected Anomaly 1: Cavitation (Day 5)
        cavit_start = int(4.375 * 24 * 3600 // self.sample_rate_sec)
        cavit_end = int(4.541 * 24 * 3600 // self.sample_rate_sec)
        press_suction[cavit_start:cavit_end] -= 0.95
        press_disch[cavit_start:cavit_end] *= 0.75
        vib_x[cavit_start:cavit_end] += np.random.exponential(3.2, cavit_end - cavit_start)
        vib_y[cavit_start:cavit_end] += np.random.exponential(2.8, cavit_end - cavit_start)
        is_anomaly[cavit_start:cavit_end] = 1
        for idx in range(cavit_start, cavit_end):
            fault_type[idx] = "Cavitation"

        # Injected Anomaly 2: Bearing Degradation / Wear (Day 10)
        bear_start = int(9.583 * 24 * 3600 // self.sample_rate_sec)
        bear_end = int(10.083 * 24 * 3600 // self.sample_rate_sec)
        ramp = np.linspace(0, 22.0, bear_end - bear_start)
        temp_bearing[bear_start:bear_end] += ramp
        vib_z[bear_start:bear_end] += np.linspace(0.5, 4.5, bear_end - bear_start) + np.random.normal(0, 0.25, bear_end - bear_start)
        is_anomaly[bear_start:bear_end] = 1
        for idx in range(bear_start, bear_end):
            fault_type[idx] = "Bearing_Wear"

        df = pd.DataFrame({
            "timestamp": time_index,
            "pump_id": "PUMP_01",
            "vibration_x": np.round(vib_x, 3).astype(np.float32),
            "vibration_y": np.round(vib_y, 3).astype(np.float32),
            "vibration_z": np.round(vib_z, 3).astype(np.float32),
            "press_suction": np.round(press_suction, 3).astype(np.float32),
            "press_disch": np.round(press_disch, 3).astype(np.float32),
            "flow_rate": np.round(flow_rate, 2).astype(np.float32),
            "temp_bearing": np.round(temp_bearing, 2).astype(np.float32),
            "temp_stator": np.round(temp_stator, 2).astype(np.float32),
            "current_L1": np.round(motor_current, 2).astype(np.float32),
            "vfd_frequency": np.round(vfd_freq, 2).astype(np.float32),
            "is_anomaly": is_anomaly,
            "fault_type": fault_type
        })
        logger.info("Generation complete. Dataset size: %s", str(df.shape))
        return df
