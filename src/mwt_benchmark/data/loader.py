"""Dataset loading, caching, and preprocessing utilities."""
import os
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mwt_benchmark.utils.safe_arrow import sanitize_arrow_registry


class TelemetryDataLoader:
    """Manages reading, caching, and temporal train-test partitioning."""

    def __init__(self, file_path: str, feature_cols: List[str], target_col: str = "is_anomaly") -> None:
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Reads CSV or Parquet safely."""
        sanitize_arrow_registry()
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Telemetry file not found: {self.file_path}")

        if self.file_path.endswith(".parquet"):
            return pd.read_parquet(self.file_path)
        return pd.read_csv(self.file_path)

    def get_temporal_splits(
        self, train_ratio: float = 0.70, test_ratio: float = 0.30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs forward chronological train/test split to avoid temporal data leakage."""
        df = self.load_data()
        n = len(df)
        split_idx = int(n * train_ratio)

        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    def scale_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fits scaler on training data and transforms test data."""
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        train_scaled[self.feature_cols] = self.scaler.fit_transform(train_df[self.feature_cols])
        test_scaled[self.feature_cols] = self.scaler.transform(test_df[self.feature_cols])
        return train_scaled, test_scaled
