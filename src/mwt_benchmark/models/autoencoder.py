"""PyTorch Multilayer Autoencoder for reconstruction-based anomaly detection."""
import time
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mwt_benchmark.utils.metrics import evaluate_detection_performance


class AutoencoderNet(nn.Module):
    """Linear Undercomplete Autoencoder."""

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PyTorchAutoencoderDetector:
    """Trains on normal operations only; flags high reconstruction error as anomaly."""

    def __init__(self, input_dim: int, latent_dim: int = 4, lr: float = 0.001, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoencoderNet(input_dim, latent_dim).to(self.device)
        self.lr = lr
        self.threshold = 0.0

    def fit(self, X_train_normal: np.ndarray, epochs: int = 10, batch_size: int = 256) -> None:
        self.model.train()
        dataset = TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_train_normal, dtype=torch.float32).to(self.device)
            recon = self.model(tensor_x)
            errors = torch.mean((tensor_x - recon) ** 2, dim=1).cpu().numpy()
            self.threshold = float(np.percentile(errors, 99.0))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        self.model.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            tensor_x = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            recon = self.model(tensor_x)
            errors = torch.mean((tensor_x - recon) ** 2, dim=1).cpu().numpy()
        inference_time = time.perf_counter() - start_time

        y_pred = (errors > self.threshold).astype(int)
        return evaluate_detection_performance(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=errors,
            latency_sec=inference_time,
            num_samples=len(X_test)
        )
