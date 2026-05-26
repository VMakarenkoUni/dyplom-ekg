"""CNN + BiLSTM hybrid over raw beat windows. PyTorch."""

from __future__ import annotations

from typing import Any

import numpy as np

from ekg.models.base import BaseClassifier
from ekg.models.cnn import TORCH_AVAILABLE, _class_weights, _require_torch

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset


class _CNNBiLSTM(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (B, 1, L)
        z = self.cnn(x)                # (B, C, L')
        z = z.transpose(1, 2)          # (B, L', C)
        _, (h, _) = self.bilstm(z)     # h: (2, B, H)
        h = torch.cat([h[0], h[1]], dim=1)  # (B, 2H)
        return self.head(h)


class CNNBiLSTMClassifier(BaseClassifier):
    name = "cnn_bilstm"

    def __init__(self, *, batch_size: int = 256, epochs: int = 25,
                 lr: float = 1e-3, device: str | None = None) -> None:
        _require_torch()
        self._kwargs = dict(batch_size=batch_size, epochs=epochs, lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _CNNBiLSTM(n_classes=self.n_classes).to(self.device)

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def _loader(self, windows: np.ndarray, labels: np.ndarray | None = None,
                shuffle: bool = False):
        x = torch.from_numpy(windows.astype(np.float32)).unsqueeze(1)
        ds = TensorDataset(x) if labels is None else TensorDataset(
            x, torch.from_numpy(labels.astype(np.int64))
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(self.device == "cuda"))

    def fit(self, *, features=None, windows: np.ndarray, labels: np.ndarray,
            verbose: bool = True, **kwargs) -> "CNNBiLSTMClassifier":
        loader = self._loader(windows, labels, shuffle=True)
        weights = _class_weights(labels, self.n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for epoch in range(self.epochs):
            running, batches = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimiser.step()
                running += float(loss.item())
                batches += 1
            if verbose:
                avg = running / max(1, batches)
                print(f"[{self.name}] epoch {epoch + 1}/{self.epochs}  loss={avg:.4f}")
        return self

    def predict_proba(self, *, features=None, windows: np.ndarray) -> np.ndarray:
        loader = self._loader(windows, None, shuffle=False)
        self.net.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                p = torch.softmax(self.net(xb), dim=1).cpu().numpy()
                outs.append(p)
        return np.concatenate(outs, axis=0)

    def get_state(self) -> dict[str, Any]:
        return {"state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}}

    def load_state(self, payload: dict[str, Any]) -> None:
        self.net.load_state_dict(payload["state_dict"])
        self.net.to(self.device)
