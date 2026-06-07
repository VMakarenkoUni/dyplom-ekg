"""1D-CNN over raw beat windows. PyTorch."""

from __future__ import annotations

from typing import Any

import numpy as np

from ekg.models.base import BaseClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed; pip install torch")


class _CNN1D(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_classes: int = 5, window_len: int = 260,
                 in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (B, 1, L)
        return self.classifier(self.features(x))


def _class_weights(y: np.ndarray, n_classes: int) -> "torch.Tensor":
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class CNN1DClassifier(BaseClassifier):
    name = "cnn"

    def __init__(self, *, window_len: int = 260, batch_size: int = 256,
                 epochs: int = 25, lr: float = 1e-3, device: str | None = None,
                 in_channels: int = 1) -> None:
        _require_torch()
        self._kwargs = dict(window_len=window_len, batch_size=batch_size,
                            epochs=epochs, lr=lr, in_channels=in_channels)
        self.window_len = window_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.in_channels = in_channels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _CNN1D(n_classes=self.n_classes, window_len=window_len,
                          in_channels=in_channels).to(self.device)

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def _loader(self, windows: np.ndarray, labels: np.ndarray | None = None,
                shuffle: bool = False) -> "DataLoader":
        w = windows.astype(np.float32)
        if w.ndim == 2:
            x = torch.from_numpy(w).unsqueeze(1)  # (N, 1, L)
        elif w.ndim == 3:
            x = torch.from_numpy(w)               # (N, C, L) — multi-channel
        else:
            raise ValueError(f"unsupported window ndim {w.ndim}")
        if labels is None:
            ds = TensorDataset(x)
        else:
            y = torch.from_numpy(labels.astype(np.int64))
            ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(self.device == "cuda"))

    def fit(self, *, features=None, windows: np.ndarray, labels: np.ndarray,
            verbose: bool = True, **kwargs) -> "CNN1DClassifier":
        loader = self._loader(windows, labels, shuffle=True)
        weights = _class_weights(labels, self.n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for epoch in range(self.epochs):
            running, batches = 0.0, 0
            for batch in loader:
                xb, yb = (b.to(self.device) for b in batch)
                optimiser.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
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
