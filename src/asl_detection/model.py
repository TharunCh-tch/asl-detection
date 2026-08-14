"""A small PyTorch MLP classifier over 63-dim hand-landmark feature vectors.

This is deliberately a small, real neural network (not a CNN over pixels):
a couple of fully-connected layers with batch normalization, dropout, and
ReLU activations, trained with a learning-rate scheduler. Landmark inputs
are only 63 numbers, so a deep/wide CNN would be massive overkill (and
would need far more data + a GPU to train well) -- an MLP is the
appropriately-sized model for this feature representation, and is fast
enough to train and run inference with on a CPU-only machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn

from asl_detection.landmarks import FEATURE_DIM


class ASLMLP(nn.Module):
    """Feed-forward classifier: 63 -> 128 -> 64 -> num_classes."""

    def __init__(self, num_classes: int, input_dim: int = FEATURE_DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FeatureScaler:
    """Standardizes features using train-set mean/std (fit once, reused everywhere)."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.ones(FEATURE_DIM, dtype=np.float32))

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1e-6
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(mean=np.array(d["mean"], dtype=np.float32), std=np.array(d["std"], dtype=np.float32))


class ASLClassifier:
    """High-level train/predict wrapper bundling the model + label encoder + scaler."""

    def __init__(self, classes: list[str], input_dim: int = FEATURE_DIM, dropout: float = 0.3):
        self.classes = list(classes)
        self.input_dim = input_dim
        self.model = ASLMLP(num_classes=len(self.classes), input_dim=input_dim, dropout=dropout)
        self.scaler = FeatureScaler()

    def _label_to_idx(self, labels: list[str]) -> np.ndarray:
        idx_map = {c: i for i, c in enumerate(self.classes)}
        return np.array([idx_map[label] for label in labels], dtype=np.int64)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: list[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[list[str]] = None,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 8,
        verbose: bool = True,
    ) -> dict:
        self.scaler.fit(X_train)
        Xt = torch.from_numpy(self.scaler.transform(X_train))
        yt = torch.from_numpy(self._label_to_idx(y_train))

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = torch.from_numpy(self.scaler.transform(X_val))
            yv = torch.from_numpy(self._label_to_idx(y_val))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss()

        n = Xt.shape[0]
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = Xt[idx], yt[idx]
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]
            epoch_loss /= n
            history["train_loss"].append(epoch_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(Xv)
                    val_loss = criterion(out, yv).item()
                    val_acc = (out.argmax(dim=1) == yv).float().mean().item()
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                    print(f"epoch {epoch:3d}  train_loss={epoch_loss:.4f}  "
                          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                    break
            elif verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:3d}  train_loss={epoch_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xs))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> list[str]:
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return [self.classes[i] for i in idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "classes": self.classes,
            "input_dim": self.input_dim,
            "state_dict": self.model.state_dict(),
            "scaler": self.scaler.to_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "ASLClassifier":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls(classes=checkpoint["classes"], input_dim=checkpoint["input_dim"])
        clf.model.load_state_dict(checkpoint["state_dict"])
        clf.model.eval()
        clf.scaler = FeatureScaler.from_dict(checkpoint["scaler"])
        return clf
