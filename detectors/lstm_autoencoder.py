"""
detectors/lstm_autoencoder.py — LSTM Autoencoder для виявлення аномалій.

Реалізація відповідає розділу 2.3.2 дипломної роботи:
  - Архітектура: LSTM(64) → LSTM(32) → Dense(16) → RepeatVector → Decoder
  - Навчання виключно на «нормальних» сегментах ряду
  - Поріг θ = μ_e + 3·σ_e (реконструкційна помилка)
  - P_LSTM(t) ∈ [0, 1] через нормовану помилку
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import RANDOM_SEED as _SEED
torch.manual_seed(_SEED); torch.backends.cudnn.deterministic = True
from collections import deque
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (LSTM_SEQ_LEN, LSTM_LATENT_DIM, LSTM_EPOCHS,
                    LSTM_BATCH_SIZE, LSTM_LR, LSTM_PATIENCE,
                    LSTM_THRESHOLD_SIGMA)


class _Encoder(nn.Module):
    """LSTM(64) → LSTM(32) → Dense(latent_dim)."""
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc    = nn.Linear(32, latent_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, (h, _) = self.lstm2(out)
        z = self.fc(h.squeeze(0))
        return z


class _Decoder(nn.Module):
    """Dense(latent_dim) → RepeatVector → LSTM(32) → LSTM(64) → output."""
    def __init__(self, n_features: int, seq_len: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc      = nn.Linear(latent_dim, 32)
        self.lstm1   = nn.LSTM(32, 32, batch_first=True)
        self.lstm2   = nn.LSTM(32, 64, batch_first=True)
        self.out     = nn.Linear(64, n_features)

    def forward(self, z):
        # RepeatVector: (batch, latent) → (batch, seq_len, 32)
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(h)
        out, _ = self.lstm2(out)
        return self.out(out)


class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, latent_dim: int):
        super().__init__()
        self.encoder = _Encoder(n_features, latent_dim)
        self.decoder = _Decoder(n_features, seq_len, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAutoencoder:
    """
    LSTM Autoencoder детектор аномалій / структурних змін.

    Parameters
    ----------
    seq_len    : int   — довжина вхідної послідовності (24 свічки)
    latent_dim : int   — розмір латентного простору d_z (16)
    epochs     : int   — кількість епох навчання
    batch_size : int   — розмір батчу
    lr         : float — швидкість навчання (Adam)
    patience   : int   — early stopping
    threshold_sigma : float — поріг θ = μ_e + sigma·σ_e
    """

    def __init__(
        self,
        seq_len: int         = LSTM_SEQ_LEN,
        latent_dim: int      = LSTM_LATENT_DIM,
        epochs: int          = LSTM_EPOCHS,
        batch_size: int      = LSTM_BATCH_SIZE,
        lr: float            = LSTM_LR,
        patience: int        = LSTM_PATIENCE,
        threshold_sigma: float = LSTM_THRESHOLD_SIGMA,
    ):
        self.seq_len         = seq_len
        self.latent_dim      = latent_dim
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.patience        = patience
        self.threshold_sigma = threshold_sigma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _LSTMAutoencoder = None
        self._threshold: float = None
        self._mu_e: float = None
        self._sigma_e: float = None
        self._is_fitted = False
        self._probabilities: list = []
        self._buffer: deque = deque(maxlen=seq_len)

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "LSTMAutoencoder":
        """
        Навчання на «нормальних» даних.
        X: array (T, n_features)
        """
        n_features = X.shape[1]
        self._model = _LSTMAutoencoder(n_features, self.seq_len, self.latent_dim)
        self._model.to(self.device)

        # Формування послідовностей
        seqs = self._make_sequences(X)
        if len(seqs) < self.batch_size:
            print("  LSTM: замало даних для навчання")
            return self

        dataset = TensorDataset(torch.FloatTensor(seqs))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        patience_cnt = 0

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self._model(batch)
                loss  = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            if epoch % 10 == 0:
                print(f"    LSTM epoch {epoch:3d}/{self.epochs}  loss={epoch_loss:.6f}")

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience_cnt = 0
                best_weights = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"    Early stopping на епосі {epoch}")
                    break

        self._model.load_state_dict(best_weights)

        # Обчислення порогу θ = μ_e + 3·σ_e
        errors = self._reconstruction_errors(X)
        self._mu_e     = float(np.mean(errors))
        self._sigma_e  = float(np.std(errors))
        self._threshold = self._mu_e + self.threshold_sigma * self._sigma_e
        self._is_fitted = True
        print(f"    Поріг θ = {self._threshold:.6f}  "
              f"(μ={self._mu_e:.6f}, σ={self._sigma_e:.6f})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Повертає P_LSTM(t) для всього ряду."""
        if not self._is_fitted:
            return np.zeros(len(X))
        errors = self._reconstruction_errors(X)
        probs  = self._error_to_prob(errors)
        self._probabilities = probs.tolist()
        return probs

    def update(self, obs: np.ndarray) -> float:
        """Онлайн-оновлення для одного спостереження."""
        if not self._is_fitted:
            return 0.0
        self._buffer.append(obs)
        if len(self._buffer) < self.seq_len:
            return 0.0
        seq = np.array(list(self._buffer))
        err = self._single_error(seq)
        prob = float(self._error_to_prob(np.array([err]))[0])
        self._probabilities.append(prob)
        return prob

    def reset(self):
        self._buffer.clear()
        self._probabilities = []

    @property
    def probabilities(self) -> list:
        return self._probabilities

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        """Формує ковзні послідовності довжиною seq_len."""
        n = len(X) - self.seq_len + 1
        return np.array([X[i:i + self.seq_len] for i in range(n)])

    @torch.no_grad()
    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """MSE між входом і реконструкцією для кожного кроку."""
        seqs = self._make_sequences(X)          # (n, seq_len, features)
        t = torch.FloatTensor(seqs).to(self.device)
        self._model.eval()
        recon = self._model(t).cpu().numpy()
        errors = np.mean((seqs - recon) ** 2, axis=(1, 2))
        # Вирівнювання: перші seq_len-1 мають ту саму помилку, що й перша
        pad = np.full(self.seq_len - 1, errors[0])
        return np.concatenate([pad, errors])

    @torch.no_grad()
    def _single_error(self, seq: np.ndarray) -> float:
        """MSE для однієї послідовності."""
        t = torch.FloatTensor(seq[np.newaxis]).to(self.device)
        self._model.eval()
        recon = self._model(t).cpu().numpy()
        return float(np.mean((seq - recon) ** 2))

    def _error_to_prob(self, errors: np.ndarray) -> np.ndarray:
        """
        Перетворення помилки реконструкції в ймовірність (формула 2.37–2.38).
        P = sigmoid(γ · (e - θ) / σ_e)
        """
        if self._threshold is None or self._sigma_e is None:
            return np.zeros_like(errors)
        normalized = (errors - self._threshold) / (self._sigma_e + 1e-9)
        return 1.0 / (1.0 + np.exp(-2.0 * normalized))


# ── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T, F = 500, 4
    X_train = np.random.normal(0, 1, (T, F)).astype(np.float32)
    X_test  = np.vstack([
        np.random.normal(0, 1, (300, F)),
        np.random.normal(3, 1, (200, F)),   # аномальний сегмент
    ]).astype(np.float32)

    det = LSTMAutoencoder(epochs=5)
    det.fit(X_train)
    probs = det.predict(X_test)
    print(f"Середня P нормальна: {probs[:300].mean():.4f}")
    print(f"Середня P аномальна: {probs[300:].mean():.4f}")
