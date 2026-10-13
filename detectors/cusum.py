"""
  - Двосторонній CUSUM (Page, 1954) з адаптивним оцінюванням μ̂, σ̂
  - Стандартизований залишок z_t = (x_t - μ̂) / σ̂
  - Сигмоїдне перетворення у ймовірність P_CUSUM(t) ∈ [0,1]
"""

import numpy as np
from collections import deque
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CUSUM_WINDOW, CUSUM_K, CUSUM_H, CUSUM_GAMMA


class CUSUMDetector:
    """
    Адаптивний двосторонній CUSUM детектор.

    Parameters
    ----------
    window : int   — вікно оцінки μ̂, σ̂ (W = 50 свічок)
    k      : float — еталонне зміщення k = δ/2
    h      : float — поріг спрацьовування h
    gamma  : float — крутість сигмоїди γ
    """

    def __init__(
        self,
        window: int = CUSUM_WINDOW,
        k: float = CUSUM_K,
        h: float = CUSUM_H,
        gamma: float = CUSUM_GAMMA,
    ):
        self.window = window
        self.k = k
        self.h = h
        self.gamma = gamma

        self._buffer: deque = deque(maxlen=window)
        self._C_pos: float = 0.0   # C⁺_t
        self._C_neg: float = 0.0   # C⁻_t
        self._probabilities: list = []
        self._changepoints: list = []
        self._t: int = 0

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def fit(self, series: np.ndarray) -> "CUSUMDetector":
        """Ініціалізація на навчальній вибірці (заповнення буфера)."""
        self.reset()
        for x in series:
            self._buffer.append(x)
        return self

    def predict(self, series: np.ndarray,
                continue_state: bool = False) -> np.ndarray:
        """
        Повертає масив ймовірностей P_CUSUM(t) для всього ряду.

        Parameters
        ----------
        continue_state : якщо True — продовжує від поточного стану після fit().
        """
        if not continue_state:
            self.reset()
        probs = []
        for x in series:
            p = self.update(x)
            probs.append(p)
        return np.array(probs)

    def update(self, x: float) -> float:
        """
        Онлайн-оновлення на одному спостереженні.
        Повертає P_CUSUM(t) ∈ [0, 1].
        """
        self._buffer.append(x)
        self._t += 1

        if len(self._buffer) < self.window:
            self._probabilities.append(0.0)
            return 0.0

        mu_hat = np.mean(self._buffer)
        sigma_hat = np.std(self._buffer) + 1e-9

        # Стандартизований залишок (формула 2.9)
        z_t = (x - mu_hat) / sigma_hat

        # Двостороннє накопичення (формули 2.4, 2.5)
        self._C_pos = max(0.0, self._C_pos + z_t - self.k)
        self._C_neg = max(0.0, self._C_neg - z_t - self.k)

        # Максимальна статистика
        C_max = max(self._C_pos, self._C_neg)

        # Перевірка на перевищення порогу (скидання)
        if C_max >= self.h:
            self._changepoints.append(self._t)
            self._C_pos = 0.0
            self._C_neg = 0.0

        # Сигмоїдне перетворення (формула 2.10)
        prob = self._sigmoid(C_max)
        self._probabilities.append(prob)
        return prob

    def reset(self):
        """Скидання стану."""
        self._buffer.clear()
        self._C_pos = 0.0
        self._C_neg = 0.0
        self._probabilities = []
        self._changepoints = []
        self._t = 0

    @property
    def changepoints(self) -> list:
        return self._changepoints

    @property
    def probabilities(self) -> list:
        return self._probabilities

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _sigmoid(self, x: float) -> float:
        """σ(γ·(x/h − 1)) — нормована сигмоїда."""
        val = self.gamma * (x / self.h - 1.0)
        return float(1.0 / (1.0 + np.exp(-val)))


# ── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    # Синтетичний ряд із точкою зміни на позиції 300
    normal = np.random.normal(0, 1, 300)
    shifted = np.random.normal(1.5, 1, 200)
    series = np.concatenate([normal, shifted])

    detector = CUSUMDetector()
    probs = detector.predict(series)

    print(f"Виявлені точки зміни: {detector.changepoints}")
    print(f"Макс. ймовірність: {probs.max():.4f}")
    print(f"Ймовірність на позиції 300-310: {probs[300:310].round(3)}")
