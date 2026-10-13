"""
ensemble/aggregator.py — Ансамблева агрегація компонентів.
  - Зважена агрегація P_final(t) = α·P_price(t) + β·P_news(t)
  - Адаптивне зважування на основі ковзної точності F₁
  - Обмеження ваг: w_i ∈ [W_MIN, W_MAX]
"""

import numpy as np
from collections import deque
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ENSEMBLE_INIT_WEIGHTS, ENSEMBLE_W_MIN, ENSEMBLE_W_MAX,
                    ENSEMBLE_F1_WIN, ALPHA_PRICE, ALPHA_NEWS,
                    THETA_HIGH, THETA_LOW, TP_TOLERANCE)


class EnsembleAggregator:
    """
    Адаптивний ансамблевий агрегатор.

    Компоненти (індекси):
      0 — CUSUM
      1 — BOCPD
      2 — HMM
      3 — LSTM AE
      4 — FinBERT (новинний)

    Parameters
    ----------
    init_weights : list  — початкові ваги [w0..w4]
    w_min, w_max : float — межі ваг
    f1_win       : int   — вікно ковзної F₁ для адаптації
    alpha_price  : float — вага цінової компоненти (0.70)
    alpha_news   : float — вага новинної компоненти (0.30)
    """

    def __init__(
        self,
        init_weights: list = None,
        w_min: float       = ENSEMBLE_W_MIN,
        w_max: float       = ENSEMBLE_W_MAX,
        f1_win: int        = ENSEMBLE_F1_WIN,
        alpha_price: float = ALPHA_PRICE,
        alpha_news: float  = ALPHA_NEWS,
    ):
        self.w_min       = w_min
        self.w_max       = w_max
        self.f1_win      = f1_win
        self.alpha_price = alpha_price
        self.alpha_news  = alpha_news

        w = init_weights or ENSEMBLE_INIT_WEIGHTS
        self._weights = np.array(w, dtype=float)
        self._weights /= self._weights.sum()

        # Буфери для адаптивного зважування
        # Зберігаємо (predicted_prob, true_label) для кожного компонента
        self._buffers = [
            deque(maxlen=f1_win) for _ in range(5)
        ]

        self._history_pfinal: list = []
        self._history_weights: list = []
        self._t = 0

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def update(
        self,
        probs: list,
        true_label: int = None,
    ) -> float:
        """
        Онлайн-оновлення агрегатора.

        Parameters
        ----------
        probs      : list [P_CUSUM, P_BOCPD, P_HMM, P_LSTM, P_FinBERT]
        true_label : int  — 0/1 (якщо відомо з ground truth)

        Returns
        -------
        float — P_final(t) ∈ [0, 1]
        """
        probs = np.clip(np.array(probs, dtype=float), 0.0, 1.0)

        # Оновлення буферів для F₁
        if true_label is not None:
            for i, p in enumerate(probs):
                pred = 1 if p >= THETA_HIGH else 0
                self._buffers[i].append((pred, true_label))

        # Адаптація ваг кожні 50 кроків
        if self._t > 0 and self._t % 50 == 0:
            self._update_weights()

        # Агрегація: цінова + новинна
        p_price = self._weighted_price(probs[:4])
        p_news  = float(probs[4])
        p_final = self.alpha_price * p_price + self.alpha_news * p_news

        self._history_pfinal.append(float(p_final))
        self._history_weights.append(self._weights.copy())
        self._t += 1
        return float(p_final)

    def aggregate_batch(
        self,
        probs_matrix: np.ndarray,
        true_labels: np.ndarray = None,
    ) -> np.ndarray:
        """
        Пакетна агрегація.

        Parameters
        ----------
        probs_matrix : array (T, 5) — ймовірності всіх компонентів
        true_labels  : array (T,)   — еталонні мітки

        Returns
        -------
        array (T,) — P_final
        """
        results = []
        for t in range(len(probs_matrix)):
            lbl = int(true_labels[t]) if true_labels is not None else None
            p = self.update(probs_matrix[t].tolist(), lbl)
            results.append(p)
        return np.array(results)

    def reset(self):
        w = list(ENSEMBLE_INIT_WEIGHTS)
        self._weights = np.array(w, dtype=float) / sum(w)
        self._buffers = [deque(maxlen=self.f1_win) for _ in range(5)]
        self._history_pfinal = []
        self._history_weights = []
        self._t = 0

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def history_pfinal(self) -> list:
        return self._history_pfinal

    @property
    def history_weights(self) -> list:
        return self._history_weights

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _weighted_price(self, price_probs: np.ndarray) -> float:
        """Зважена агрегація цінових компонентів (CUSUM, BOCPD, HMM, LSTM)."""
        w = self._weights[:4]
        w = w / w.sum()
        return float(np.dot(w, price_probs))

    def _update_weights(self):
        """
        Оновлення ваг на основі ковзної F₁ кожного компонента.
        w_i ∝ F₁_i, з обмеженням [w_min, w_max].
        """
        f1_scores = []
        for buf in self._buffers:
            if len(buf) < 10:
                f1_scores.append(0.1)
                continue
            preds  = [x[0] for x in buf]
            labels = [x[1] for x in buf]
            f1_scores.append(self._f1(preds, labels))

        f1_arr = np.array(f1_scores, dtype=float) + 1e-6
        new_w  = f1_arr / f1_arr.sum()

        # Обмеження
        new_w  = np.clip(new_w, self.w_min, self.w_max)
        new_w /= new_w.sum()
        self._weights = new_w

    @staticmethod
    def _f1(preds: list, labels: list) -> float:
        """Обчислення F₁-score."""
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        return 2 * precision * recall / (precision + recall + 1e-9)


# ── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T = 200
    probs_matrix = np.random.rand(T, 5)
    labels = (np.random.rand(T) > 0.9).astype(int)

    agg = EnsembleAggregator()
    p_final = agg.aggregate_batch(probs_matrix, labels)
    print(f"P_final: min={p_final.min():.3f}, max={p_final.max():.3f}, "
          f"mean={p_final.mean():.3f}")
    print(f"Фінальні ваги: {agg.weights.round(3)}")
