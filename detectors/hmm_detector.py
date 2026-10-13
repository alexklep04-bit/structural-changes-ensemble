"""
detectors/hmm_detector.py — Прихована марківська модель (HMM).

Реалізація відповідає розділу 2.3.1 дипломної роботи:
  - 3 прихованих стани: Bullish (0), Range (1), Bearish (2)
  - Вектор спостережень: [log_return, rv, log_vol, atr_range]
  - Gaussian emission, навчання алгоритмом Баума–Велча
  - Перенавчання кожні HMM_RETRAIN_INT свічок на вікні HMM_TRAIN_WIN
  - P_HMM(t) — ймовірність переходу до іншого стану
"""

import numpy as np
from hmmlearn import hmm
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (HMM_N_STATES, HMM_RETRAIN_INT,
                    HMM_TRAIN_WIN, HMM_FEATURES)

REGIME_NAMES = {0: "Bullish", 1: "Range", 2: "Bearish"}


class HMMDetector:
    """
    HMM детектор ринкових режимів та структурних змін.

    Parameters
    ----------
    n_states    : int — кількість прихованих станів (3)
    retrain_int : int — інтервал перенавчання (250 свічок)
    train_win   : int — розмір навчального вікна (500 свічок)
    features    : list — назви ознак у DataFrame
    """

    def __init__(
        self,
        n_states: int    = HMM_N_STATES,
        retrain_int: int = HMM_RETRAIN_INT,
        train_win: int   = HMM_TRAIN_WIN,
        features: list   = None,
    ):
        self.n_states    = n_states
        self.retrain_int = retrain_int
        self.train_win   = train_win
        self.features    = features or HMM_FEATURES
        self._model      = None
        self._is_fitted  = False
        self._prev_state = None
        self._probabilities: list = []
        self._regimes: list = []
        self._t = 0

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "HMMDetector":
        """
        Навчання моделі алгоритмом Баума–Велча.
        X: array (T, n_features)
        """
        self._model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            random_state=42,
            init_params="stmc",   # ініціалізуємо всі параметри
            min_covar=1e-4,       # регуляризація — мінімальна дисперсія
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X)

        # Регуляризація transmat_ — усуваємо нульові рядки
        row_sums = self._model.transmat_.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self._model.transmat_ = self._model.transmat_ / row_sums

        self._is_fitted = True
        self._state_map = {i: i for i in range(self.n_states)}
        self._sort_states(X)

        # Калібрування log-likelihood на навчальних даних для аномалій
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_ll = self._model.score_samples(X)
            self._ll_mean = float(np.mean(train_ll))
            self._ll_std  = float(np.std(train_ll)) + 1e-9
        except Exception:
            self._ll_mean = 0.0
            self._ll_std  = 1.0

        # Зберігаємо останній стан для безперервного predict
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                last_post = self._model.predict_proba(X[-1:])[0]
            self._prev_state = int(np.argmax(last_post))
        except Exception:
            self._prev_state = None

        return self

    def predict(self, X: np.ndarray, use_pretrained: bool = False) -> np.ndarray:
        """
        Прогноз ймовірностей P_HMM(t) для масиву X.

        Parameters
        ----------
        X             : array (T, n_features)
        use_pretrained: якщо True — використовує вже навчену модель (не перенавчає на X).
                        Використовується в walk-forward режимі після fit(train_data).
        """
        self._probabilities = []
        self._regimes = []
        self._t = 0
        if not use_pretrained:
            self._prev_state = None  # скидаємо лише в автономному режимі

        if not use_pretrained or not self._is_fitted:
            # Режим self-supervised: навчання на початковому вікні X
            init_win = min(self.train_win, len(X))
            self.fit(X[:init_win])

        raw_probs = []
        raw_regimes = []
        for i, obs in enumerate(X):
            # Перенавчання лише в автономному (не pretrained) режимі
            if (not use_pretrained and i > 0
                    and i % self.retrain_int == 0
                    and i >= self.train_win):
                start = i - self.train_win
                self.fit(X[start:i])

            p, regime = self._step(obs.reshape(1, -1))
            raw_probs.append(p)
            raw_regimes.append(regime)
            self._t += 1

        # Перенормалізація відносно медіани вікна (усуваємо систематичне зміщення)
        # Гарантує: типовий бар → 0.5; аномальний → > 0.5
        probs_arr = np.array(raw_probs)
        med = np.median(probs_arr)
        mad = np.median(np.abs(probs_arr - med)) + 1e-9
        z   = (probs_arr - med) / (1.4826 * mad)
        probs_arr = 1.0 / (1.0 + np.exp(-np.clip(z, -5, 5)))

        self._probabilities = probs_arr.tolist()
        self._regimes = raw_regimes
        return probs_arr

    def update(self, obs: np.ndarray) -> float:
        """
        Онлайн-оновлення для одного спостереження obs (n_features,).
        Повертає P_HMM(t).
        """
        if not self._is_fitted:
            self._probabilities.append(0.0)
            self._regimes.append(1)
            return 0.0
        p, regime = self._step(obs.reshape(1, -1))
        self._probabilities.append(p)
        self._regimes.append(regime)
        self._t += 1
        return p

    def get_regime(self, t: int = -1) -> str:
        """Повертає назву режиму на позиції t."""
        if not self._regimes:
            return "Range"
        idx = self._regimes[t] if t != -1 else self._regimes[-1]
        return REGIME_NAMES.get(idx, "Range")

    def reset(self):
        self._is_fitted = False
        self._prev_state = None
        self._probabilities = []
        self._regimes = []
        self._t = 0

    @property
    def probabilities(self) -> list:
        return self._probabilities

    @property
    def regimes(self) -> list:
        return [REGIME_NAMES.get(r, "Range") for r in self._regimes]

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _step(self, obs: np.ndarray):
        """
        Один крок аномалії HMM. Повертає (P_HMM, mapped_state).

        P_HMM = sigmoid нормованого від'ємного log-likelihood.
        Висока P_HMM → спостереження аномальне відносно навченого розподілу
        → ймовірна структурна зміна.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                log_ll = self._model.score_samples(obs)  # (1,)
                ll = float(log_ll[0])
                posteriors = self._model.predict_proba(obs)[0]
                current_state = int(np.argmax(posteriors))
            except Exception:
                self._prev_state = None
                return 0.3, 1  # нейтральне значення при помилці

        # Нормований Z-score від'ємного LL → sigmoid → P_HMM
        ll_mean = getattr(self, "_ll_mean", 0.0)
        ll_std  = getattr(self, "_ll_std",  1.0)
        z = -(ll - ll_mean) / ll_std   # від'ємний: низький LL = аномалія = позитивний z
        prob = float(1.0 / (1.0 + np.exp(-np.clip(z, -5, 5))))

        self._prev_state = current_state
        mapped = getattr(self, "_state_map", {}).get(current_state, current_state)
        return prob, mapped

    def _sort_states(self, X: np.ndarray):
        """
        Упорядкування станів: 0=Bullish (max return), 1=Range, 2=Bearish.
        Зберігаємо mapping окремо, не переставляємо матриці моделі
        (уникаємо проблем з валідацією covars у hmmlearn).
        """
        means_0 = self._model.means_[:, 0]
        perm = list(np.argsort(means_0)[::-1])  # найбільший mean → Bullish
        # Зберігаємо mapping: внутрішній стан → семантична назва
        self._state_map = {perm[0]: 0, perm[1]: 1, perm[2]: 2}  # 0=Bull,1=Range,2=Bear


# ── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T = 600
    X = np.column_stack([
        np.concatenate([np.random.normal(0.001, 0.01, 300),
                        np.random.normal(-0.001, 0.02, 300)]),
        np.abs(np.random.normal(0.01, 0.005, T)),
        np.random.normal(10, 1, T),
        np.random.normal(0.005, 0.001, T),
    ])

    det = HMMDetector()
    probs = det.predict(X)
    print(f"Форма виходу: {probs.shape}")
    print(f"Режими (останні 5): {det.regimes[-5:]}")
    print(f"Макс. ймовірність: {probs.max():.4f}")
