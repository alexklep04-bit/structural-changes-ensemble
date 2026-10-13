"""
detectors/bocpd.py — Байєсівське онлайн-виявлення точок зміни (BOCPD).

Реалізація відповідає розділу 2.2 дипломної роботи:
  - Алгоритм Адамса–МакКея (2007)
  - Normal-Gamma спряжений пріор
  - Усічення гіпотез run-length до R_MAX
  - Вихід: P_BOCPD(t) ∈ [0, 1]
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BOCPD_LAMBDA, BOCPD_R_MAX,
                    BOCPD_MU0, BOCPD_KAPPA0, BOCPD_ALPHA0, BOCPD_BETA0)


class BOCPDDetector:
    """
    Байєсівське онлайн-виявлення точок зміни (BOCPD).

    Parameters
    ----------
    lam      : int   — очікувана довжина сегмента (λ = 250 свічок)
    r_max    : int   — максимальний run-length (усічення)
    mu0      : float — гіперпараметр пріору μ₀
    kappa0   : float — гіперпараметр пріору κ₀
    alpha0   : float — гіперпараметр пріору α₀
    beta0    : float — гіперпараметр пріору β₀
    """

    def __init__(
        self,
        lam: int   = BOCPD_LAMBDA,
        r_max: int = BOCPD_R_MAX,
        mu0: float    = BOCPD_MU0,
        kappa0: float = BOCPD_KAPPA0,
        alpha0: float = BOCPD_ALPHA0,
        beta0: float  = BOCPD_BETA0,
    ):
        self.lam    = lam
        self.r_max  = r_max
        self.mu0    = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0  = beta0

        self._reset_state()

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def fit(self, series: np.ndarray) -> "BOCPDDetector":
        """
        Прогрів детектора на навчальній вибірці.
        Залишає стан "прогрітим" — не скидає після обробки.
        """
        self._reset_state()
        for x in series[-self.r_max:]:
            self.update(x)
        # НЕ скидаємо стан — він тепер прогрітий для test-фази
        return self

    def predict(self, series: np.ndarray,
                continue_state: bool = False) -> np.ndarray:
        """
        Повертає масив P_BOCPD(t) для всього ряду.

        Parameters
        ----------
        series         : масив спостережень
        continue_state : якщо True — продовжує від поточного стану (після fit),
                         не скидає стан. Використовується в walk-forward.
        """
        if not continue_state:
            self._reset_state()

        raw = np.array([self.update(x) for x in series])
        # Нормалізація відносно очікуваної ймовірності 1/lam.
        # При raw ≈ 1/lam → prob ≈ 0.5; при raw >> 1/lam → prob → 1.0
        # sigmoid(lam * raw - 1): [0→0.27, 1/lam→0.50, 2/lam→0.73, 3/lam→0.88]
        z = self.lam * raw - 1.0
        normalized = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
        self._probabilities = normalized.tolist()
        return normalized

    def update(self, x: float) -> float:
        """
        Онлайн-оновлення. Повертає P_BOCPD(t) ∈ [0, 1].

        Реалізує рекурсивне оновлення апостеріорного розподілу
        над run-length r_t (формули 2.11–2.14 дипломної роботи).
        """
        # Масиви параметрів для кожної гіпотези run-length
        # R: (r_max+1,) — поточний апостеріорний розподіл P(r_t | x_{1:t})

        # 1. Обчислення правдоподібності для кожної гіпотези (Student-t)
        pred_probs = self._predictive_prob(x)  # (r_max+1,)

        # 2. Оновлення гіпотез (growth probabilities)
        h = self._hazard(self._run_lengths[:-1])
        growth = self._R[:-1] * pred_probs[:-1] * (1.0 - h)

        # 3. Ймовірність точки зміни (changepoint probability)
        cp_prob = np.sum(self._R[:-1] * pred_probs[:-1] * h)


        # 4. Нова апостеріорна (зсув на 1 + нова гіпотеза run_length=0)
        R_new = np.zeros(self.r_max + 1)
        R_new[1:len(growth) + 1] = growth
        R_new[0] = cp_prob

        # 5. Нормування
        Z = R_new.sum()
        if Z > 0:
            R_new /= Z
        else:
            R_new[0] = 1.0

        # 6. Усічення до r_max
        self._R = R_new[:self.r_max + 1]

        # 7. Оновлення параметрів Normal-Gamma пріору для кожного сегмента
        self._update_sufficient_stats(x)

        # 8. P_BOCPD = ймовірність run_length = 0 (точка зміни)
        prob = float(self._R[0])
        self._probabilities.append(prob)
        self._t += 1
        return prob

    def reset(self):
        self._reset_state()

    @property
    def probabilities(self) -> list:
        return self._probabilities

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _reset_state(self):
        self._R = np.zeros(self.r_max + 1)
        self._R[0] = 1.0  # спочатку run_length = 0 з ймовірністю 1

        # Normal-Gamma параметри для кожної гіпотези r ∈ [0, r_max]
        self._mu    = np.full(self.r_max + 1, self.mu0)
        self._kappa = np.full(self.r_max + 1, self.kappa0)
        self._alpha = np.full(self.r_max + 1, self.alpha0)
        self._beta  = np.full(self.r_max + 1, self.beta0)

        self._run_lengths = np.arange(self.r_max + 1, dtype=float)
        self._probabilities = []
        self._t = 0

    def _hazard(self, r: np.ndarray) -> np.ndarray:
        """Модель небезпеки: геометричний пріор h(r) = 1/λ (формула 2.12)."""
        return np.full_like(r, 1.0 / self.lam, dtype=float)

    def _predictive_prob(self, x: float) -> np.ndarray:
        """
        Передбачувальна ймовірність P(x_t | r_t, x_{1:t-1}).
        Student-t розподіл з параметрами Normal-Gamma (формула 2.15).
        """
        df     = 2.0 * self._alpha
        loc    = self._mu
        scale  = np.sqrt(self._beta * (self._kappa + 1.0) /
                         (self._alpha * self._kappa))
        # Обчислення log-pdf для стабільності
        log_p = (gammaln((df + 1) / 2) - gammaln(df / 2)
                 - 0.5 * np.log(df * np.pi * scale**2)
                 - (df + 1) / 2 * np.log(1 + ((x - loc) / scale)**2 / df))
        p = np.exp(np.clip(log_p, -500, 0))
        return p

    def _update_sufficient_stats(self, x: float):
        """
        Оновлення параметрів Normal-Gamma після спостереження x
        (формули 2.15–2.18 дипломної роботи).
        """
        mu_new    = (self._kappa * self._mu + x) / (self._kappa + 1)
        kappa_new = self._kappa + 1.0
        alpha_new = self._alpha + 0.5
        beta_new  = (self._beta
                     + 0.5 * self._kappa / (self._kappa + 1)
                     * (x - self._mu)**2)

        # Зсув параметрів (нові значення для run_length r+1)
        self._mu    = np.roll(mu_new,    1)
        self._kappa = np.roll(kappa_new, 1)
        self._alpha = np.roll(alpha_new, 1)
        self._beta  = np.roll(beta_new,  1)

        # Скидання параметрів для r=0 (нова точка зміни)
        self._mu[0]    = self.mu0
        self._kappa[0] = self.kappa0
        self._alpha[0] = self.alpha0
        self._beta[0]  = self.beta0


# ── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    normal  = np.random.normal(0,   1, 300)
    shifted = np.random.normal(1.5, 1, 200)
    series  = np.concatenate([normal, shifted])

    detector = BOCPDDetector()
    probs = detector.predict(series)

    peak_idx = np.argmax(probs[250:]) + 250
    print(f"Пік ймовірності: позиція {peak_idx}, P = {probs[peak_idx]:.4f}")
    print(f"P на позиціях 295-310: {probs[295:310].round(4)}")
