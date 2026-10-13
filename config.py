"""
config.py — Централізована конфігурація системи.
Всі параметри моделей відповідають дипломній роботі.
"""

import random, numpy as np, os
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

from pathlib import Path

# ── Шляхи ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DB_PATH    = RESULTS_DIR / "trades.db"

# ── Інструменти та дані ───────────────────────────────────────────────────────
SYMBOLS = {
    "XAU/USD": "GLD",        # SPDR Gold ETF (денні дані 2019-2024)
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
INTERVAL   = "1d"            # денні свічки (повний період 2019-2024)

# ── Вектор ознак ──────────────────────────────────────────────────────────────
RV_WINDOW     = 20    # вікно для реалізованої волатильності (20 торг. днів)
BBW_WINDOW    = 20    # вікно смуг Боллінджера
ADX_WINDOW    = 14    # вікно ADX
NORM_WINDOW   = 120   # вікно ковзного мін-макс нормування (6 місяців)

# ── CUSUM ─────────────────────────────────────────────────────────────────────
CUSUM_WINDOW  = 30    # вікно оцінки μ̂, σ̂ (30 торг. днів)
CUSUM_K       = 0.5   # k = δ/2 (оптимальний поріг)
CUSUM_H       = 4.0   # поріг h (налаштовується grid search)
CUSUM_GAMMA   = 2.0   # крутість сигмоїди

# ── BOCPD ─────────────────────────────────────────────────────────────────────
BOCPD_LAMBDA  = 60    # очікувана довжина сегмента (~3 місяці торг. днів)
BOCPD_R_MAX   = 120   # усічення гіпотез run-length
# Normal-Gamma пріор
BOCPD_MU0     = 0.0
BOCPD_KAPPA0  = 1.0
BOCPD_ALPHA0  = 1.0
BOCPD_BETA0   = 1.0

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES    = 3     # Bullish, Bearish, Range
HMM_RETRAIN_INT = 60    # інтервал перенавчання (~3 місяці)
HMM_TRAIN_WIN   = 252   # розмір навчального вікна (~1 рік торг. днів)
HMM_FEATURES    = ["log_return_norm", "rv_norm", "log_vol_norm", "atr_range_norm"]

# ── LSTM Autoencoder ──────────────────────────────────────────────────────────
LSTM_SEQ_LEN    = 20    # довжина вхідної послідовності (20 торг. днів)
LSTM_LATENT_DIM = 16    # розмір латентного простору (d_z)
LSTM_EPOCHS     = 50
LSTM_BATCH_SIZE = 64
LSTM_LR         = 0.001
LSTM_PATIENCE   = 10    # early stopping
LSTM_THRESHOLD_SIGMA = 3.0  # θ = μ_e + 3σ_e

# ── FinBERT ───────────────────────────────────────────────────────────────────
FINBERT_MODEL   = "ProsusAI/finbert"
FINBERT_WINDOW  = 1     # часове вікно агрегації новин (днів)
FINBERT_MA_WIN  = 10    # вікно ковзного середнього сентименту (2 тижні)
FINBERT_COSINE_THRESH = 0.85  # поріг дедублікації

# ── Ансамбль ──────────────────────────────────────────────────────────────────
# Початкові ваги [CUSUM, BOCPD, HMM, LSTM, FinBERT]
ENSEMBLE_INIT_WEIGHTS = [0.20, 0.25, 0.25, 0.25, 0.05]
ENSEMBLE_W_MIN  = 0.05
ENSEMBLE_W_MAX  = 0.50
ENSEMBLE_F1_WIN = 100   # вікно ковзної F₁ для адаптації ваг
ALPHA_PRICE     = 0.70  # вага цінової компоненти
ALPHA_NEWS      = 0.30  # вага новинної компоненти

# ── Торговий модуль ───────────────────────────────────────────────────────────
THETA_LOW       = 0.35
THETA_HIGH      = 0.50
ATR_WINDOW      = 14
K_SL            = 1.5   # множник ATR для стоп-лосс
K_TP            = 2.5   # множник ATR для тейк-профіт
R_MAX           = 0.01  # максимальний ризик на угоду (1% балансу)
MAX_POSITIONS   = 3
PARTIAL_EXIT    = 0.50  # частка позиції для часткового закриття

# ── Бектест ───────────────────────────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0  # USD
SPREAD_PIPS = {
    "XAU/USD": 0.30,
    "EUR/USD": 0.00010,
    "GBP/USD": 0.00013,
}
WALK_FORWARD_STEP_MONTHS = 6
WALK_FORWARD_TEST_MONTHS = 6

# ── PELT (ground truth) ───────────────────────────────────────────────────────
PELT_PENALTY    = "BIC"
PELT_MODEL      = "rbf"

# ── Метрики ───────────────────────────────────────────────────────────────────
RISK_FREE_RATE  = 0.045 / 252  # 4.5% річних / 252 торгові дні
TP_TOLERANCE    = 5             # ±5 свічок для True Positive
