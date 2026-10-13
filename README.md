# Ensemble Detection of Structural Change Points in Financial Time Series

Bachelor's thesis project — NTU "Kharkiv Polytechnic Institute", 2026  
Department of System Analysis and Information-Analytical Technologies  
**Author:** Oleksandr Klepachevskyi

---

## Overview

This project implements an ensemble system for online detection of structural change points in financial time series. The system combines five heterogeneous detectors into a unified probabilistic framework, applied to forex and commodity markets (XAU/USD, EUR/USD, GBP/USD) over the period 2019–2024.

The core idea: no single method simultaneously provides low false alarm rate, online detection, and the ability to capture nonlinear market dynamics. A heterogeneous ensemble addresses all of these requirements at once.

---

## System Architecture

```
OHLCV Price Data ──┬──▶ CUSUM Detector     ──┐
                   ├──▶ BOCPD Detector     ──┤
                   ├──▶ HMM Detector       ──┼──▶ EnsembleAggregator ──▶ P_final ──▶ Signal
                   └──▶ LSTM Autoencoder   ──┤         (adaptive F₁ weights)
News Headlines  ──────▶ FinBERT Sentiment  ──┘
```

**Ensemble formula:**
```
P_final(t) = 0.70 · P_price(t) + 0.30 · P_news(t)
P_price(t) = 0.20·P_CUSUM + 0.25·P_BOCPD + 0.25·P_HMM + 0.25·P_LSTM
```

Signal is generated when `P_final > 0.50`. Trade direction is determined by HMM regime (Bullish → Buy, Bearish → Sell).

---

## Detectors

### 1. CUSUM (Page, 1954) — Statistical
Bilateral cumulative sum control chart with adaptive mean/variance estimation.
- `C⁺ₜ = max(0, C⁺ₜ₋₁ + zₜ − k)` ; `C⁻ₜ = max(0, C⁻ₜ₋₁ − zₜ − k)`
- Parameters: `k=0.5`, `h=4.0`, window `W=30`
- Output: `P_CUSUM = σ(γ · max(C⁺ₜ, C⁻ₜ))`, `γ=2.0`

### 2. BOCPD (Adams & MacKay, 2007) — Bayesian
Maintains a posterior distribution over run-length (time since last regime change). Updated via Bayes' theorem on every bar.
- Normal-Gamma conjugate prior, `λ=60` bars expected segment length
- Output: `P_BOCPD = P(R=1 | data)` — probability of a new segment starting today

### 3. HMM — Markov
3-state Gaussian HMM (Bullish / Range / Bearish), trained with Baum-Welch on a 252-bar rolling window, retrained every 60 bars.
- Output: sigmoid-transformed negative log-likelihood anomaly score

### 4. LSTM Autoencoder — Deep Learning
Sequence-to-sequence autoencoder trained on normal market windows. Anomaly = high reconstruction error.
- Architecture: `LSTM(64) → LSTM(32) → Dense(16) → RepeatVector → Decoder`
- Input: windows of 6 features (OHLCV + realized volatility), `W=50`
- Threshold: `θ = μₑ + 3σₑ` ; Output: `P_LSTM = min(1, eₜ/θ)`

### 5. FinBERT — NLP
FinBERT-based sentiment scoring of financial news headlines. Synthetic sentiment generated from log-returns for the 2019–2024 backtest period (no real news archive available).

---

## Adaptive Weighting

Component weights are updated every **50 bars** based on rolling F₁ score (window `N=100`):

```python
w_i ∝ F₁_i,  w_i ∈ [W_MIN, W_MAX]
```

See [`ensemble/aggregator.py`](ensemble/aggregator.py).

---

## Risk Management

ATR-based position sizing and stop management:
- **Stop-loss:** `1.5 × ATR(14)`
- **Take-profit:** `2.5 × ATR(14)`
- **Position size:** `(balance × 1%) / (1.5 × ATR)`

Risk/reward ratio ≥ 1:1.67 ensures profitability even below 50% win rate.

---

## Validation: Walk-Forward Backtesting

Expanding-window walk-forward methodology (no look-ahead bias):
- Minimum training window: 24 months
- Test window: 6 months per fold
- **7 folds × 3 instruments = 21 independent test cycles**

Ground truth labels generated offline with PELT (`ruptures`, `rbf` model, `pen=1.0`) — approximately 39 change points per instrument over 5 years.

---

## Results Summary

| Instrument | F₁ (avg) | SR (avg) | MDD (avg) | PF (avg) |
|------------|----------|----------|-----------|----------|
| XAU/USD    | 0.168    | −0.91    | 6.3%      | 0.93     |
| EUR/USD    | 0.202    | −1.28    | 6.4%      | 1.26     |
| GBP/USD    | 0.099    | −0.99    | 5.2%      | 0.88     |
| **Average**| **0.156**| **−1.06**| **6.0%**  | **1.02** |

**Key findings:**
- MDD < 10% in all 21 walk-forward cycles ✅
- SR target (≥1.50) met in best cycles: EUR/USD cycle 3 (+2.56), XAU/USD cycle 2 (+1.05)
- Low F₁ explained by aggressive PELT labeling + inherent online detection lag (3–10 bars)
- Main limitation: HMM direction errors dominate losses more than detection misses

---

## Project Structure

```
fincode/
├── config.py                  # Centralized configuration
├── main.py                    # Entry point
├── requirements.txt
├── data/
│   └── download_data.py       # yfinance download + PELT labeling
├── detectors/
│   ├── cusum.py               # Bilateral CUSUM
│   ├── bocpd.py               # Bayesian Online CPD
│   ├── hmm_detector.py        # Hidden Markov Model
│   ├── lstm_autoencoder.py    # LSTM Autoencoder
│   └── finbert_sentiment.py   # FinBERT sentiment
├── ensemble/
│   └── aggregator.py          # Adaptive ensemble aggregation
├── backtest/
│   └── walk_forward.py        # Walk-forward engine
├── trading/                   # Signal generation + ATR risk management
├── utils/                     # Plotting, metrics
└── results/                   # Output CSVs and figures
```

---

## Installation

```bash
git clone https://github.com/klep/structural-changes-ensemble
cd structural-changes-ensemble/fincode
pip install -r requirements.txt
```

**Python 3.11+ recommended.**

---

## Usage

```bash
cd fincode
python main.py
```

This will:
1. Download OHLCV data for XAU/USD, EUR/USD, GBP/USD via yfinance (cached to `.parquet`)
2. Label change points with PELT
3. Run walk-forward backtest across all 7 folds × 3 instruments
4. Save results to `results/`

---

## Configuration

All parameters are centralized in [`config.py`](config.py):

```python
CUSUM_K = 0.5          # Reference value k
CUSUM_H = 4.0          # Detection threshold h
BOCPD_LAMBDA = 60      # Expected segment length
HMM_N_STATES = 3       # Number of hidden states
LSTM_SEQ_LEN = 50      # Input window length
THETA_HIGH = 0.50      # Signal threshold
THETA_LOW = 0.35       # Hold zone lower bound
ATR_WINDOW = 14        # ATR period
PELT_PENALTY = 1.0     # PELT penalty parameter
ENSEMBLE_F1_WIN = 100  # Rolling F₁ buffer size
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `yfinance` | Market data download |
| `pandas` / `numpy` | Data processing |
| `PyTorch` | LSTM Autoencoder |
| `hmmlearn` | Hidden Markov Model |
| `ruptures` | PELT ground truth labeling |
| `transformers` | FinBERT sentiment |
| `scikit-learn` | Metrics (F₁, precision, recall) |
| `ta` | ATR, Bollinger Bands, ADX |

---

## License

MIT
