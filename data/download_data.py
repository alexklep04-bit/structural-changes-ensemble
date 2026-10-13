"""
data/download_data.py — Завантаження та підготовка даних.

Завантажує годинні OHLCV-дані для XAU/USD, EUR/USD, GBP/USD
через yfinance та формує вектор ознак відповідно до дипломної роботи.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
import ruptures as rpt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


def download_ohlcv(symbol_key: str, symbol_ticker: str) -> pd.DataFrame:
    """Завантажує годинні свічки через yfinance."""
    print(f"  Завантаження {symbol_key} ({symbol_ticker})...")
    df = yf.download(
        symbol_ticker,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"Немає даних для {symbol_ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "datetime"
    df.dropna(inplace=True)
    print(f"    -> {len(df)} свічок завантажено")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формує вектор ознак відповідно до розділу 3.2 дипломної роботи:
      - log_return    : логарифмічний приріст
      - rv            : реалізована волатильність на вікні RV_WINDOW
      - log_vol       : ln(volume + 1)
      - bbw           : Bollinger Band Width
      - adx           : Average Directional Index
      - atr           : Average True Range (для SL/TP)
      - atr_range     : нормований ATR (ATR / close)
    """
    df = df.copy()

    # Логарифмічний приріст
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Реалізована волатильність (стд. лог-приростів на вікні)
    df["rv"] = df["log_return"].rolling(RV_WINDOW).std()

    # Логарифм обсягу
    df["log_vol"] = np.log(df["volume"].clip(lower=1))

    # Bollinger Band Width
    bb = BollingerBands(close=df["close"], window=BBW_WINDOW, window_dev=2)
    df["bbw"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    # ADX
    adx_ind = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=ADX_WINDOW
    )
    df["adx"] = adx_ind.adx()

    # ATR (для торгового модуля)
    atr_ind = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=ATR_WINDOW
    )
    df["atr"] = atr_ind.average_true_range()
    df["atr_range"] = df["atr"] / df["close"]

    df.dropna(inplace=True)
    return df


def normalize_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Ковзна мін-макс нормалізація на вікні NORM_WINDOW (розділ 2.4).
    x_norm = (x - min_W) / (max_W - min_W + ε)
    """
    df = df.copy()
    for col in cols:
        roll_min = df[col].rolling(NORM_WINDOW).min()
        roll_max = df[col].rolling(NORM_WINDOW).max()
        df[f"{col}_norm"] = (df[col] - roll_min) / (roll_max - roll_min + 1e-9)
    df.dropna(inplace=True)
    return df


def label_changepoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Еталонна розмітка точок зміни алгоритмом PELT (розділ 3.2).
    Повертає колонку 'is_changepoint' (0/1).
    """
    # Використовуємо нормований сигнал (z-score) для стабільної роботи PELT
    raw = df["log_return"].values
    signal = (raw - raw.mean()) / (raw.std() + 1e-9)
    algo = rpt.Pelt(model="rbf", min_size=10, jump=2).fit(signal)
    try:
        bkps = algo.predict(pen=1.0)
    except Exception:
        bkps = []

    labels = np.zeros(len(df), dtype=int)
    for bkp in bkps:
        if bkp < len(df):
            labels[bkp] = 1
    df["is_changepoint"] = labels
    print(f"    -> PELT виявив {labels.sum()} точок зміни")
    return df


def load_or_download(symbol_key: str) -> pd.DataFrame:
    """Завантажує дані або читає з кешу."""
    cache_path = DATA_DIR / f"{symbol_key.replace('/', '_')}_features.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"  Читаємо кеш: {cache_path.name}")
        return pd.read_parquet(cache_path)

    ticker = SYMBOLS[symbol_key]
    df = download_ohlcv(symbol_key, ticker)
    df = build_features(df)

    feat_cols = ["log_return", "rv", "log_vol", "bbw", "adx", "atr_range"]
    df = normalize_features(df, feat_cols)
    df = label_changepoints(df)

    df.to_parquet(cache_path)
    print(f"  Збережено: {cache_path}")
    return df


def load_all_symbols() -> dict:
    """Повертає словник {symbol: DataFrame} для всіх інструментів."""
    print("\n=== Завантаження даних ===")
    data = {}
    for sym in SYMBOLS:
        data[sym] = load_or_download(sym)
        print(f"  {sym}: {len(data[sym])} рядків, "
              f"{data[sym]['is_changepoint'].sum()} точок зміни")
    return data


if __name__ == "__main__":
    data = load_all_symbols()
    for sym, df in data.items():
        print(f"\n{sym}:")
        print(df[["close", "log_return", "rv", "is_changepoint"]].tail(5))
