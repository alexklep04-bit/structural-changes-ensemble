"""
trading/signal_generator.py — Генератор торгових сигналів.
  - Сигнали: s_t ∈ {Buy=1, Sell=-1, Hold=0}
  - Пороги: θ_low=0.30, θ_high=0.70
  - SL/TP на основі ATR(14)
  - Фіксований ризик R_max=1% балансу
  - Максимум 3 одночасні позиції
  - Журналювання угод у SQLite
"""

import numpy as np
import pandas as pd
import sqlite3
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (THETA_LOW, THETA_HIGH, K_SL, K_TP, R_MAX,
                    MAX_POSITIONS, PARTIAL_EXIT, DB_PATH, INITIAL_BALANCE,
                    SPREAD_PIPS)


@dataclass
class Position:
    """Відкрита торгова позиція."""
    id: int
    symbol: str
    direction: int          # +1 Buy, -1 Sell
    entry_price: float
    size: float             # лоти
    sl: float               # стоп-лосс
    tp: float               # тейк-профіт
    tp_half: float          # 50% TP для часткового закриття
    open_time: pd.Timestamp
    partial_closed: bool = False
    pnl: float = 0.0


@dataclass
class Trade:
    """Закрита угода."""
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    reason: str             # "sl", "tp", "tp_half", "signal", "end"


class SignalGenerator:
    """
    Генератор торгових сигналів та управління позиціями.

    Parameters
    ----------
    symbol        : str   — інструмент
    initial_balance: float — початковий баланс
    theta_low     : float — нижній поріг (Hold зона)
    theta_high    : float — верхній поріг (активна зона)
    k_sl, k_tp   : float — множники ATR для SL/TP
    r_max         : float — максимальний ризик на угоду
    max_positions : int   — максимум одночасних позицій
    """

    def __init__(
        self,
        symbol: str           = "XAU/USD",
        initial_balance: float = INITIAL_BALANCE,
        theta_low: float      = THETA_LOW,
        theta_high: float     = THETA_HIGH,
        k_sl: float           = K_SL,
        k_tp: float           = K_TP,
        r_max: float          = R_MAX,
        max_positions: int    = MAX_POSITIONS,
    ):
        self.symbol        = symbol
        self.balance       = initial_balance
        self.initial_balance = initial_balance
        self.theta_low     = theta_low
        self.theta_high    = theta_high
        self.k_sl          = k_sl
        self.k_tp          = k_tp
        self.r_max         = r_max
        self.max_positions = max_positions
        self.spread        = SPREAD_PIPS.get(symbol, 0.0001)

        self._positions: list  = []
        self._trades: list     = []
        self._equity_curve: list = [initial_balance]
        self._pos_id: int      = 0
        self._signals: list    = []
        self._price_history: list = []  # останні 10 цін для визначення тренду

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def step(
        self,
        p_final: float,
        regime: str,
        price: float,
        atr: float,
        timestamp: pd.Timestamp,
    ) -> int:
        """
        Один крок торгового рішення.

        Parameters
        ----------
        p_final   : float — ансамблева ймовірність структурної зміни
        regime    : str   — поточний режим ("Bullish", "Bearish", "Range")
        price     : float — ціна закриття поточної свічки
        atr       : float — ATR(14)
        timestamp : pd.Timestamp

        Returns
        -------
        int — сигнал {1=Buy, -1=Sell, 0=Hold}
        """
        # 1. Оновлення відкритих позицій
        self._update_positions(price, timestamp)

        # 2. Генерація нового сигналу
        signal = self._decide(p_final, regime, price, atr, timestamp)
        self._signals.append(signal)

        # 3. Оновлення equity
        current_equity = self._calc_equity(price)
        self._equity_curve.append(current_equity)

        return signal

    def run(self, df: pd.DataFrame, p_final: np.ndarray, regimes: list) -> dict:
        """
        Повний прохід по даним.

        Parameters
        ----------
        df       : DataFrame з колонками close, atr, datetime index
        p_final  : array (T,) ансамблеві ймовірності
        regimes  : list (T,) режими

        Returns
        -------
        dict з результатами бектесту
        """
        self.reset()
        for i, (ts, row) in enumerate(df.iterrows()):
            self.step(
                p_final=float(p_final[i]),
                regime=regimes[i] if i < len(regimes) else "Range",
                price=float(row["close"]),
                atr=float(row.get("atr", row["close"] * 0.001)),
                timestamp=ts,
            )

        return self._compile_results()

    def reset(self):
        self._positions = []
        self._trades = []
        self._equity_curve = [self.initial_balance]
        self.balance = self.initial_balance
        self._pos_id = 0
        self._signals = []
        self._price_history = []

    def save_to_db(self, db_path: Path = DB_PATH):
        """Збереження угод у SQLite."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        records = [
            (t.symbol, t.direction, t.entry_price, t.exit_price,
             t.size, t.pnl, str(t.open_time), str(t.close_time), t.reason)
            for t in self._trades
        ]
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                symbol TEXT, direction INTEGER, entry_price REAL,
                exit_price REAL, size REAL, pnl REAL,
                open_time TEXT, close_time TEXT, reason TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)", records
        )
        conn.commit()
        conn.close()

    @property
    def equity_curve(self) -> list:
        return self._equity_curve

    @property
    def trades(self) -> list:
        return self._trades

    @property
    def signals(self) -> list:
        return self._signals

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _decide(self, p_final, regime, price, atr, timestamp) -> int:
        """Логіка формування торгового рішення (розділ 3.4.1)."""
        # Оновлення короткострокової ціної историї (вікно 10 барів)
        self._price_history.append(price)
        if len(self._price_history) > 10:
            self._price_history.pop(0)

        if len(self._positions) >= self.max_positions:
            return 0

        if p_final >= self.theta_high:
            # Структурна зміна — визначаємо напрям
            if regime == "Bullish":
                direction = 1
            elif regime == "Bearish":
                direction = -1
            else:
                # Range: за трендом на вікні 10 барів (зміна ціни)
                if len(self._price_history) >= 6:
                    # Лінійна регресія нахилу ціни за останніми 10 барами
                    y = np.array(self._price_history[-10:]) if len(self._price_history) >= 10 \
                        else np.array(self._price_history)
                    n = len(y)
                    x = np.arange(n, dtype=float)
                    slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (n * np.dot(x, x) - x.sum()**2 + 1e-9)
                    direction = 1 if slope > 0 else -1
                else:
                    return 0
            return self._open_position(direction, price, atr, timestamp)

        elif p_final <= self.theta_low:
            return 0  # стабільний режим — Hold
        else:
            return 0  # нейтральна зона — Hold

    def _open_position(self, direction: int, price: float,
                       atr: float, timestamp: pd.Timestamp) -> int:
        """Відкриття позиції з ATR-based SL/TP."""
        spread_adj = self.spread * direction
        entry = price + spread_adj

        sl = entry - direction * self.k_sl * atr
        tp = entry + direction * self.k_tp * atr
        tp_half = entry + direction * (self.k_tp / 2) * atr

        # Розмір позиції: L = (Balance * R_max) / (k_SL * ATR)
        risk_amount = self.balance * self.r_max
        size = risk_amount / (self.k_sl * atr + 1e-9)
        size = max(0.01, round(size, 2))

        pos = Position(
            id=self._pos_id,
            symbol=self.symbol,
            direction=direction,
            entry_price=entry,
            size=size,
            sl=sl,
            tp=tp,
            tp_half=tp_half,
            open_time=timestamp,
        )
        self._positions.append(pos)
        self._pos_id += 1
        return direction

    def _update_positions(self, price: float, timestamp: pd.Timestamp):
        """Перевірка SL/TP для відкритих позицій."""
        closed = []
        for pos in self._positions:
            # Часткове закриття на половині TP
            if not pos.partial_closed:
                if ((pos.direction == 1 and price >= pos.tp_half) or
                        (pos.direction == -1 and price <= pos.tp_half)):
                    pnl_half = pos.direction * (price - pos.entry_price) * pos.size * PARTIAL_EXIT
                    self.balance += pnl_half
                    pos.pnl += pnl_half
                    pos.size *= (1 - PARTIAL_EXIT)
                    pos.sl = pos.entry_price  # стоп на беззбитковість
                    pos.partial_closed = True

            # SL
            if ((pos.direction == 1 and price <= pos.sl) or
                    (pos.direction == -1 and price >= pos.sl)):
                pnl = pos.direction * (price - pos.entry_price) * pos.size
                self.balance += pnl
                pos.pnl += pnl
                self._trades.append(Trade(
                    pos.symbol, pos.direction, pos.entry_price, price,
                    pos.size, pos.pnl, pos.open_time, timestamp, "sl"
                ))
                closed.append(pos)

            # TP
            elif ((pos.direction == 1 and price >= pos.tp) or
                  (pos.direction == -1 and price <= pos.tp)):
                pnl = pos.direction * (price - pos.entry_price) * pos.size
                self.balance += pnl
                pos.pnl += pnl
                self._trades.append(Trade(
                    pos.symbol, pos.direction, pos.entry_price, price,
                    pos.size, pos.pnl, pos.open_time, timestamp, "tp"
                ))
                closed.append(pos)

        self._positions = [p for p in self._positions if p not in closed]

    def _calc_equity(self, price: float) -> float:
        """Поточний equity = баланс + нереалізований PnL."""
        unrealized = sum(
            p.direction * (price - p.entry_price) * p.size
            for p in self._positions
        )
        return self.balance + unrealized

    def _compile_results(self) -> dict:
        """Зведення результатів після прогону."""
        # Закрити всі залишкові позиції за останньою ціною
        if self._equity_curve:
            final_equity = self._equity_curve[-1]
        else:
            final_equity = self.balance

        return {
            "trades":       self._trades,
            "equity_curve": self._equity_curve,
            "signals":      self._signals,
            "n_trades":     len(self._trades),
            "final_equity": final_equity,
            "total_return": (final_equity - self.initial_balance) / self.initial_balance,
        }
