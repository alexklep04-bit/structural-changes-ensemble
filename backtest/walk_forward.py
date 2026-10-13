"""
backtest/walk_forward.py — Walk-forward бектест з розширюваним вікном.
  - Expanding window: навчання завжди з 2019 року, розширюється по 6 місяців
  - Тестове вікно: 6 місяців
  - Всі компоненти перенавчаються виключно на навчальних даних
  - Метрики: F₁, SR, MDD, PF, CS
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (WALK_FORWARD_STEP_MONTHS, WALK_FORWARD_TEST_MONTHS,
                    INITIAL_BALANCE, TP_TOLERANCE, RISK_FREE_RATE,
                    HMM_FEATURES, LSTM_SEQ_LEN)
from detectors import (CUSUMDetector, BOCPDDetector, HMMDetector,
                       LSTMAutoencoder, FinBERTSentiment)
from ensemble import EnsembleAggregator
from trading import SignalGenerator
from utils.metrics import compute_all_metrics


class WalkForwardBacktest:
    """
    Walk-forward бектест ансамблевої системи.

    Parameters
    ----------
    symbol : str — інструмент ("XAU/USD", "EUR/USD", "GBP/USD")
    use_finbert_transformers : bool — True = FinBERT, False = lexicon
    verbose : bool — виводити прогрес
    """

    def __init__(
        self,
        symbol: str = "XAU/USD",
        use_finbert_transformers: bool = False,
        verbose: bool = True,
    ):
        self.symbol   = symbol
        self.verbose  = verbose
        self.use_transformers = use_finbert_transformers
        self._fold_results: list = []

    # ── Публічний інтерфейс ─────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict:
        """
        Запуск повного walk-forward бектесту.

        Parameters
        ----------
        df : DataFrame з колонками close, log_return, rv, log_vol,
             atr_range, atr, bbw, adx, is_changepoint + DatetimeIndex

        Returns
        -------
        dict з агрегованими метриками по всіх циклах
        """
        self._fold_results = []
        start_date = df.index[0]
        end_date   = df.index[-1]

        # Генерація циклів walk-forward
        folds = self._generate_folds(start_date, end_date)
        if self.verbose:
            print(f"\n=== Walk-Forward Бектест: {self.symbol} ===")
            print(f"  Циклів: {len(folds)}")

        for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
            if self.verbose:
                print(f"\n--- Цикл {fold_idx+1}/{len(folds)} ---")
                print(f"  Train: {start_date.date()} → {train_end.date()}")
                print(f"  Test:  {test_start.date()} → {test_end.date()}")

            train_df = df[df.index <= train_end]
            test_df  = df[(df.index > train_end) & (df.index <= test_end)]

            if len(train_df) < 200 or len(test_df) < 20:
                if self.verbose:
                    print("  Пропускаємо: замало даних")
                continue

            fold_result = self._run_fold(fold_idx, train_df, test_df)
            self._fold_results.append(fold_result)

            if self.verbose:
                m = fold_result["metrics"]
                print(f"  F₁={m.get('f1',0):.3f}  "
                      f"SR={m.get('sharpe',0):.2f}  "
                      f"MDD={m.get('mdd',0):.3f}  "
                      f"PF={m.get('profit_factor',0):.2f}")

        return self._aggregate_results()

    @property
    def fold_results(self) -> list:
        return self._fold_results

    # ── Внутрішні методи ────────────────────────────────────────────────────

    def _generate_folds(self, start, end):
        """Генерація пар (train_end, test_start, test_end)."""
        folds = []
        # Мінімальне навчальне вікно: 24 місяці (2019-2020)
        train_end = start + relativedelta(months=24)
        while train_end < end:
            test_start = train_end
            test_end   = min(
                test_start + relativedelta(months=WALK_FORWARD_TEST_MONTHS),
                end
            )
            if test_end <= test_start:
                break
            folds.append((train_end, test_start, test_end))
            train_end += relativedelta(months=WALK_FORWARD_STEP_MONTHS)
        return folds

    def _run_fold(self, fold_idx: int, train_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> dict:
        """Один цикл: навчання + тест."""

        # ── Ініціалізація компонентів ─────────────────────────────────────
        cusum  = CUSUMDetector()
        bocpd  = BOCPDDetector()
        hmm    = HMMDetector()
        lstm   = LSTMAutoencoder(epochs=30)
        finbert = FinBERTSentiment(
            symbol=self.symbol,
            use_transformers=self.use_transformers
        )
        agg    = EnsembleAggregator()
        trader = SignalGenerator(
            symbol=self.symbol,
            initial_balance=INITIAL_BALANCE
        )

        # ── Навчання (тільки на train_df) ────────────────────────────────
        train_ret  = train_df["log_return"].values
        train_feat = train_df[HMM_FEATURES].values.astype(np.float32)

        if self.verbose:
            print("  Навчання CUSUM...")
        cusum.fit(train_ret)

        if self.verbose:
            print("  Навчання BOCPD...")
        bocpd.fit(train_ret)

        if self.verbose:
            print("  Навчання HMM...")
        hmm.fit(train_feat)

        # Ознаки для LSTM (нормовані)
        lstm_cols = ["log_return", "rv", "log_vol", "atr_range"]
        norm_cols = [c + "_norm" for c in lstm_cols if c + "_norm" in train_df.columns]
        if not norm_cols:
            norm_cols = lstm_cols

        train_lstm = train_df[norm_cols].values.astype(np.float32)
        if self.verbose:
            print("  Навчання LSTM Autoencoder...")
        lstm.fit(train_lstm)

        if self.verbose:
            print("  Підготовка FinBERT...")
        finbert.fit(train_df)

        # ── Тестування (тільки на test_df) ───────────────────────────────
        if self.verbose:
            print("  Тестування...")

        test_ret   = test_df["log_return"].values
        test_feat  = test_df[HMM_FEATURES].values.astype(np.float32)
        test_lstm  = test_df[norm_cols].values.astype(np.float32)
        true_labels = test_df["is_changepoint"].values.astype(int)

        # Прогнози детекторів (use_pretrained/continue_state = зберігаємо навчені параметри)
        p_cusum  = cusum.predict(test_ret, continue_state=True)
        p_bocpd  = bocpd.predict(test_ret, continue_state=True)
        p_hmm    = hmm.predict(test_feat, use_pretrained=True)
        p_lstm   = lstm.predict(test_lstm)
        p_news   = finbert.predict(test_df)

        probs_matrix = np.column_stack([p_cusum, p_bocpd, p_hmm, p_lstm, p_news])

        # Ансамблева агрегація
        p_final = agg.aggregate_batch(probs_matrix, true_labels)

        # Торговий прогін
        regimes = hmm.regimes if len(hmm.regimes) == len(test_df) else ["Range"] * len(test_df)
        trade_result = trader.run(test_df, p_final, regimes)

        # ── Метрики ────────────────────────────────────────────────────────
        metrics = compute_all_metrics(
            p_final=p_final,
            true_labels=true_labels,
            equity_curve=trade_result["equity_curve"],
            trades=trade_result["trades"],
        )

        return {
            "fold":         fold_idx + 1,
            "train_size":   len(train_df),
            "test_size":    len(test_df),
            "metrics":      metrics,
            "p_final":      p_final,
            "true_labels":  true_labels,
            "equity_curve": trade_result["equity_curve"],
            "weights":      agg.history_weights[-1] if agg.history_weights else None,
            "n_trades":     trade_result["n_trades"],
        }

    def _aggregate_results(self) -> dict:
        """Агрегація метрик по всіх циклах."""
        if not self._fold_results:
            return {}

        metric_keys = ["f1", "precision", "recall", "sharpe",
                       "mdd", "profit_factor", "composite_score",
                       "total_return", "n_trades"]

        agg = {}
        for key in metric_keys:
            vals = [r["metrics"].get(key, 0.0) for r in self._fold_results]
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }

        # Загальна equity-крива (послідовна)
        all_equity = []
        for r in self._fold_results:
            all_equity.extend(r["equity_curve"])

        agg["equity_curve"]  = all_equity
        agg["n_folds"]       = len(self._fold_results)
        agg["fold_results"]  = self._fold_results

        if self.verbose:
            print(f"\n=== Агреговані результати ({self.symbol}) ===")
            print(f"  F₁:     {agg['f1']['mean']:.3f} ± {agg['f1']['std']:.3f}")
            print(f"  SR:     {agg['sharpe']['mean']:.2f} ± {agg['sharpe']['std']:.2f}")
            print(f"  MDD:    {agg['mdd']['mean']:.3f} ± {agg['mdd']['std']:.3f}")
            print(f"  PF:     {agg['profit_factor']['mean']:.2f}")
            print(f"  Return: {agg['total_return']['mean']*100:.1f}%")

        return agg
