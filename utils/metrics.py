"""
utils/metrics.py — Метрики ефективності системи.

Реалізація відповідає розділу 3.5.2 дипломної роботи:
  - F₁, Precision, Recall (виявлення точок зміни)
  - Sharpe Ratio (SR)
  - Maximum Drawdown (MDD)
  - Profit Factor (PF)
  - Composite Score (CS = 0.4·F₁ + 0.3·SR_norm + 0.2·(1-MDD) + 0.1·PF_norm)
"""

import numpy as np
from typing import List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RISK_FREE_RATE, TP_TOLERANCE, THETA_HIGH


def compute_f1(
    p_final: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = THETA_HIGH,
    tolerance: int   = TP_TOLERANCE,
) -> dict:
    """
    F₁-score для виявлення точок зміни з допуском ±tolerance свічок.

    True Positive: виявлена точка зміни потрапляє у вікно ±tolerance
                   навколо еталонної точки.
    """
    pred_cps = np.where(p_final >= threshold)[0]
    true_cps = np.where(true_labels == 1)[0]

    if len(true_cps) == 0 and len(pred_cps) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "far": 0.0, "tp": 0, "fp": 0, "fn": 0,
                "n_pred_cp": 0, "n_true_cp": 0}
    if len(true_cps) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 1.0}
    if len(pred_cps) == 0:
        return {"f1": 0.0, "precision": 1.0, "recall": 0.0}

    matched_pred = set()
    matched_true = set()

    for ti, tcp in enumerate(true_cps):
        for pi, pcp in enumerate(pred_cps):
            if abs(pcp - tcp) <= tolerance and pi not in matched_pred:
                matched_pred.add(pi)
                matched_true.add(ti)
                break

    tp = len(matched_true)
    fp = len(pred_cps) - len(matched_pred)
    fn = len(true_cps) - tp

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # FAR (False Alarm Rate)
    far = fp / (fp + tp + 1e-9)

    return {
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "far":       float(far),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "n_pred_cp": len(pred_cps),
        "n_true_cp": len(true_cps),
    }


def sharpe_ratio(equity_curve: list, rf: float = RISK_FREE_RATE) -> float:
    """
    Коефіцієнт Шарпа (SR).
    SR = (R̄_p - R_f) / (σ_p · √252)
    """
    eq = np.array(equity_curve)
    if len(eq) < 2:
        return 0.0
    returns = np.diff(eq) / (eq[:-1] + 1e-9)
    excess  = returns - rf
    if excess.std() < 1e-9:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def max_drawdown(equity_curve: list) -> float:
    """
    Максимальне просідання (MDD).
    MDD = max_{0≤τ≤t≤T} (V_τ - V_t) / V_τ
    """
    eq = np.array(equity_curve)
    if len(eq) < 2:
        return 0.0
    peak   = np.maximum.accumulate(eq)
    dd     = (peak - eq) / (peak + 1e-9)
    return float(dd.max())


def profit_factor(trades: list) -> float:
    """
    Фактор прибутку: PF = Σ(прибутки) / |Σ(збитки)|
    """
    profits = sum(t.pnl for t in trades if t.pnl > 0)
    losses  = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if losses < 1e-9:
        return float("inf") if profits > 0 else 1.0
    return float(profits / losses)


def total_return(equity_curve: list, initial: float = None) -> float:
    """Загальна прибутковість."""
    eq = np.array(equity_curve)
    if len(eq) < 2:
        return 0.0
    start = initial if initial else eq[0]
    return float((eq[-1] - start) / (start + 1e-9))


def composite_score(f1: float, sr: float, mdd: float, pf: float) -> float:
    """
    Композитна оцінка (CS):
    CS = 0.4·F₁ + 0.3·norm(SR) + 0.2·(1-MDD) + 0.1·norm(PF)
    """
    sr_norm = np.clip(sr / 3.0, 0, 1)          # нормуємо SR відносно 3.0
    pf_norm = np.clip((pf - 1) / 4.0, 0, 1)    # нормуємо PF відносно 5.0
    mdd_score = 1.0 - np.clip(mdd, 0, 1)
    cs = 0.4 * f1 + 0.3 * sr_norm + 0.2 * mdd_score + 0.1 * pf_norm
    return float(np.clip(cs, 0, 1))


def compute_all_metrics(
    p_final: np.ndarray,
    true_labels: np.ndarray,
    equity_curve: list,
    trades: list,
) -> dict:
    """Обчислення всіх метрик одним викликом."""
    f1_metrics = compute_f1(p_final, true_labels)
    sr  = sharpe_ratio(equity_curve)
    mdd = max_drawdown(equity_curve)
    pf  = profit_factor(trades)
    ret = total_return(equity_curve)
    cs  = composite_score(f1_metrics["f1"], sr, mdd, pf)

    n_trades     = len(trades)
    win_trades   = [t for t in trades if t.pnl > 0]
    win_rate     = len(win_trades) / (n_trades + 1e-9)
    avg_win      = np.mean([t.pnl for t in win_trades]) if win_trades else 0
    loss_trades  = [t for t in trades if t.pnl <= 0]
    avg_loss     = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0

    return {
        **f1_metrics,
        "sharpe":          sr,
        "mdd":             mdd,
        "profit_factor":   pf,
        "total_return":    ret,
        "composite_score": cs,
        "n_trades":        n_trades,
        "win_rate":        float(win_rate),
        "avg_win":         float(avg_win),
        "avg_loss":        float(avg_loss),
    }
