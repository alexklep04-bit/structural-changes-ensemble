"""
utils/visualization.py — Візуалізація результатів бектесту.

Генерує графіки для дипломної роботи:
  - Equity curve по циклах
  - P_final vs true changepoints
  - Динаміка ваг ансамблю
  - Зведена таблиця метрик
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, THETA_HIGH, THETA_LOW

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COLORS = {
    "CUSUM":  "#1565c0",
    "BOCPD":  "#2e7d32",
    "HMM":    "#f57f17",
    "LSTM":   "#6a1b9a",
    "FinBERT":"#c62828",
    "Ensemble": "#37474f",
}


def plot_results(results: dict, symbol: str, show: bool = False) -> Path:
    """
    Головна функція — генерує зведений рисунок результатів.
    Повертає шлях до збереженого PNG.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Результати ансамблевої системи — {symbol}",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.4, wspace=0.3)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    _plot_equity(ax1, results)

    # 2. P_final vs changepoints (останній цикл)
    ax2 = fig.add_subplot(gs[1, :])
    _plot_pfinal(ax2, results)

    # 3. Метрики по циклах
    ax3 = fig.add_subplot(gs[2, 0])
    _plot_fold_metrics(ax3, results)

    # 4. Зведена таблиця
    ax4 = fig.add_subplot(gs[2, 1])
    _plot_summary_table(ax4, results, symbol)

    path = RESULTS_DIR / f"results_{symbol.replace('/', '_')}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Збережено: {path}")
    return path


def plot_detector_comparison(
    p_cusum: np.ndarray,
    p_bocpd: np.ndarray,
    p_hmm: np.ndarray,
    p_lstm: np.ndarray,
    p_news: np.ndarray,
    p_final: np.ndarray,
    true_labels: np.ndarray,
    symbol: str,
) -> Path:
    """Порівняльний графік виходів всіх детекторів."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"Виходи детекторів — {symbol}", fontweight="bold")

    data_pairs = [
        ("CUSUM",   p_cusum,  COLORS["CUSUM"]),
        ("BOCPD",   p_bocpd,  COLORS["BOCPD"]),
        ("HMM",     p_hmm,    COLORS["HMM"]),
        ("LSTM AE", p_lstm,   COLORS["LSTM"]),
        ("FinBERT", p_news,   COLORS["FinBERT"]),
        ("P_final (Ensemble)", p_final, COLORS["Ensemble"]),
    ]

    cp_idx = np.where(true_labels == 1)[0]

    for ax, (name, probs, color) in zip(axes, data_pairs):
        ax.plot(probs, color=color, lw=1.2, alpha=0.8, label=name)
        ax.axhline(THETA_HIGH, color="red",  lw=0.8, ls="--", alpha=0.5)
        ax.axhline(THETA_LOW,  color="green", lw=0.8, ls="--", alpha=0.5)
        for cp in cp_idx:
            ax.axvline(cp, color="orange", lw=0.5, alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(name, fontsize=8)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Свічки")
    plt.tight_layout()

    path = RESULTS_DIR / f"detectors_{symbol.replace('/', '_')}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Збережено: {path}")
    return path


def plot_metrics_table(all_results: dict) -> Path:
    """Зведена таблиця метрик для всіх інструментів."""
    rows = []
    for sym, res in all_results.items():
        rows.append({
            "Інструмент": sym,
            "F₁":   f"{res.get('f1', {}).get('mean', 0):.3f}",
            "SR":    f"{res.get('sharpe', {}).get('mean', 0):.2f}",
            "MDD":   f"{res.get('mdd', {}).get('mean', 0):.3f}",
            "PF":    f"{res.get('profit_factor', {}).get('mean', 0):.2f}",
            "Return":f"{res.get('total_return', {}).get('mean', 0)*100:.1f}%",
            "CS":    f"{res.get('composite_score', {}).get('mean', 0):.3f}",
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Заголовок
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#1565c0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Зведені метрики ефективності системи",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()

    path = RESULTS_DIR / "metrics_table_all.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Збережено: {path}")
    return path


# ── Допоміжні функції ────────────────────────────────────────────────────────

def _plot_equity(ax, results: dict):
    eq = results.get("equity_curve", [10000])
    ax.plot(eq, color="#1565c0", lw=1.5)
    ax.fill_between(range(len(eq)), 10000, eq,
                    where=[e >= 10000 for e in eq],
                    alpha=0.15, color="green")
    ax.fill_between(range(len(eq)), 10000, eq,
                    where=[e < 10000 for e in eq],
                    alpha=0.15, color="red")
    ax.axhline(10000, color="grey", lw=0.8, ls="--")
    ax.set_title("Equity Curve (walk-forward)", fontweight="bold")
    ax.set_ylabel("USD")
    ax.set_xlabel("Свічки")


def _plot_pfinal(ax, results: dict):
    folds = results.get("fold_results", [])
    if not folds:
        ax.text(0.5, 0.5, "Немає даних", ha="center", va="center",
                transform=ax.transAxes)
        return
    last = folds[-1]
    p    = last.get("p_final", np.array([]))
    lbl  = last.get("true_labels", np.array([]))

    ax.plot(p, color="#37474f", lw=1.2, label="P_final(t)", alpha=0.8)
    ax.axhline(THETA_HIGH, color="red",   lw=1, ls="--", label=f"θ_high={THETA_HIGH}")
    ax.axhline(THETA_LOW,  color="green", lw=1, ls="--", label=f"θ_low={THETA_LOW}")
    if len(lbl) == len(p):
        cp_idx = np.where(lbl == 1)[0]
        ax.scatter(cp_idx, p[cp_idx], color="orange", s=20, zorder=5,
                   label="Еталонні точки зміни")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"P_final(t) — останній цикл ({len(folds)})",
                 fontweight="bold")
    ax.set_xlabel("Свічки")
    ax.legend(loc="upper right", fontsize=8)


def _plot_fold_metrics(ax, results: dict):
    folds = results.get("fold_results", [])
    if not folds:
        return
    x     = [f["fold"] for f in folds]
    f1    = [f["metrics"].get("f1", 0) for f in folds]
    sr    = [min(f["metrics"].get("sharpe", 0) / 3, 1) for f in folds]
    mdd   = [f["metrics"].get("mdd", 0) for f in folds]

    ax.plot(x, f1,  "o-", color="#1565c0", label="F₁")
    ax.plot(x, sr,  "s-", color="#2e7d32", label="SR/3 (norm)")
    ax.plot(x, mdd, "^-", color="#c62828", label="MDD")
    ax.axhline(0.7, color="#1565c0", ls=":", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xlabel("Цикл")
    ax.set_title("Метрики по циклах", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)


def _plot_summary_table(ax, results: dict, symbol: str):
    ax.axis("off")
    metrics_labels = [
        ("F₁ (mean)",   "f1"),
        ("SR (mean)",   "sharpe"),
        ("MDD (mean)",  "mdd"),
        ("PF (mean)",   "profit_factor"),
        ("Return",      "total_return"),
        ("CS (mean)",   "composite_score"),
    ]
    rows = []
    for label, key in metrics_labels:
        val = results.get(key, {})
        if isinstance(val, dict):
            mean = val.get("mean", 0)
            std  = val.get("std",  0)
            if key == "total_return":
                rows.append([label, f"{mean*100:.1f}% ± {std*100:.1f}%"])
            else:
                rows.append([label, f"{mean:.3f} ± {std:.3f}"])
        else:
            rows.append([label, f"{val:.3f}"])

    tbl = ax.table(
        cellText=rows,
        colLabels=["Метрика", "Значення"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for j in range(2):
        tbl[0, j].set_facecolor("#37474f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Метрики {symbol}", fontweight="bold")
