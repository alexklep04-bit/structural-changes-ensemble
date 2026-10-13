"""
main.py — Головний файл запуску системи.

Запуск:
    python main.py                    # всі інструменти
    python main.py --symbol XAU/USD   # один інструмент
    python main.py --no-lstm          # без LSTM (швидко)
    python main.py --finbert          # з реальним FinBERT
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Додаємо кореневу директорію до шляху
sys.path.insert(0, str(Path(__file__).parent))

from config import SYMBOLS, RESULTS_DIR
from data.download_data import load_all_symbols, load_or_download
from backtest.walk_forward import WalkForwardBacktest
from utils.visualization import plot_results, plot_metrics_table
from utils.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ансамблева система виявлення структурних змін"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Інструмент (XAU/USD, EUR/USD, GBP/USD). За замовчуванням — всі."
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="Вимкнути LSTM (пришвидшує запуск)"
    )
    parser.add_argument(
        "--finbert", action="store_true",
        help="Використовувати реальний FinBERT (потребує трансформери)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ігнорувати кеш даних, завантажити заново"
    )
    return parser.parse_args()


def run_single(symbol: str, use_finbert: bool = False) -> dict:
    """Запуск бектесту для одного інструменту."""
    print(f"\n{'='*60}")
    print(f"  Інструмент: {symbol}")
    print(f"{'='*60}")

    # Завантаження даних
    df = load_or_download(symbol)
    print(f"  Даних: {len(df)} свічок "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Точок зміни (PELT): {df['is_changepoint'].sum()}")

    # Walk-forward бектест
    backtester = WalkForwardBacktest(
        symbol=symbol,
        use_finbert_transformers=use_finbert,
        verbose=True,
    )
    results = backtester.run(df)

    if not results:
        print("  Помилка: бектест не дав результатів")
        return {}

    # Візуалізація
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(results, symbol)

    # Детальний звіт по циклах
    print_fold_report(results, symbol)

    # Збереження CSV
    save_results_csv(results, symbol)

    return results


def print_fold_report(results: dict, symbol: str):
    """Виведення таблиці результатів по циклах."""
    folds = results.get("fold_results", [])
    if not folds:
        return

    print(f"\n{'─'*70}")
    print(f"  {'Цикл':>5} {'Train':>6} {'Test':>6} "
          f"{'F₁':>6} {'SR':>6} {'MDD':>6} {'PF':>6} {'Угод':>5}")
    print(f"{'─'*70}")

    for f in folds:
        m = f["metrics"]
        print(
            f"  {f['fold']:>5} "
            f"{f['train_size']:>6} "
            f"{f['test_size']:>6} "
            f"{m.get('f1', 0):>6.3f} "
            f"{m.get('sharpe', 0):>6.2f} "
            f"{m.get('mdd', 0):>6.3f} "
            f"{min(m.get('profit_factor', 0), 99):>6.2f} "
            f"{f.get('n_trades', 0):>5}"
        )

    print(f"{'─'*70}")
    print(
        f"  {'MEAN':>5} "
        f"{'':>6} "
        f"{'':>6} "
        f"{results['f1']['mean']:>6.3f} "
        f"{results['sharpe']['mean']:>6.2f} "
        f"{results['mdd']['mean']:>6.3f} "
        f"{min(results['profit_factor']['mean'], 99):>6.2f} "
        f"{results['n_trades']['mean']:>5.0f}"
    )
    print(f"{'─'*70}")

    # Перевірка вимог дипломної роботи
    print(f"\n  Перевірка вимог:")
    reqs = {
        "F₁ ≥ 0.70":   results["f1"]["mean"] >= 0.70,
        "SR ≥ 1.50":   results["sharpe"]["mean"] >= 1.50,
        "MDD ≤ 0.15":  results["mdd"]["mean"] <= 0.15,
        "PF ≥ 1.50":   results["profit_factor"]["mean"] >= 1.50,
    }
    for req, ok in reqs.items():
        status = "✓" if ok else "✗"
        print(f"    {status} {req}")


def save_results_csv(results: dict, symbol: str):
    """Збереження метрик у CSV для дипломної роботи."""
    folds = results.get("fold_results", [])
    if not folds:
        return

    rows = []
    for f in folds:
        m = f["metrics"]
        rows.append({
            "Цикл": f["fold"],
            "Розмір навчання": f["train_size"],
            "Розмір тесту":   f["test_size"],
            "F1":             round(m.get("f1", 0), 4),
            "Precision":      round(m.get("precision", 0), 4),
            "Recall":         round(m.get("recall", 0), 4),
            "FAR":            round(m.get("far", 0), 4),
            "Sharpe":         round(m.get("sharpe", 0), 4),
            "MDD":            round(m.get("mdd", 0), 4),
            "PF":             round(min(m.get("profit_factor", 0), 99), 4),
            "Return_%":       round(m.get("total_return", 0) * 100, 2),
            "CS":             round(m.get("composite_score", 0), 4),
            "Угод":           m.get("n_trades", 0),
        })

    df_out = pd.DataFrame(rows)
    path = RESULTS_DIR / f"metrics_{symbol.replace('/', '_')}.csv"
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n  Метрики збережено: {path}")


def main():
    args = parse_args()

    # Очищення кешу
    if args.no_cache:
        import glob
        for f in glob.glob(str(Path(__file__).parent / "data" / "*.parquet")):
            Path(f).unlink()
            print(f"  Видалено кеш: {f}")

    # Список інструментів
    symbols = [args.symbol] if args.symbol else list(SYMBOLS.keys())
    all_results = {}

    for sym in symbols:
        try:
            res = run_single(sym, use_finbert=args.finbert)
            all_results[sym] = res
        except Exception as e:
            print(f"\n  ПОМИЛКА для {sym}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Зведена таблиця
    if len(all_results) > 1:
        print("\n\n=== ЗВЕДЕНІ РЕЗУЛЬТАТИ ПО ВСІХ ІНСТРУМЕНТАХ ===")
        print(f"{'Інструмент':>10} {'F₁':>6} {'SR':>6} "
              f"{'MDD':>6} {'PF':>6} {'Return':>8}")
        print("─" * 55)
        for sym, res in all_results.items():
            if res:
                print(
                    f"{sym:>10} "
                    f"{res['f1']['mean']:>6.3f} "
                    f"{res['sharpe']['mean']:>6.2f} "
                    f"{res['mdd']['mean']:>6.3f} "
                    f"{min(res['profit_factor']['mean'], 99):>6.2f} "
                    f"{res['total_return']['mean']*100:>7.1f}%"
                )
        plot_metrics_table(all_results)

    print("\n✓ Готово! Результати збережено у папці results/")


if __name__ == "__main__":
    main()
