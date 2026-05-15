"""Run Phase 2 baseline experiment.

Replays TWAP and VWAP-following strategies across multiple tickers × dates
for a fixed liquidation scenario. Writes:
- reports/phase2_baseline.csv  (one row per (ticker, date, strategy))
- reports/phase2_baseline.md   (summary tables + observations)

Per NO Silent Fallback: any single (ticker, date) failure is recorded with
its exception in the CSV `error` column; we keep going so partial results
are still useful, but failures are reported up-front in the markdown.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.vwap import compute_lookback_volume_profile  # noqa: E402
from hft.backtest.engine import BacktestEngine                 # noqa: E402
from hft.data import AVAILABLE_DATES                           # noqa: E402
from hft.strategies.base import ParentOrder                    # noqa: E402
from hft.strategies.twap import TWAPStrategy                   # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy  # noqa: E402

NS_PER_HOUR = 3600 * 1_000_000_000

# 10 large-cap, highly-liquid NDX-100 names (all confirmed in dataset)
TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "AMD", "NFLX", "ADBE", "AVGO", "INTC"]
DATES = list(AVAILABLE_DATES)

# Standardised liquidation scenario
PARENT_QTY = 10_000
WINDOW_START_NS = int(10 * NS_PER_HOUR)   # 10:00 ET
WINDOW_END_NS = int(11 * NS_PER_HOUR)     # 11:00 ET
NUM_SLICES = 60                            # 1 child / minute
SIDE = "sell"


def run_one(ticker: str, date: str) -> list[dict]:
    """Run TWAP and VWAP-following on a single (ticker, date). Returns list of result dicts."""
    rows = []
    try:
        engine = BacktestEngine(ticker, date)
    except Exception as e:
        return [{
            "ticker": ticker, "date": date, "strategy": "(load)",
            "error": f"{type(e).__name__}: {e}",
        }]

    parent = ParentOrder(
        ticker=ticker, date=date, side=SIDE,
        quantity=PARENT_QTY,
        start_ns=WINDOW_START_NS, end_ns=WINDOW_END_NS,
    )

    # Leave-one-out lookback baseline: average volume profile across all
    # other days for this ticker (excludes the current eval date — no leak).
    lookback = [d for d in DATES if d != date]
    try:
        profile = compute_lookback_volume_profile(
            ticker, lookback_dates=lookback, bin_minutes=5,
        )
    except Exception as e:
        return [{
            "ticker": ticker, "date": date, "strategy": "(lookback_profile)",
            "error": f"{type(e).__name__}: {e}",
        }]

    for strat in [TWAPStrategy(num_slices=NUM_SLICES), VWAPFollowingStrategy(num_slices=NUM_SLICES)]:
        try:
            ctx = engine.market_context(bin_minutes=5, volume_profile_override=profile)
            res = engine.run(parent, strat, market_context=ctx)
            rows.append({
                "ticker": ticker, "date": date, "strategy": strat.name,
                "arrival_mid": res.arrival_mid,
                "market_vwap": res.market_vwap,
                "executed_avg": float(
                    (res.fills["price"] * res.fills["quantity"]).sum() / res.fills["quantity"].sum()
                ),
                "fills": int(res.fills.shape[0]),
                "oversize_count": res.oversize_count,
                "null_nbbo_count": res.null_nbbo_count,
                "vwap_slip_bps": res.metrics.vwap_slip_bps,
                "is_bps": res.metrics.is_bps,
                "effective_spread_bps": res.metrics.effective_spread_bps,
                "realized_spread_bps": res.metrics.realized_spread_bps,
                "markout_1s_bps": res.metrics.markout_1s_bps,
                "markout_10s_bps": res.metrics.markout_10s_bps,
                "markout_60s_bps": res.metrics.markout_60s_bps,
                "reversion_5m_bps": res.metrics.reversion_5m_bps,
                "pov": res.metrics.pov,
                "price_var": res.metrics.price_var,
                "schedule_rmse": res.metrics.schedule_rmse,
                "hit_ratio_nbbo": res.metrics.hit_ratio_nbbo,
                "error": "",
            })
        except Exception as e:
            rows.append({
                "ticker": ticker, "date": date, "strategy": strat.name,
                "error": f"{type(e).__name__}: {e}",
                "_traceback": traceback.format_exc(),
            })

    return rows


def main():
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    print(f"Running {len(TICKERS)} tickers × {len(DATES)} dates × 2 strategies = {len(TICKERS)*len(DATES)*2} backtests")
    for ticker in TICKERS:
        for date in DATES:
            print(f"  {ticker} {date}...")
            rows = run_one(ticker, date)
            all_rows.extend(rows)

    df = pl.DataFrame(all_rows)
    csv_path = out_dir / "phase2_baseline.csv"
    df.write_csv(csv_path)
    print(f"\nWrote {csv_path}")

    # Summary
    failures = df.filter(pl.col("error") != "")
    successes = df.filter(pl.col("error") == "")

    md_lines = [
        "# Phase 2 Baseline Report",
        "",
        f"**Setup**: sell {PARENT_QTY:,} shares between 10:00 and 11:00 ET, {NUM_SLICES} child orders.",
        f"**Tickers**: {', '.join(TICKERS)}",
        f"**Dates**: {', '.join(DATES)}",
        "",
        f"**Total runs**: {len(all_rows)} ({len(successes)} success / {len(failures)} failure)",
        "",
    ]

    if not failures.is_empty():
        md_lines.append("## ⚠️ Failures (NO Silent Fallback — these were not glossed over)")
        md_lines.append("")
        for row in failures.iter_rows(named=True):
            md_lines.append(f"- **{row['ticker']} {row['date']} {row['strategy']}**: {row['error']}")
        md_lines.append("")

    if not successes.is_empty():
        # Per-strategy summary stats
        md_lines.append("## Per-strategy summary (median across all runs)")
        md_lines.append("")
        md_lines.append("| Strategy | VWAP slip (bps) | IS (bps) | Eff spread (bps) | Markout 60s (bps) | Reversion 5m (bps) | POV | Hit ratio |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for strat_name in successes["strategy"].unique().to_list():
            sub = successes.filter(pl.col("strategy") == strat_name)
            md_lines.append(
                f"| **{strat_name}** | "
                f"{sub['vwap_slip_bps'].median():+.2f} | "
                f"{sub['is_bps'].median():+.2f} | "
                f"{sub['effective_spread_bps'].median():.2f} | "
                f"{sub['markout_60s_bps'].median():+.2f} | "
                f"{sub['reversion_5m_bps'].median():+.2f} | "
                f"{sub['pov'].median()*100:.2f}% | "
                f"{sub['hit_ratio_nbbo'].median()*100:.1f}% |"
            )
        md_lines.append("")

        # Strategy comparison: pivot
        md_lines.append("## VWAP slippage per (ticker, date, strategy)")
        md_lines.append("")
        md_lines.append("| Ticker | Date | TWAP slip (bps) | VWAP-follow slip (bps) | TWAP wins? |")
        md_lines.append("|---|---|---|---|---|")
        twap = successes.filter(pl.col("strategy") == "twap")
        vwap_f = successes.filter(pl.col("strategy") == "vwap_following")
        for ticker in TICKERS:
            for date in DATES:
                t_row = twap.filter((pl.col("ticker") == ticker) & (pl.col("date") == date))
                v_row = vwap_f.filter((pl.col("ticker") == ticker) & (pl.col("date") == date))
                if t_row.is_empty() or v_row.is_empty():
                    continue
                t = float(t_row["vwap_slip_bps"][0])
                v = float(v_row["vwap_slip_bps"][0])
                wins = "✅" if t < v else "❌"
                md_lines.append(f"| {ticker} | {date} | {t:+.2f} | {v:+.2f} | {wins} |")
        md_lines.append("")

        md_lines.append("## Notes")
        md_lines.append("")
        md_lines.append("- All metrics: positive = cost (worse than benchmark), negative = better.")
        md_lines.append("- Backtester known limitations (per NO Silent Fallback transparency):")
        md_lines.append("  1. Marketable child orders fill at NBBO with **no self-impact modelling**.")
        md_lines.append("  2. **Oversize fills** at best price (size > NBBO size at that level) are recorded in `oversize_count` not silently truncated.")
        md_lines.append("  3. VWAP-following uses **same-day volume profile** (look-ahead bias). Phase 3+ will use rolling estimate from prior days.")
        md_lines.append("  4. Both strategies share the same simulator → relative comparison is fair even with these biases.")

    md_path = out_dir / "phase2_baseline.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
