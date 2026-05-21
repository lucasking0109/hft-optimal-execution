"""Phase G.2 — Parent size sweep.

Sweeps 3 parent sizes (% of ADV) × 5 RTH windows × 104 tickers × 6 strategies.
Tests whether RL maintains its edge over VWAP-follow as parent size grows.

CAVEAT: at 5% ADV in a 1-hour window, POV cap is heavily violated
(~23% participation per bucket). Results in that regime are "stress test
under POV cap violation", NOT production scenarios.

Outputs:
  - reports/phase_g_parent_size.csv  (per-(ticker, window, size, strategy))
  - reports/phase_g_parent_size.md
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase_g_common import (                                    # noqa: E402
    OOS_DATE,
    RL_MODEL_NAME,
    TRAIN_DATES,
    WINDOWS,
    list_eval_tickers,
    load_rl_strategy,
    make_strategies,
)
from hft.analysis.vwap import compute_lookback_volume_profile    # noqa: E402
from hft.backtest.engine import BacktestEngine                   # noqa: E402
from hft.data import load_eq_taq                                 # noqa: E402
from hft.simulators.adv_cache import get_adv                     # noqa: E402
from hft.strategies.base import ParentOrder                      # noqa: E402


# 3 parent sizes as % of 4-day ADV.
SIZES_PCT_ADV = [
    ("0_1pct_adv", 0.001),   # 0.1% — no-impact regime
    ("1pct_adv",   0.01),    # 1%   — borderline
    ("5pct_adv",   0.05),    # 5%   — stress test (POV cap violated)
]
MIN_PARENT_QTY = 100   # skip very-small parents on low-ADV names


def main():
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tickers = list_eval_tickers()
    rl_strat = load_rl_strategy()
    n_strats = 6 if rl_strat is not None else 5

    print("=" * 70)
    print("Phase G.2 — Parent size sweep (% of ADV)")
    print(f"OOS date: {OOS_DATE}  |  Tickers: {len(tickers)}")
    print(f"Sizes: {[s[0] for s in SIZES_PCT_ADV]}")
    print(f"Windows: {len(WINDOWS)}  |  Strategies: {n_strats}")
    print(f"Approx backtests: {len(tickers) * len(WINDOWS) * len(SIZES_PCT_ADV) * n_strats}")
    print("=" * 70)

    rows = []
    t0 = time.perf_counter()
    n_done = n_failed = n_skipped = 0
    for size_label, size_pct in SIZES_PCT_ADV:
        print(f"\n══ Size: {size_label} ({size_pct*100:.1f}% ADV) ══")
        for window_name, start_sec, end_sec in WINDOWS:
            start_ns = start_sec * int(1e9)
            end_ns = end_sec * int(1e9)
            for ticker in tickers:
                try:
                    df = load_eq_taq(ticker, OOS_DATE)
                    engine = BacktestEngine(ticker, OOS_DATE, market_df=df)
                    profile = compute_lookback_volume_profile(
                        ticker, lookback_dates=TRAIN_DATES, bin_minutes=5,
                    )
                    adv = get_adv(ticker, exclude_dates=[OOS_DATE])
                    parent_qty = int(round(adv * size_pct))
                    if parent_qty < MIN_PARENT_QTY:
                        n_skipped += 1
                        continue
                    ctx = engine.market_context(
                        bin_minutes=5, volume_profile_override=profile,
                    )
                    ctx["adv_shares"] = adv
                except Exception:
                    continue
                parent = ParentOrder(
                    ticker=ticker, date=OOS_DATE, side="sell",
                    quantity=parent_qty,
                    start_ns=start_ns, end_ns=end_ns,
                )
                for label, strat in make_strategies(ticker, adv, rl_strat):
                    try:
                        res = engine.run(parent, strat, market_context=ctx)
                        rows.append({
                            "size": size_label, "size_pct_adv": size_pct,
                            "window": window_name,
                            "ticker": ticker, "strategy": label,
                            "parent_qty": parent_qty,
                            "is_bps": res.metrics.is_bps,
                            "vwap_slip_bps": res.metrics.vwap_slip_bps,
                            "eff_spread_bps": res.metrics.effective_spread_bps,
                            "markout_60s_bps": res.metrics.markout_60s_bps,
                        })
                        n_done += 1
                    except Exception:
                        n_failed += 1
            print(f"  ✓ {size_label}/{window_name}: done {n_done}, "
                  f"failed {n_failed}, skipped {n_skipped}")

    elapsed = time.perf_counter() - t0
    print(f"\n📊 Parent size sweep done in {elapsed/60:.1f} min — "
          f"{n_done} done, {n_failed} failed, {n_skipped} skipped")

    df_results = pl.DataFrame(rows)
    csv_path = reports_dir / "phase_g_parent_size.csv"
    df_results.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # Per-(size, window, strategy) aggregates.
    agg = (
        df_results
        .group_by(["size", "window", "strategy"])
        .agg(
            pl.col("is_bps").median().alias("median_is"),
            pl.col("is_bps").std().alias("std_is"),
            pl.len().alias("n"),
        )
        .sort(["size", "window", "median_is"])
    )

    # RL vs VWAP-follow per (size, window) — the key signal.
    rl_vs_vwap_rows = []
    for size_label, size_pct in SIZES_PCT_ADV:
        for window_name, _, _ in WINDOWS:
            rl_m = agg.filter(
                (pl.col("size") == size_label)
                & (pl.col("window") == window_name)
                & (pl.col("strategy") == RL_MODEL_NAME)
            )["median_is"]
            vwap_m = agg.filter(
                (pl.col("size") == size_label)
                & (pl.col("window") == window_name)
                & (pl.col("strategy") == "vwap_following")
            )["median_is"]
            if rl_m.is_empty() or vwap_m.is_empty():
                continue
            # Win rate per ticker
            slice_df = df_results.filter(
                (pl.col("size") == size_label) & (pl.col("window") == window_name)
            )
            vwap_per_ticker = (
                slice_df.filter(pl.col("strategy") == "vwap_following")
                .select(["ticker", "is_bps"]).rename({"is_bps": "vwap_is"})
            )
            rl_per_ticker = (
                slice_df.filter(pl.col("strategy") == RL_MODEL_NAME)
                .select(["ticker", "is_bps"]).rename({"is_bps": "rl_is"})
            )
            joined = rl_per_ticker.join(vwap_per_ticker, on="ticker", how="inner")
            win_rate = float((joined["rl_is"] < joined["vwap_is"]).mean() or 0.0)
            rl_vs_vwap_rows.append({
                "size": size_label, "size_pct_adv": size_pct,
                "window": window_name,
                "rl_median_is": rl_m[0],
                "vwap_median_is": vwap_m[0],
                "delta_bps": rl_m[0] - vwap_m[0],
                "rl_win_rate_vs_vwap": win_rate,
                "n_tickers": joined.height,
            })

    rl_vs_vwap = pl.DataFrame(rl_vs_vwap_rows) if rl_vs_vwap_rows else None

    # Markdown
    md = ["# Phase G.2 — Parent Size Sweep (% of ADV)\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**OOS date**: {OOS_DATE}  |  **Tickers**: {len(tickers)}")
    md.append(f"**Sizes**: " + ", ".join([f"{s[0]} ({s[1]*100:.1f}% ADV)" for s in SIZES_PCT_ADV]))
    md.append(f"**Windows**: {', '.join([w[0] for w in WINDOWS])}")
    md.append(f"**Total backtests**: {n_done} done / {n_failed} failed / {n_skipped} skipped\n")

    md.append("## Caveats\n")
    md.append("- **0.1% ADV** → no-impact assumption safe, results directly comparable")
    md.append("- **1% ADV** → borderline; POV 5% cap occasionally binds on illiquid names")
    md.append("- **5% ADV × 1-hour** → POV 5% cap **heavily violated** (per-bucket "
              "participation ~20-30%). force_completion water-fill makes every child "
              "~4-5× POV cap. Use this row as **stress test**, not production scenario.")
    md.append("- Self-impact NOT modelled — all results assume price-taker. At 1%+ ADV, "
              "real fills would push the market. Phase H candidate.\n")

    if rl_vs_vwap is not None:
        md.append("## RL vs VWAP-following per (size, window) — key signal\n")
        md.append("| Size | Window | RL median IS | VWAP-follow median IS | Δ (bps) | RL win-rate vs VWAP |")
        md.append("|---|---|---|---|---|---|")
        for r in rl_vs_vwap.iter_rows(named=True):
            md.append(
                f"| {r['size']} | {r['window']} | {r['rl_median_is']:+.2f} | "
                f"{r['vwap_median_is']:+.2f} | **{r['delta_bps']:+.2f}** | "
                f"{r['rl_win_rate_vs_vwap']*100:.0f}% |"
            )

    md.append("\n## Per-(size, window, strategy) median IS — full breakdown\n")
    md.append("| Size | Window | Strategy | n | Median IS (bps) | Std |")
    md.append("|---|---|---|---|---|---|")
    for row in agg.iter_rows(named=True):
        md.append(
            f"| {row['size']} | {row['window']} | {row['strategy']} | "
            f"{row['n']} | **{row['median_is']:+.2f}** | {row['std_is']:.1f} |"
        )

    md_path = reports_dir / "phase_g_parent_size.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    print("\n" + "=" * 70)
    print("Phase G.2 parent size sweep done — see reports/phase_g_parent_size.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
