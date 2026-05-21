"""Phase G.1 — Multi-window OOS eval (104 ticker × 5 RTH windows × 6 strategies).

Tests whether the Phase E v3 RL policy generalises across time-of-day
(open / early-mid / mid / late-mid / close). Parent size fixed at 10k.

OOS = Day 5 (20200117). All other days form the leave-one-out lookback
baseline for volume_profile (no leakage).

Outputs:
  - reports/phase_g_multi_window.csv  (per-(ticker, window, strategy))
  - reports/phase_g_multi_window.md
  - reports/figures/phase_g_multi_window_violin.png
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
    DEFAULT_PARENT_QTY,
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


def main():
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tickers = list_eval_tickers()
    rl_strat = load_rl_strategy()
    n_strats = 6 if rl_strat is not None else 5

    print("=" * 70)
    print(f"Phase G.1 — Multi-window OOS eval")
    print(f"OOS date: {OOS_DATE}  |  Tickers: {len(tickers)}  |  Windows: {len(WINDOWS)}")
    print(f"Parent: {DEFAULT_PARENT_QTY:,} shares sell")
    if rl_strat is not None:
        print(f"RL model: {RL_MODEL_NAME} (loaded)")
    else:
        print(f"⚠️  RL model missing — skipping {RL_MODEL_NAME}")
    print(f"Total backtests: {len(tickers) * len(WINDOWS) * n_strats}")
    print("=" * 70)

    rows = []
    t0 = time.perf_counter()
    n_done = n_failed = 0
    for window_name, start_sec, end_sec in WINDOWS:
        print(f"\n── Window: {window_name} ({start_sec//3600:02d}:"
              f"{(start_sec%3600)//60:02d} → {end_sec//3600:02d}:"
              f"{(end_sec%3600)//60:02d}) ──")
        start_ns = start_sec * int(1e9)
        end_ns = end_sec * int(1e9)
        for ticker in tickers:
            try:
                df = load_eq_taq(ticker, OOS_DATE)
                engine = BacktestEngine(ticker, OOS_DATE, market_df=df)
                # Causal lookback (Day 1-4 only) — no leakage.
                profile = compute_lookback_volume_profile(
                    ticker, lookback_dates=TRAIN_DATES, bin_minutes=5,
                )
                adv = get_adv(ticker, exclude_dates=[OOS_DATE])
                ctx = engine.market_context(
                    bin_minutes=5, volume_profile_override=profile,
                )
                ctx["adv_shares"] = adv
            except Exception as e:
                print(f"  ❌ {ticker}: load failed — {e}")
                continue
            parent = ParentOrder(
                ticker=ticker, date=OOS_DATE, side="sell",
                quantity=DEFAULT_PARENT_QTY,
                start_ns=start_ns, end_ns=end_ns,
            )
            for label, strat in make_strategies(ticker, adv, rl_strat):
                try:
                    res = engine.run(parent, strat, market_context=ctx)
                    m = res.metrics
                    rows.append({
                        "window": window_name,
                        "ticker": ticker, "strategy": label,
                        "is_bps": m.is_bps,
                        "vwap_slip_bps": m.vwap_slip_bps,
                        "eff_spread_bps": m.effective_spread_bps,
                        "markout_60s_bps": m.markout_60s_bps,
                    })
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    print(f"  ❌ {ticker} {label}: {type(e).__name__}: "
                          f"{str(e)[:80]}")
        print(f"  ✓ {window_name}: {n_done} done so far, {n_failed} failed")

    elapsed = time.perf_counter() - t0
    print(f"\n📊 Multi-window sweep done in {elapsed/60:.1f} min — "
          f"{n_done} done, {n_failed} failed")

    df_results = pl.DataFrame(rows)
    csv_path = reports_dir / "phase_g_multi_window.csv"
    df_results.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # Per-(window, strategy) aggregates: median IS, std, win-rate vs VWAP-follow.
    agg = (
        df_results
        .group_by(["window", "strategy"])
        .agg(
            pl.col("is_bps").median().alias("median_is"),
            pl.col("is_bps").std().alias("std_is"),
            pl.len().alias("n"),
        )
        .sort(["window", "median_is"])
    )

    # Win rate vs VWAP-following within each window.
    win_rates: dict[tuple[str, str], float] = {}
    for window_name, _, _ in WINDOWS:
        win_df = df_results.filter(pl.col("window") == window_name)
        vwap = (
            win_df.filter(pl.col("strategy") == "vwap_following")
            .select(["ticker", "is_bps"])
            .rename({"is_bps": "vwap_is"})
        )
        for s in win_df["strategy"].unique().to_list():
            if s == "vwap_following":
                continue
            joined = (
                win_df.filter(pl.col("strategy") == s)
                .select(["ticker", "is_bps"])
                .join(vwap, on=["ticker"], how="inner")
            )
            wr = (joined["is_bps"] < joined["vwap_is"]).mean()
            win_rates[(window_name, s)] = float(wr) if wr is not None else 0.0

    # Sanity check: RL early_mid median IS should be in reasonable range.
    # For v3 specifically we expected +1.55 (Phase E.3 baseline). For v4 we
    # allow drift since training pool differs (overlapping windows).
    sanity_ok = True
    sanity_msg = ""
    if rl_strat is not None:
        em_rl_median = (
            df_results.filter(
                (pl.col("window") == "early_mid")
                & (pl.col("strategy") == RL_MODEL_NAME)
            )["is_bps"].median()
        )
        if em_rl_median is None:
            sanity_ok = False
            sanity_msg = f"FAIL: no {RL_MODEL_NAME} data in early_mid window"
        elif abs(em_rl_median) > 30:
            sanity_ok = False
            sanity_msg = (
                f"FAIL: {RL_MODEL_NAME} early_mid median IS = {em_rl_median:+.3f} "
                f"(out of plausible range [-30, +30]). Eval pipeline regressed?"
            )
        else:
            sanity_msg = (f"{RL_MODEL_NAME} early_mid median IS = "
                          f"{em_rl_median:+.3f} (within plausible range)")
        print(f"\nSanity: {sanity_msg}")

    # Build markdown report
    md = ["# Phase G.1 — Multi-Window OOS Eval (104 ticker × 5 RTH windows)\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**OOS date**: {OOS_DATE}  |  **Tickers**: {len(tickers)}")
    md.append(f"**Parent**: {DEFAULT_PARENT_QTY:,} shares sell, "
              f"1-hour window, NBBO routing")
    md.append(f"**Lookback volume profile**: Day 1-4 average (causal, no leak)\n")

    md.append("## Regression sanity\n")
    md.append(f"{sanity_msg}\n")

    md.append("## Per-(window, strategy) median IS\n")
    md.append("| Window | Strategy | n | Median IS (bps) | Std | Win-rate vs VWAP-follow |")
    md.append("|---|---|---|---|---|---|")
    for row in agg.iter_rows(named=True):
        s = row["strategy"]
        wr_key = (row["window"], s)
        wr_str = f"{win_rates[wr_key]*100:.0f}%" if wr_key in win_rates else "—"
        md.append(
            f"| {row['window']} | {s} | {row['n']} | "
            f"**{row['median_is']:+.2f}** | {row['std_is']:.1f} | {wr_str} |"
        )

    if rl_strat is not None:
        md.append("\n## RL v3 across windows (key generalization check)\n")
        md.append("| Window | RL median IS | VWAP-follow median IS | Δ | Win-rate vs VWAP |")
        md.append("|---|---|---|---|---|")
        for window_name, _, _ in WINDOWS:
            rl_m = agg.filter(
                (pl.col("window") == window_name) & (pl.col("strategy") == RL_MODEL_NAME)
            )["median_is"]
            vwap_m = agg.filter(
                (pl.col("window") == window_name) & (pl.col("strategy") == "vwap_following")
            )["median_is"]
            if rl_m.is_empty() or vwap_m.is_empty():
                continue
            delta = rl_m[0] - vwap_m[0]
            wr = win_rates.get((window_name, RL_MODEL_NAME), 0.0)
            md.append(
                f"| {window_name} | {rl_m[0]:+.2f} | {vwap_m[0]:+.2f} | "
                f"**{delta:+.2f}** | {wr*100:.0f}% |"
            )

        # G.4 trigger evaluation
        close_wr = win_rates.get(("close", RL_MODEL_NAME), None)
        md.append("\n## G.4 retrain trigger check\n")
        if close_wr is None:
            md.append("- RL win-rate in close window unavailable")
        elif close_wr >= 0.55:
            md.append(f"- RL win-rate vs VWAP-follow in close = {close_wr*100:.0f}% "
                      f"— no retrain needed (>=55% threshold)")
        elif close_wr >= 0.45:
            mean_wr = np.mean([win_rates.get((w[0], RL_MODEL_NAME), 0.0) for w in WINDOWS])
            if mean_wr >= 0.50:
                md.append(f"- RL win-rate in close = {close_wr*100:.0f}% borderline, "
                          f"but mean across windows = {mean_wr*100:.0f}% >= 50% "
                          f"— no retrain needed")
            else:
                md.append(f"- RL win-rate in close = {close_wr*100:.0f}% borderline, "
                          f"mean = {mean_wr*100:.0f}% < 50% — retrain v4 recommended")
        else:
            md.append(f"- RL win-rate vs VWAP-follow in close = {close_wr*100:.0f}% "
                      f"< 45% — generalization failed, retrain v4 (G.4)")

    md.append("\n## Caveats\n")
    md.append("- Single OOS day (20200117).")
    md.append(f"- 1-hour parent window across {len(WINDOWS)} time-of-day points.")
    md.append("- Spread cost in BacktestEngine (NBB/NBO fills).")
    md.append("- RL trained on 20 tickers × Day 1-4 across 6 non-overlapping 60-min windows starting 09:30–14:30. The close window (15:00–16:00) was NOT in the training pool — this eval tests generalization to unseen time-of-day.\n")

    md_path = reports_dir / "phase_g_multi_window.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Distribution figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(WINDOWS), figsize=(20, 4), sharey=True)
        strats = sorted(df_results["strategy"].unique().to_list())
        for ax, (window_name, _, _) in zip(axes, WINDOWS):
            win_df = df_results.filter(pl.col("window") == window_name)
            data = [
                win_df.filter(pl.col("strategy") == s)["is_bps"].to_list()
                for s in strats
            ]
            parts = ax.violinplot(data, showmedians=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(strats) + 1))
            ax.set_xticklabels(strats, rotation=20, ha="right", fontsize=8)
            ax.set_title(window_name)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("IS (bps; lower = better)")
        fig.suptitle(f"Phase G.1 — IS across {len(tickers)} tickers per window")
        plt.tight_layout()
        fig_path = figures_dir / "phase_g_multi_window_violin.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    if not sanity_ok:
        print(f"\n🚨 SANITY CHECK FAILED: {sanity_msg}")
        print("   Investigate before trusting downstream results.")
    print("\n" + "=" * 70)
    print("Phase G.1 multi-window eval done — see reports/phase_g_multi_window.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
