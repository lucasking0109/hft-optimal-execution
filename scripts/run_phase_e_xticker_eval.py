"""Phase E — 97-ticker × Day 5 OOS sweep: v3 RL vs 5 classical baselines.

Tests whether the ticker-agnostic Phase E policy (trained on a 20-ticker
pool) generalises to the full 97-ticker universe on Day 5 (held out).

Parent spec matches Phase D / Phase B for direct comparability:
10k shares sell, 10:00–11:00 ET, NBBO routing.

Outputs:
  - reports/phase_e_xticker.csv      — per-(ticker, strategy) results
  - reports/phase_e_xticker.md       — narrative + ranking
  - reports/figures/phase_e_xticker_distribution.png
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.impact import (                                 # noqa: E402
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
)
from hft.analysis.vwap import compute_lookback_volume_profile    # noqa: E402
from hft.backtest.engine import BacktestEngine                   # noqa: E402
from hft.data import load_eq_taq                                 # noqa: E402
from hft.simulators.adv_cache import get_adv                     # noqa: E402
from hft.strategies.almgren_chriss import AlmgrenChrissStrategy  # noqa: E402
from hft.strategies.base import ParentOrder                      # noqa: E402
from hft.strategies.pov import POVStrategy                       # noqa: E402
from hft.strategies.rl_agent import RLAgentStrategy              # noqa: E402
from hft.strategies.toth import TothStrategy                     # noqa: E402
from hft.strategies.twap import TWAPStrategy                     # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy  # noqa: E402

OOS_DATE = "20200117"
TAQ_ROOT = ROOT / "data" / "processed" / "eq_taq"
RL_MODEL_PATH = ROOT / "rl" / "checkpoints" / "ppo_v3_xticker" / "model.zip"

NUM_SLICES = 60
LAMBDA_RISK = 1e-3
SIGMA_DEFAULT = 1.0
PARENT_QTY = 10_000
START_TIME_NS = 10 * 3600 * int(1e9)
END_TIME_NS = 11 * 3600 * int(1e9)


def all_5_days(ticker: str) -> bool:
    return all(
        (TAQ_ROOT / d / f"{ticker}.parquet").exists()
        for d in ["20200113", "20200114", "20200115", "20200116", OOS_DATE]
    )


def list_all_tickers() -> list[str]:
    if not (TAQ_ROOT / OOS_DATE).exists():
        return []
    candidates = sorted(
        f.replace(".parquet", "")
        for f in os.listdir(TAQ_ROOT / OOS_DATE) if f.endswith(".parquet")
    )
    return [t for t in candidates if all_5_days(t)]


def make_strategies(ticker: str, rl_strat: RLAgentStrategy | None) -> list[tuple[str, object]]:
    eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
    gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
    try:
        adv = get_adv(ticker, exclude_dates=[OOS_DATE])
    except Exception:
        adv = 1e7
    strategies: list[tuple[str, object]] = [
        ("twap",           TWAPStrategy(num_slices=NUM_SLICES)),
        ("ac_ra",          AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
            lambda_risk=LAMBDA_RISK, adv_shares=adv,
        )),
        ("vwap_following", VWAPFollowingStrategy(num_slices=NUM_SLICES)),
        ("pov_5pct_cap",   POVStrategy(target_pov=None, cap_pov=0.05)),
        ("toth",           TothStrategy(participation_cap=0.10)),
    ]
    if rl_strat is not None:
        strategies.append(("rl_v3", rl_strat))
    return strategies


def main():
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tickers = list_all_tickers()
    print("=" * 70)
    print(f"Phase E — Cross-ticker OOS eval (Day 5 = {OOS_DATE})")
    print(f"Tickers with full 5-day coverage: {len(tickers)}")
    rl_strat = None
    if RL_MODEL_PATH.exists():
        rl_strat = RLAgentStrategy(
            model_path=str(RL_MODEL_PATH),
            slice_minutes=60, step_seconds=30, n_steps=120,
            observation_mode="v3", max_action_per_step=0.05,
        )
        rl_strat.name = "rl_v3"
        print(f"RL model: {RL_MODEL_PATH}")
    else:
        print(f"⚠️  {RL_MODEL_PATH} missing — RL strategy skipped")

    n_strats = 6 if rl_strat is not None else 5
    print(f"Strategies per ticker: {n_strats}  →  total backtests: "
          f"{len(tickers) * n_strats}")
    print("=" * 70)

    rows = []
    t0 = time.perf_counter()
    n_done = n_failed = 0
    # Causal lookback: only Day 1-4 (NEVER include the OOS day = Day 5).
    TRAIN_DATES = ["20200113", "20200114", "20200115", "20200116"]
    for ticker in tickers:
        try:
            df = load_eq_taq(ticker, OOS_DATE)
            engine = BacktestEngine(ticker, OOS_DATE, market_df=df)
            profile = compute_lookback_volume_profile(
                ticker, lookback_dates=TRAIN_DATES, bin_minutes=5,
            )
            ctx = engine.market_context(
                bin_minutes=5, volume_profile_override=profile,
            )
            ctx["adv_shares"] = get_adv(ticker, exclude_dates=[OOS_DATE])
        except Exception as e:
            print(f"  ❌ {ticker}: load failed — {e}")
            continue
        parent = ParentOrder(
            ticker=ticker, date=OOS_DATE, side="sell",
            quantity=PARENT_QTY, start_ns=START_TIME_NS, end_ns=END_TIME_NS,
        )
        for label, strat in make_strategies(ticker, rl_strat):
            try:
                res = engine.run(parent, strat, market_context=ctx)
                m = res.metrics
                rows.append({
                    "ticker": ticker, "strategy": label,
                    "is_bps": m.is_bps,
                    "vwap_slip_bps": m.vwap_slip_bps,
                    "eff_spread_bps": m.effective_spread_bps,
                    "markout_60s_bps": m.markout_60s_bps,
                })
                n_done += 1
            except Exception as e:
                n_failed += 1
                print(f"  ❌ {ticker} {label}: {type(e).__name__}: {str(e)[:80]}")

    elapsed = time.perf_counter() - t0
    print(f"\n📊 Sweep done in {elapsed/60:.1f} min — {n_done} done, {n_failed} failed")

    df_results = pl.DataFrame(rows)
    csv_path = reports_dir / "phase_e_xticker.csv"
    df_results.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # Per-strategy aggregates
    agg = (
        df_results
        .group_by("strategy")
        .agg(
            pl.col("is_bps").median().alias("median_is"),
            pl.col("is_bps").mean().alias("mean_is"),
            pl.col("is_bps").std().alias("std_is"),
            pl.col("vwap_slip_bps").median().alias("median_vwap_slip"),
            pl.len().alias("n"),
        )
        .sort("median_is")
    )

    # Per-ticker win rate vs vwap_following
    vwap_runs = df_results.filter(pl.col("strategy") == "vwap_following").select(
        ["ticker", "is_bps"]
    ).rename({"is_bps": "vwap_is"})
    win_rates: dict[str, float] = {}
    for s in df_results["strategy"].unique().to_list():
        if s == "vwap_following":
            continue
        joined = (
            df_results.filter(pl.col("strategy") == s)
            .select(["ticker", "is_bps"])
            .join(vwap_runs, on=["ticker"], how="inner")
        )
        wr = (joined["is_bps"] < joined["vwap_is"]).mean()
        win_rates[s] = float(wr) if wr is not None else 0.0

    # Markdown
    md = ["# Phase E — Cross-Ticker OOS Eval (97 tickers × Day 5)\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**OOS date**: {OOS_DATE}  |  **Tickers with full data**: {len(tickers)}")
    md.append(f"**Parent**: {PARENT_QTY:,} shares sell, 10:00–11:00 ET, NBBO routing")
    md.append(f"**RL training pool**: 20 tickers sampled across ADV deciles (Day 1-4); "
              f"OOS evaluates on ALL {len(tickers)} tickers (most never seen during training)")
    md.append(f"**Total backtests**: {n_done} succeeded / {n_failed} failed\n")

    md.append("## Per-strategy aggregates (sorted by median IS, best first)\n")
    md.append("| Strategy | n | Median IS (bps) | Mean IS | Std IS | Median VWAP slip | Win-rate vs VWAP-follow |")
    md.append("|---|---|---|---|---|---|---|")
    for row in agg.iter_rows(named=True):
        s = row["strategy"]
        wr_str = f"{win_rates[s]*100:.0f}%" if s != "vwap_following" else "—"
        md.append(
            f"| {s} | {row['n']} | **{row['median_is']:+.2f}** | "
            f"{row['mean_is']:+.2f} | {row['std_is']:.1f} | "
            f"{row['median_vwap_slip']:+.2f} | {wr_str} |"
        )

    # RL vs VWAP-following per ticker breakdown (top 10 best/worst RL deltas)
    if "rl_v3" in df_results["strategy"].unique().to_list():
        md.append("\n## RL vs VWAP-following per ticker (delta in bps, negative = RL better for sell)\n")
        rl_runs = df_results.filter(pl.col("strategy") == "rl_v3").select(
            ["ticker", "is_bps"]
        ).rename({"is_bps": "rl_is"})
        rl_vs_vwap = rl_runs.join(vwap_runs, on=["ticker"]).with_columns(
            (pl.col("rl_is") - pl.col("vwap_is")).alias("rl_minus_vwap_bps")
        ).sort("rl_minus_vwap_bps")
        rl_wins = rl_vs_vwap.filter(pl.col("rl_minus_vwap_bps") < 0).height
        rl_losses = rl_vs_vwap.filter(pl.col("rl_minus_vwap_bps") > 0).height
        rl_ties = rl_vs_vwap.height - rl_wins - rl_losses
        md.append(f"- RL beats VWAP-following on **{rl_wins}/{rl_vs_vwap.height}** tickers "
                  f"({rl_wins/rl_vs_vwap.height*100:.0f}%)")
        md.append(f"- RL loses to VWAP-following on {rl_losses} tickers, ties on {rl_ties}")
        md.append(f"- Median delta (RL − VWAP-follow): "
                  f"**{rl_vs_vwap['rl_minus_vwap_bps'].median():+.3f} bps**")
        md.append(f"- Mean delta: {rl_vs_vwap['rl_minus_vwap_bps'].mean():+.3f} bps")
        md.append(f"\n**Top 10 RL wins:**")
        md.append("| Ticker | RL IS | VWAP-follow IS | Δ |")
        md.append("|---|---|---|---|")
        for r in rl_vs_vwap.head(10).iter_rows(named=True):
            md.append(f"| {r['ticker']} | {r['rl_is']:+.2f} | {r['vwap_is']:+.2f} | "
                      f"**{r['rl_minus_vwap_bps']:+.2f}** |")
        md.append(f"\n**Top 10 RL losses:**")
        md.append("| Ticker | RL IS | VWAP-follow IS | Δ |")
        md.append("|---|---|---|---|")
        for r in rl_vs_vwap.tail(10).iter_rows(named=True):
            md.append(f"| {r['ticker']} | {r['rl_is']:+.2f} | {r['vwap_is']:+.2f} | "
                      f"**{r['rl_minus_vwap_bps']:+.2f}** |")

    md.append("\n## Caveats\n")
    md.append("- Single OOS day (Day 5 / 20200117). 5 days isn't a lot.")
    md.append("- Parent fixed at 10k shares, 10:00–11:00 ET window. Open/close regimes may differ.")
    md.append("- Spread cost in BacktestEngine (NBB / NBO fills), so all strategies pay half-spread realistically.")
    md.append("- RL trained on 20 tickers sampled across ADV deciles from the 97; ~77 tickers in OOS are unseen during training.\n")

    md_path = reports_dir / "phase_e_xticker.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Distribution plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        strats = sorted(df_results["strategy"].unique().to_list())
        data = [
            df_results.filter(pl.col("strategy") == s)["is_bps"].to_list()
            for s in strats
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(strats) + 1))
        ax.set_xticklabels(strats, rotation=20, ha="right")
        ax.set_ylabel("IS (bps; lower = better for sell)")
        ax.set_title(f"Phase E — IS distribution across {len(tickers)} tickers (Day 5 OOS)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = figures_dir / "phase_e_xticker_distribution.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print("Phase E cross-ticker eval done — see reports/phase_e_xticker.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
