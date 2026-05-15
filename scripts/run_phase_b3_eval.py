"""Phase B.3 Step 4 — 9-strategy benchmark adding ACML to Phase B.7 zoo.

Reuses the Phase B.7 sweep config (5 tickers × 5 days) but adds ACML.
Compares ACML vs static AC-RA per (ticker, date).

Outputs:
  - reports/phase_b3_ac_ml.csv
  - reports/phase_b3_ac_ml.md
  - reports/figures/phase_b3_ac_ml_distribution.png
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.impact import (                                    # noqa: E402
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
)
from hft.analysis.vwap import compute_lookback_volume_profile        # noqa: E402
from hft.backtest.engine import BacktestEngine                       # noqa: E402
from hft.data import load_eq_taq                                    # noqa: E402
from hft.strategies.ac_ml import ACMLStrategy                       # noqa: E402
from hft.strategies.almgren_chriss import AlmgrenChrissStrategy     # noqa: E402
from hft.strategies.base import ParentOrder                          # noqa: E402
from hft.strategies.cvxpy_optimal import CVXPYConstrainedAC         # noqa: E402
from hft.strategies.pov import POVStrategy                           # noqa: E402
from hft.strategies.toth import TothStrategy                         # noqa: E402
from hft.strategies.twap import TWAPStrategy                         # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy     # noqa: E402

TICKERS = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATES = ["20200113", "20200114", "20200115", "20200116", "20200117"]
NUM_SLICES = 60
LAMBDA_RISK = 1e-3
SIGMA_DEFAULT = 1.0
PARENT_QTY = 10_000
START_TIME_NS = 10 * 3600 * int(1e9)
END_TIME_NS = 11 * 3600 * int(1e9)
MODEL_PATH = ROOT / "rl" / "checkpoints" / "eta_ml_v0" / "model.json"

ADV_LOOKUP = {
    "AAPL": 33_809_763.0, "AMZN": 2_773_115.0, "AMD": 42_871_012.0,
    "TSLA": 20_365_299.4, "NVDA": 7_715_207.0,
}


def make_strategies(ticker: str) -> list[tuple[str, object]]:
    eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
    gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
    adv = ADV_LOOKUP[ticker]
    strats = [
        ("twap",              TWAPStrategy(num_slices=NUM_SLICES)),
        ("vwap_following",    VWAPFollowingStrategy(num_slices=NUM_SLICES)),
        ("ac_rn",             AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
            lambda_risk=0.0, adv_shares=adv,
        )),
        ("ac_ra",             AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
            lambda_risk=LAMBDA_RISK, adv_shares=adv,
        )),
        ("pov_auto",          POVStrategy(target_pov=None, cap_pov=0.20)),
        ("pov_5pct_cap",      POVStrategy(target_pov=None, cap_pov=0.05)),
        ("toth",              TothStrategy(participation_cap=0.10)),
        ("cvxpy_constrained", CVXPYConstrainedAC(
            num_slices=NUM_SLICES, eta=1.0, sigma=SIGMA_DEFAULT,
            lambda_risk=LAMBDA_RISK, pov_cap=0.20,
        )),
    ]
    if MODEL_PATH.exists():
        strats.append((
            "ac_ml",
            ACMLStrategy(
                model_path=MODEL_PATH,
                adv_shares=adv,
                sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
                lambda_risk=LAMBDA_RISK,
                num_slices=NUM_SLICES,
                gamma_bps_per_pct_adv=gamma,
            ),
        ))
    return strats


def main():
    if not MODEL_PATH.exists():
        print(f"🔴 Model not found at {MODEL_PATH}. Run train_eta_ml_model.py first.")
        sys.exit(1)

    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase B.3 Step 4 — 9-strategy benchmark with ACML")
    print(f"5 tickers × 5 days × 9 strategies")
    print(f"Parent: {PARENT_QTY:,} shares sell, 10:00-11:00 ET")
    print("=" * 70)

    rows = []
    t0 = time.perf_counter()
    n_done = n_failed = 0

    for ticker in TICKERS:
        for date in DATES:
            try:
                df = load_eq_taq(ticker, date)
                engine = BacktestEngine(ticker, date, market_df=df)
                # Leave-one-out lookback volume profile (no same-day leak).
                lookback = [d for d in DATES if d != date]
                profile = compute_lookback_volume_profile(
                    ticker, lookback_dates=lookback, bin_minutes=5,
                )
                ctx = engine.market_context(
                    bin_minutes=5, volume_profile_override=profile,
                )
                ctx["adv_shares"] = ADV_LOOKUP[ticker]
                ctx["market_df"] = df  # ACML needs full df
            except Exception as e:
                print(f"  ❌ {ticker} {date}: load failed: {e}")
                continue
            parent = ParentOrder(
                ticker=ticker, date=date, side="sell",
                quantity=PARENT_QTY, start_ns=START_TIME_NS, end_ns=END_TIME_NS,
            )
            for label, strat in make_strategies(ticker):
                try:
                    res = engine.run(parent, strat, market_context=ctx)
                    m = res.metrics
                    rows.append({
                        "ticker": ticker, "date": date, "strategy": label,
                        "is_bps": m.is_bps, "vwap_slip_bps": m.vwap_slip_bps,
                        "eff_spread_bps": m.effective_spread_bps,
                        "markout_60s_bps": m.markout_60s_bps,
                        "reversion_5m_bps": m.reversion_5m_bps,
                        "price_var": m.price_var, "pov_pct": m.pov,
                        "n_fills": len(res.fills),
                    })
                    if label == "ac_ml":
                        rows[-1]["predicted_eta"] = getattr(strat, "last_predicted_eta", None)
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    print(f"  ❌ {ticker} {date} {label}: {type(e).__name__}: {str(e)[:80]}")
            print(f"  ✓ {ticker} {date}: {n_done} done, {n_failed} failed")

    elapsed = time.perf_counter() - t0
    print(f"\n📊 Sweep done in {elapsed/60:.1f} min — {n_done} done, {n_failed} failed")

    df_results = pl.DataFrame(rows)
    csv_path = reports_dir / "phase_b3_ac_ml.csv"
    df_results.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # Aggregate per strategy
    agg = (
        df_results
        .group_by("strategy")
        .agg(
            pl.col("is_bps").median().alias("median_is"),
            pl.col("is_bps").std().alias("std_is"),
            pl.col("vwap_slip_bps").median().alias("median_vwap_slip"),
            pl.col("eff_spread_bps").median().alias("median_eff_spread"),
            pl.col("price_var").median().alias("median_price_var"),
            pl.len().alias("n_runs"),
        )
        .sort("median_is", descending=False)
    )
    print("\nAggregate (median across 5 tickers × 5 days):")
    print(agg)

    # Compute ACML vs ac_ra delta
    acml_row = agg.filter(pl.col("strategy") == "ac_ml")
    acra_row = agg.filter(pl.col("strategy") == "ac_ra")
    if not acml_row.is_empty() and not acra_row.is_empty():
        acml_is = acml_row["median_is"][0]
        acra_is = acra_row["median_is"][0]
        acml_std = acml_row["std_is"][0]
        acra_std = acra_row["std_is"][0]
        is_delta = acml_is - acra_is
        std_pct = (acml_std - acra_std) / acra_std * 100 if acra_std > 0 else 0
        print(f"\nACML vs static AC-RA:")
        print(f"  Median IS Δ: {is_delta:+.3f} bps  (negative = ACML better)")
        print(f"  Std change : {std_pct:+.1f}%       (negative = ACML lower variance)")

        if is_delta < -1.0 or std_pct < -10:
            verdict = "✅ WIN — ACML beats AC-RA materially"
        elif abs(is_delta) <= 0.5 and abs(std_pct) <= 5:
            verdict = "🟡 NEUTRAL — within noise; ML-η doesn't help on this scale"
        else:
            verdict = "🔴 LOSE — ACML worse; static η preferred"
        print(f"\n  Verdict: {verdict}")
    else:
        verdict = "Could not compute ACML vs AC-RA delta"
        is_delta = None
        std_pct = None

    # Markdown report
    md = ["# Phase B.3 — ACML (AC + ML-predicted η) Eval Report\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Sweep**: {n_done} done / {n_failed} failed across "
              f"{len(TICKERS)} tickers × {len(DATES)} days × 9 strategies")
    md.append(f"**Parent**: {PARENT_QTY:,} shares sell 10:00-11:00 ET, NBBO routing\n")

    md.append("## Aggregate (median across 5 tickers × 5 days), ranked by IS\n")
    md.append("| Strategy | n | Median IS | IS std | Median VWAP slip | Median eff spread |")
    md.append("|---|---|---|---|---|---|")
    for r in agg.iter_rows(named=True):
        md.append(f"| {r['strategy']} | {r['n_runs']} | "
                  f"**{r['median_is']:+.2f}** | {r['std_is']:.1f} | "
                  f"{r['median_vwap_slip']:+.2f} | {r['median_eff_spread']:.2f} |")

    md.append(f"\n## ACML vs static AC-RA (key comparison)\n")
    md.append(f"- **Median IS Δ**: {is_delta:+.3f} bps" if is_delta is not None else "- (data missing)")
    md.append(f"- **Std change**: {std_pct:+.1f}%" if std_pct is not None else "")
    md.append(f"- **Verdict**: {verdict}\n")

    md.append("## Caveats\n")
    md.append("- 5-day training data is small for ML; OOS R² known to be marginal.")
    md.append("- ACML predicts η once at parent.start_ns (not per child); per-child")
    md.append("  prediction would be closer to ideal but adds compute.")
    md.append("- σ default 1 bps/√sec; not per-ticker calibrated.")
    md.append("- Lambda_risk=1e-3 same for ACML and static AC-RA for fair comparison.\n")

    md_path = reports_dir / "phase_b3_ac_ml.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Distribution plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 6))
        strategies = agg["strategy"].to_list()
        data = [df_results.filter(pl.col("strategy") == s)["is_bps"].to_list()
                for s in strategies]
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor('#4a90d9')
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(strategies) + 1))
        ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("IS (bps)")
        ax.set_title(f"Phase B.3 — 9-strategy IS distribution "
                     f"(5 tickers × 5 days, ACML highlighted)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = figures_dir / "phase_b3_ac_ml_distribution.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print("Phase B.3 done")
    print("=" * 70)


if __name__ == "__main__":
    main()
