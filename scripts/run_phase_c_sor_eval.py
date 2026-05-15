"""Phase C — SOR routing comparison sweep.

Compares execution strategies under different routing modes:
  Routing modes:
    - 'nbbo'     : baseline, fill at NBBO (no venue routing) — legacy
    - 'naive'    : route by historical volume share weighted random
    - 'top1'     : route all children to highest composite-score venue
    - 'weighted' : route by softmax(composite_score) weighted random

  Base strategies: TWAP / VWAP-following / AC-RN / AC-RA (4)
  Tickers: AAPL, AMZN, AMD, TSLA, NVDA (5)
  Dates: 20200113 – 20200117 (5)

= 5 × 5 × 4 × 4 = **400 backtests** (was 300; added 'nbbo' baseline).

Output:
  - reports/phase_c_sor_routing.csv  — full results
  - reports/phase_c_sor_routing.md   — summary + improvement table
  - reports/figures/phase_c_routing_violin.png
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

from hft.analysis.impact import (                                    # noqa: E402
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
)
from hft.analysis.sor import (                                      # noqa: E402
    compute_composite_score,
    naive_volume_allocation,
    sor_score_allocation,
)
from hft.analysis.venue_metrics import compute_all_venue_metrics    # noqa: E402
from hft.analysis.vwap import compute_lookback_volume_profile        # noqa: E402
from hft.backtest.engine import BacktestEngine                       # noqa: E402
from hft.data import load_eq_taq                                    # noqa: E402
from hft.strategies.almgren_chriss import (                          # noqa: E402
    AlmgrenChrissStrategy,
)
from hft.strategies.base import ParentOrder                          # noqa: E402
from hft.strategies.sor_routing import SORRoutingStrategy            # noqa: E402
from hft.strategies.twap import TWAPStrategy                         # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy     # noqa: E402

TICKERS = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATES = ["20200113", "20200114", "20200115", "20200116", "20200117"]
ROUTING_MODES = ["nbbo", "naive", "top1", "weighted"]
BASE_STRATEGIES = ["twap", "vwap_following", "ac_rn", "ac_ra"]
PARENT_QTY = 10_000   # match Phase 3
NUM_SLICES = 60
LAMBDA_RISK_AVERSE = 1e-3
SIGMA_DEFAULT = 1.0  # bps/sqrt(sec) — placeholder; calibrated per ticker if available
START_TIME_NS = 10 * 3600 * int(1e9)        # 10:00 ET
END_TIME_NS = 11 * 3600 * int(1e9)          # 11:00 ET

# Cache ADV per ticker so we don't recompute per cell
ADV_CACHE: dict[str, float] = {}


def get_adv(ticker: str) -> float:
    if ticker not in ADV_CACHE:
        try:
            from hft.analysis.impact import average_daily_volume_from_history
            ADV_CACHE[ticker] = average_daily_volume_from_history(ticker, lookback_days=60)
        except Exception:
            ADV_CACHE[ticker] = 1e7   # fallback default if history not available
    return ADV_CACHE[ticker]


def make_base_strategy(name: str, ticker: str):
    if name == "twap":
        return TWAPStrategy(num_slices=NUM_SLICES)
    if name == "vwap_following":
        return VWAPFollowingStrategy(num_slices=NUM_SLICES)
    if name in ("ac_rn", "ac_ra"):
        eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
        gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
        adv = get_adv(ticker)
        lam = 0.0 if name == "ac_rn" else LAMBDA_RISK_AVERSE
        return AlmgrenChrissStrategy(
            num_slices=NUM_SLICES,
            eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma,
            sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
            lambda_risk=lam,
            adv_shares=adv,
        )
    raise ValueError(f"Unknown base strategy: {name}")


def build_allocations(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Compute the 3 SOR allocations for a given (ticker, date) df."""
    metrics = compute_all_venue_metrics(df)
    metrics_with_score = compute_composite_score(metrics)
    naive_alloc = naive_volume_allocation(metrics, top_k=5)
    score_alloc = sor_score_allocation(metrics_with_score, top_k=5)
    # top1 = highest-score routable venue (re-using sor_score_allocation top-1)
    if score_alloc:
        top1_venue = max(score_alloc, key=score_alloc.get)
        top1_alloc = {top1_venue: 1.0}
    else:
        top1_alloc = {}
    return {
        "naive": naive_alloc,
        "top1": top1_alloc,
        "weighted": score_alloc,
    }


def run_one(engine, parent, base_name, mode, allocations, market_context):
    base_strat = make_base_strategy(base_name, parent.ticker)
    if mode == "nbbo":
        strat = base_strat
        venue_aware = False
    else:
        alloc = allocations.get(mode, {})
        if not alloc:
            return None  # no routable venues
        strat = SORRoutingStrategy(base_strat, alloc, mode=mode, seed=42)
        venue_aware = True
    res = engine.run(parent, strat, market_context=market_context,
                     venue_aware_fills=venue_aware)
    return res


def main():
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase C — SOR routing sweep")
    print(f"Tickers: {TICKERS}")
    print(f"Dates:   {DATES}")
    print(f"Strategies: {BASE_STRATEGIES}")
    print(f"Routing modes: {ROUTING_MODES}")
    print(f"Total runs: {len(TICKERS) * len(DATES) * len(BASE_STRATEGIES) * len(ROUTING_MODES)}")
    print("=" * 70)

    rows = []
    t0_total = time.perf_counter()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for ticker in TICKERS:
        for date in DATES:
            try:
                df = load_eq_taq(ticker, date)
                engine = BacktestEngine(ticker, date, market_df=df)
                allocations = build_allocations(df)
                # Leave-one-out lookback volume profile (no same-day leak).
                lookback = [d for d in DATES if d != date]
                profile = compute_lookback_volume_profile(
                    ticker, lookback_dates=lookback, bin_minutes=5,
                )
                ctx = engine.market_context(
                    bin_minutes=5, volume_profile_override=profile,
                )
            except Exception as e:
                print(f"  ❌ {ticker} {date}: load/alloc failed: {e}")
                n_failed += 1
                continue

            parent = ParentOrder(
                ticker=ticker, date=date, side="sell",
                quantity=PARENT_QTY, start_ns=START_TIME_NS, end_ns=END_TIME_NS,
            )

            for base_name in BASE_STRATEGIES:
                for mode in ROUTING_MODES:
                    try:
                        res = run_one(engine, parent, base_name, mode, allocations, ctx)
                        if res is None:
                            n_skipped += 1
                            continue
                        m = res.metrics
                        rows.append({
                            "ticker": ticker, "date": date,
                            "base_strategy": base_name, "routing": mode,
                            "median_is_bps": m.is_bps,
                            "vwap_slip_bps": m.vwap_slip_bps,
                            "eff_spread_bps": m.effective_spread_bps,
                            "markout_60s_bps": m.markout_60s_bps,
                            "reversion_5m_bps": m.reversion_5m_bps,
                            "price_var": m.price_var,
                            "pov_pct": m.pov,
                            "oversize_count": res.oversize_count,
                            "null_nbbo_count": res.null_nbbo_count,
                            "n_fills": len(res.fills),
                            "arrival_mid": res.arrival_mid,
                            "market_vwap": res.market_vwap,
                        })
                        n_done += 1
                    except Exception as e:
                        n_failed += 1
                        print(f"  ❌ {ticker} {date} {base_name}/{mode}: {type(e).__name__}: {str(e)[:80]}")
            print(f"  ✓ {ticker} {date}: {n_done} runs done, {n_skipped} skipped, {n_failed} failed")

    elapsed = time.perf_counter() - t0_total
    print(f"\n📊 Sweep complete in {elapsed/60:.1f} min — "
          f"{n_done} done, {n_skipped} skipped (no routable venues), {n_failed} failed")

    df_results = pl.DataFrame(rows)
    csv_path = reports_dir / "phase_c_sor_routing.csv"
    df_results.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # Aggregate per (base_strategy, routing) with median across (ticker, date)
    agg = (
        df_results
        .group_by(["base_strategy", "routing"])
        .agg(
            pl.col("median_is_bps").median().alias("median_is_bps"),
            pl.col("vwap_slip_bps").median().alias("median_vwap_slip"),
            pl.col("eff_spread_bps").median().alias("median_eff_spread"),
            pl.col("price_var").median().alias("median_price_var"),
            pl.col("median_is_bps").std().alias("std_is_bps"),
            pl.len().alias("n_runs"),
        )
        .sort(["base_strategy", "routing"])
    )

    print("\nAggregate (median across 5 tickers × 5 days):")
    print(agg)

    # Compute SOR improvement vs nbbo baseline per base strategy
    improvements = {}
    for base in BASE_STRATEGIES:
        nbbo_row = agg.filter(
            (pl.col("base_strategy") == base) & (pl.col("routing") == "nbbo")
        )
        if nbbo_row.is_empty():
            continue
        nbbo_is = nbbo_row["median_is_bps"][0]
        for mode in ["naive", "top1", "weighted"]:
            mode_row = agg.filter(
                (pl.col("base_strategy") == base) & (pl.col("routing") == mode)
            )
            if mode_row.is_empty():
                continue
            mode_is = mode_row["median_is_bps"][0]
            improvements[(base, mode)] = mode_is - nbbo_is

    # Markdown report
    md = ["# Phase C — SOR Routing Comparison\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Sweep**: {len(TICKERS)} tickers × {len(DATES)} dates × {len(BASE_STRATEGIES)} strategies × {len(ROUTING_MODES)} routings = {len(TICKERS)*len(DATES)*len(BASE_STRATEGIES)*len(ROUTING_MODES)} backtests")
    md.append(f"**Parent order**: {PARENT_QTY:,} shares, sell, 10:00–11:00 ET")
    md.append(f"**Runs done / skipped / failed**: {n_done} / {n_skipped} / {n_failed}\n")

    md.append("## Aggregate median (across 5 tickers × 5 days)\n")
    md.append("| base_strategy | routing | median IS (bps) | std IS | median VWAP slip | median eff spread |")
    md.append("|---|---|---|---|---|---|")
    for row in agg.iter_rows(named=True):
        md.append(f"| {row['base_strategy']} | {row['routing']} | "
                  f"{row['median_is_bps']:+.3f} | {row['std_is_bps']:.2f} | "
                  f"{row['median_vwap_slip']:+.3f} | {row['median_eff_spread']:.3f} |")

    md.append("\n## SOR improvement vs NBBO baseline (median IS Δbps)\n")
    md.append("Positive = SOR routing better (less negative median IS = less slippage)\n")
    md.append("| base_strategy | naive | top1 | weighted |")
    md.append("|---|---|---|---|")
    for base in BASE_STRATEGIES:
        row = [f"| {base} |"]
        for mode in ["naive", "top1", "weighted"]:
            d = improvements.get((base, mode))
            row.append(f" {d:+.3f} |" if d is not None else " — |")
        md.append(" ".join(row))

    md.append("\n## Caveats\n")
    md.append("- Small parent (10k shares) on liquid large-caps: SOR effects expected to be small (<1 bps).")
    md.append("- 'top1' uses single highest-score venue per (ticker,date); could exhaust depth at large size.")
    md.append("- Venue fallback to NBBO occurs when target venue has stale quote.")

    md_path = reports_dir / "phase_c_sor_routing.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Violin plot: median_is_bps × routing for each base strategy
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(BASE_STRATEGIES), figsize=(16, 5), sharey=True)
        for ax, base in zip(axes, BASE_STRATEGIES):
            data_rows = df_results.filter(pl.col("base_strategy") == base)
            datasets = [
                data_rows.filter(pl.col("routing") == m)["median_is_bps"].to_list()
                for m in ROUTING_MODES
            ]
            datasets = [d if d else [0.0] for d in datasets]
            parts = ax.violinplot(datasets, showmedians=True, widths=0.7)
            for pc, c in zip(parts['bodies'], ['#888888', '#aa66cc', '#44aa44', '#dd6644']):
                pc.set_facecolor(c)
                pc.set_alpha(0.65)
            ax.set_xticks(range(1, len(ROUTING_MODES) + 1))
            ax.set_xticklabels(ROUTING_MODES, fontsize=9)
            ax.set_title(base)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Median IS (bps)")
        fig.suptitle("Phase C — SOR routing × base strategy")
        plt.tight_layout()
        fig_path = figures_dir / "phase_c_routing_violin.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print("Phase C complete — see reports/phase_c_sor_routing.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
