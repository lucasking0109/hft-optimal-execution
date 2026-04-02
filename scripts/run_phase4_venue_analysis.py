"""Phase 4: Multi-venue routing analysis.

For each (ticker, date) compute the full venue-metrics DataFrame, then aggregate
across the 5 days to produce a stable per-venue ranking. The Naive (volume-share)
vs SOR (composite-score) allocations are compared.

Outputs:
    reports/phase4_venue_metrics.csv       — per (ticker, date, venue) row
    reports/phase4_sor.md                  — narrative summary + ranked tables
    reports/phase4_naive_vs_sor_alloc.csv  — allocation weights for both rules
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.sor import (                       # noqa: E402
    compute_composite_score,
    naive_volume_allocation,
    sor_score_allocation,
)
from hft.analysis.venue_metrics import (              # noqa: E402
    NON_ROUTABLE_VENUES,
    compute_all_venue_metrics,
)
from hft.data import AVAILABLE_DATES, load_eq_taq    # noqa: E402

TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "AMD"]
DATES = list(AVAILABLE_DATES)


def run_one(ticker: str, date: str) -> pl.DataFrame | None:
    try:
        df = load_eq_taq(ticker, date)
    except Exception as e:
        print(f"  [{ticker} {date}] load failed: {e}")
        return None
    metrics = compute_all_venue_metrics(df, horizon_seconds=60)
    metrics = metrics.with_columns(
        pl.lit(ticker).alias("ticker"),
        pl.lit(date).alias("date"),
    )
    return metrics


def main():
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    all_rows: list[pl.DataFrame] = []
    for ticker in TICKERS:
        for date in DATES:
            print(f"  {ticker} {date}…")
            res = run_one(ticker, date)
            if res is not None:
                all_rows.append(res)

    if not all_rows:
        print("No data — aborting")
        return

    full = pl.concat(all_rows, how="diagonal")
    full.write_csv(out_dir / "phase4_venue_metrics.csv")
    print(f"\nWrote {out_dir / 'phase4_venue_metrics.csv'} ({len(full)} rows)")

    # Aggregate across days per (ticker, venue), then average across tickers
    agg = (
        full.group_by(["ticker", "Exchange"])
        .agg([
            pl.col("nbbo_share_bid_pct").mean().alias("nbbo_share_bid_pct"),
            pl.col("nbbo_share_ask_pct").mean().alias("nbbo_share_ask_pct"),
            pl.col("volume_share_pct").mean().alias("volume_share_pct"),
            pl.col("avg_depth_at_best").mean().alias("avg_depth_at_best"),
            pl.col("adverse_selection_bps").mean().alias("adverse_selection_bps"),
            pl.col("n_trades").sum().alias("n_trades"),
            pl.col("routable").first().alias("routable"),
        ])
    )

    cross_ticker = (
        agg.group_by("Exchange")
        .agg([
            pl.col("nbbo_share_bid_pct").mean().alias("nbbo_share_bid_pct"),
            pl.col("nbbo_share_ask_pct").mean().alias("nbbo_share_ask_pct"),
            pl.col("volume_share_pct").mean().alias("volume_share_pct"),
            pl.col("avg_depth_at_best").mean().alias("avg_depth_at_best"),
            pl.col("adverse_selection_bps").mean().alias("adverse_selection_bps"),
            pl.col("n_trades").sum().alias("n_trades"),
            pl.col("routable").first().alias("routable"),
        ])
        .sort("volume_share_pct", descending=True)
    )

    scored = compute_composite_score(cross_ticker)

    # Allocations
    naive = naive_volume_allocation(cross_ticker, top_k=5)
    sor = sor_score_allocation(scored, top_k=5)

    alloc_rows = []
    for venue in set(naive) | set(sor):
        alloc_rows.append({
            "venue": venue,
            "naive_pct": naive.get(venue, 0.0) * 100,
            "sor_pct": sor.get(venue, 0.0) * 100,
            "diff_bps": (sor.get(venue, 0.0) - naive.get(venue, 0.0)) * 10000,
        })
    alloc_df = pl.DataFrame(alloc_rows).sort("naive_pct", descending=True)
    alloc_df.write_csv(out_dir / "phase4_naive_vs_sor_alloc.csv")

    # ---- markdown ----
    md = ["# Phase 4: Multi-Venue SOR Routing Analysis\n"]
    md.append(f"**Tickers**: {', '.join(TICKERS)}; **Dates**: {', '.join(DATES)}; "
              f"**Total venue-day rows**: {len(full)}\n")
    md.append("\n## Cross-ticker venue ranking (averaged over 5 tickers × 5 days)\n")
    md.append("| Venue | Routable | Volume share | NBBO share (bid) | NBBO share (ask) | Avg depth | Adverse selection (bps) | Composite score |")
    md.append("|---|---|---|---|---|---|---|---|")
    for row in scored.iter_rows(named=True):
        md.append(
            f"| {row['Exchange']} | {'✅' if row['routable'] else '❌'} | "
            f"{row['volume_share_pct']:.2f}% | "
            f"{row['nbbo_share_bid_pct']:.2f}% | "
            f"{row['nbbo_share_ask_pct']:.2f}% | "
            f"{row['avg_depth_at_best']:.0f} | "
            f"{row['adverse_selection_bps']:+.2f} | "
            f"{row['composite_score']:+.2f} |"
        )

    md.append("\n## Top-5 routable allocation comparison\n")
    md.append("| Venue | Naive (volume %) | SOR (score %) | SOR − Naive |")
    md.append("|---|---|---|---|")
    for row in alloc_df.iter_rows(named=True):
        md.append(f"| {row['venue']} | {row['naive_pct']:.2f}% | {row['sor_pct']:.2f}% | "
                  f"{(row['sor_pct'] - row['naive_pct']):+.2f}pp |")

    md.append("\n## Notes (NO Silent Fallback)\n")
    md.append(f"- **Non-routable venues** (excluded from SOR): {sorted(NON_ROUTABLE_VENUES)}.")
    md.append("  - FINRA isn't an exchange — it reports off-exchange (dark pool / internalizer) trades. SOR can't physically route there.")
    md.append("- Adverse selection uses Lee-Ready signed markout @ 60 sec. Lower (more negative) = better.")
    md.append("- Composite score weights: 0.5 volume + 0.3 depth − 0.2 adverse selection (z-scored across routable venues).")
    md.append("- Lookback is just 5 days — production SOR rules need months of data to stabilise. Use this analysis as methodology demo, not as live rule.")
    md.append("- Composite score is *relative* (z-scored), so absolute values aren't comparable across stocks.")

    md_path = out_dir / "phase4_sor.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
