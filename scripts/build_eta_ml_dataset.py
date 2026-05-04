"""Phase B.3 Step 1 — build per-event impact dataset for ML eta prediction.

For each (ticker, date), walks all large-trade events (top decile by qty)
and computes:
  TARGET:  signed_impact_bps (mid drift over 5s after the trade, signed by
           Lee-Ready aggressor direction)
  FEATURES (per event):
    - size_pct_adv          (already used in Phase 3 η regression)
    - seconds_from_open     (time-of-day signal)
    - recent_vol_bps_15m    (15-min rolling std of mid log returns)
    - recent_spread_bps_15m (15-min rolling avg NBBO spread)
    - recent_aggressor_imb_15m (signed flow / total flow over last 15m)
    - recent_volume_15m_pct_adv (last-15m trades / ADV)
    - ticker_idx            (categorical 0..4 for AAPL/AMZN/AMD/TSLA/NVDA)

Outputs `data/processed/eta_ml_dataset.parquet`.

⚠️ Distributional caveat (read before using the model)
------------------------------------------------------
The training set is filtered to **top-decile trade events**
(`PERCENTILE_THRESHOLD = 0.90` below). Typical training-row
`size_pct_adv` ranges roughly 0.5–2%.

Any downstream inference (e.g. `ACMLStrategy.predict_eta`) that queries
the model with much smaller `size_pct_adv` (e.g. ~0.025% for a 167-share
child) is asking the model to extrapolate to a regime it never saw.
See `reports/phase_b3_ac_ml.md` "Root cause analysis" for empirical
evidence (feature importance shows `size_pct_adv` ranks lowest).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.nbbo_lookup import NBBOLookup                       # noqa: E402
from hft.data import load_eq_taq                                      # noqa: E402
from hft.data.timeparse import add_eq_ns_of_day, filter_rth           # noqa: E402

TICKERS = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATES = ["20200113", "20200114", "20200115", "20200116", "20200117"]
TICKER_IDX = {t: i for i, t in enumerate(TICKERS)}

# ADV from Phase 3 calibration (reports/calibration.csv) — avoid re-computing
ADV_LOOKUP: dict[str, float] = {
    "AAPL": 33_809_763.0,
    "AMZN":  2_773_115.0,
    "AMD":  42_871_012.0,
    "TSLA": 20_365_299.4,
    "NVDA":  7_715_207.0,
}

PERCENTILE_THRESHOLD = 0.90
IMPACT_WINDOW_SEC = 5
WINDOW_15M_NS = 15 * 60 * 1_000_000_000
RTH_OPEN_NS = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)


def build_for(ticker: str, date: str) -> pl.DataFrame:
    """Build per-event feature DataFrame for one (ticker, date)."""
    df = load_eq_taq(ticker, date)
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    df = filter_rth(df, src="Timestamp")

    adv = ADV_LOOKUP[ticker]
    lookup = NBBOLookup(df)

    # Sorted trades for both target events and rolling-feature aggregation
    trades = (
        df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
        .sort("ns_of_day")
    )
    if trades.is_empty():
        return pl.DataFrame()

    # Pre-compute trade arrays for fast 15-min rolling features
    trade_ns = trades["ns_of_day"].to_numpy()
    trade_qty = trades["Quantity"].to_numpy()
    trade_px = trades["Price"].to_numpy()

    # Mid before each trade (for Lee-Ready aggressor sign for ALL trades)
    # We'll compute lazily in loop below to avoid O(n²) blowup.

    # Large-trade subset (top decile by qty)
    threshold = trades["Quantity"].quantile(PERCENTILE_THRESHOLD)
    large = trades.filter(pl.col("Quantity") >= threshold)

    horizon_ns = IMPACT_WINDOW_SEC * 1_000_000_000

    # NBBO spread + mid quotes (for rolling features)
    bid = (
        df.filter(pl.col("EventType") == "QUOTE BID NB")
        .select("ns_of_day", "Price")
        .sort("ns_of_day")
    )
    ask = (
        df.filter(pl.col("EventType") == "QUOTE ASK NB")
        .select("ns_of_day", "Price")
        .sort("ns_of_day")
    )
    bid_ns = bid["ns_of_day"].to_numpy()
    bid_px = bid["Price"].to_numpy()
    ask_ns = ask["ns_of_day"].to_numpy()
    ask_px = ask["Price"].to_numpy()

    rows: list[dict] = []
    for ev in large.iter_rows(named=True):
        ns = ev["ns_of_day"]
        qty = ev["Quantity"]
        px = ev["Price"]
        if qty is None or qty <= 0 or px is None or px <= 0:
            continue
        if ns < RTH_OPEN_NS:
            continue   # skip pre-market events

        # Pre-trade mid (1ms before to avoid contemporaneous overlap)
        mid_before = lookup.mid_at(ns - 1_000_000)
        mid_after = lookup.mid_at(ns + horizon_ns)
        if mid_before is None or mid_after is None or mid_before <= 0:
            continue

        # Lee-Ready sign
        if px > mid_before * (1 + 1e-9):
            sign = +1
        elif px < mid_before * (1 - 1e-9):
            sign = -1
        else:
            continue
        impact_bps = sign * (mid_after - mid_before) / mid_before * 10000

        # Features
        size_pct_adv = qty / adv * 100
        seconds_from_open = (ns - RTH_OPEN_NS) / 1e9

        # 15-min rolling: vol of log mid returns at 1-sec sampling
        # Use bid/ask arrays to reconstruct mid at sample times
        win_start = max(ns - WINDOW_15M_NS, RTH_OPEN_NS)
        # Sample mid every 1 sec in [win_start, ns)
        n_samples = max(1, int((ns - win_start) / 1e9))
        sample_times = np.linspace(win_start, ns - 1, num=min(n_samples, 600), dtype=np.int64)
        sampled_mids = []
        for st in sample_times:
            m = lookup.mid_at(int(st))
            if m is not None and m > 0:
                sampled_mids.append(m)
        if len(sampled_mids) < 5:
            recent_vol_bps_15m = 0.0
            recent_spread_bps_15m = 0.0
        else:
            log_rets = np.diff(np.log(np.array(sampled_mids)))
            recent_vol_bps_15m = float(np.std(log_rets) * 1e4)
            # Spread proxy: use mid bid/ask values around mid
            i_bid = np.searchsorted(bid_ns, ns, side="right") - 1
            i_ask = np.searchsorted(ask_ns, ns, side="right") - 1
            if 0 <= i_bid < len(bid_px) and 0 <= i_ask < len(ask_px):
                b, a = bid_px[i_bid], ask_px[i_ask]
                if b > 0 and a > 0 and a > b:
                    recent_spread_bps_15m = (a - b) / ((a + b) / 2) * 1e4
                else:
                    recent_spread_bps_15m = 0.0
            else:
                recent_spread_bps_15m = 0.0

        # 15-min rolling: aggressor imbalance + volume
        i_lo = np.searchsorted(trade_ns, win_start, side="left")
        i_hi = np.searchsorted(trade_ns, ns, side="left")
        if i_hi > i_lo:
            window_qtys = trade_qty[i_lo:i_hi]
            window_pxs = trade_px[i_lo:i_hi]
            window_ns = trade_ns[i_lo:i_hi]
            # Quick Lee-Ready for window trades using existing mid lookups
            # Vectorize: use mid at each window trade time minus 1ms
            window_signs = []
            for k in range(len(window_qtys)):
                m = lookup.mid_at(int(window_ns[k] - 1_000_000))
                if m is None or m <= 0:
                    window_signs.append(0)
                    continue
                if window_pxs[k] > m * (1 + 1e-9):
                    window_signs.append(+1)
                elif window_pxs[k] < m * (1 - 1e-9):
                    window_signs.append(-1)
                else:
                    window_signs.append(0)
            window_signs = np.array(window_signs)
            total_vol = float(window_qtys.sum())
            signed_vol = float((window_qtys * window_signs).sum())
            if total_vol > 0:
                recent_aggressor_imb_15m = signed_vol / total_vol
            else:
                recent_aggressor_imb_15m = 0.0
            recent_volume_15m_pct_adv = total_vol / adv * 100
        else:
            recent_aggressor_imb_15m = 0.0
            recent_volume_15m_pct_adv = 0.0

        rows.append({
            "ticker": ticker, "date": date,
            "ns_of_day": ns,
            "size_pct_adv": float(size_pct_adv),
            "signed_impact_bps": float(impact_bps),
            "seconds_from_open": float(seconds_from_open),
            "recent_vol_bps_15m": float(recent_vol_bps_15m),
            "recent_spread_bps_15m": float(recent_spread_bps_15m),
            "recent_aggressor_imb_15m": float(recent_aggressor_imb_15m),
            "recent_volume_15m_pct_adv": float(recent_volume_15m_pct_adv),
            "ticker_idx": TICKER_IDX[ticker],
        })

    return pl.DataFrame(rows)


def main():
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase B.3 Step 1 — build eta-ML training dataset")
    print(f"Tickers: {TICKERS}")
    print(f"Dates:   {DATES}")
    print("=" * 70)

    all_dfs = []
    t0 = time.perf_counter()
    for ticker in TICKERS:
        for date in DATES:
            t_cell = time.perf_counter()
            try:
                df_cell = build_for(ticker, date)
            except Exception as e:
                print(f"  ❌ {ticker} {date}: {type(e).__name__}: {str(e)[:80]}")
                continue
            t_sec = time.perf_counter() - t_cell
            print(f"  ✓ {ticker} {date}: {len(df_cell):>5} events ({t_sec:.1f}s)")
            if len(df_cell):
                all_dfs.append(df_cell)

    if not all_dfs:
        print("🔴 No events extracted — aborting")
        sys.exit(2)

    full = pl.concat(all_dfs, how="diagonal")
    out_path = out_dir / "eta_ml_dataset.parquet"
    full.write_parquet(out_path)
    elapsed = time.perf_counter() - t0
    print(f"\n💾 Wrote {out_path} — {len(full):,} rows in {elapsed/60:.1f} min")

    # Quick sanity stats
    print(f"\nFeature stats:")
    print(full.describe())


if __name__ == "__main__":
    main()
