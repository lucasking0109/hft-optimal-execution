"""Step 0 — HerderAgent parameter calibration from real AAPL data.

Outputs `data/synthetic/herder_params.json` with:
  - HerderAgent default params (entry_threshold_bps, max_size, lookback range)
    measured from real AAPL micro-dynamics
  - Real 5-min benchmark distribution (autocorr / kurtosis / total_dist IQR
    for synthetic cells to be compared against — apples-to-apples)

Per Stage 3 plan Step 0. NO Silent Fallback: any unreasonable measurement
(e.g., P75 < 0.5 bps or > 50 bps) raises explicit error for user review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.data import load_eq_taq                                       # noqa: E402
from hft.data.timeparse import add_eq_ns_of_day, filter_rth            # noqa: E402
from hft.simulators.stylized_facts import (                            # noqa: E402
    compute_stylized_facts,
    extract_real_taq_series,
    facts_distance,
    total_calibration_distance,
)


# ---------------------------------------------------------------------------
# Sanity ranges (NO Silent Fallback)
# ---------------------------------------------------------------------------

# Per NotebookLM lit: entry threshold typically 5-50 bps for high-frequency
# herders. We require P75 to fall in a reasonable range.
SANE_THRESHOLD_BPS_RANGE = (0.5, 50.0)

# trade_size median for AAPL is typically 50-500. Anything outside is fishy.
SANE_TRADE_SIZE_RANGE = (10.0, 1000.0)


# ---------------------------------------------------------------------------
# 5-sec drift measurement
# ---------------------------------------------------------------------------

def measure_5sec_drift_bps(mids_1hz: np.ndarray) -> dict:
    """Compute distribution of 5-second mid-price drift in bps.

    drift_bps_t = (mid_t - mid_{t-5}) / mid_{t-5} * 1e4

    Returns stats: count, median, P75, P90, P95 of |drift_bps|.
    """
    if len(mids_1hz) < 60:
        raise ValueError(f"Need ≥ 60 mid samples for 5-sec drift, got {len(mids_1hz)}")
    drift = (mids_1hz[5:] - mids_1hz[:-5]) / mids_1hz[:-5] * 1e4
    abs_drift = np.abs(drift[np.isfinite(drift)])
    abs_drift = abs_drift[abs_drift > 0]
    return {
        "n_samples": int(len(abs_drift)),
        "median": float(np.median(abs_drift)),
        "p75": float(np.percentile(abs_drift, 75)),
        "p90": float(np.percentile(abs_drift, 90)),
        "p95": float(np.percentile(abs_drift, 95)),
        "max": float(np.max(abs_drift)),
    }


# ---------------------------------------------------------------------------
# 5-min slice bootstrap (apples-to-apples benchmark)
# ---------------------------------------------------------------------------

def bootstrap_5min_slices(
    taq_df: pl.DataFrame,
    *,
    n_slices: int = 20,
    slice_minutes: int = 5,
    sample_seconds: int = 1,
    seed: int = 1234,
) -> list[dict]:
    """Draw n_slices random 5-min windows from RTH, compute stylized facts on each.

    Returns list of fact summaries (one per slice).
    """
    df = taq_df
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    df = filter_rth(df, src="Timestamp")

    rth_start_ns = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)        # 09:30:00
    rth_end_ns = 16 * 3600 * int(1e9)                              # 16:00:00
    slice_ns = slice_minutes * 60 * int(1e9)

    rng = np.random.default_rng(seed)
    summaries: list[dict] = []
    attempts = 0
    while len(summaries) < n_slices and attempts < n_slices * 3:
        attempts += 1
        # Random start ∈ [rth_start, rth_end - slice_ns]
        start_ns = int(rng.integers(rth_start_ns, rth_end_ns - slice_ns))
        end_ns = start_ns + slice_ns
        slice_df = df.filter(
            (pl.col("ns_of_day") >= start_ns) & (pl.col("ns_of_day") < end_ns)
        )
        if slice_df.is_empty():
            continue
        try:
            mids, spread, trades = extract_real_taq_series(
                slice_df, sample_seconds=sample_seconds, rth_only=False,
            )
            facts = compute_stylized_facts(
                mid_prices=mids, spread_bps=spread, trade_sizes=trades,
                metadata={"slice_start_ns": start_ns},
            )
        except ValueError as e:
            # Slice too sparse (e.g., low-volume periods); skip
            print(f"  slice @ {start_ns/1e9/3600:.2f}h skipped: {e}")
            continue
        summary = facts.summary()
        summary["slice_start_hours"] = start_ns / 1e9 / 3600
        summaries.append(summary)
        print(f"  slice {len(summaries)}/{n_slices} @ {summary['slice_start_hours']:.2f}h: "
              f"ac_l1={summary['vol_autocorr_lag_1']:.3f} kurt={summary['excess_kurtosis']:.1f}")

    if len(summaries) < n_slices // 2:
        raise RuntimeError(f"Only {len(summaries)}/{n_slices} slices succeeded — data too sparse")
    return summaries


def aggregate_slice_distribution(summaries: list[dict]) -> dict:
    """Compute IQR / median / range for each metric across slices."""
    metrics = [
        "excess_kurtosis", "hill_tail_index",
        "vol_autocorr_lag_1", "vol_autocorr_lag_10", "vol_autocorr_lag_50",
        "spread_bps_median", "spread_bps_p95",
        "trade_size_median", "trade_size_p95",
    ]
    out: dict = {"n_slices": len(summaries)}
    for m in metrics:
        vals = [s[m] for s in summaries if s.get(m) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[m] = {
            "median": float(np.median(arr)),
            "iqr_low": float(np.percentile(arr, 25)),
            "iqr_high": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 0 — HerderAgent parameter calibration from real AAPL")
    print("=" * 70)

    print("\n📥 Loading AAPL 2020-01-13...")
    taq_df = load_eq_taq("AAPL", "20200113")
    print(f"   Loaded {len(taq_df):,} rows")

    # ─── 1. 5-sec drift measurement (entry_threshold_bps source) ───
    print("\n🔬 Measuring 5-sec mid drift distribution (full RTH, 1Hz mids)...")
    mids_1hz, _, trades = extract_real_taq_series(
        taq_df, sample_seconds=1, rth_only=True,
    )
    print(f"   {len(mids_1hz):,} 1-Hz mid samples")

    drift_stats = measure_5sec_drift_bps(mids_1hz)
    print(f"   |drift| n={drift_stats['n_samples']:,}")
    print(f"   |drift|_median = {drift_stats['median']:.3f} bps")
    print(f"   |drift|_P75    = {drift_stats['p75']:.3f} bps  ← entry_threshold_bps")
    print(f"   |drift|_P90    = {drift_stats['p90']:.3f} bps")
    print(f"   |drift|_P95    = {drift_stats['p95']:.3f} bps")

    entry_threshold_bps = drift_stats["p75"]
    if not (SANE_THRESHOLD_BPS_RANGE[0] <= entry_threshold_bps <= SANE_THRESHOLD_BPS_RANGE[1]):
        raise ValueError(
            f"P75 drift = {entry_threshold_bps:.3f} bps falls outside sane range "
            f"{SANE_THRESHOLD_BPS_RANGE}. Refusing to use as herder threshold. "
            "Inspect drift distribution before continuing (NO Silent Fallback)."
        )

    # ─── 2. Trade size median (max_size source) ───
    trade_size_median = float(np.median(trades))
    print(f"\n   Real trade_size median = {trade_size_median:.0f}")
    if not (SANE_TRADE_SIZE_RANGE[0] <= trade_size_median <= SANE_TRADE_SIZE_RANGE[1]):
        raise ValueError(
            f"Trade size median = {trade_size_median:.0f} outside sane range "
            f"{SANE_TRADE_SIZE_RANGE} (NO Silent Fallback)."
        )
    # 10% of real median, min 1
    max_size = max(1, int(round(trade_size_median * 0.1)))
    print(f"   max_size = {max_size} (10% of real median)")

    # ─── 3. 5-min benchmark slice distribution (apples-to-apples) ───
    print(f"\n🔬 Bootstrapping 20 random 5-min slices for benchmark IQR...")
    summaries = bootstrap_5min_slices(taq_df, n_slices=20, slice_minutes=5,
                                       sample_seconds=1, seed=1234)
    benchmark = aggregate_slice_distribution(summaries)
    print(f"\n   Benchmark IQR (n={benchmark['n_slices']}):")
    print(f"   excess_kurtosis: median={benchmark['excess_kurtosis']['median']:.1f}, "
          f"IQR=[{benchmark['excess_kurtosis']['iqr_low']:.1f}, "
          f"{benchmark['excess_kurtosis']['iqr_high']:.1f}]")
    print(f"   vol_autocorr_lag_1: median={benchmark['vol_autocorr_lag_1']['median']:.3f}, "
          f"IQR=[{benchmark['vol_autocorr_lag_1']['iqr_low']:.3f}, "
          f"{benchmark['vol_autocorr_lag_1']['iqr_high']:.3f}]")
    print(f"   vol_autocorr_lag_50: median={benchmark['vol_autocorr_lag_50']['median']:.3f}, "
          f"IQR=[{benchmark['vol_autocorr_lag_50']['iqr_low']:.3f}, "
          f"{benchmark['vol_autocorr_lag_50']['iqr_high']:.3f}]")

    # ─── 4. Output JSON ───
    payload = {
        "metadata": {
            "ticker": "AAPL",
            "date": "20200113",
            "sample_seconds": 1,
            "n_total_mids": int(len(mids_1hz)),
        },
        "herder_params": {
            "lookback_window_secs_min": 3.0,
            "lookback_window_secs_max": 30.0,
            "entry_threshold_bps": entry_threshold_bps,
            "max_size": max_size,
            "position_cap": 50,
            "tolerance_ticks": 5,
            "computation_delay_ns": 1_000_000,
        },
        "drift_stats": drift_stats,
        "real_trade_size_median": trade_size_median,
        "benchmark_distribution": benchmark,
        "all_slice_summaries": summaries,
    }
    out_path = out_dir / "herder_params.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n💾 Wrote {out_path}")

    print("\n" + "=" * 70)
    print(f"✅ Step 0 complete. HerderAgent params:")
    print(f"   entry_threshold_bps = {entry_threshold_bps:.3f}")
    print(f"   max_size            = {max_size}")
    print(f"   lookback range      = [3.0, 30.0] sec (heterogeneous per agent)")
    print("=" * 70)


if __name__ == "__main__":
    main()
