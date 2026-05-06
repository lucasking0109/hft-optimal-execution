"""Cross-day rolling-average intraday volume profile.

Phase D — feeds the v2 observation feature `vol_profile_pct`: at any
anchor_ns within RTH, what fraction of a typical day's cumulative trade
volume has happened by this time?

Built from N reference days (e.g. Day 1–4 train pool); excludes the OOS
day (Day 5) to prevent leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RTH_OPEN_NS = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)
RTH_CLOSE_NS = 16 * 3600 * int(1e9)
DEFAULT_BIN_MIN = 5
NS_PER_MIN = 60 * 1_000_000_000


@dataclass
class VolProfileBaseline:
    """Cumulative-volume curve normalised to [0, 1] across RTH.

    `bucket_starts_ns[i]` is the start of bucket i (5-min by default);
    `cum_pct[i]` is the average cumulative-volume fraction up to that point.
    """

    bucket_starts_ns: np.ndarray   # shape (N+1,) covering RTH
    cum_pct: np.ndarray            # shape (N+1,), monotone in [0, 1]
    n_days: int
    ticker: str

    def cumulative_pct_at(self, anchor_ns: int) -> float:
        """Fraction of typical-day volume already traded by `anchor_ns`."""
        if anchor_ns <= self.bucket_starts_ns[0]:
            return 0.0
        if anchor_ns >= self.bucket_starts_ns[-1]:
            return 1.0
        idx = int(np.searchsorted(self.bucket_starts_ns, anchor_ns, side="right")) - 1
        idx = max(0, min(idx, len(self.cum_pct) - 1))
        return float(self.cum_pct[idx])

    @classmethod
    def for_ticker(
        cls,
        *,
        ticker: str,
        exclude_dates: list[str] | None = None,
        bin_minutes: int = DEFAULT_BIN_MIN,
        cache_root: Path | None = None,
    ) -> "VolProfileBaseline":
        """Build baseline by averaging volume profile across all cached days
        for `ticker`, excluding `exclude_dates`.

        Falls back loudly (NO Silent Fallback) if no eligible days remain.
        """
        excluded = set(exclude_dates or [])
        root = (cache_root or (PROJECT_ROOT / "data" / "processed" / "eq_taq"))
        if not root.exists():
            raise RuntimeError(f"VolProfileBaseline: cache root {root} not found")
        eligible_dates = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name not in excluded
            and (d / f"{ticker}.parquet").exists()
        )
        if not eligible_dates:
            raise RuntimeError(
                f"VolProfileBaseline: no eligible days for {ticker} "
                f"(excluded={sorted(excluded)})"
            )

        # Build the bucket grid covering RTH: 09:30 → 16:00 in `bin_minutes` steps.
        n_buckets = (RTH_CLOSE_NS - RTH_OPEN_NS) // (bin_minutes * NS_PER_MIN)
        bucket_starts_ns = np.array(
            [RTH_OPEN_NS + i * bin_minutes * NS_PER_MIN for i in range(n_buckets + 1)],
            dtype=np.int64,
        )

        per_day_cum: list[np.ndarray] = []
        for date in eligible_dates:
            try:
                df = pl.read_parquet(root / date / f"{ticker}.parquet")
            except Exception:
                continue
            try:
                cum = _cumulative_volume_at_buckets(df, bucket_starts_ns)
            except ValueError:
                continue
            per_day_cum.append(cum)

        if not per_day_cum:
            raise RuntimeError(
                f"VolProfileBaseline: every parquet failed to load for {ticker}"
            )

        avg_cum = np.mean(per_day_cum, axis=0)
        # Normalise to [0, 1] using end-of-day cumulative.
        total = avg_cum[-1] if avg_cum[-1] > 0 else 1.0
        cum_pct = (avg_cum / total).astype(np.float64)
        # Force monotonicity (rolling avg should already be monotone but be safe).
        cum_pct = np.maximum.accumulate(cum_pct)
        cum_pct = np.clip(cum_pct, 0.0, 1.0)

        return cls(
            bucket_starts_ns=bucket_starts_ns,
            cum_pct=cum_pct,
            n_days=len(per_day_cum),
            ticker=ticker,
        )


def _cumulative_volume_at_buckets(
    df: pl.DataFrame, bucket_starts_ns: np.ndarray,
) -> np.ndarray:
    """Compute cumulative trade volume at each bucket boundary."""
    if "ns_of_day" not in df.columns:
        from hft.data.timeparse import add_eq_ns_of_day
        df = add_eq_ns_of_day(df)
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"])).sort("ns_of_day")
    if trades.is_empty():
        raise ValueError("no trades")
    ns = trades["ns_of_day"].to_numpy()
    qty = trades["Quantity"].to_numpy().astype(np.float64)
    cum_qty = np.cumsum(qty)
    # For each bucket boundary, find the cumulative volume up to that ns.
    out = np.zeros(len(bucket_starts_ns), dtype=np.float64)
    idx = np.searchsorted(ns, bucket_starts_ns, side="right") - 1
    for i, j in enumerate(idx):
        if j < 0:
            out[i] = 0.0
        elif j >= len(cum_qty):
            out[i] = cum_qty[-1]
        else:
            out[i] = cum_qty[j]
    return out
