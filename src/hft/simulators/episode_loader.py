"""Episode loaders for ExecutionEnv.

Two episode sources:
  1. Synthetic — pre-generated 5-min ABIDES episodes
     (`data/synthetic/aapl/episode_seed{N}.parquet`, 300 floats × 1 Hz).
  2. Real — slice from real TAQ tick data via `hft.data.load_eq_taq` +
     `hft.simulators.stylized_facts.extract_real_taq_series`
     (sample_seconds=1).

`EpisodeData.mid_prices` is 1 Hz; length = slice_minutes × 60. Synthetic
is always 300 (5 min). Real supports configurable slice length (5, 30, 60
min etc.) via the `slice_minutes` argument to `load_real_episode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class EpisodeData:
    """An execution episode (5–60 min) replayed from synthetic or real data."""

    mid_prices: np.ndarray            # 1-D 1 Hz mid prices; length = slice_seconds
    arrival_mid: float                 # = mid_prices[0]
    source: str                        # 'synthetic_seed1042' or 'real_AAPL_20200113_09:30'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mid_prices.ndim != 1:
            raise ValueError(
                f"mid_prices must be 1-D, got shape {self.mid_prices.shape} "
                f"(source={self.source})"
            )
        if len(self.mid_prices) < 60:
            raise ValueError(
                f"mid_prices must have ≥ 60 samples, got {len(self.mid_prices)} "
                f"(source={self.source})"
            )
        if self.arrival_mid <= 0:
            raise ValueError(f"arrival_mid must be > 0, got {self.arrival_mid}")


# ---------------------------------------------------------------------------
# Synthetic episodes
# ---------------------------------------------------------------------------

def list_synthetic_episodes(
    *,
    include_aapl: bool = True,
    include_multi_ticker: bool = True,
) -> list[Path]:
    """Return parquet paths for available synthetic episodes.

    By default returns AAPL batch (Phase 5C, ~89 successful seeds) plus
    multi-ticker validation (4 tickers × 25 seeds each).
    """
    paths: list[Path] = []
    aapl_dir = PROJECT_ROOT / "data" / "synthetic" / "aapl"
    multi_dir = PROJECT_ROOT / "data" / "synthetic" / "multi_ticker"

    if include_aapl and aapl_dir.exists():
        paths.extend(sorted(aapl_dir.glob("episode_seed*.parquet")))
    if include_multi_ticker and multi_dir.exists():
        for ticker_dir in sorted(multi_dir.glob("*_episodes")):
            paths.extend(sorted(ticker_dir.glob("episode_seed*.parquet")))
    return paths


def load_synthetic_episode(parquet_path: Path) -> EpisodeData:
    """Load a synthetic episode parquet (`mid_price_1hz` column)."""
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    df = pl.read_parquet(parquet_path)
    if "mid_price_1hz" not in df.columns:
        raise ValueError(
            f"{parquet_path} missing required column 'mid_price_1hz'. "
            f"Has: {df.columns}"
        )
    mids = df["mid_price_1hz"].to_numpy().astype(float)
    # Some sims may return slightly under 300 samples (boundary edge);
    # forward-fill the last valid value to enforce shape (300,).
    if len(mids) < 300:
        pad = np.full(300 - len(mids), mids[-1] if len(mids) else 1.0)
        mids = np.concatenate([mids, pad])
    elif len(mids) > 300:
        mids = mids[:300]
    if not np.all(mids > 0):
        # Replace any non-positive values with prior valid (defensive)
        for i in range(len(mids)):
            if mids[i] <= 0 and i > 0:
                mids[i] = mids[i - 1]
    return EpisodeData(
        mid_prices=mids,
        arrival_mid=float(mids[0]),
        source=f"synthetic_{parquet_path.stem}",
        metadata={"path": str(parquet_path)},
    )


# ---------------------------------------------------------------------------
# Real episodes
# ---------------------------------------------------------------------------

def list_real_5min_slices(
    *,
    ticker: str = "AAPL",
    date: str = "20200113",
    step_minutes: int = 5,
    window_minutes: int | None = None,
) -> list[int]:
    """Return list of valid `start_ns` for slices within RTH.

    RTH = 09:30:00 to 16:00:00 ET.

    Args:
        step_minutes: gap between consecutive slice start times.
        window_minutes: slice length. If None (default), uses step_minutes
            (i.e., non-overlapping slices, backward-compat behavior).
            If different from step_minutes, slices overlap.

    Phase G use case: `step_minutes=30, window_minutes=60` gives 12 overlapping
    60-min slices starting every 30 min from 09:30 to 15:00 — covers full
    RTH including 15:00–16:00 (which non-overlapping 60-min step would miss).
    """
    rth_start_ns = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)        # 09:30:00
    rth_end_ns = 16 * 3600 * int(1e9)                              # 16:00:00
    if window_minutes is None:
        window_minutes = step_minutes
    slice_ns = window_minutes * 60 * int(1e9)
    step_ns = step_minutes * 60 * int(1e9)
    starts: list[int] = []
    t = rth_start_ns
    while t + slice_ns <= rth_end_ns:
        starts.append(t)
        t += step_ns
    return starts


def load_real_episode(
    *,
    ticker: str,
    date: str,
    start_ns: int,
    slice_minutes: int = 5,
) -> EpisodeData:
    """Load real TAQ episode from `start_ns` for `slice_minutes` minutes."""
    from hft.data import load_eq_taq
    from hft.data.timeparse import add_eq_ns_of_day, filter_rth
    from hft.simulators.stylized_facts import extract_real_taq_series

    df = load_eq_taq(ticker, date)
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    df = filter_rth(df, src="Timestamp")

    end_ns = start_ns + slice_minutes * 60 * int(1e9)
    slice_df = df.filter(
        (pl.col("ns_of_day") >= start_ns) & (pl.col("ns_of_day") < end_ns)
    )
    if slice_df.is_empty():
        raise ValueError(
            f"Real slice {ticker} {date} from {start_ns/1e9/3600:.2f}h is empty"
        )
    mids, _, _ = extract_real_taq_series(
        slice_df, sample_seconds=1, rth_only=False,
    )
    if len(mids) < 60:
        raise ValueError(
            f"Real slice {ticker} {date} from {start_ns/1e9/3600:.2f}h has only "
            f"{len(mids)} 1-Hz mid samples (need ≥ 60)"
        )
    # Pad/truncate to slice_minutes × 60 (1-Hz target length).
    target_len = slice_minutes * 60
    if len(mids) < target_len:
        pad = np.full(target_len - len(mids), mids[-1])
        mids = np.concatenate([mids, pad])
    elif len(mids) > target_len:
        mids = mids[:target_len]

    hours = start_ns / 1e9 / 3600
    src = f"real_{ticker}_{date}_{int(hours):02d}:{int((hours % 1) * 60):02d}"
    return EpisodeData(
        mid_prices=mids,
        arrival_mid=float(mids[0]),
        source=src,
        metadata={"ticker": ticker, "date": date, "start_ns": start_ns},
    )
