"""Cached per-ticker ADV (average daily volume in shares).

Computed from the 5 days of cached TAQ tick data. Persists to disk so we
only pay the parquet scan once per ticker. Used by Phase E v3 observation
feature `log_adv_norm`.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "adv_cache.json"
TAQ_ROOT = PROJECT_ROOT / "data" / "processed" / "eq_taq"


def _compute_adv_from_taq(ticker: str, exclude_dates: list[str] | None = None) -> float:
    """Compute ADV by summing TRADE NB volume across all cached days (minus
    excluded ones). Raises if no eligible days remain.
    """
    excluded = set(exclude_dates or [])
    eligible_dates = sorted(
        d.name for d in TAQ_ROOT.iterdir()
        if d.is_dir() and d.name not in excluded
        and (d / f"{ticker}.parquet").exists()
    )
    if not eligible_dates:
        raise RuntimeError(
            f"ADV: no eligible days for {ticker} (excluded={sorted(excluded)})"
        )
    totals: list[int] = []
    for date in eligible_dates:
        try:
            df = pl.read_parquet(TAQ_ROOT / date / f"{ticker}.parquet")
        except Exception:
            continue
        trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
        totals.append(int(trades["Quantity"].sum()))
    if not totals:
        raise RuntimeError(f"ADV: all parquet failed to load for {ticker}")
    return sum(totals) / len(totals)


def _load_cache() -> dict[str, float]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, float]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_adv(ticker: str, *, exclude_dates: list[str] | None = None) -> float:
    """Return ADV for `ticker` (cached). `exclude_dates` is part of the key,
    so OOS-day exclusion produces a separate cache slot.
    """
    cache = _load_cache()
    key = f"{ticker}|{','.join(sorted(exclude_dates or []))}"
    if key in cache:
        return float(cache[key])
    adv = _compute_adv_from_taq(ticker, exclude_dates=exclude_dates)
    cache[key] = adv
    _save_cache(cache)
    return adv


def warm_cache(tickers: list[str], *, exclude_dates: list[str] | None = None) -> None:
    """Pre-compute ADV for a batch of tickers (useful before training)."""
    for t in tickers:
        try:
            get_adv(t, exclude_dates=exclude_dates)
        except Exception as e:
            print(f"  ⚠️  ADV failed for {t}: {e}")
