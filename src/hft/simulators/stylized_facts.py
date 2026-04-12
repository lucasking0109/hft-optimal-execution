"""Stylized facts of a financial time series — used to compare synthetic
abides output against real AAPL tick.

The five stylized facts we compute (per Cont 2001 + standard literature):

1. **Excess kurtosis of log returns** — fat-tail strength
   (Gaussian = 0; AAPL high-freq typically 5-20)

2. **Hill tail index (left tail)** — power-law decay alpha
   smaller → fatter tail (real AAPL roughly 3-5)

3. **Volatility autocorrelation** — autocorr of |returns| at lags 1-100
   long-memory persistence is a hallmark of real markets

4. **Spread distribution** (bps) — empirical CDF of NBBO spread
   used directly in KS test against synthetic

5. **Trade size distribution** — empirical CDF of trade quantities

Comparison via `facts_distance()` returns:
- per-metric KS distance + Wasserstein distance + ratio (for kurtosis/Hill)

NO Silent Fallback: if input series is empty / insufficient → raise
explicitly with which metric failed and how many samples we had.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as st


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class StylizedFacts:
    """Bundle of five stylized facts."""

    excess_kurtosis: float
    hill_tail_index: float
    vol_autocorr: list[float]      # length 100 (lags 1..100)
    spread_bps: np.ndarray         # raw values for KS test
    trade_sizes: np.ndarray        # raw values for KS test
    n_returns: int
    n_spread_obs: int
    n_trades: int
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """A flat dict suitable for JSON / DataFrame storage."""
        return {
            "excess_kurtosis": float(self.excess_kurtosis),
            "hill_tail_index": float(self.hill_tail_index),
            "vol_autocorr_lag_1": float(self.vol_autocorr[0]) if self.vol_autocorr else None,
            "vol_autocorr_lag_10": float(self.vol_autocorr[9]) if len(self.vol_autocorr) > 9 else None,
            "vol_autocorr_lag_50": float(self.vol_autocorr[49]) if len(self.vol_autocorr) > 49 else None,
            "spread_bps_median": float(np.median(self.spread_bps)) if len(self.spread_bps) else None,
            "spread_bps_p95": float(np.percentile(self.spread_bps, 95)) if len(self.spread_bps) else None,
            "trade_size_median": float(np.median(self.trade_sizes)) if len(self.trade_sizes) else None,
            "trade_size_p95": float(np.percentile(self.trade_sizes, 95)) if len(self.trade_sizes) else None,
            "n_returns": self.n_returns,
            "n_spread_obs": self.n_spread_obs,
            "n_trades": self.n_trades,
        }


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def _hill_estimator(absolute_returns: np.ndarray, top_pct: float = 0.05) -> float:
    """Hill estimator for the tail index alpha of |returns|.

    Fits on top `top_pct` of values; alpha = 1 / mean(log(x_i / x_threshold)).
    Smaller alpha → fatter tail. AAPL typical 3-5.
    """
    if len(absolute_returns) < 50:
        raise ValueError(f"Hill estimator needs ≥ 50 samples, got {len(absolute_returns)}")
    x = np.sort(absolute_returns)
    x = x[x > 0]
    k = max(2, int(len(x) * top_pct))
    threshold = x[-k]
    tail = x[-k:]
    log_ratios = np.log(tail / threshold)
    log_ratios = log_ratios[np.isfinite(log_ratios)]
    log_ratios = log_ratios[log_ratios > 0]
    if len(log_ratios) < 2:
        raise ValueError("Hill estimator: too few positive log-ratios after filtering")
    return float(1.0 / np.mean(log_ratios))


def _vol_autocorr(returns: np.ndarray, max_lag: int = 100) -> list[float]:
    """Autocorrelation of |returns| at lags 1..max_lag."""
    if len(returns) < max_lag + 10:
        raise ValueError(
            f"Vol autocorr needs ≥ {max_lag + 10} returns, got {len(returns)}"
        )
    abs_r = np.abs(returns)
    abs_r = abs_r - abs_r.mean()
    var = (abs_r ** 2).mean()
    if var <= 0:
        raise ValueError("Returns have zero variance — cannot autocorrelate")
    out = []
    n = len(abs_r)
    for lag in range(1, max_lag + 1):
        cov = (abs_r[:-lag] * abs_r[lag:]).mean()
        out.append(float(cov / var))
    return out


def compute_stylized_facts(
    *,
    mid_prices: np.ndarray,
    spread_bps: np.ndarray,
    trade_sizes: np.ndarray,
    return_horizon_steps: int = 1,
    metadata: dict | None = None,
) -> StylizedFacts:
    """Compute all five stylized facts from pre-extracted series.

    Args:
        mid_prices: float array of mid prices (sequential samples)
        spread_bps: float array of spread observations in basis points
        trade_sizes: int/float array of trade quantities
        return_horizon_steps: lag for return calculation (1 = consecutive)
        metadata: optional dict (e.g., source, ticker, date)

    Raises ValueError with specific reason if any metric can't be computed.
    """
    metadata = metadata or {}

    # ── log returns
    if len(mid_prices) < 100:
        raise ValueError(
            f"mid_prices has only {len(mid_prices)} samples — need ≥ 100 for stable kurtosis/Hill"
        )
    mid = np.asarray(mid_prices, dtype=float)
    mid = mid[mid > 0]
    if len(mid) < 100:
        raise ValueError(f"After filtering positives only {len(mid)} mids remain")
    log_mid = np.log(mid)
    returns = np.diff(log_mid, n=return_horizon_steps)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        raise ValueError(f"After differencing only {len(returns)} returns")

    # ── kurtosis
    kurt = float(st.kurtosis(returns, fisher=True, bias=False))

    # ── Hill tail index
    abs_returns = np.abs(returns)
    abs_returns = abs_returns[abs_returns > 0]
    hill = _hill_estimator(abs_returns)

    # ── vol autocorr lags 1..100
    autocorr = _vol_autocorr(returns)

    # ── spread / trade size distributions
    spread_arr = np.asarray(spread_bps, dtype=float)
    spread_arr = spread_arr[(spread_arr >= 0) & np.isfinite(spread_arr)]
    if len(spread_arr) < 50:
        raise ValueError(f"Need ≥ 50 spread observations, got {len(spread_arr)}")

    trade_arr = np.asarray(trade_sizes, dtype=float)
    trade_arr = trade_arr[(trade_arr > 0) & np.isfinite(trade_arr)]
    if len(trade_arr) < 50:
        raise ValueError(f"Need ≥ 50 trade size observations, got {len(trade_arr)}")

    return StylizedFacts(
        excess_kurtosis=kurt,
        hill_tail_index=hill,
        vol_autocorr=autocorr,
        spread_bps=spread_arr,
        trade_sizes=trade_arr,
        n_returns=len(returns),
        n_spread_obs=len(spread_arr),
        n_trades=len(trade_arr),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Adapters: real TAQ → series  /  abides output → series
# ---------------------------------------------------------------------------

def extract_real_taq_series(
    taq_df: pl.DataFrame,
    *,
    sample_seconds: int = 1,
    rth_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From a real eq_taq DataFrame extract (mid_prices, spread_bps, trade_sizes).

    Mid is taken from QUOTE BID NB / QUOTE ASK NB (NBBO published).
    Spread is computed at every NBBO update.
    Trade sizes from TRADE / TRADE NB.
    """
    from hft.data.timeparse import add_eq_ns_of_day, filter_rth

    df = taq_df
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    if rth_only:
        df = filter_rth(df, src="Timestamp")

    # NBBO mid sampled every `sample_seconds`
    nbb = df.filter(pl.col("EventType") == "QUOTE BID NB").select(
        "ns_of_day", pl.col("Price").alias("bid")
    )
    nbo = df.filter(pl.col("EventType") == "QUOTE ASK NB").select(
        "ns_of_day", pl.col("Price").alias("ask")
    )
    if nbb.is_empty() or nbo.is_empty():
        raise ValueError("No NBBO rows in real TAQ slice")
    merged = nbb.join(nbo, on="ns_of_day", how="full", coalesce=True).sort("ns_of_day")
    merged = merged.with_columns(
        pl.col("bid").forward_fill(),
        pl.col("ask").forward_fill(),
    ).filter(
        pl.col("bid").is_not_null() & pl.col("ask").is_not_null()
        & (pl.col("bid") > 0) & (pl.col("ask") > 0) & (pl.col("ask") > pl.col("bid"))
    )

    mids = ((merged["bid"] + merged["ask"]) / 2.0).to_numpy()
    spread = (
        (merged["ask"] - merged["bid"]) / ((merged["bid"] + merged["ask"]) / 2.0) * 10000
    ).to_numpy()

    # Sample mid every `sample_seconds`
    if sample_seconds > 0:
        ns_arr = merged["ns_of_day"].to_numpy()
        step_ns = sample_seconds * 1_000_000_000
        first_ns = ns_arr[0]
        target = first_ns
        sampled_idx: list[int] = []
        i = 0
        while i < len(ns_arr) and target <= ns_arr[-1]:
            while i < len(ns_arr) and ns_arr[i] < target:
                i += 1
            if i < len(ns_arr):
                sampled_idx.append(i)
            target += step_ns
        mids = mids[sampled_idx]

    # Trade sizes
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    trade_arr = trades["Quantity"].to_numpy()

    return mids, spread, trade_arr.astype(float)


def extract_abides_series(
    abides_data: dict,
    *,
    sample_seconds: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From abides log dict (output of `load_abides_run`) extract series."""
    bbo = abides_data["bbo"]
    if bbo.is_empty():
        raise ValueError("abides BBO is empty")

    # Reconstruct mid + spread at each BBO update.
    # bbo schema: time, side, price_cents, price_dollars, qty
    bid_rows = bbo.filter(pl.col("side") == "BID").select(
        "time", pl.col("price_cents").alias("bid_c")
    )
    ask_rows = bbo.filter(pl.col("side") == "ASK").select(
        "time", pl.col("price_cents").alias("ask_c")
    )
    if bid_rows.is_empty() or ask_rows.is_empty():
        raise ValueError("abides has no bid or ask BBO rows")

    merged = bid_rows.join(ask_rows, on="time", how="full", coalesce=True).sort("time")
    merged = merged.with_columns(
        pl.col("bid_c").forward_fill(),
        pl.col("ask_c").forward_fill(),
    ).filter(
        pl.col("bid_c").is_not_null() & pl.col("ask_c").is_not_null()
        & (pl.col("ask_c") > pl.col("bid_c"))
    )

    bids = merged["bid_c"].to_numpy()
    asks = merged["ask_c"].to_numpy()
    mids = (bids + asks) / 200.0    # cents → dollars
    spread_bps = (asks - bids) / ((bids + asks) / 2.0) * 10000

    if sample_seconds > 0:
        # Sample mid every sample_seconds
        times = merged["time"].to_numpy()
        # Convert to ns
        ns_arr = times.astype("datetime64[ns]").astype(np.int64)
        step_ns = sample_seconds * 1_000_000_000
        target = ns_arr[0]
        sampled_idx: list[int] = []
        i = 0
        while i < len(ns_arr) and target <= ns_arr[-1]:
            while i < len(ns_arr) and ns_arr[i] < target:
                i += 1
            if i < len(ns_arr):
                sampled_idx.append(i)
            target += step_ns
        mids = mids[sampled_idx]

    trades = abides_data["trades"]
    if trades.is_empty():
        raise ValueError("abides has no trades")
    trade_arr = trades["quantity"].to_numpy().astype(float)

    return mids, spread_bps, trade_arr


# ---------------------------------------------------------------------------
# Compare two stylized fact bundles
# ---------------------------------------------------------------------------

def facts_distance(real: StylizedFacts, synth: StylizedFacts) -> dict:
    """Return per-metric distance dict.

    Metrics:
      kurtosis_ratio:  synth/real (1.0 = perfect; aim 0.5-2.0)
      hill_ratio:      synth/real (aim 0.5-2.0)
      spread_ks_dist:  Kolmogorov-Smirnov distance on spread bps
      spread_wass:     Wasserstein-1 distance on spread bps
      trade_size_ks_dist: KS on trade size
      autocorr_l2:     L2 distance on lag-1..100 vol autocorr vectors
    """
    out = {}

    if real.excess_kurtosis != 0:
        out["kurtosis_ratio"] = float(synth.excess_kurtosis / real.excess_kurtosis)
    else:
        out["kurtosis_ratio"] = float("nan")
    if real.hill_tail_index != 0:
        out["hill_ratio"] = float(synth.hill_tail_index / real.hill_tail_index)
    else:
        out["hill_ratio"] = float("nan")

    ks_spread = st.ks_2samp(synth.spread_bps, real.spread_bps)
    out["spread_ks_dist"] = float(ks_spread.statistic)
    out["spread_ks_pvalue"] = float(ks_spread.pvalue)
    out["spread_wass"] = float(st.wasserstein_distance(synth.spread_bps, real.spread_bps))

    ks_trades = st.ks_2samp(synth.trade_sizes, real.trade_sizes)
    out["trade_size_ks_dist"] = float(ks_trades.statistic)
    out["trade_size_ks_pvalue"] = float(ks_trades.pvalue)

    a_real = np.asarray(real.vol_autocorr)
    a_synth = np.asarray(synth.vol_autocorr)
    L = min(len(a_real), len(a_synth))
    out["autocorr_l2"] = float(np.linalg.norm(a_real[:L] - a_synth[:L]))

    return out


def total_calibration_distance(distances: dict) -> float:
    """Weighted L2-style aggregate of the distances. Used to pick best cell.

    Lower is better. Weights chosen to make all components ~unit scale.
    """
    components = [
        abs(np.log(distances["kurtosis_ratio"])) if distances["kurtosis_ratio"] > 0 else 5.0,
        abs(np.log(distances["hill_ratio"])) if distances["hill_ratio"] > 0 else 5.0,
        distances["spread_ks_dist"] * 2.0,
        distances["trade_size_ks_dist"] * 2.0,
        distances["autocorr_l2"] * 0.5,
    ]
    components = [c if np.isfinite(c) else 5.0 for c in components]
    return float(np.sqrt(sum(c ** 2 for c in components)))
