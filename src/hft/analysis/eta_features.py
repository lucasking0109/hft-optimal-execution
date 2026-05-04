"""Feature extraction for ML-η prediction at scheduling time.

Mirrors the per-event feature engineering in
`scripts/build_eta_ml_dataset.py` but computes a single feature vector
for one (ticker, date, anchor_ns) — used by ACMLStrategy at schedule
time before running the AC sinh formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hft.analysis.nbbo_lookup import NBBOLookup
from hft.data.timeparse import add_eq_ns_of_day, filter_rth


WINDOW_15M_NS = 15 * 60 * 1_000_000_000
RTH_OPEN_NS = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)

TICKER_IDX = {"AAPL": 0, "AMZN": 1, "AMD": 2, "TSLA": 3, "NVDA": 4}

FEATURE_NAMES = [
    "size_pct_adv",
    "seconds_from_open",
    "recent_vol_bps_15m",
    "recent_spread_bps_15m",
    "recent_aggressor_imb_15m",
    "recent_volume_15m_pct_adv",
    "ticker_idx",
]


@dataclass
class EtaFeatures:
    feature_dict: dict[str, float]

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.feature_dict[name] for name in FEATURE_NAMES],
            dtype=np.float64,
        ).reshape(1, -1)


def compute_eta_features(
    *,
    ticker: str,
    df: pl.DataFrame,
    anchor_ns: int,
    parent_qty: int,
    adv: float,
) -> EtaFeatures:
    """Compute the 7-feature vector at a specific anchor timestamp.

    Args:
      df: TAQ DataFrame for the (ticker, date)
      anchor_ns: nanoseconds-of-day where we want the feature snapshot
        (typically the parent's start_ns).
      parent_qty: parent order qty in shares (used for size_pct_adv)
      adv: average daily volume in shares
    """
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    df = filter_rth(df, src="Timestamp")
    lookup = NBBOLookup(df)

    # Trades for window aggregation
    trades = (
        df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
        .sort("ns_of_day")
    )
    trade_ns = trades["ns_of_day"].to_numpy()
    trade_qty = trades["Quantity"].to_numpy()
    trade_px = trades["Price"].to_numpy()

    # Bids/asks
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

    win_start = max(anchor_ns - WINDOW_15M_NS, RTH_OPEN_NS)
    seconds_from_open = (anchor_ns - RTH_OPEN_NS) / 1e9
    size_pct_adv = parent_qty / adv * 100

    # Vol from sampled mids
    n_samples = max(1, int((anchor_ns - win_start) / 1e9))
    sample_times = np.linspace(win_start, max(win_start, anchor_ns - 1),
                                num=min(n_samples, 600), dtype=np.int64)
    sampled_mids: list[float] = []
    for st in sample_times:
        m = lookup.mid_at(int(st))
        if m is not None and m > 0:
            sampled_mids.append(m)
    if len(sampled_mids) >= 5:
        log_rets = np.diff(np.log(np.array(sampled_mids)))
        recent_vol_bps_15m = float(np.std(log_rets) * 1e4)
    else:
        recent_vol_bps_15m = 0.0

    # Spread proxy from latest bid/ask near anchor
    i_bid = np.searchsorted(bid_ns, anchor_ns, side="right") - 1
    i_ask = np.searchsorted(ask_ns, anchor_ns, side="right") - 1
    if 0 <= i_bid < len(bid_px) and 0 <= i_ask < len(ask_px):
        b, a = bid_px[i_bid], ask_px[i_ask]
        if b > 0 and a > 0 and a > b:
            recent_spread_bps_15m = (a - b) / ((a + b) / 2) * 1e4
        else:
            recent_spread_bps_15m = 0.0
    else:
        recent_spread_bps_15m = 0.0

    # Aggressor imbalance + volume in window
    i_lo = np.searchsorted(trade_ns, win_start, side="left")
    i_hi = np.searchsorted(trade_ns, anchor_ns, side="left")
    recent_aggressor_imb_15m = 0.0
    recent_volume_15m_pct_adv = 0.0
    if i_hi > i_lo:
        window_qtys = trade_qty[i_lo:i_hi]
        window_pxs = trade_px[i_lo:i_hi]
        window_ns = trade_ns[i_lo:i_hi]
        signs = []
        for k in range(len(window_qtys)):
            m = lookup.mid_at(int(window_ns[k] - 1_000_000))
            if m is None or m <= 0:
                signs.append(0)
                continue
            if window_pxs[k] > m * (1 + 1e-9):
                signs.append(+1)
            elif window_pxs[k] < m * (1 - 1e-9):
                signs.append(-1)
            else:
                signs.append(0)
        signs = np.array(signs)
        total_vol = float(window_qtys.sum())
        if total_vol > 0:
            signed_vol = float((window_qtys * signs).sum())
            recent_aggressor_imb_15m = signed_vol / total_vol
            recent_volume_15m_pct_adv = total_vol / adv * 100

    return EtaFeatures(feature_dict={
        "size_pct_adv": float(size_pct_adv),
        "seconds_from_open": float(seconds_from_open),
        "recent_vol_bps_15m": float(recent_vol_bps_15m),
        "recent_spread_bps_15m": float(recent_spread_bps_15m),
        "recent_aggressor_imb_15m": float(recent_aggressor_imb_15m),
        "recent_volume_15m_pct_adv": float(recent_volume_15m_pct_adv),
        "ticker_idx": float(TICKER_IDX.get(ticker, 0)),
    })
