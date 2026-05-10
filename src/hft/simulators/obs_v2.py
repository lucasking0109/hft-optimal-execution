"""13-dim microstructure observation builder for Phase D ExecutionEnv.

Combines mid-derived features (carried over from v1) with NBBO depth,
microprice, per-venue concentration, trade-flow imbalance, normalised
trade rate, intraday volume-profile percentile, and a ticker index.

All features are point-in-time. The observation builder runs in O(1) per
step given pre-built lookups (NBBOLookup, VenueBBOLookup,
VolProfileBaseline) and a sorted trades DataFrame slice.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hft.analysis.nbbo_lookup import NBBOLookup, VenueBBOLookup
from hft.simulators.vol_profile_baseline import VolProfileBaseline


TICKER_IDX: dict[str, int] = {"AAPL": 0, "AMZN": 1, "AMD": 2, "TSLA": 3, "NVDA": 4}

# v3 (Phase E): drop ticker_idx, add log_adv_norm. Same 13-dim shape.
OBS_V3_NAMES = [
    "inv_pct_left",          # 0
    "time_pct_left",         # 1
    "recent_vol_bps_1m",     # 2
    "mid_drift_5m_bps",      # 3
    "schedule_lag",          # 4
    "spread_bps",            # 5
    "nbbo_size_imbalance",   # 6
    "microprice_drift_5m_bps",  # 7
    "venue_concentration",   # 8
    "aggressor_imb_5m",      # 9
    "trade_rate_5m_norm",    # 10
    "vol_profile_pct",       # 11
    "log_adv_norm",          # 12  (replaces ticker_idx)
]

OBS_V3_LOW = np.array(
    [0.0, 0.0, 0.0, -100.0, -1.0, 0.0, -1.0, -100.0, 0.0, -1.0, -10.0, 0.0, -2.0],
    dtype=np.float32,
)
OBS_V3_HIGH = np.array(
    [1.0, 1.0, 100.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 1.0, 10.0, 1.0, 2.0],
    dtype=np.float32,
)

OBS_V2_NAMES = [
    "inv_pct_left",          # 0
    "time_pct_left",         # 1
    "recent_vol_bps_1m",     # 2
    "mid_drift_5m_bps",      # 3
    "schedule_lag",          # 4
    "spread_bps",            # 5
    "nbbo_size_imbalance",   # 6
    "microprice_drift_5m_bps",  # 7
    "venue_concentration",   # 8
    "aggressor_imb_5m",      # 9
    "trade_rate_5m_norm",    # 10
    "vol_profile_pct",       # 11
    "ticker_idx",            # 12
]

OBS_V2_LOW = np.array(
    [0.0, 0.0, 0.0, -100.0, -1.0, 0.0, -1.0, -100.0, 0.0, -1.0, -10.0, 0.0, 0.0],
    dtype=np.float32,
)
OBS_V2_HIGH = np.array(
    [1.0, 1.0, 100.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 1.0, 10.0, 1.0, 4.0],
    dtype=np.float32,
)


WIN_5M_NS = 5 * 60 * 1_000_000_000
RTH_OPEN_NS = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)


def build_obs_v2(
    *,
    inventory_left: float,
    total_qty: int,
    step_idx: int,
    n_steps: int,
    cumulative_filled: float,
    anchor_ns: int,
    episode_arrival_mid: float,
    episode_mids: np.ndarray,
    step_seconds: int,
    ticker_idx: int,
    nbbo_lookup: NBBOLookup,
    venue_bbo_lookup: VenueBBOLookup,
    trades_df: pl.DataFrame,
    vol_profile_baseline: VolProfileBaseline,
) -> np.ndarray:
    """Compute 13-dim observation vector clipped into the box bounds."""
    # ---- 0..4: carried over from v1 (mid-only) ----
    current_mid_idx = min(step_idx * step_seconds, len(episode_mids) - 1)

    inv_pct_left = float(np.clip(inventory_left / total_qty, 0.0, 1.0))
    time_pct_left = float(np.clip(1.0 - step_idx / n_steps, 0.0, 1.0))

    recent_vol_bps_1m = 0.0
    if current_mid_idx >= 61:
        win = episode_mids[current_mid_idx - 60: current_mid_idx + 1]
        win = win[win > 0]
        if len(win) >= 2:
            log_rets = np.diff(np.log(win))
            recent_vol_bps_1m = float(np.std(log_rets) * 1e4)
    recent_vol_bps_1m = float(np.clip(recent_vol_bps_1m, 0.0, 100.0))

    mid_drift_5m_bps = 0.0
    if current_mid_idx >= 300:
        m_then = episode_mids[current_mid_idx - 300]
        m_now = episode_mids[current_mid_idx]
        if m_then > 0:
            mid_drift_5m_bps = (m_now - m_then) / m_then * 1e4
    mid_drift_5m_bps = float(np.clip(mid_drift_5m_bps, -100.0, 100.0))

    schedule_lag = float(np.clip(
        step_idx / n_steps - cumulative_filled / total_qty, -1.0, 1.0
    ))

    # ---- 5..7: NBBO + microprice ----
    snap = nbbo_lookup.at(anchor_ns)
    spread_bps = 0.0
    nbbo_size_imbalance = 0.0
    microprice_now = None
    if snap.spread_bps is not None and snap.spread_bps >= 0:
        spread_bps = float(np.clip(snap.spread_bps, 0.0, 100.0))
    if snap.nbb_size and snap.nbo_size:
        denom = snap.nbb_size + snap.nbo_size
        if denom > 0:
            nbbo_size_imbalance = float(
                np.clip((snap.nbb_size - snap.nbo_size) / denom, -1.0, 1.0)
            )
        if snap.nbb and snap.nbo and snap.nbb > 0 and snap.nbo > 0:
            microprice_now = (
                snap.nbb_size * snap.nbo + snap.nbo_size * snap.nbb
            ) / denom

    microprice_drift_5m_bps = 0.0
    if microprice_now is not None and anchor_ns - WIN_5M_NS >= 0:
        snap_5m = nbbo_lookup.at(anchor_ns - WIN_5M_NS)
        if (
            snap_5m.nbb_size and snap_5m.nbo_size
            and snap_5m.nbb and snap_5m.nbo
            and (snap_5m.nbb_size + snap_5m.nbo_size) > 0
        ):
            mp_then = (
                snap_5m.nbb_size * snap_5m.nbo + snap_5m.nbo_size * snap_5m.nbb
            ) / (snap_5m.nbb_size + snap_5m.nbo_size)
            if mp_then > 0:
                microprice_drift_5m_bps = float(
                    np.clip((microprice_now - mp_then) / mp_then * 1e4, -100.0, 100.0)
                )

    # ---- 8: venue concentration (# of venues currently at NBBO / total) ----
    venue_bbos = venue_bbo_lookup.at(anchor_ns)
    n_venues = max(1, len(venue_bbos))
    n_at_nbbo = 0
    if snap.nbb is not None and snap.nbo is not None:
        for vbo in venue_bbos.values():
            at_nbb = (
                vbo.bid is not None and vbo.bid > 0
                and abs(vbo.bid - snap.nbb) <= max(1e-4, snap.nbb * 1e-6)
            )
            at_nbo = (
                vbo.ask is not None and vbo.ask > 0
                and abs(vbo.ask - snap.nbo) <= max(1e-4, snap.nbo * 1e-6)
            )
            if at_nbb or at_nbo:
                n_at_nbbo += 1
    venue_concentration = float(np.clip(n_at_nbbo / n_venues, 0.0, 1.0))

    # ---- 9..10: trade flow over last 5 min ----
    aggressor_imb_5m = 0.0
    trade_rate_5m_norm = 0.0
    win_start_ns = max(0, anchor_ns - WIN_5M_NS)
    trade_ns = trades_df["ns_of_day"].to_numpy()
    if len(trade_ns):
        i_lo = int(np.searchsorted(trade_ns, win_start_ns, side="left"))
        i_hi = int(np.searchsorted(trade_ns, anchor_ns, side="left"))
        n_trades_in_win = max(0, i_hi - i_lo)
        if n_trades_in_win > 0:
            window_qty = trades_df["Quantity"].to_numpy()[i_lo:i_hi].astype(np.float64)
            window_px = trades_df["Price"].to_numpy()[i_lo:i_hi].astype(np.float64)
            window_ns = trade_ns[i_lo:i_hi]
            # Lee-Ready aggressor sign vs mid 1ms before each trade.
            signs = np.zeros(n_trades_in_win, dtype=np.float64)
            for k in range(n_trades_in_win):
                m = nbbo_lookup.mid_at(int(window_ns[k]) - 1_000_000)
                if m is None or m <= 0:
                    continue
                if window_px[k] > m * (1 + 1e-9):
                    signs[k] = +1
                elif window_px[k] < m * (1 - 1e-9):
                    signs[k] = -1
            total_vol = float(window_qty.sum())
            if total_vol > 0:
                aggressor_imb_5m = float(np.clip(
                    float((window_qty * signs).sum()) / total_vol, -1.0, 1.0
                ))
        # Normalise rate against the day-so-far average (trades up to anchor /
        # seconds since open). No look-ahead: only past data used.
        n_past = max(1, i_hi)
        seconds_since_open = max(60, (anchor_ns - RTH_OPEN_NS) // 1_000_000_000)
        avg_rate_per_sec = n_past / seconds_since_open
        win_rate_per_sec = n_trades_in_win / max(1, WIN_5M_NS // 1_000_000_000)
        if avg_rate_per_sec > 0:
            trade_rate_5m_norm = float(np.clip(
                (win_rate_per_sec - avg_rate_per_sec) / avg_rate_per_sec,
                -10.0, 10.0,
            ))

    # ---- 11: vol profile percentile ----
    vol_profile_pct = float(np.clip(
        vol_profile_baseline.cumulative_pct_at(anchor_ns), 0.0, 1.0,
    ))

    # ---- 12: ticker idx ----
    ticker_idx_clipped = float(np.clip(ticker_idx, 0, 4))

    return np.array(
        [
            inv_pct_left, time_pct_left, recent_vol_bps_1m, mid_drift_5m_bps, schedule_lag,
            spread_bps, nbbo_size_imbalance, microprice_drift_5m_bps,
            venue_concentration, aggressor_imb_5m, trade_rate_5m_norm,
            vol_profile_pct, ticker_idx_clipped,
        ],
        dtype=np.float32,
    )


def build_obs_v3(
    *,
    inventory_left: float,
    total_qty: int,
    step_idx: int,
    n_steps: int,
    cumulative_filled: float,
    anchor_ns: int,
    episode_arrival_mid: float,
    episode_mids: np.ndarray,
    step_seconds: int,
    log_adv_norm: float,
    nbbo_lookup: NBBOLookup,
    venue_bbo_lookup: VenueBBOLookup,
    trades_df: pl.DataFrame,
    vol_profile_baseline: VolProfileBaseline,
) -> np.ndarray:
    """Same 12 microstructure features as v2 but with `log_adv_norm` at
    dim 12 instead of `ticker_idx`. This makes the policy ticker-agnostic
    — it can be applied to any ticker (the log_adv_norm value distinguishes
    high- vs low-liquidity names).
    """
    v2 = build_obs_v2(
        inventory_left=inventory_left, total_qty=total_qty,
        step_idx=step_idx, n_steps=n_steps,
        cumulative_filled=cumulative_filled, anchor_ns=anchor_ns,
        episode_arrival_mid=episode_arrival_mid,
        episode_mids=episode_mids, step_seconds=step_seconds,
        ticker_idx=0,   # placeholder; we replace dim 12 below
        nbbo_lookup=nbbo_lookup, venue_bbo_lookup=venue_bbo_lookup,
        trades_df=trades_df, vol_profile_baseline=vol_profile_baseline,
    )
    v3 = v2.copy()
    v3[12] = float(np.clip(log_adv_norm, -2.0, 2.0))
    return v3
