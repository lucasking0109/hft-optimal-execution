"""Loader for abides-sim simulation output.

abides-sim writes 5 bz2-compressed pickle files per run. This module loads
each into a polars DataFrame with consistent schema, and provides helpers
to reconstruct DOM snapshots at any time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Default ABIDES log dir
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ABIDES_LOG = PROJECT_ROOT / "log" / "test_output"


# ---------------------------------------------------------------------------
# Trade events
# ---------------------------------------------------------------------------

def _coerce_event_columns(expanded: pd.DataFrame) -> pd.DataFrame:
    """abides events have mixed bool/int in is_buy_order — make them clean ints
    so polars / pyarrow conversion works."""
    if "is_buy_order" in expanded.columns:
        # bool → int, int → int, NaN → -1 (sentinel)
        expanded["is_buy_order"] = (
            expanded["is_buy_order"].fillna(-1).astype(int)
        )
    for c in ("quantity", "fill_price", "limit_price", "agent_id", "order_id"):
        if c in expanded.columns:
            expanded[c] = pd.to_numeric(expanded[c], errors="coerce")
    if "tag" in expanded.columns:
        expanded["tag"] = expanded["tag"].astype(str)
    return expanded


def load_executed_trades(log_dir: Path = DEFAULT_ABIDES_LOG) -> pl.DataFrame:
    """ORDER_EXECUTED events with parsed columns.

    Columns:
        time, agent_id, symbol, quantity, is_buy_order (1/0), order_id,
        fill_price (cents), limit_price (cents), price_dollars, side ('buy'/'sell')
    """
    df = pd.read_pickle(log_dir / "EXCHANGE_AGENT.bz2", compression="bz2")
    oe = df[df["EventType"] == "ORDER_EXECUTED"].copy()
    if oe.empty:
        return pl.DataFrame()
    expanded = pd.json_normalize(oe["Event"])
    expanded.insert(0, "time", oe.index.values)
    expanded = expanded.dropna(subset=["time"])
    expanded = _coerce_event_columns(expanded)
    expanded["price_dollars"] = expanded["fill_price"] / 100.0
    expanded["side"] = expanded["is_buy_order"].map({1: "buy", 0: "sell"}).fillna("unknown")
    return pl.from_pandas(expanded)


def load_cancels(log_dir: Path = DEFAULT_ABIDES_LOG) -> pl.DataFrame:
    """ORDER_CANCELLED events."""
    df = pd.read_pickle(log_dir / "EXCHANGE_AGENT.bz2", compression="bz2")
    cn = df[df["EventType"] == "ORDER_CANCELLED"].copy()
    if cn.empty:
        return pl.DataFrame()
    expanded = pd.json_normalize(cn["Event"])
    expanded.insert(0, "time", cn.index.values)
    expanded = expanded.dropna(subset=["time"])
    expanded = _coerce_event_columns(expanded)
    expanded["price_dollars"] = expanded["limit_price"] / 100.0
    expanded["side"] = expanded["is_buy_order"].map({1: "buy", 0: "sell"}).fillna("unknown")
    return pl.from_pandas(expanded)


def load_limit_orders(log_dir: Path = DEFAULT_ABIDES_LOG) -> pl.DataFrame:
    """LIMIT_ORDER events (placements)."""
    df = pd.read_pickle(log_dir / "EXCHANGE_AGENT.bz2", compression="bz2")
    lo = df[df["EventType"] == "LIMIT_ORDER"].copy()
    if lo.empty:
        return pl.DataFrame()
    expanded = pd.json_normalize(lo["Event"])
    expanded.insert(0, "time", lo.index.values)
    expanded = expanded.dropna(subset=["time"])
    expanded = _coerce_event_columns(expanded)
    expanded["price_dollars"] = expanded["limit_price"] / 100.0
    expanded["side"] = expanded["is_buy_order"].map({1: "buy", 0: "sell"}).fillna("unknown")
    return pl.from_pandas(expanded)


# ---------------------------------------------------------------------------
# BBO history
# ---------------------------------------------------------------------------

def load_bbo(log_dir: Path = DEFAULT_ABIDES_LOG) -> pl.DataFrame:
    """Top-of-book updates (BEST_BID and BEST_ASK)."""
    df = pd.read_pickle(log_dir / "EXCHANGE_AGENT.bz2", compression="bz2")
    bbo = df[df["EventType"].isin(["BEST_BID", "BEST_ASK"])].copy()
    parsed = bbo["Event"].str.split(",", expand=True)
    parsed.columns = ["symbol", "price_cents", "qty"]
    parsed["price_cents"] = parsed["price_cents"].astype(int)
    parsed["qty"] = parsed["qty"].astype(int)
    parsed["side"] = bbo["EventType"].map({"BEST_BID": "BID", "BEST_ASK": "ASK"})
    parsed["time"] = bbo.index
    parsed["price_dollars"] = parsed["price_cents"] / 100.0
    return pl.from_pandas(
        parsed[["time", "side", "price_cents", "price_dollars", "qty"]]
    ).sort("time")


# ---------------------------------------------------------------------------
# Order book snapshots
# ---------------------------------------------------------------------------

def load_orderbook(log_dir: Path = DEFAULT_ABIDES_LOG) -> pd.DataFrame:
    """Full L2 snapshots (still pandas — wide format with price columns)."""
    return pd.read_pickle(log_dir / "ORDERBOOK_ABM_FULL.bz2", compression="bz2")


def orderbook_snapshot_at(
    orderbook: pd.DataFrame, time: pd.Timestamp | str
) -> list[dict]:
    """Return DOM at given time as list of {price_dollars, side, qty} dicts."""
    if isinstance(time, str):
        time = pd.Timestamp(time)
    idx = orderbook.index.searchsorted(time)
    if idx >= len(orderbook):
        idx = len(orderbook) - 1
    if idx < 0:
        idx = 0
    row = orderbook.iloc[idx]
    nz = row[row != 0]
    return [
        {
            "price_cents": int(p),
            "price_dollars": int(p) / 100.0,
            "side": "BID" if q < 0 else "ASK",
            "qty": abs(int(q)),
        }
        for p, q in nz.items()
    ]


# ---------------------------------------------------------------------------
# Fundamental + summary
# ---------------------------------------------------------------------------

def load_fundamental(log_dir: Path = DEFAULT_ABIDES_LOG) -> pl.DataFrame:
    """The 'true value' time series."""
    df = pd.read_pickle(log_dir / "fundamental_ABM.bz2", compression="bz2")
    df = df.reset_index()
    df["price_dollars"] = df["FundamentalValue"] / 100.0
    return pl.from_pandas(df.rename(columns={"FundamentalTime": "time"}))


def load_agent_summary(log_dir: Path = DEFAULT_ABIDES_LOG) -> pl.DataFrame:
    """Per-agent lifecycle events."""
    df = pd.read_pickle(log_dir / "summary_log.bz2", compression="bz2")
    return pl.from_pandas(df)


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------

def load_abides_run(log_dir: Path = DEFAULT_ABIDES_LOG) -> dict:
    """Load all artefacts from one abides run.

    Returns dict with keys: trades, cancels, orders, bbo, orderbook,
    fundamental, agents.
    """
    log_dir = Path(log_dir)
    return {
        "trades": load_executed_trades(log_dir),
        "cancels": load_cancels(log_dir),
        "orders": load_limit_orders(log_dir),
        "bbo": load_bbo(log_dir),
        "orderbook": load_orderbook(log_dir),
        "fundamental": load_fundamental(log_dir),
        "agents": load_agent_summary(log_dir),
    }
