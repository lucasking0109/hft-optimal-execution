"""AgGrid (HTML <table>) renderers for DOM ladder + Time & Sales.

Why AgGrid not Plotly:
    Plotly DOM ladder is a canvas chart — every slider change forces a full
    canvas redraw (~100-300ms flicker). AgGrid renders real HTML cells and
    does cell-level delta updates → near-zero flicker when only a few values
    change between snapshots.

This module produces two pieces:
    1. `make_dom_ladder_df(snapshot, ...)` — a pandas DataFrame, one row per
       price tick, columns suitable for AgGrid rendering.
    2. `dom_grid_options(...)` — a `GridOptionsBuilder` config with cell
       styling (best bid/ask highlight, intensity-coloured size cells).
And similar pair for Time & Sales tape.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
from st_aggrid import GridOptionsBuilder, JsCode

# ---------------------------------------------------------------------------
# DOM ladder
# ---------------------------------------------------------------------------

def make_dom_ladder_df(
    snapshot: list[dict],
    *,
    n_levels: int = 15,
    filter_boundary_qty: int = 5000,
    show_filtered: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build a DataFrame of price-ladder rows.

    Returns:
        (df, meta) where:
            df has columns: bid_size, price, ask_size, _is_best_bid, _is_best_ask
                (bid_size and ask_size empty for levels with no orders)
            meta dict has best_bid_cents, best_ask_cents, max_qty (for cell intensity)
    """
    if not snapshot:
        return pd.DataFrame(columns=["bid_size", "price", "ask_size",
                                     "_is_best_bid", "_is_best_ask"]), {}

    if not show_filtered:
        snapshot = [s for s in snapshot if s["qty"] <= filter_boundary_qty]

    bid_lookup = {s["price_cents"]: s["qty"] for s in snapshot if s["side"] == "BID"}
    ask_lookup = {s["price_cents"]: s["qty"] for s in snapshot if s["side"] == "ASK"}

    if not bid_lookup and not ask_lookup:
        return pd.DataFrame(columns=["bid_size", "price", "ask_size",
                                     "_is_best_bid", "_is_best_ask"]), {}

    best_bid = max(bid_lookup) if bid_lookup else None
    best_ask = min(ask_lookup) if ask_lookup else None

    if best_bid is not None and best_ask is not None:
        top = best_ask + n_levels
        bottom = best_bid - n_levels
    elif best_bid is not None:
        top, bottom = best_bid + 5, best_bid - n_levels
    else:
        top, bottom = best_ask + n_levels, best_ask - 5

    if top - bottom > 60 and best_bid and best_ask:
        mid = (best_bid + best_ask) // 2
        top = min(top, mid + 30)
        bottom = max(bottom, mid - 30)

    rows = []
    for p in range(top, bottom - 1, -1):
        rows.append({
            "bid_size": bid_lookup.get(p, ""),
            "price": f"{p/100:.2f}",
            "ask_size": ask_lookup.get(p, ""),
            "_is_best_bid": p == best_bid,
            "_is_best_ask": p == best_ask,
        })
    df = pd.DataFrame(rows)

    max_qty = max(
        max(bid_lookup.values(), default=1),
        max(ask_lookup.values(), default=1),
    )
    meta = {
        "best_bid_cents": best_bid,
        "best_ask_cents": best_ask,
        "max_qty": max_qty,
    }
    return df, meta


def dom_grid_options(df: pd.DataFrame, meta: dict) -> dict:
    """Construct AgGrid options dict for the DOM ladder.

    Cell styling (via JsCode):
      - bid_size: blue background, intensity ~ qty / max_qty, right-aligned
      - price:    centered, mono font; best_bid → deep blue, best_ask → deep red
      - ask_size: red background, intensity, left-aligned
    """
    max_qty = meta.get("max_qty", 1) or 1

    # Cell colors are SOLID DARK (not pale washes). Intensity scales lightness
    # within a high-contrast range so white text always readable.
    bid_cell_style = JsCode(f"""
        function(params) {{
            const v = params.value;
            const max_qty = {max_qty};
            if (v === '' || v === null || v === undefined) {{
                return {{
                    backgroundColor: '#0a0a0a',
                    color: '#222',
                }};
            }}
            // intensity 0..1, mapped to dark→bright blue range
            const intensity = Math.min(1.0, Math.max(0.05, v / max_qty));
            // RGB lightness scaling: dark base (15, 50, 95) → vivid blue (45, 145, 220)
            const r = Math.round(15 + intensity * 30);
            const g = Math.round(50 + intensity * 95);
            const b = Math.round(95 + intensity * 125);
            return {{
                backgroundColor: 'rgb(' + r + ',' + g + ',' + b + ')',
                color: 'white',
                textAlign: 'right',
                paddingRight: '10px',
                fontFamily: 'Menlo, monospace',
                fontWeight: '700',
                fontSize: '13px',
            }};
        }}
    """)

    ask_cell_style = JsCode(f"""
        function(params) {{
            const v = params.value;
            const max_qty = {max_qty};
            if (v === '' || v === null || v === undefined) {{
                return {{
                    backgroundColor: '#0a0a0a',
                    color: '#222',
                }};
            }}
            const intensity = Math.min(1.0, Math.max(0.05, v / max_qty));
            // dark base (90, 20, 25) → vivid red (220, 60, 65)
            const r = Math.round(90 + intensity * 130);
            const g = Math.round(20 + intensity * 40);
            const b = Math.round(25 + intensity * 40);
            return {{
                backgroundColor: 'rgb(' + r + ',' + g + ',' + b + ')',
                color: 'white',
                textAlign: 'left',
                paddingLeft: '10px',
                fontFamily: 'Menlo, monospace',
                fontWeight: '700',
                fontSize: '13px',
            }};
        }}
    """)

    price_cell_style = JsCode("""
        function(params) {
            const isBestBid = params.data._is_best_bid;
            const isBestAsk = params.data._is_best_ask;
            let bg = '#1a1a1a';
            let color = '#aaa';
            let weight = '500';
            let size = '12px';
            if (isBestBid) {
                bg = '#1e6abf';   // bright vivid blue
                color = 'white';
                weight = '800';
                size = '14px';
            } else if (isBestAsk) {
                bg = '#bf2530';   // bright vivid red
                color = 'white';
                weight = '800';
                size = '14px';
            }
            return {
                backgroundColor: bg,
                color: color,
                textAlign: 'center',
                fontFamily: 'Menlo, monospace',
                fontWeight: weight,
                fontSize: size,
                borderLeft: '1px solid #000',
                borderRight: '1px solid #000',
            };
        }
    """)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        editable=False, sortable=False, filter=False, resizable=False,
    )
    gb.configure_column(
        "bid_size", header_name="Bid Size",
        cellStyle=bid_cell_style, width=120,
    )
    gb.configure_column(
        "price", header_name="Price",
        cellStyle=price_cell_style, width=110,
    )
    gb.configure_column(
        "ask_size", header_name="Ask Size",
        cellStyle=ask_cell_style, width=120,
    )
    gb.configure_column("_is_best_bid", hide=True)
    gb.configure_column("_is_best_ask", hide=True)
    gb.configure_grid_options(
        domLayout="normal",
        rowHeight=24,
        headerHeight=32,
        suppressHorizontalScroll=True,
        suppressMovableColumns=True,
        suppressMenuHide=True,
        suppressContextMenu=True,
    )
    return gb.build()


# ---------------------------------------------------------------------------
# Time & Sales tape
# ---------------------------------------------------------------------------

def make_trade_tape_df(trades: pl.DataFrame, *, n_recent: int = 30) -> pd.DataFrame:
    """Latest n trades as DataFrame (newest first)."""
    if trades.is_empty():
        return pd.DataFrame(columns=["time", "qty", "price", "side"])

    recent = trades.tail(n_recent).reverse().to_pandas()
    return pd.DataFrame({
        "time": [t.strftime("%H:%M:%S.%f")[:-3] for t in recent["time"]],
        "qty": recent["quantity"].astype(int),
        "price": [f"${p:.2f}" for p in recent["price_dollars"]],
        "side": recent["side"].str.upper(),
    })


def trade_tape_grid_options(df: pd.DataFrame) -> dict:
    """AgGrid options for the trade tape."""
    side_cell_style = JsCode("""
        function(params) {
            const v = params.value;
            if (v === 'BUY') {
                return {
                    backgroundColor: '#1f7a1f',
                    color: 'white',
                    fontWeight: '700',
                    textAlign: 'center',
                };
            } else if (v === 'SELL') {
                return {
                    backgroundColor: '#a82a2f',
                    color: 'white',
                    fontWeight: '700',
                    textAlign: 'center',
                };
            }
            return {textAlign: 'center'};
        }
    """)

    side_renderer = JsCode("""
        function(params) {
            if (params.value === 'BUY') return '▲ BUY';
            if (params.value === 'SELL') return '▼ SELL';
            return params.value;
        }
    """)

    # Force dark background + bright text — don't rely on theme inheritance
    mono_style = JsCode("""
        function(params) {
            return {
                fontFamily: 'Menlo, monospace',
                color: 'white',
                backgroundColor: '#1a1a1a',
                fontWeight: '600',
            };
        }
    """)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        editable=False, sortable=False, filter=False, resizable=False,
    )
    gb.configure_column("time", header_name="Time", width=125, cellStyle=mono_style)
    gb.configure_column("qty",  header_name="Qty",  width=70,  cellStyle=mono_style)
    gb.configure_column("price", header_name="Price", width=100, cellStyle=mono_style)
    gb.configure_column(
        "side", header_name="Side", width=90,
        cellStyle=side_cell_style, cellRenderer=side_renderer,
    )
    gb.configure_grid_options(
        domLayout="normal",
        rowHeight=24,
        headerHeight=32,
        suppressHorizontalScroll=True,
        suppressMovableColumns=True,
        suppressContextMenu=True,
    )
    return gb.build()
