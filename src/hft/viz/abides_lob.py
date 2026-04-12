"""Trading-platform-style visualisations for abides-sim output.

These mirror what professional trading software shows:
- DOM ladder (Depth of Market)
- Trade tape (Time & Sales)
- Cumulative aggressor flow (signed volume cumsum)
- Volume profile by price level
- Cancel rate heatmap
- Order arrival rate over time
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT, PALETTE, TITLE_FONT_SIZE


# ---------------------------------------------------------------------------
# 1. DOM Ladder (like Bookmap / NinjaTrader DOM)
# ---------------------------------------------------------------------------

def plot_dom_ladder(
    snapshot: list[dict],
    *,
    n_levels: int = 15,
    title: str = "DOM Ladder",
    presentation: bool = False,
    filter_boundary_qty: int = 5000,
    show_filtered: bool = False,
) -> go.Figure:
    """Trading-platform-style DOM with **price ladder centered**.

    Design (matching commercial trading software like NinjaTrader / ATAS):
    - Every 1-cent price tick is a row in the ladder (uniform spacing)
    - BID bars extend LEFT from x=0
    - ASK bars extend RIGHT from x=0
    - **Price labels are placed at x=0 (center column)**, not on the y-axis
    - Best bid / best ask price labels highlighted with coloured background
    - All bars same height (because all rows are 1 cent apart, uniform)

    abides-sim places huge "boundary guard" orders (qty 50k-100k). These are
    not real liquidity and would squash visualisation. Filtered by default.
    """
    if not snapshot:
        fig = go.Figure()
        fig.add_annotation(text="Empty book", showarrow=False, font=dict(size=18))
        return fig

    if not show_filtered:
        snapshot = [s for s in snapshot if s["qty"] <= filter_boundary_qty]
        if not snapshot:
            fig = go.Figure()
            fig.add_annotation(
                text=f"All levels filtered (qty > {filter_boundary_qty})",
                showarrow=False, font=dict(size=14),
            )
            return fig

    # Build lookups
    bid_lookup = {s["price_cents"]: s["qty"] for s in snapshot if s["side"] == "BID"}
    ask_lookup = {s["price_cents"]: s["qty"] for s in snapshot if s["side"] == "ASK"}

    if not bid_lookup and not ask_lookup:
        fig = go.Figure()
        fig.add_annotation(text="Empty book after filter", showarrow=False, font=dict(size=18))
        return fig

    best_bid_cents = max(bid_lookup.keys()) if bid_lookup else None
    best_ask_cents = min(ask_lookup.keys()) if ask_lookup else None

    # === Build the price ladder — uniform 1-cent spacing ===
    if best_bid_cents is not None and best_ask_cents is not None:
        top = best_ask_cents + n_levels
        bottom = best_bid_cents - n_levels
    elif best_bid_cents is not None:
        top = best_bid_cents + 5
        bottom = best_bid_cents - n_levels
    else:
        top = best_ask_cents + n_levels
        bottom = best_ask_cents - 5

    # Cap total rows for readability (max 60 ticks visible)
    if top - bottom > 60:
        if best_bid_cents and best_ask_cents:
            mid = (best_bid_cents + best_ask_cents) // 2
            top = min(top, mid + 30)
            bottom = max(bottom, mid - 30)

    ladder_prices_cents = list(range(top, bottom - 1, -1))

    # Bars only for nonzero levels — uniform 0.85-cent width gives consistent thickness
    bid_y = [p for p in ladder_prices_cents if p in bid_lookup]
    bid_qty = [bid_lookup[p] for p in bid_y]
    bid_custom = [f"{p/100:.2f}" for p in bid_y]

    ask_y = [p for p in ladder_prices_cents if p in ask_lookup]
    ask_qty = [ask_lookup[p] for p in ask_y]
    ask_custom = [f"{p/100:.2f}" for p in ask_y]

    # Symmetric x range
    max_q = max(max(bid_qty, default=1), max(ask_qty, default=1))
    x_pad = max(max_q * 1.4, 80)

    fig = go.Figure()

    # BID bars (extend leftward — negative x)
    if bid_y:
        fig.add_trace(go.Bar(
            y=bid_y,
            x=[-q for q in bid_qty],
            orientation="h",
            name="BID",
            marker_color="rgba(31, 119, 180, 0.85)",
            marker_line_color="rgba(31, 119, 180, 1)",
            marker_line_width=1,
            text=[str(q) for q in bid_qty],
            textposition="outside",
            textfont=dict(color="rgba(150, 200, 255, 1)", size=11),
            cliponaxis=False,
            width=0.85,                 # 0.85 of 1 cent → slight visual gap
            hovertemplate="<b>BID</b> @ $%{customdata}<br>Qty: %{text}<extra></extra>",
            customdata=bid_custom,
        ))

    # ASK bars (extend rightward — positive x)
    if ask_y:
        fig.add_trace(go.Bar(
            y=ask_y,
            x=ask_qty,
            orientation="h",
            name="ASK",
            marker_color="rgba(214, 39, 40, 0.85)",
            marker_line_color="rgba(214, 39, 40, 1)",
            marker_line_width=1,
            text=[str(q) for q in ask_qty],
            textposition="outside",
            textfont=dict(color="rgba(255, 180, 180, 1)", size=11),
            cliponaxis=False,
            width=0.85,
            hovertemplate="<b>ASK</b> @ $%{customdata}<br>Qty: %{x}<extra></extra>",
            customdata=ask_custom,
        ))

    # === Price column annotations (centered at x=0) ===
    for p_cents in ladder_prices_cents:
        is_best_bid = p_cents == best_bid_cents
        is_best_ask = p_cents == best_ask_cents
        if is_best_bid:
            bg = "rgba(30, 80, 130, 1)"
            color = "white"
            label = f"${p_cents/100:.2f}"
        elif is_best_ask:
            bg = "rgba(130, 35, 40, 1)"
            color = "white"
            label = f"${p_cents/100:.2f}"
        elif p_cents in bid_lookup:
            bg = "rgba(35, 50, 70, 0.95)"
            color = "rgba(180, 200, 220, 1)"
            label = f"{p_cents/100:.2f}"
        elif p_cents in ask_lookup:
            bg = "rgba(70, 35, 40, 0.95)"
            color = "rgba(220, 180, 180, 1)"
            label = f"{p_cents/100:.2f}"
        else:
            bg = "rgba(35, 35, 35, 0.85)"
            color = "rgba(140, 140, 140, 1)"
            label = f"{p_cents/100:.2f}"

        fig.add_annotation(
            x=0, y=p_cents,
            text=label,
            showarrow=False,
            font=dict(size=11, color=color, family="Menlo, monospace"),
            bgcolor=bg,
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            borderpad=4,
            xanchor="center", yanchor="middle",
            width=55,   # narrower price column (was 70)
            height=18,
        )

    # Best bid / ask horizontal subtle lines
    if best_bid_cents is not None:
        fig.add_hline(
            y=best_bid_cents,
            line_color="rgba(31, 119, 180, 0.4)",
            line_width=1, line_dash="dot",
        )
    if best_ask_cents is not None:
        fig.add_hline(
            y=best_ask_cents,
            line_color="rgba(214, 39, 40, 0.4)",
            line_width=1, line_dash="dot",
        )

    title_size = 22 if presentation else TITLE_FONT_SIZE
    spread_label = ""
    if best_bid_cents is not None and best_ask_cents is not None:
        bb = best_bid_cents / 100
        ba = best_ask_cents / 100
        spread = ba - bb
        spread_bps = (spread / ((bb + ba) / 2)) * 10000 if (bb + ba) > 0 else 0
        spread_label = f"  ·  Spread ${spread:.2f} ({spread_bps:.1f} bps)"

    fig.update_layout(
        title=dict(
            text=title + spread_label + "  ·  🟦 BID = 買單   🟥 ASK = 賣單",
            font=dict(size=title_size),
            x=0.5,
        ),
        barmode="overlay",
        height=max(550, 22 * len(ladder_prices_cents) + 100),
        xaxis=dict(
            title="Quantity (← BID  |  ASK →)",
            range=[-x_pad, x_pad],
            tickfont=dict(size=11),
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            range=[bottom - 0.5, top + 0.5],
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="#0e0e0e",
        paper_bgcolor="#0e0e0e",
        showlegend=False,           # ← 拿掉重複的 legend，color hint 寫在 title
        margin=dict(l=10, r=10, t=70, b=50),
        bargap=0.0,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Trade Tape (Time & Sales)
# ---------------------------------------------------------------------------

def plot_trade_tape(
    trades: pl.DataFrame,
    *,
    n_recent: int = 30,
    presentation: bool = False,
) -> go.Figure:
    """Scrolling list of recent trades with aggressor side colored.

    Resembles the 'Time & Sales' window in trading software.
    """
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades", showarrow=False, font=dict(size=18))
        return fig

    recent = trades.tail(n_recent)
    # Reverse for newest-on-top
    recent = recent.reverse()

    times = [t.strftime("%H:%M:%S.%f")[:-3] for t in recent["time"].to_list()]
    qtys = recent["quantity"].to_list()
    prices = recent["price_dollars"].to_list()
    sides = recent["side"].to_list()
    colors = ["#2ca02c" if s == "buy" else "#d62728" for s in sides]
    side_labels = ["▲ BUY" if s == "buy" else "▼ SELL" for s in sides]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[3, 1, 2, 2],
                header=dict(
                    values=["<b>Time</b>", "<b>Qty</b>", "<b>Price</b>", "<b>Side</b>"],
                    fill_color="#222",
                    font=dict(color="white", size=13),
                    align="center",
                    height=32,
                ),
                cells=dict(
                    values=[times, qtys, [f"${p:.2f}" for p in prices], side_labels],
                    fill_color=[
                        "#1a1a1a",  # time
                        "#1a1a1a",  # qty
                        "#1a1a1a",  # price
                        [c for c in colors],  # side colored
                    ],
                    font=dict(
                        color=[
                            ["white"] * len(times),
                            ["white"] * len(qtys),
                            ["#aaa"] * len(prices),
                            ["white"] * len(sides),
                        ],
                        size=12,
                        family="Menlo, monospace",
                    ),
                    align=["left", "right", "right", "center"],
                    height=24,
                ),
            )
        ]
    )
    title_size = 18 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"Time &amp; Sales (last {n_recent})", font=dict(size=title_size)),
        height=min(700, 55 + 26 * n_recent),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Cumulative Aggressor Flow + Price overlay
# ---------------------------------------------------------------------------

def plot_aggressor_with_price(
    trades: pl.DataFrame,
    fundamental: pl.DataFrame,
    *,
    sample_every: int = 50,
    presentation: bool = False,
) -> go.Figure:
    """Two-panel: top = price (trades scattered + fundamental line),
    bottom = cumulative buy − sell aggressor volume.
    """
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades", showarrow=False, font=dict(size=18))
        return fig

    # Sign volume
    signed = trades.with_columns(
        pl.when(pl.col("is_buy_order") == 1)
        .then(pl.col("quantity"))
        .otherwise(-pl.col("quantity"))
        .alias("signed_qty")
    ).sort("time")
    signed = signed.with_columns(pl.col("signed_qty").cum_sum().alias("cum_aggressor"))

    sampled = signed.gather_every(max(1, sample_every))
    trades_sample = trades.gather_every(max(1, sample_every))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        subplot_titles=("成交價 + 真實價值", "累積 Aggressor 流（綠＞0 → 主動買壓累積）"),
        shared_xaxes=True,
    )

    # Top: trades by side
    buys = trades_sample.filter(pl.col("side") == "buy")
    sells = trades_sample.filter(pl.col("side") == "sell")
    fig.add_trace(go.Scatter(
        x=buys["time"].to_list(),
        y=buys["price_dollars"].to_list(),
        mode="markers",
        marker=dict(size=4, color="#2ca02c", opacity=0.65),
        name="BUY trades",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells["time"].to_list(),
        y=sells["price_dollars"].to_list(),
        mode="markers",
        marker=dict(size=4, color="#d62728", opacity=0.65),
        name="SELL trades",
    ), row=1, col=1)

    # Fundamental overlay
    if not fundamental.is_empty():
        fund_sample = fundamental.gather_every(max(1, len(fundamental) // 200))
        fig.add_trace(go.Scatter(
            x=fund_sample["time"].to_list(),
            y=fund_sample["price_dollars"].to_list(),
            mode="lines",
            line=dict(color="#888", width=1, dash="dash"),
            name="Fundamental ('true value')",
        ), row=1, col=1)

    # Bottom: cumulative aggressor
    fig.add_trace(go.Scatter(
        x=sampled["time"].to_list(),
        y=sampled["cum_aggressor"].to_list(),
        mode="lines",
        line=dict(color="#2ca02c", width=2),
        fill="tozeroy",
        fillcolor="rgba(44,160,44,0.18)",
        name="Cum aggressor",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666", row=2, col=1)

    title_size = 22 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text="價格 + 訂單流時序", font=dict(size=title_size)),
        height=600,
        margin=dict(l=70, r=30, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cum signed qty", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# 4. Volume Profile (price level → traded qty)
# ---------------------------------------------------------------------------

def plot_volume_profile_by_price(
    trades: pl.DataFrame,
    *,
    presentation: bool = False,
) -> go.Figure:
    """Horizontal bars: each price level → total volume traded there,
    split by aggressor side (buy=green, sell=red). 'Bookmap' style."""
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades", showarrow=False, font=dict(size=18))
        return fig

    grp = (
        trades.group_by(["price_dollars", "side"])
        .agg(pl.col("quantity").sum().alias("vol"))
    )
    buys = grp.filter(pl.col("side") == "buy").sort("price_dollars")
    sells = grp.filter(pl.col("side") == "sell").sort("price_dollars")

    fig = go.Figure()
    if not buys.is_empty():
        fig.add_trace(go.Bar(
            y=buys["price_dollars"].to_list(),
            x=buys["vol"].to_list(),
            orientation="h",
            name="BUY-aggressor vol",
            marker_color="#2ca02c",
        ))
    if not sells.is_empty():
        fig.add_trace(go.Bar(
            y=sells["price_dollars"].to_list(),
            x=[-v for v in sells["vol"].to_list()],
            orientation="h",
            name="SELL-aggressor vol",
            marker_color="#d62728",
        ))

    fig.add_vline(x=0, line_color="#888")

    title_size = 18 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text="Volume Profile (by price level, signed by aggressor)",
                   font=dict(size=title_size)),
        height=600,
        barmode="overlay",
        xaxis=dict(title="Volume (BUY → ← → SELL)", zeroline=True),
        yaxis=dict(title="Price ($)"),
        margin=dict(l=80, r=20, t=70, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Cancel Heatmap (cancels per minute × price level)
# ---------------------------------------------------------------------------

def plot_cancel_heatmap(
    cancels: pl.DataFrame,
    *,
    bin_seconds: int = 30,
    presentation: bool = False,
) -> go.Figure:
    """Heatmap: time bin × price level → number of cancels.
    Reveals where in the book + when traders are pulling orders."""
    if cancels.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No cancels", showarrow=False, font=dict(size=18))
        return fig

    df = cancels.with_columns([
        pl.col("time").dt.truncate(f"{bin_seconds}s").alias("time_bin"),
        pl.col("price_dollars").round(2).alias("price_round"),
    ])
    grid = (
        df.group_by(["time_bin", "price_round"])
        .agg(pl.len().alias("n_cancels"))
    )

    # Pivot
    times = sorted(grid["time_bin"].unique().to_list())
    prices = sorted(grid["price_round"].unique().to_list(), reverse=True)
    z = []
    for p in prices:
        row = []
        for t in times:
            cell = grid.filter(
                (pl.col("time_bin") == t) & (pl.col("price_round") == p)
            )
            row.append(int(cell["n_cancels"][0]) if not cell.is_empty() else 0)
        z.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[t.strftime("%H:%M:%S") for t in times],
            y=[f"${p:.2f}" for p in prices],
            colorscale="OrRd",
            colorbar=dict(title="Cancels"),
            hovertemplate="%{y}<br>%{x}<br>%{z} cancels<extra></extra>",
        )
    )
    title_size = 18 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"Cancels heatmap (per {bin_seconds}s × price)",
                   font=dict(size=title_size)),
        height=550,
        margin=dict(l=70, r=30, t=70, b=60),
        xaxis=dict(title="Time"),
        yaxis=dict(title="Price"),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Order arrival rate (orders/sec over time, by side)
# ---------------------------------------------------------------------------

def plot_order_arrival_rate(
    orders: pl.DataFrame,
    cancels: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    bin_seconds: int = 5,
    presentation: bool = False,
) -> go.Figure:
    """Stacked area: orders placed / orders cancelled / orders executed per second."""
    def bin_count(df: pl.DataFrame, label: str) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({"time_bin": [], "n": [], "kind": []})
        return (
            df.with_columns(pl.col("time").dt.truncate(f"{bin_seconds}s").alias("time_bin"))
            .group_by("time_bin").agg(pl.len().alias("n"))
            .with_columns(pl.lit(label).alias("kind"))
            .sort("time_bin")
        )

    placed = bin_count(orders, "PLACED")
    cancelled = bin_count(cancels, "CANCELLED")
    executed = bin_count(trades, "EXECUTED")

    fig = go.Figure()
    for sub, color, name in [
        (placed, "#1f77b4", "PLACED"),
        (cancelled, "#ff7f0e", "CANCELLED"),
        (executed, "#2ca02c", "EXECUTED"),
    ]:
        if sub.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=sub["time_bin"].to_list(),
            y=sub["n"].to_list(),
            mode="lines",
            stackgroup="one",
            name=name,
            line=dict(color=color, width=0),
            fillcolor=color,
        ))

    title_size = 18 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"Order Event Rate (per {bin_seconds}s)",
                   font=dict(size=title_size)),
        height=400,
        xaxis=dict(title="Time"),
        yaxis=dict(title="Events"),
        margin=dict(l=70, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    return fig
