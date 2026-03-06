"""Module 1: Price + Bid/Ask Band time series.

The previous version used Timestamp strings as categorical x-axis, which
caused traces with different timestamp sets to be concatenated horizontally
(producing the "two halves" bug Lucas pointed out). We now convert
Timestamp → seconds-of-day numeric and use HH:MM hour-mark tick labels.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from hft.data.timeparse import (
    add_eq_ns_of_day,
    add_eq_seconds_of_day,
    filter_rth,
    hour_tick_labels,
)
from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT, PALETTE, TITLE_FONT_SIZE

RTH_START_SEC = 9 * 3600 + 30 * 60
RTH_END_SEC = 16 * 3600
FULL_START_SEC = 4 * 3600
FULL_END_SEC = 20 * 3600


def _format_date_label(date: str) -> str:
    return f"{date[:4]}-{date[4:6]}-{date[6:8]}"


def _x_axis_range(rth_only: bool) -> tuple[float, float]:
    return (RTH_START_SEC, RTH_END_SEC) if rth_only else (FULL_START_SEC, FULL_END_SEC)


def plot_price_with_band(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    rth_only: bool = True,
    sample_every: int = 100,
    presentation: bool = False,
) -> go.Figure:
    """Price + bid/ask band only (single panel). Kept for backward compat."""
    if rth_only:
        df = filter_rth(df, src="Timestamp")
    df = add_eq_ns_of_day(df)
    df = add_eq_seconds_of_day(df)

    nbb = df.filter(pl.col("EventType") == "QUOTE BID NB").select("sec_of_day", pl.col("Price").alias("nbb"))
    nbo = df.filter(pl.col("EventType") == "QUOTE ASK NB").select("sec_of_day", pl.col("Price").alias("nbo"))
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"])).select(
        "sec_of_day", pl.col("Price").alias("trade_px")
    )

    if nbb.is_empty() or nbo.is_empty() or trades.is_empty():
        nbb = df.filter(pl.col("EventType") == "QUOTE BID").select("sec_of_day", pl.col("Price").alias("nbb"))
        nbo = df.filter(pl.col("EventType") == "QUOTE ASK").select("sec_of_day", pl.col("Price").alias("nbo"))
        trades = df.filter(pl.col("EventType") == "TRADE").select("sec_of_day", pl.col("Price").alias("trade_px"))

    if sample_every > 1:
        nbb = nbb.gather_every(sample_every)
        nbo = nbo.gather_every(sample_every)
        trades = trades.gather_every(sample_every)

    fig = go.Figure()
    if not nbo.is_empty():
        fig.add_trace(go.Scatter(
            x=nbo["sec_of_day"].to_list(), y=nbo["nbo"].to_list(),
            line=dict(color=PALETTE["ask"], width=0.7), name="Best Ask",
            hovertemplate="Ask: %{y:.2f}<extra></extra>",
        ))
    if not nbb.is_empty():
        fig.add_trace(go.Scatter(
            x=nbb["sec_of_day"].to_list(), y=nbb["nbb"].to_list(),
            fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
            line=dict(color=PALETTE["bid"], width=0.7), name="Best Bid",
            hovertemplate="Bid: %{y:.2f}<extra></extra>",
        ))
    if not trades.is_empty():
        fig.add_trace(go.Scatter(
            x=trades["sec_of_day"].to_list(), y=trades["trade_px"].to_list(),
            mode="markers", marker=dict(size=2, color=PALETTE["mid"], opacity=0.5),
            name="Trade", hovertemplate="%{y:.2f}<extra></extra>",
        ))

    xstart, xend = _x_axis_range(rth_only)
    tickvals, ticktext = hour_tick_labels(xstart, xend, every_seconds=1800)

    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    rth_note = "（盤中 09:30–16:00）" if rth_only else "（含盤前盤後 04:00–20:00）"
    fig.update_layout(
        title=dict(text=f"{ticker} 在 {_format_date_label(date)} 的成交與買賣價走勢{rth_note}",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT,
        xaxis=dict(title="時間（ET）", tickvals=tickvals, ticktext=ticktext,
                   range=[xstart, xend], tickfont=dict(size=axis_size)),
        yaxis=dict(title="價格 ($)", tickfont=dict(size=axis_size)),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig


def plot_price_volume_combined(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    rth_only: bool = True,
    sample_every: int = 100,
    bin_minutes: int = 5,
    presentation: bool = False,
    show_trades: bool = True,
) -> go.Figure:
    """Combined chart: price + bid/ask band on top, volume bars on bottom,
    sharing the same x-axis so you can visually align price moves with volume.

    This is the version Lucas wants for Tab 1 (價格 & 量).
    """
    if rth_only:
        df = filter_rth(df, src="Timestamp")
    df = add_eq_ns_of_day(df)
    df = add_eq_seconds_of_day(df)

    nbb = df.filter(pl.col("EventType") == "QUOTE BID NB").select("sec_of_day", pl.col("Price").alias("nbb"))
    nbo = df.filter(pl.col("EventType") == "QUOTE ASK NB").select("sec_of_day", pl.col("Price").alias("nbo"))
    trades_full = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"])).select(
        "sec_of_day", pl.col("Price").alias("trade_px"), pl.col("Quantity").alias("qty")
    )

    # Fallback if no NB rows (some illiquid tickers)
    if nbb.is_empty() or nbo.is_empty() or trades_full.is_empty():
        nbb = df.filter(pl.col("EventType") == "QUOTE BID").select("sec_of_day", pl.col("Price").alias("nbb"))
        nbo = df.filter(pl.col("EventType") == "QUOTE ASK").select("sec_of_day", pl.col("Price").alias("nbo"))
        trades_full = df.filter(pl.col("EventType") == "TRADE").select(
            "sec_of_day", pl.col("Price").alias("trade_px"), pl.col("Quantity").alias("qty")
        )

    nbb_plot = nbb.gather_every(sample_every) if sample_every > 1 else nbb
    nbo_plot = nbo.gather_every(sample_every) if sample_every > 1 else nbo
    trades_plot = trades_full.gather_every(sample_every) if sample_every > 1 else trades_full

    # Bucket FULL trades (not sampled) for accurate volume profile
    bin_sec = bin_minutes * 60
    bucketed = (
        trades_full.with_columns(
            (pl.col("sec_of_day") // bin_sec).cast(pl.Int64).alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.col("qty").sum().alias("volume"))
        .sort("bucket")
    )
    if bucketed.is_empty():
        bucket_centers, volumes = [], []
    else:
        bucket_centers = [int(b * bin_sec + bin_sec / 2) for b in bucketed["bucket"].to_list()]
        volumes = bucketed["volume"].to_list()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    # === Row 1: price + bid/ask band ===
    if not nbo_plot.is_empty():
        fig.add_trace(go.Scatter(
            x=nbo_plot["sec_of_day"].to_list(), y=nbo_plot["nbo"].to_list(),
            line=dict(color=PALETTE["ask"], width=0.7), name="Best Ask",
            hovertemplate="Ask %{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if not nbb_plot.is_empty():
        fig.add_trace(go.Scatter(
            x=nbb_plot["sec_of_day"].to_list(), y=nbb_plot["nbb"].to_list(),
            fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
            line=dict(color=PALETTE["bid"], width=0.7), name="Best Bid",
            hovertemplate="Bid %{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if show_trades and not trades_plot.is_empty():
        fig.add_trace(go.Scatter(
            x=trades_plot["sec_of_day"].to_list(), y=trades_plot["trade_px"].to_list(),
            mode="markers", marker=dict(size=2, color=PALETTE["mid"], opacity=0.45),
            name="Trade", hovertemplate="Trade %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # === Row 2: volume bars ===
    fig.add_trace(go.Bar(
        x=bucket_centers, y=volumes,
        marker_color=PALETTE["mid"], width=bin_sec * 0.85,
        name=f"成交量 ({bin_minutes}m)",
        hovertemplate="vol %{y:,.0f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    xstart, xend = _x_axis_range(rth_only)
    tickvals, ticktext = hour_tick_labels(xstart, xend, every_seconds=1800)

    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    rth_note = "（盤中 09:30–16:00）" if rth_only else "（全日 04:00–20:00）"
    fig.update_layout(
        title=dict(text=f"{ticker} 在 {_format_date_label(date)} — 價格 + 量{rth_note}",
                   font=dict(size=title_size)),
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=70, r=40, t=90, b=60),
        bargap=0.08,
    )
    fig.update_xaxes(range=[xstart, xend], row=1, col=1)
    fig.update_xaxes(range=[xstart, xend], tickvals=tickvals, ticktext=ticktext,
                     title_text="時間（ET）", row=2, col=1)
    fig.update_yaxes(title_text="價格 ($)", row=1, col=1, tickfont=dict(size=axis_size))
    fig.update_yaxes(title_text="量（股）", tickformat=",", row=2, col=1, tickfont=dict(size=axis_size))
    return fig


def plot_trade_size_distribution(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    presentation: bool = False,
) -> go.Figure:
    """STRETCH: log-binned histogram of trade sizes with P90/P99 markers."""
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades found", showarrow=False, font=dict(size=18))
        return fig

    qty = np.array(trades["Quantity"].to_list(), dtype=float)
    qty = qty[qty > 0]
    if len(qty) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No positive trade sizes", showarrow=False, font=dict(size=18))
        return fig

    log_min = math.log10(qty.min())
    log_max = math.log10(qty.max())
    bins = np.logspace(log_min, log_max, 60)
    counts, edges = np.histogram(qty, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    p50 = float(np.percentile(qty, 50))
    p90 = float(np.percentile(qty, 90))
    p99 = float(np.percentile(qty, 99))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=counts, width=widths,
        marker_color=PALETTE["mid"], name="Trade size",
        hovertemplate="size %{x:,.0f}<br>count %{y}<extra></extra>",
    ))
    fig.add_vline(x=p50, line_dash="dot", line_color="#888",
                  annotation_text=f"P50={int(p50)}", annotation_position="top",
                  annotation=dict(yshift=4))
    fig.add_vline(x=p90, line_dash="dash", line_color="orange",
                  annotation_text=f"P90={int(p90)}", annotation_position="top",
                  annotation=dict(yshift=22))
    fig.add_vline(x=p99, line_dash="dash", line_color="red",
                  annotation_text=f"P99={int(p99)}", annotation_position="top",
                  annotation=dict(yshift=40))

    title_size = 22 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"{ticker} 單筆成交量分布（{_format_date_label(date)}） — P50/P90/P99 標記",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT,
        xaxis=dict(title="成交數量（股）", type="log"),
        yaxis=dict(title="頻率"),
        margin=dict(l=70, r=30, t=110, b=60),
        bargap=0.0,
    )
    return fig
