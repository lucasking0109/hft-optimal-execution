"""Module 5: Aggressor Flow (futures only).

Uses a continuous datetime axis so that NQ's 23-hour overnight session
(spanning two calendar dates) renders as a single monotonic timeline
instead of wrapping at midnight.

CME data is native CT; we shift +1h for ET display.
"""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from hft.data.timeparse import add_fut_ns_of_day
from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT_TALL, PALETTE, TITLE_FONT_SIZE


def _add_fut_seconds(df: pl.DataFrame) -> pl.DataFrame:
    """LocalTime is 'HHMMSSnnnnnnnnn' (15 chars). Convert to seconds-of-day."""
    if "ns_of_day" not in df.columns:
        df = add_fut_ns_of_day(df, src="LocalTime")
    return df.with_columns((pl.col("ns_of_day") / 1e9).alias("sec_of_day"))


def _add_fut_datetime(df: pl.DataFrame, *, hour_offset_for_display: int = 0) -> pl.DataFrame:
    """Build a continuous datetime column ('dt') that is monotonic across
    LocalDate boundaries. CT native; add hour_offset_for_display to convert
    to a different timezone label (1 = ET).
    """
    if "ns_of_day" not in df.columns:
        df = add_fut_ns_of_day(df, src="LocalTime")

    df = df.with_columns(
        pl.col("LocalDate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("_d")
    )
    df = df.with_columns(
        (
            pl.col("_d").cast(pl.Datetime("ns"))
            + pl.duration(nanoseconds=pl.col("ns_of_day"))
            + pl.duration(hours=hour_offset_for_display)
        ).alias("dt")
    ).drop("_d")
    return df


def _hour_ticks_for_session(min_sec: float, max_sec: float, step: int = 3600,
                            hour_offset: int = 0):
    """Build (vals, text) ticks at hour marks (legacy wrap-around mode)."""
    first = int(min_sec // step) * step
    if first < min_sec:
        first += step
    vals = list(range(first, int(max_sec) + 1, step))
    text = [
        f"{((t // 3600) + hour_offset) % 24:02d}:{(t % 3600) // 60:02d}"
        for t in vals
    ]
    return vals, text


def plot_aggressor_flow(
    df: pl.DataFrame,
    contract: str,
    date: str,
    *,
    sample_every: int = 50,
    presentation: bool = False,
    timezone: str = "ET",   # ET (default, user in Boston) or CT (CME native)
) -> go.Figure:
    """Two-panel: trade price (top), cumulative aggressor flow (bottom).
    Shared time axis with HH:MM hour-mark labels.
    """
    hour_offset = 1 if timezone.upper() == "ET" else 0
    axis_label = "美東時間 (ET)" if hour_offset else "中部時間 (CT, CME native)"

    df = df.sort(["LocalDate", "LocalTime"]).with_columns(
        pl.col("Type").cast(pl.Utf8).alias("_type_str")
    )
    df = _add_fut_datetime(df, hour_offset_for_display=hour_offset)

    trades = df.filter(pl.col("_type_str").str.contains("AGRESSOR"))
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No aggressor-tagged trades", showarrow=False, font=dict(size=18))
        return fig

    trades = trades.with_columns(
        pl.when(pl.col("_type_str") == "TRADE AGRESSOR ON BUY")
        .then(pl.col("Quantity"))
        .otherwise(-pl.col("Quantity"))
        .alias("signed_qty")
    )
    trades = trades.with_columns(pl.col("signed_qty").cum_sum().alias("cum_aggressor"))

    trades_plot = trades.gather_every(max(1, sample_every))
    trades_for_price = df.filter(pl.col("_type_str").str.starts_with("TRADE")).gather_every(max(1, sample_every))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.10,
        subplot_titles=("成交價", "累積 Aggressor 訊號（綠＞0 → 主動買壓）"),
        shared_xaxes=True,
    )

    if not trades_for_price.is_empty():
        fig.add_trace(go.Scatter(
            x=trades_for_price["dt"].to_list(),
            y=trades_for_price["Price"].to_list(),
            mode="markers",
            marker=dict(size=2.5, color="#888", opacity=0.7),
            name="Trade price",
            hovertemplate="%{x|%H:%M:%S}<br>%{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=trades_plot["dt"].to_list(),
        y=trades_plot["cum_aggressor"].to_list(),
        mode="lines",
        line=dict(color=PALETTE["buy_aggressor"], width=2),
        name="Cum aggressor",
        fill="tozeroy",
        fillcolor="rgba(44,160,44,0.15)",
        showlegend=False,
        hovertemplate="%{x|%a %m-%d %H:%M}<br>%{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#888", row=2, col=1)

    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    fig.update_layout(
        title=dict(text=f"{contract} 期貨 Aggressor 訊號 — Trade Date {date_label}",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT_TALL,
        showlegend=False,
        margin=dict(l=70, r=30, t=90, b=60),
    )
    fig.update_yaxes(title_text="價格", row=1, col=1, tickfont=dict(size=axis_size))
    fig.update_yaxes(title_text="累積帶號量", row=2, col=1, tickfont=dict(size=axis_size))
    fig.update_xaxes(title_text=axis_label, row=2, col=1,
                     tickformat="%a %m-%d\n%H:%M",
                     tickfont=dict(size=axis_size))
    return fig
