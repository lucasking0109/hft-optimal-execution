"""Module 2: Intraday Volume Profile (the U-shape!).

Bins TRADE volume into N-minute buckets across the trading day so the
characteristic morning surge / midday lull / closing rush is visible.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from hft.data.timeparse import add_eq_minute_bucket, filter_rth
from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT, PALETTE, TITLE_FONT_SIZE


def _bucket_to_label(bucket: int, bin_minutes: int) -> str:
    minute_of_day = bucket * bin_minutes
    h = minute_of_day // 60
    m = minute_of_day % 60
    return f"{h:02d}:{m:02d}"


def plot_intraday_volume(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    bin_minutes: int = 5,
    rth_only: bool = True,
    presentation: bool = False,
) -> go.Figure:
    """Bin trades into N-minute buckets and bar-chart the volume profile.

    Adds annotations marking the U-shape pattern.
    """
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if rth_only:
        trades = filter_rth(trades, src="Timestamp")
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades available", showarrow=False, font=dict(size=18))
        return fig

    bucketed = add_eq_minute_bucket(trades, bin_minutes=bin_minutes, src="Timestamp", dst="bucket_min")
    profile = (
        bucketed.group_by("bucket_min")
        .agg(pl.col("Quantity").sum().alias("volume"))
        .sort("bucket_min")
    )

    buckets = profile["bucket_min"].to_list()
    volumes = profile["volume"].to_list()
    labels = [_bucket_to_label(int(b), bin_minutes) for b in buckets]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=volumes,
            marker_color=PALETTE["mid"],
            name=f"成交量（每 {bin_minutes} 分鐘）",
            hovertemplate="%{x}<br>%{y:,} 股<extra></extra>",
        )
    )

    # Add U-shape annotations
    if rth_only and len(volumes) > 6:
        # Find approximate morning peak (first 30 min after 9:30) and afternoon peak (last 30 min)
        max_idx = volumes.index(max(volumes))
        min_idx = volumes.index(min(volumes))
        annot_font = 14 if presentation else 12
        fig.add_annotation(
            x=labels[max_idx], y=volumes[max_idx],
            text=f"📈 最高量：{labels[max_idx]}",
            showarrow=True, arrowhead=2, ax=-40, ay=-40,
            font=dict(size=annot_font, color="#444"),
            bgcolor="rgba(255,255,255,0.8)",
        )
        fig.add_annotation(
            x=labels[min_idx], y=volumes[min_idx],
            text=f"📉 最低量：{labels[min_idx]}",
            showarrow=True, arrowhead=2, ax=0, ay=-30,
            font=dict(size=annot_font, color="#444"),
            bgcolor="rgba(255,255,255,0.8)",
        )

    rth_note = "（盤中 09:30–16:00）" if rth_only else "（全日 04:00–20:00）"
    date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    title = f"{ticker} 日內成交量分布 — {date_label}{rth_note}"

    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size)),
        height=CHART_HEIGHT,
        xaxis=dict(title="時間（ET）", tickfont=dict(size=axis_size)),
        yaxis=dict(title="成交量（股）", tickfont=dict(size=axis_size), tickformat=","),
        bargap=0.05,
        margin=dict(l=70, r=30, t=80, b=60),
    )
    return fig


def plot_average_volume_profile(
    profiles: list[tuple[str, pl.DataFrame]],  # list of (label, trades_df)
    *,
    bin_minutes: int = 5,
    rth_only: bool = True,
    presentation: bool = False,
) -> go.Figure:
    """Overlay multiple days' or tickers' volume profiles for comparison."""
    fig = go.Figure()
    palette_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (label, trades) in enumerate(profiles):
        if rth_only:
            trades = filter_rth(trades, src="Timestamp")
        if trades.is_empty():
            continue
        bucketed = add_eq_minute_bucket(trades, bin_minutes=bin_minutes, src="Timestamp", dst="bucket_min")
        profile = bucketed.group_by("bucket_min").agg(pl.col("Quantity").sum().alias("volume")).sort("bucket_min")
        labels = [_bucket_to_label(int(b), bin_minutes) for b in profile["bucket_min"].to_list()]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=profile["volume"].to_list(),
                mode="lines",
                name=label,
                line=dict(color=palette_cycle[i % len(palette_cycle)], width=2),
            )
        )

    title_size = 22 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text="成交量分布對比（多日/多 ticker）", font=dict(size=title_size)),
        height=CHART_HEIGHT,
        xaxis=dict(title="時間"),
        yaxis=dict(title="成交量"),
        margin=dict(l=70, r=30, t=80, b=60),
    )
    return fig
