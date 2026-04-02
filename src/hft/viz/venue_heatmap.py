"""Module 4: Multi-Venue Liquidity Heatmap.

For each (venue, time-bin) cell, displays that venue's share of total trade
volume in that period. Highlights when each venue dominates — e.g. NASDAQ
likely owns AAPL throughout, but IEX may pick up fraction during stress.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from hft.data.timeparse import add_eq_minute_bucket, filter_rth
from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT_TALL, TITLE_FONT_SIZE


def _bucket_to_label(bucket: int, bin_minutes: int) -> str:
    minute_of_day = bucket * bin_minutes
    h = minute_of_day // 60
    m = minute_of_day % 60
    return f"{h:02d}:{m:02d}"


def plot_venue_heatmap(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    bin_minutes: int = 15,
    rth_only: bool = True,
    presentation: bool = False,
) -> go.Figure:
    """Heatmap of venue volume share over the trading day."""
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if rth_only:
        trades = filter_rth(trades, src="Timestamp")
    if trades.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades to chart venues", showarrow=False, font=dict(size=18))
        return fig

    bucketed = add_eq_minute_bucket(trades, bin_minutes=bin_minutes, src="Timestamp", dst="bucket_min")
    grouped = (
        bucketed.group_by(["bucket_min", "Exchange"])
        .agg(pl.col("Quantity").sum().alias("vol"))
    )
    total_per_bucket = (
        grouped.group_by("bucket_min").agg(pl.col("vol").sum().alias("total"))
    )
    grouped = grouped.join(total_per_bucket, on="bucket_min").with_columns(
        (pl.col("vol") / pl.col("total") * 100).alias("share_pct")
    )

    buckets = sorted(grouped["bucket_min"].unique().to_list())
    venues = sorted(grouped["Exchange"].unique().to_list(), key=lambda v: -grouped.filter(pl.col("Exchange") == v)["vol"].sum())

    bucket_labels = [_bucket_to_label(int(b), bin_minutes) for b in buckets]

    # Build matrix [venue][bucket] = share_pct
    matrix = []
    for venue in venues:
        row = []
        venue_data = grouped.filter(pl.col("Exchange") == venue)
        for bucket in buckets:
            cell = venue_data.filter(pl.col("bucket_min") == bucket)
            if cell.is_empty():
                row.append(0.0)
            else:
                row.append(float(cell["share_pct"][0]))
        matrix.append(row)

    fig = go.Figure(
        go.Heatmap(
            x=bucket_labels,
            y=venues,
            z=matrix,
            colorscale="Viridis",
            colorbar=dict(title="占比 %"),
            hovertemplate="%{y}<br>%{x}<br>%{z:.1f}%<extra></extra>",
        )
    )

    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    fig.update_layout(
        title=dict(text=f"{ticker} 各交易所成交量占比 — {date_label}",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT_TALL,
        xaxis=dict(title="時間（每 {} 分鐘）".format(bin_minutes), tickfont=dict(size=axis_size)),
        yaxis=dict(title="交易所", tickfont=dict(size=axis_size), autorange="reversed"),
        margin=dict(l=110, r=40, t=80, b=60),
    )
    return fig


def plot_venue_summary_bar(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    rth_only: bool = True,
    presentation: bool = False,
) -> go.Figure:
    """Horizontal bar chart of total volume by venue (one day, one ticker)."""
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if rth_only:
        trades = filter_rth(trades, src="Timestamp")
    if trades.is_empty():
        return go.Figure()

    summary = (
        trades.group_by("Exchange")
        .agg(pl.col("Quantity").sum().alias("vol"))
        .sort("vol", descending=True)
    )
    total = summary["vol"].sum()
    summary = summary.with_columns((pl.col("vol") / total * 100).alias("share"))

    shares = summary["share"].to_list()
    max_share = max(shares) if shares else 100.0

    fig = go.Figure(
        go.Bar(
            y=summary["Exchange"].to_list(),
            x=shares,
            orientation="h",
            marker_color="#1f77b4",
            text=[f"{s:.1f}%" for s in shares],
            textposition="auto",   # auto puts labels inside long bars, outside short ones
            insidetextanchor="end",
            hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
        )
    )
    title_size = 22 if presentation else TITLE_FONT_SIZE
    date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    fig.update_layout(
        title=dict(text=f"{ticker} 各交易所總占比 — {date_label}",
                   font=dict(size=title_size)),
        height=460,
        # Extend x-axis range so even outside text isn't clipped
        xaxis=dict(title="占比 (%)", range=[0, max_share * 1.15]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=110, t=80, b=60),
    )
    return fig
