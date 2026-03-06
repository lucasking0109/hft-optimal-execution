"""Module 3: Bid/Ask Spread time series + distribution.

Same fix as price_charts: numeric x-axis (sec_of_day) instead of string
categorical so traces with different timestamps don't get split.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from hft.data.timeparse import (
    add_eq_ns_of_day,
    add_eq_seconds_of_day,
    filter_rth,
    hour_tick_labels,
)
from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT_TALL, PALETTE, TITLE_FONT_SIZE

RTH_START_SEC = 9 * 3600 + 30 * 60
RTH_END_SEC = 16 * 3600
FULL_START_SEC = 4 * 3600
FULL_END_SEC = 20 * 3600


def _compute_spread_df(df: pl.DataFrame) -> pl.DataFrame:
    """Compute spread bps using sec_of_day as the time axis."""
    nbb = df.filter(pl.col("EventType") == "QUOTE BID NB").select(
        "sec_of_day", pl.col("Price").alias("nbb")
    )
    nbo = df.filter(pl.col("EventType") == "QUOTE ASK NB").select(
        "sec_of_day", pl.col("Price").alias("nbo")
    )
    if nbb.is_empty() or nbo.is_empty():
        return pl.DataFrame(schema={"sec_of_day": pl.Float64, "spread_bps": pl.Float64})

    merged = nbb.join(nbo, on="sec_of_day", how="full", coalesce=True).sort("sec_of_day")
    merged = merged.with_columns(
        pl.col("nbb").forward_fill(),
        pl.col("nbo").forward_fill(),
    ).filter(
        pl.col("nbb").is_not_null() & pl.col("nbo").is_not_null()
        & (pl.col("nbb") > 0) & (pl.col("nbo") > 0) & (pl.col("nbo") > pl.col("nbb"))
    )
    return merged.with_columns(
        ((pl.col("nbb") + pl.col("nbo")) / 2.0).alias("mid"),
        ((pl.col("nbo") - pl.col("nbb")) / ((pl.col("nbb") + pl.col("nbo")) / 2.0) * 10000).alias("spread_bps"),
    )


def plot_spread(
    df: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    rth_only: bool = True,
    sample_every: int = 200,
    presentation: bool = False,
) -> go.Figure:
    """Two-panel: spread time series (top) + spread distribution (bottom)."""
    if rth_only:
        df = filter_rth(df, src="Timestamp")
    df = add_eq_ns_of_day(df)
    df = add_eq_seconds_of_day(df)

    spread_df = _compute_spread_df(df)
    if spread_df.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No NBBO available to compute spread",
                           showarrow=False, font=dict(size=18))
        return fig

    plotted = spread_df.gather_every(max(1, sample_every))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.18,
        subplot_titles=("價差隨時間變化（bps）", "價差分布"),
    )

    fig.add_trace(go.Scatter(
        x=plotted["sec_of_day"].to_list(),
        y=plotted["spread_bps"].to_list(),
        mode="lines",
        line=dict(color="#5b3b8c", width=1),
        name="Spread (bps)",
        hovertemplate="%{y:.2f} bps<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    full_spread = spread_df["spread_bps"].to_list()
    fig.add_trace(go.Histogram(
        x=full_spread,
        nbinsx=60,
        marker_color="#8b6bb0",
        name="分布",
        showlegend=False,
    ), row=2, col=1)

    median = float(spread_df["spread_bps"].median())
    p95 = float(spread_df["spread_bps"].quantile(0.95))

    # Use yshift to keep median/P95 labels from overlapping each other
    fig.add_vline(
        x=median, line_dash="dash", line_color="orange",
        annotation_text=f"中位數 {median:.2f} bps",
        annotation_position="top",
        annotation=dict(yshift=2),
        row=2, col=1,
    )
    fig.add_vline(
        x=p95, line_dash="dash", line_color="red",
        annotation_text=f"P95 {p95:.2f} bps",
        annotation_position="top",
        annotation=dict(yshift=22),
        row=2, col=1,
    )

    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    xstart, xend = (RTH_START_SEC, RTH_END_SEC) if rth_only else (FULL_START_SEC, FULL_END_SEC)
    tickvals, ticktext = hour_tick_labels(xstart, xend, every_seconds=1800)

    date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    fig.update_layout(
        title=dict(text=f"{ticker} 買賣價差 — {date_label}", font=dict(size=title_size)),
        height=CHART_HEIGHT_TALL,
        showlegend=False,
        margin=dict(l=70, r=30, t=90, b=70),
    )
    fig.update_yaxes(title_text="bps", row=1, col=1, tickfont=dict(size=axis_size))
    fig.update_yaxes(title_text="頻率", row=2, col=1)
    fig.update_xaxes(title_text="時間（ET）",
                     tickvals=tickvals, ticktext=ticktext,
                     range=[xstart, xend], row=1, col=1)
    fig.update_xaxes(title_text="價差 (bps)", row=2, col=1)
    return fig
