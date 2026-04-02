"""Phase 4 visualisations for venue routing analysis."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from hft.viz import AXIS_FONT_SIZE, CHART_HEIGHT_TALL, TITLE_FONT_SIZE


def plot_venue_score_heatmap(
    hourly: pl.DataFrame,   # from compute_venue_nbbo_share_hourly
    ticker: str,
    date: str,
    *,
    presentation: bool = False,
) -> go.Figure:
    """Heatmap: hour (x) × venue (y), colour = NBBO share %."""
    if hourly.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No hourly NBBO data", showarrow=False, font=dict(size=18))
        return fig

    hours = sorted(hourly["hour_et"].unique().to_list())
    venues = (
        hourly.group_by("Exchange").agg(pl.col("share_pct").mean().alias("avg"))
        .sort("avg", descending=True)["Exchange"].to_list()
    )
    grid: list[list[float]] = []
    for v in venues:
        row = []
        sub = hourly.filter(pl.col("Exchange") == v)
        for h in hours:
            cell = sub.filter(pl.col("hour_et") == h)
            row.append(float(cell["share_pct"][0]) if not cell.is_empty() else 0.0)
        grid.append(row)

    fig = go.Figure(
        go.Heatmap(
            x=[f"{h:02d}:00" for h in hours], y=venues, z=grid,
            colorscale="Viridis", colorbar=dict(title="NBBO 占比 %"),
            hovertemplate="%{y}<br>%{x}<br>%{z:.1f}%<extra></extra>",
        )
    )
    title_size = 22 if presentation else TITLE_FONT_SIZE
    axis_size = 16 if presentation else AXIS_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"{ticker} 各交易所 NBBO 提供率（每小時 ET） — {date[:4]}-{date[4:6]}-{date[6:8]}",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT_TALL,
        xaxis=dict(title="ET 小時", tickfont=dict(size=axis_size)),
        yaxis=dict(title="交易所", autorange="reversed", tickfont=dict(size=axis_size)),
        margin=dict(l=130, r=40, t=80, b=60),
    )
    return fig


def plot_adverse_selection_bar(
    metrics: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    presentation: bool = False,
) -> go.Figure:
    """Horizontal bar: per-venue adverse selection cost (bps).
    Negative values = trades on this venue followed by FAVOURABLE moves.
    Routable venues coloured; FINRA / non-routable greyed.
    """
    sub = metrics.filter(pl.col("n_trades") > 50).sort("adverse_selection_bps")
    if sub.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No venues have enough trades", showarrow=False, font=dict(size=18))
        return fig

    colors = ["#888" if not r else "#1f77b4" for r in sub["routable"].to_list()]
    venues = sub["Exchange"].to_list()
    vals = sub["adverse_selection_bps"].to_list()

    fig = go.Figure(go.Bar(
        y=venues, x=vals, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in vals], textposition="auto",
        hovertemplate="%{y}<br>%{x:+.2f} bps<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#bbb")

    title_size = 22 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"{ticker} 各交易所 60-秒逆向選擇成本（bps，越小越好） — {date[:4]}-{date[4:6]}-{date[6:8]}",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT_TALL,
        xaxis=dict(title="逆向選擇 bps（負值＝成交後價格反向，賺）"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=130, r=80, t=90, b=60),
    )
    return fig


def plot_composite_score_bar(
    scored: pl.DataFrame,
    ticker: str,
    date: str,
    *,
    presentation: bool = False,
) -> go.Figure:
    """Bar of composite SOR score per venue."""
    sub = scored.filter(pl.col("routable")).sort("composite_score", descending=True)
    if sub.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No routable venues", showarrow=False, font=dict(size=18))
        return fig

    fig = go.Figure(go.Bar(
        y=sub["Exchange"].to_list(),
        x=sub["composite_score"].to_list(),
        orientation="h",
        marker_color="#2ca02c",
        text=[f"{s:+.2f}" for s in sub["composite_score"].to_list()],
        textposition="auto",
        hovertemplate="%{y}<br>%{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#bbb")

    title_size = 22 if presentation else TITLE_FONT_SIZE
    fig.update_layout(
        title=dict(text=f"{ticker} SOR Composite Score（越大越好） — {date[:4]}-{date[4:6]}-{date[6:8]}",
                   font=dict(size=title_size)),
        height=CHART_HEIGHT_TALL,
        xaxis=dict(title="Composite score"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=130, r=60, t=90, b=60),
    )
    return fig
