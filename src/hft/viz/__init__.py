"""Visualization modules for Phase 1 dashboard.

Design principles (per plan §Phase 1 presentation guidelines):
- Self-explanatory storytelling titles (not "Volume Profile" but
  "AAPL 在 X 日的成交量分布 — 注意 9:30 開盤的爆量")
- In-chart annotations (label morning surge / midday lull / closing rush)
- Large fonts (>=18pt titles), color-blind safe palettes
- Each function returns a plotly Figure callers can embed in Streamlit
  (so we don't couple visualization to dashboard framework)
"""

# Color-blind safe palette — viridis-derived
PALETTE = {
    "twap": "#7f7f7f",          # gray
    "vwap_follow": "#1f77b4",   # blue
    "ac": "#ff7f0e",            # orange
    "rl": "#d62728",            # red
    "mid": "#2ca02c",           # green
    "bid": "#1f77b4",
    "ask": "#d62728",
    "buy_aggressor": "#2ca02c",
    "sell_aggressor": "#d62728",
}

CHART_HEIGHT = 480
CHART_HEIGHT_TALL = 600
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 13
