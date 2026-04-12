"""ABIDES LOB Explorer — Phase 5 visualization dashboard.

Trading-platform-style view of abides-sim simulation output:
- DOM ladder (Depth of Market) with HFT-grade time slider
- Time & Sales (trade tape)
- Volume profile by price level
- Order event rate over time
- Cumulative aggressor flow + price chart
- Cancel heatmap

Architecture note:
    DOM time sliders live INSIDE @st.fragment so changes only re-run the DOM
    section, eliminating full-page flash that breaks visual continuity.

Run:
    cd "/Users/lucasking/Projects/HFT Date Analysis"
    uv run streamlit run dashboards/03_abides_lob_explorer.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from st_aggrid import AgGrid

from hft.data.abides_loader import (
    DEFAULT_ABIDES_LOG,
    load_abides_run,
    orderbook_snapshot_at,
)
from hft.viz.abides_lob import (
    plot_aggressor_with_price,
    plot_cancel_heatmap,
    plot_order_arrival_rate,
    plot_volume_profile_by_price,
)
from hft.viz.abides_lob_grid import (
    dom_grid_options,
    make_dom_ladder_df,
    make_trade_tape_df,
    trade_tape_grid_options,
)

st.set_page_config(
    page_title="ABIDES LOB Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached data load (won't re-run on slider changes)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="載入 abides 模擬輸出…", max_entries=4)
def cached_abides_run(log_dir_str: str):
    return load_abides_run(Path(log_dir_str))


# ---------------------------------------------------------------------------
# Top-level sidebar (only data-loading + mode — rare changes, OK to full re-run)
# ---------------------------------------------------------------------------

st.sidebar.title("📊 ABIDES LOB Explorer")
st.sidebar.caption("Phase 5 — 模擬器訂單簿 + 訂單流視覺化")

mode = st.sidebar.radio(
    "顯示模式",
    options=["研究模式", "Demo 模式"],
    index=0,
    horizontal=True,
)
presentation = mode == "Demo 模式"

log_dir = st.sidebar.text_input(
    "ABIDES log dir",
    value=str(DEFAULT_ABIDES_LOG),
    help="跑完 abides 後的 log 目錄。預設 = smoke test output",
)

try:
    data = cached_abides_run(log_dir)
except Exception as e:
    st.error(f"❌ 載入失敗（NO Silent Fallback）：{e}")
    st.stop()

trades_df: pl.DataFrame = data["trades"]
cancels_df: pl.DataFrame = data["cancels"]
orders_df: pl.DataFrame = data["orders"]
fund_df: pl.DataFrame = data["fundamental"]
ob_df = data["orderbook"]

if trades_df.is_empty():
    st.error("❌ 此 log 沒有成交事件，無法顯示")
    st.stop()

t_min: pd.Timestamp = pd.Timestamp(trades_df["time"].min())
t_max: pd.Timestamp = pd.Timestamp(trades_df["time"].max())
total_seconds = int((t_max - t_min).total_seconds())

st.sidebar.markdown(f"**時間範圍**: {t_min.strftime('%H:%M:%S')} → {t_max.strftime('%H:%M:%S')}")
st.sidebar.markdown(f"**總成交數**: {len(trades_df):,} 筆")

# ─── DOM controls (Tab 2) — IN SIDEBAR (no fragment, AgGrid handles flicker) ───
if "snap_override" not in st.session_state:
    st.session_state.snap_override = None

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱ Tab 2 — DOM 時間（HFT 級）")

elapsed_sec = st.sidebar.slider(
    "秒（整秒）", 0, max(total_seconds, 1),
    value=min(60, total_seconds), step=1, key="dom_sec",
)
elapsed_ms = st.sidebar.slider(
    "毫秒（0-999 ms）", 0, 999, value=0, step=1, key="dom_ms",
    help="毫秒級微調 — sub-second 訂單簿動態",
)
elapsed_us = st.sidebar.slider(
    "微秒（0-999 μs）", 0, 999, value=0, step=10, key="dom_us",
    help="HFT 級解析度",
)

snap_time = (
    t_min
    + pd.Timedelta(seconds=elapsed_sec)
    + pd.Timedelta(milliseconds=elapsed_ms)
    + pd.Timedelta(microseconds=elapsed_us)
)

# Event navigation
st.sidebar.markdown("**📍 事件跳轉**")
nav_a, nav_b = st.sidebar.columns(2)
with nav_a:
    if st.button("⏮ 前", key="prev_event", use_container_width=True):
        snap_ts = snap_time.to_datetime64()
        prev = trades_df.filter(pl.col("time") < snap_ts).tail(1)
        if not prev.is_empty():
            st.session_state.snap_override = pd.Timestamp(prev["time"][0])
with nav_b:
    if st.button("後 ⏭", key="next_event", use_container_width=True):
        snap_ts = snap_time.to_datetime64()
        nxt = trades_df.filter(pl.col("time") > snap_ts).head(1)
        if not nxt.is_empty():
            st.session_state.snap_override = pd.Timestamp(nxt["time"][0])

if st.session_state.snap_override is not None:
    snap_time = st.session_state.snap_override
    st.sidebar.caption(f"⤵ 跳到 `{snap_time.strftime('%H:%M:%S.%f')[:-3]}`")
    if st.sidebar.button("清除跳轉", key="clear_override", use_container_width=True):
        st.session_state.snap_override = None
        st.rerun()

st.sidebar.markdown("**🪜 DOM 設定**")
n_levels = st.sidebar.slider("DOM 顯示深度", 5, 40, 15, key="dom_depth")
filter_qty = st.sidebar.slider(
    "過濾守護單 qty>", 500, 20000, 5000, step=500, key="dom_filter_qty",
)
show_filtered = st.sidebar.checkbox(
    "顯示守護單", value=False, key="dom_show_filtered",
)
n_recent = st.sidebar.slider(
    "Time & Sales 顯示筆數", 10, 80, 30, key="dom_n_recent",
)

st.sidebar.markdown("---")
bin_sec = st.sidebar.slider(
    "Tab 3/4 時間分桶（秒）",
    min_value=1, max_value=60, value=5,
    help="用於訂單流分析 + 撤單熱圖",
)


# ---------------------------------------------------------------------------
# Force dark theme everywhere (covers AgGrid container white-block issue)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0e0e0e !important;
}
iframe[title="streamlit_aggrid.agGrid"] {
    background-color: #0e0e0e !important;
}
.element-container {
    background-color: transparent !important;
}
/* AgGrid theme background fix */
.ag-theme-balham-dark, .ag-theme-balham-dark .ag-root-wrapper,
.ag-theme-balham-dark .ag-body, .ag-theme-balham-dark .ag-row {
    background-color: #0e0e0e !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Static top
# ---------------------------------------------------------------------------

st.title("📊 ABIDES Limit Order Book Explorer")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 市場景觀",
    "🪜 DOM Ladder + 成交帶",
    "📊 訂單流分析",
    "⚙️ 撤單模式",
])

# ---------------------------------------------------------------------------
# Tab 1: 市場景觀 (no time slider needed — static)
# ---------------------------------------------------------------------------

with tab1:
    st.markdown(
        "### 📌 這個 tab 告訴你什麼\n"
        "- **上半**：每筆成交（綠=主動買，紅=主動賣）+ 模擬器的「真實價值」虛線\n"
        "- **下半**：累積買賣壓 — 上升 = 主動買壓佔上風\n"
    )
    fig = plot_aggressor_with_price(trades_df, fund_df, sample_every=20,
                                    presentation=presentation)
    st.plotly_chart(fig, use_container_width=True, key="tab1_aggressor")


# ---------------------------------------------------------------------------
# Tab 2: DOM Ladder — wrapped in @st.fragment for partial re-runs
#   → time slider changes ONLY re-render this fragment, no whole-page flash
# ---------------------------------------------------------------------------

with tab2:
    st.markdown(
        "### 📌 核心 trading-platform 介面\n"
        "- **左**：訂單簿即時快照（DOM ladder） — bid 在中線左側、ask 右側\n"
        "- **右**：Time & Sales — 最近成交流，**綠 = 主動買，紅 = 主動賣**\n"
        "- 用左側 **sidebar 的時間 slider** 控制 DOM 快照時間（HFT 解析度）"
    )

    # ─── Compute snapshot data + event density ───
    window = pd.Timedelta(milliseconds=100)
    ws = (snap_time - window).to_datetime64()
    we = (snap_time + window).to_datetime64()
    snap_ns_str = (
        snap_time.strftime("%H:%M:%S.%f")[:-3]
        + f".{snap_time.nanosecond:03d}"
    )

    snapshot = orderbook_snapshot_at(ob_df, snap_time)

    # ─── 2-column layout: DOM (left/wide) + Tape (right) ───
    dom_col, tape_col = st.columns([3, 2], gap="medium")

    # === LEFT: DOM ladder ===
    with dom_col:
        n_t = trades_df.filter((pl.col("time") >= ws) & (pl.col("time") <= we)).height
        n_c = (cancels_df.filter((pl.col("time") >= ws) & (pl.col("time") <= we)).height
               if not cancels_df.is_empty() else 0)
        n_o = (orders_df.filter((pl.col("time") >= ws) & (pl.col("time") <= we)).height
               if not orders_df.is_empty() else 0)
        st.markdown(
            f"#### DOM @ {snap_time.strftime('%H:%M:%S.%f')[:-3]}  ·  "
            f"📍 **{snap_ns_str}**"
        )

        dom_df, dom_meta = make_dom_ladder_df(
            snapshot,
            n_levels=n_levels,
            filter_boundary_qty=filter_qty,
            show_filtered=show_filtered,
        )
        best_bid = dom_meta.get("best_bid_cents")
        best_ask = dom_meta.get("best_ask_cents")
        spread_label = ""
        if best_bid is not None and best_ask is not None:
            spread = (best_ask - best_bid) / 100
            mid = ((best_bid + best_ask) / 2) / 100
            spread_bps = spread / mid * 10000 if mid > 0 else 0
            spread_label = f"Spread ${spread:.2f} ({spread_bps:.1f} bps)"

        st.caption(
            f"{spread_label}  ·  ±100ms 內: 成交 {n_t} / 撤單 {n_c} / 下單 {n_o}"
        )

        if dom_df.empty:
            st.info("沒有可顯示的訂單（可能全被守護單過濾掉）")
        else:
            grid_opts = dom_grid_options(dom_df, dom_meta)
            grid_height = max(420, 24 * len(dom_df) + 40)
            AgGrid(
                dom_df,
                gridOptions=grid_opts,
                height=grid_height,
                fit_columns_on_grid_load=True,
                theme="balham-dark",
                allow_unsafe_jscode=True,
                update_mode="NO_UPDATE",
                key="dom_aggrid",
                reload_data=True,
            )

        n_filtered = sum(1 for s in snapshot if s["qty"] > filter_qty)
        if n_filtered > 0 and not show_filtered:
            st.caption(f"⚙️ 過濾了 {n_filtered} 個守護單（qty > {filter_qty}）")

    # === RIGHT: Trade tape rendered with AgGrid ===
    with tape_col:
        st.markdown("#### Time & Sales")
        snap_ts = snap_time.to_datetime64()
        recent_trades = trades_df.filter(pl.col("time") <= snap_ts)
        tape_df = make_trade_tape_df(recent_trades, n_recent=n_recent)
        if tape_df.empty:
            st.info("尚無成交事件")
        else:
            tape_opts = trade_tape_grid_options(tape_df)
            tape_height = max(420, 24 * len(tape_df) + 40)
            AgGrid(
                tape_df,
                gridOptions=tape_opts,
                height=tape_height,
                fit_columns_on_grid_load=True,
                theme="balham-dark",
                allow_unsafe_jscode=True,
                update_mode="NO_UPDATE",
                key="tape_aggrid",
                reload_data=True,
            )

    # ─── Below the 2-column layout: nearby events + volume profile ───
    with st.expander("🔍 附近事件（±100ms 內）", expanded=False):
        nearby_trades = trades_df.filter(
            (pl.col("time") >= ws) & (pl.col("time") <= we)
        ).select(["time", "side", "quantity", "price_dollars", "agent_id"])
        if not nearby_trades.is_empty():
            st.markdown(f"**{len(nearby_trades)} 筆 trade**")
            st.dataframe(nearby_trades, hide_index=True, height=180)
        if not cancels_df.is_empty():
            nearby_cancels = cancels_df.filter(
                (pl.col("time") >= ws) & (pl.col("time") <= we)
            ).select(["time", "side", "quantity", "price_dollars", "agent_id"])
            if not nearby_cancels.is_empty():
                st.markdown(f"**{len(nearby_cancels)} 筆 cancel**")
                st.dataframe(nearby_cancels, hide_index=True, height=140)

    st.markdown("---")
    st.markdown("#### 量加權分布（按價位累積成交量）")
    fig_vp = plot_volume_profile_by_price(trades_df, presentation=presentation)
    st.plotly_chart(fig_vp, use_container_width=True, key="vol_profile")


# ---------------------------------------------------------------------------
# Tab 3: 訂單流分析
# ---------------------------------------------------------------------------

with tab3:
    st.markdown(
        "### 📌 這個 tab 告訴你什麼\n"
        "- **placed** = 新下單湧入率\n"
        "- **cancelled** = 撤單率（高於 placed = 虛假流動性）\n"
        "- **executed** = 真正成交率"
    )
    fig_rate = plot_order_arrival_rate(orders_df, cancels_df, trades_df,
                                       bin_seconds=bin_sec,
                                       presentation=presentation)
    st.plotly_chart(fig_rate, use_container_width=True, key="tab3_rate")

    st.markdown("#### 數字摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總下單", f"{len(orders_df):,}")
    c2.metric("總撤單", f"{len(cancels_df):,}")
    c3.metric("總成交", f"{len(trades_df):,}")
    if len(orders_df):
        c4.metric("撤單率", f"{len(cancels_df) / len(orders_df) * 100:.1f}%")


# ---------------------------------------------------------------------------
# Tab 4: 撤單模式
# ---------------------------------------------------------------------------

with tab4:
    st.markdown(
        "### 📌 這個 tab 告訴你什麼\n"
        "- 哪些價位、什麼時段最容易被撤單\n"
        "- 大量撤單 = information arrival 偵測\n"
        "- 反覆撤掉的 best price = spoofing / iceberg"
    )
    fig_cancel = plot_cancel_heatmap(cancels_df, bin_seconds=max(bin_sec, 10),
                                     presentation=presentation)
    st.plotly_chart(fig_cancel, use_container_width=True, key="tab4_cancel")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    f"資料來源: `{log_dir}` · "
    "abides-sim rmsc03 multi-agent simulation · "
    f"模擬時長 {total_seconds} 秒"
)
