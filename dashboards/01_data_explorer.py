"""HFT Data Explorer — Phase 1 Streamlit dashboard.

Run:
    cd "/Users/lucasking/Projects/HFT Date Analysis"
    uv run streamlit run dashboards/01_data_explorer.py

Design (per plan §Phase 1 presentation guidelines):
- Default view shows AAPL × first day × full RTH so you see something useful
  immediately.
- Each tab opens with a "📌 這張圖告訴你什麼" key insight callout.
- A sidebar toggle switches Research / Demo modes (Demo = larger fonts,
  hidden sidebar after selection).
- NO silent fallback: if data load fails, show the exact error and let the
  user decide what to do.
"""

from __future__ import annotations

import time

import polars as pl
import streamlit as st

from hft.analysis.sor import (
    SORWeights,
    compute_composite_score,
    naive_volume_allocation,
    sor_score_allocation,
)
from hft.analysis.venue_metrics import (
    compute_all_venue_metrics,
    compute_venue_nbbo_share_hourly,
)
from hft.data import AVAILABLE_DATES, list_eq_taq_tickers, load_eq_taq
from hft.data.orderbook import KNOWN_CONTRACTS, load_fut_taq
from hft.viz.futures_charts import plot_aggressor_flow
from hft.viz.price_charts import (
    plot_price_volume_combined,
    plot_price_with_band,
    plot_trade_size_distribution,
)
from hft.viz.spread import plot_spread
from hft.viz.venue_heatmap import plot_venue_heatmap, plot_venue_summary_bar
from hft.viz.venue_routing import (
    plot_adverse_selection_bar,
    plot_composite_score_bar,
    plot_venue_score_heatmap,
)
from hft.viz.volume_profile import plot_intraday_volume

st.set_page_config(
    page_title="HFT Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Caching helpers — Streamlit's @cache_data hashes args, so we feed it strings.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="載入股票 tick 資料中…", max_entries=10)
def cached_eq_taq(ticker: str, date: str) -> pl.DataFrame:
    return load_eq_taq(ticker, date)


@st.cache_data(show_spinner="載入期貨 tick 資料中…", max_entries=10)
def cached_fut_taq(contract: str, date: str) -> pl.DataFrame:
    return load_fut_taq(contract, date)


@st.cache_data(show_spinner="檢索可用 ticker…", max_entries=10)
def cached_ticker_list(date: str) -> list[str]:
    return list_eq_taq_tickers(date)


@st.cache_data(show_spinner="計算 venue metrics…", max_entries=20)
def cached_venue_metrics(ticker: str, date: str):
    df = cached_eq_taq(ticker, date)
    return compute_all_venue_metrics(df, horizon_seconds=60)


@st.cache_data(show_spinner="計算 hourly NBBO share…", max_entries=20)
def cached_venue_hourly(ticker: str, date: str):
    df = cached_eq_taq(ticker, date)
    return compute_venue_nbbo_share_hourly(df)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 HFT Data Explorer")
st.sidebar.caption("Phase 1 — 資料探索 dashboard")

mode = st.sidebar.radio(
    "顯示模式",
    options=["研究模式", "Demo 模式"],
    index=0,
    horizontal=True,
    help="Demo 模式採大字、簡潔版，適合 presentation 截圖",
)
presentation = mode == "Demo 模式"

data_type = st.sidebar.radio("資料類型", ["Equity (股票)", "Futures (期貨)"], index=0)

date = st.sidebar.selectbox("日期", AVAILABLE_DATES, index=0)

if data_type.startswith("Equity"):
    try:
        all_tickers = cached_ticker_list(date)
    except Exception as e:
        st.sidebar.error(f"無法列出 ticker：{e}")
        st.stop()
    default_idx = all_tickers.index("AAPL") if "AAPL" in all_tickers else 0
    ticker = st.sidebar.selectbox("Ticker", all_tickers, index=default_idx)
    fut_timezone = "ET"  # not used for equity, but keep var defined
else:
    ticker = st.sidebar.selectbox("Contract", list(KNOWN_CONTRACTS), index=0)
    fut_timezone = st.sidebar.radio(
        "期貨時區顯示",
        options=["ET (美東)", "CT (CME 原生)"],
        index=0,
        help="ET 比 CT 快 1 小時。CME 原始資料是 CT，但 dashboard 預設顯示 ET",
    ).split()[0]

rth_only = st.sidebar.checkbox(
    "只看 RTH（9:30–16:00）",
    value=True,
    help="關閉後會看到盤前 4:00–9:30 與盤後 16:00–20:00",
)

bin_minutes = st.sidebar.slider("Volume 分桶（分鐘）", min_value=1, max_value=30, value=5)
heatmap_bin = st.sidebar.slider("交易所熱力圖分桶（分鐘）", min_value=5, max_value=60, value=15)

# Sample-every for plotting (controls dashboard responsiveness)
sample_every = st.sidebar.slider(
    "繪圖採樣間隔（取每 N 筆）",
    min_value=1, max_value=2000, value=300,
    help="原始 5M 筆 plotly 跑不動；越大越快但較粗略",
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
st.title(f"📊 {ticker} — {date_label}")

# Load data
load_start = time.perf_counter()
try:
    if data_type.startswith("Equity"):
        df = cached_eq_taq(ticker, date)
    else:
        df = cached_fut_taq(ticker, date)
except FileNotFoundError as e:
    st.error(f"❌ 找不到檔案：{e}")
    st.stop()
except Exception as e:
    st.error(f"❌ 載入失敗（**NO Silent Fallback**：直接呈報原因，請決定下一步）\n\n```\n{e}\n```")
    st.stop()
load_elapsed = time.perf_counter() - load_start

st.caption(
    f"載入時間 {load_elapsed:.2f} 秒 · 共 {len(df):,} 筆事件 · "
    f"{'盤中時段' if rth_only else '全日含盤前盤後'}"
)

# ---------------------------------------------------------------------------
# Equity tabs
# ---------------------------------------------------------------------------

if data_type.startswith("Equity"):
    tab1, tab2, tab3, tab5, tab4 = st.tabs(
        ["📈 價格 & 量", "📐 微結構（價差）", "🏢 多交易所占比",
         "🧭 SOR 路由分析", "🔬 大單分布（stretch）"]
    )

    with tab1:
        st.markdown(
            "### 📌 這個 tab 告訴你什麼\n"
            "- **上半**：mid price 走勢 + 半透明買賣帶（藍/紅）+ 綠點為實際成交\n"
            "- **下半**：每 5 分鐘成交量 → **U-shape**（開盤爆量 → 中午低谷 → 收盤回升）\n"
            "- 兩張圖共享時間軸，可以直觀對應「這時段的價格動 vs 量」"
        )
        show_trades = st.checkbox("顯示成交點（綠點，可能讓圖較雜）", value=False)
        fig = plot_price_volume_combined(
            df, ticker, date,
            rth_only=rth_only, sample_every=sample_every,
            bin_minutes=bin_minutes, show_trades=show_trades,
            presentation=presentation,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(
            "### 📌 這個 tab 告訴你什麼\n"
            "- 買賣價差越窄 → 你拋單時跨價成本越低\n"
            "- 開盤、收盤、新聞發佈時 spread 會擴大 → 拋單避開這些時點\n"
            "- 中位數 vs P95 落差越大 → 流動性越不穩"
        )
        fig = plot_spread(df, ticker, date, rth_only=rth_only,
                          sample_every=sample_every, presentation=presentation)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown(
            "### 📌 這個 tab 告訴你什麼\n"
            "- 14 個交易所各自的成交量占比\n"
            "- 越分散 → smart order routing 益處越大\n"
            "- 看哪些 venue 在哪些時段最活躍 → Phase 4 SOR 的設計依據"
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = plot_venue_heatmap(df, ticker, date,
                                     bin_minutes=heatmap_bin, rth_only=rth_only,
                                     presentation=presentation)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = plot_venue_summary_bar(df, ticker, date, rth_only=rth_only,
                                          presentation=presentation)
            st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.markdown(
            "### 📌 這個 tab 告訴你什麼（Phase 4）\n"
            "- 比較 14 個交易所在這檔股票的執行品質\n"
            "- **Composite score**：高 = 適合 routing（融合流動性、深度、逆向選擇）\n"
            "- **Naive vs SOR**：對比簡單依量分配 vs 智慧分配\n"
            "- ⚠️ FINRA 是場外成交回報，不可路由（已標灰排除）"
        )
        try:
            metrics = cached_venue_metrics(ticker, date)
            scored = compute_composite_score(metrics)

            c1, c2 = st.columns(2)
            with c1:
                fig_score = plot_composite_score_bar(scored, ticker, date,
                                                     presentation=presentation)
                st.plotly_chart(fig_score, use_container_width=True)
            with c2:
                fig_adv = plot_adverse_selection_bar(metrics, ticker, date,
                                                     presentation=presentation)
                st.plotly_chart(fig_adv, use_container_width=True)

            st.markdown("#### 各小時 NBBO 提供率（誰在何時設 best price）")
            hourly = cached_venue_hourly(ticker, date)
            fig_heat = plot_venue_score_heatmap(hourly, ticker, date,
                                                presentation=presentation)
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("#### 配置對比 (top-5)")
            naive = naive_volume_allocation(metrics, top_k=5)
            sor = sor_score_allocation(scored, top_k=5, min_volume_pct=3.0)
            alloc_rows = []
            for v in sorted(set(naive) | set(sor), key=lambda v: -naive.get(v, 0)):
                alloc_rows.append({
                    "venue": v,
                    "naive (%)": f"{naive.get(v, 0)*100:.1f}",
                    "sor (%)": f"{sor.get(v, 0)*100:.1f}",
                    "diff (pp)": f"{(sor.get(v,0)-naive.get(v,0))*100:+.1f}",
                })
            st.dataframe(pl.DataFrame(alloc_rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Venue analysis 失敗：{e}")

    with tab4:
        st.markdown(
            "### 📌 這個 tab 告訴你什麼（stretch goal）\n"
            "- 單筆成交量分布；長尾大單會推升市場衝擊\n"
            "- P90 / P99 標記極端大單頻率"
        )
        fig = plot_trade_size_distribution(df, ticker, date, presentation=presentation)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Futures tabs
# ---------------------------------------------------------------------------

else:
    tab1, tab2 = st.tabs(["📈 價格 + Aggressor", "ℹ️ Notes"])
    with tab1:
        st.markdown(
            "### 📌 這個 tab 告訴你什麼\n"
            "- **上半**：成交價走勢\n"
            "- **下半**：累積 aggressor 訊號 → 綠色上升 = 主動買壓累積\n"
            "  - 訊號加速向上 → 趨勢買壓建立\n"
            "  - 訊號掉頭 → 賣壓接管"
        )
        fig = plot_aggressor_flow(df, ticker, date, sample_every=sample_every,
                                  presentation=presentation, timezone=fut_timezone)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(
            "### 期貨 Tick 資料說明\n"
            "- `TRADE AGRESSOR ON BUY` / `TRADE AGRESSOR ON SELL` 直接標記主動方\n"
            "- 訂單簿重建（`reconstruct_book`）目前在跨會話陳舊 quotes 下會回報 `is_crossed`\n"
            "  → Phase 1 EDA 之後會深入研究正確的 reconstruction\n"
            "- `EMPTY BOOK FINAL` 事件可能標示 session 邊界，是後續可探索的線索"
        )

# Footer
st.markdown("---")
st.caption(
    "📂 Phase 1 / 5 個 v1 必做模組已完成（價格&量、價差、多交易所、aggressor、stretch 大單分布）。"
    "Dashboard 載入慢請調大 sidebar「繪圖採樣間隔」。"
)
