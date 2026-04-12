# Phase 5B Baseline — abides default vs real AAPL stylized facts

**Generated**: 2026-05-03
**Real**: AAPL 2020-01-13 (full RTH)
**Synth**: rmsc03 default config, 5-min smoke run

> 這是 calibration 開始前的「未校準」狀態，作為改善基準。

---

## Per-metric values

| Metric | Real AAPL | abides (rmsc03 default) | Notes |
|---|---|---|---|
| n_returns | 23,399 | 299 | Real: 5-min granularity full day; abides: 5-min sim only |
| n_spread_obs | 431,291 | 30,236 | Real has more BBO updates (14 venues) |
| n_trades | 246,275 | 12,920 | Real has 19x trades |
| **excess_kurtosis** | 248.5 | 248.7 | ✅ similar magnitude (both very fat tail) |
| **hill_tail_index** | ~3.85 | ~1.23 | abides tail too fat (alpha low) |
| **vol_autocorr lag 1** | ~0.45 | 0.009 | ⚠️ abides lacks volatility clustering |
| **vol_autocorr lag 10** | ~0.30 | 0.007 | ⚠️ |
| **vol_autocorr lag 50** | 0.137 | -0.012 | ⚠️ |
| **spread_bps median** | **0.642** | **0.100** | abides tighter most of time |
| **spread_bps P95** | 1.284 | **8.608** | abides has WIDE tail (bimodal) |
| **trade_size median** | 71 | 18 | abides trades too small |
| **trade_size P95** | 300 | 35 | abides trades too small |

---

## Distance summary

```
kurtosis_ratio:        15.21   (synth/real; aim 0.5-2.0)  ⚠️
hill_ratio:             0.32   (aim 0.5-2.0)              ⚠️
spread_ks_dist:         0.83   (aim < 0.3)                ⚠️ ← biggest issue
spread_wass:            1.35   bps unit
trade_size_ks_dist:     0.58   (aim < 0.3)                ⚠️
autocorr_l2:            1.61   (aim < 0.5)                ⚠️
```

**Total calibration distance**: **3.665**
**Target after Phase 5B**: **< 1.5** (ideal) or < 2.5 (acceptable)

---

## Diagnostic interpretation

1. **Spread distribution**: abides 是「**雙峰**」 — tight 大部分時間 + wide 偶爾。真實 AAPL 是「單峰窄」。原因：abides 的 noise agents 撤單再下單造成偶發 wide gap。
   → Calibration 目標：增加 noise agent count 讓 spread 更穩定

2. **Volatility autocorrelation**: abides 完全沒有 clustering（接近 0 在所有 lag）。真實市場 lag 50 還有 0.13。
   → 原因：abides 的價值代理（ValueAgent）反應 fundamental，但缺乏動量回饋
   → 可能需要更多 momentum agents

3. **Trade size**: abides 預設大多數成交是 noise agent 對 noise agent 的小單。
   → Calibration 目標：放大 noise agent 平均下單量參數

4. **Hill tail index**: abides 尾巴比真實更肥（alpha=1.23 vs 3.85）。**這是 fundamental volatility 設太高的訊號** — 偶發大跳動造成超肥尾。
   → Calibration 目標：降 fund_volatility

---

## 校準維度與預期效果（3D Latin hypercube）

| 維度 | 範圍 | 期望降低哪個指標 |
|---|---|---|
| `noise_agent_count` | 500 / 5000 / 25000 | 減少 spread KS（更穩定 BBO 競爭） |
| `fund_volatility` | 1e-5 / 1e-4 / 1e-3 | 降低 Hill tail 過肥（小 vol → 較少跳動） |
| `mm_aggressiveness` | timid / balanced / aggressive | 影響 spread 中位數 + 雙峰分布 |

---

## 後續

- 5B.2: 寫 `calib_aapl.py` config，加 CLI 參數
- 5B.3: 9-cell runner（subprocess driver）
- 5B.4: 跑 9 cells，輸出每 cell distance
- 5B.5: 此報告升級成完整對比（heatmap + 4 metrics × 9 cells）

**STOP 條件**：若 9 cell 全部 total distance > 2.5，STOP 呈報，給用戶 (A) 接受 caveat (B) 擴 27 cells (C) 改評估方式 三選項。
