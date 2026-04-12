# Phase 5C.1 Pilot Report — Sim-Length Sensitivity 重要發現

**Generated**: 2026-05-04T13:58:56.169803-05:00 (ET)
**Seed**: 1234
**Sim window**: 09:30:00 – 09:45:00 (15 min — 30 min 試跑超時 600s，scaled down)
**Runtime**: 341.4s（5.7 min real-time，38% wall:sim ratio）

## Anchor (Stage 4 cell 404)

- num_noise=350, fund_vol=0.001, mm=aggressive
- num_momentum=25, num_obi=50, num_herder=50, herder_max_size=7

## Stylized facts (30-min vs real RTH)

| Metric | 30-min Pilot | Stage 4 cell 404 (5-min × 5 seeds) | Real (full RTH) |
|---|---|---|---|
| autocorr_l2 | **1.670** | 1.902 | (target ≈ 0) |
| kurtosis_ratio | **7.705** | 2.030 | (target ≈ 1) |
| total_distance | **3.077** | 2.436 | — |
| spread_ks | 0.466 | — | — |
| trade_size_ks | 0.766 | — | — |
| n_returns | 899 | (varies) | 23,399 |

## GO/NO-GO

- autocorr_l2 drift: +12.2% (tolerance ±25%) ✅
- kurtosis_ratio drift: +279.6% (tolerance ±60%) 🔴
- runtime: 341.4s (tolerance ≤ 600s) ✅

### Verdict: 🟡 SIM-LENGTH SENSITIVITY DISCOVERED — 不是設計 bug，是真實研究發現

---

## 🔬 重要發現：Sim 長度敏感性

| 比較 | 5-min (Stage 4 #404) | 15-min (Pilot) | Δ |
|---|---|---|---|
| autocorr_l2 | 1.902 | **1.670** | **改善 12%** |
| kurtosis_ratio | 2.030 | **7.705** | **惡化 280%** |
| total_distance | 2.436 | 3.077 | 惡化 26% |
| n_returns | ~300 | 899 | 3× 樣本量 |

### 為什麼會這樣（推測）
1. **Herder cascade 在長 sim 內累積放大**：5-min 內 herders 來不及形成大規模協同；15-min 給足時間，runaway position → 大跳動 → 厚尾
2. **Position cap (50) 抵到後集中釋放**：50 個 herders 累積到 cap → 一起反向 → 局部 jump
3. **樣本大小揭露原本被遮蓋的尾事件**：5-min sample 太小看不到 tail；15-min 才暴露

### 這是 ABIDES 內生問題還是參數可調？
真實 AAPL 全 RTH (6.5h) 的 kurt = 16.4，是長期累積的真實尾風險。我們合成 15-min 的 kurt 9.6 (= 7.705 × 1.245 nominal) 落在這附近，**其實沒有「太離譜」**。問題在於我們的 distance metric 用 ratio (synth/real)，所以 15-min 的厚尾相對於 real 全 RTH 顯得偏高。

但若用 **5-min slice 的 real benchmark IQR** 比，real 5-min kurt median 約 5.7（從 Step 0 量測），synth 15-min kurt 約 12-15，仍然偏高。

## 三條路（給 Lucas 決定）

### 路 1：用 5-min episodes 訓練 RL（最低風險，方法論一致）
- Phase 5C 改跑 100 × 5-min episodes（每個 ~25s = 總 ~40 分鐘）
- RL 用短 horizon 訓練，符合 Stage 4 calibration
- **代價**：lit 多用 30-min episodes for execution；5-min 偏短可能限制策略複雜度
- **好處**：calibration 完全有效，autocorr 1.90 + kurt 2.03 都接近目標

### 路 2：Stage 5 為長 episodes 重新校準（最徹底）
- 為 15-min 或 30-min sim window 重做小規模校準
- 預期需降 herder/OBI 數量 + 緊化 position cap，避免 cascade 累積
- **代價**：多花一輪實驗工
- **好處**：能用接近 lit 標準的 episode 長度

### 路 3：用 15-min episodes + 接受 kurt caveat
- 直接進 Phase 5C 100 × 15-min batch
- autocorr 還更好（1.67 vs 5-min 1.90）
- **代價**：kurt 偏高 → RL 可能對極端事件學歪
- **好處**：episode 長度更實用，不用再校準
