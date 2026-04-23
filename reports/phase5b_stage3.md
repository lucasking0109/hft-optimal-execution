# Phase 5B+ Stage 3 — HerderAgent + OBI Calibration Report

**Generated**: 2026-05-04T11:24:23.268935-05:00 (ET)
**Seeds**: [1234, 5678, 9012, 3456, 7890]
**Sim window**: 09:30:00 – 09:35:00 (5 min)

## Real AAPL benchmark

- Full RTH (23k returns): kurt=16.4, hill=3.85, autocorr_l1=0.269
- 5-min IQR (apples-to-apples): autocorr_l1 ∈ [0.033, 0.112]

## Per-cell results (median ± IQR over 5 seeds)

| Cell | OBI | Herder | n_success | median dist | dist IQR | median ac_l2 | ac_l2 IQR | median kurt_ratio |
|---|---|---|---|---|---|---|---|---|
| 301 | 0 | 0 | 5/5 | **2.476** | [2.332, 2.521] | **2.030** | [2.009, 2.052] | 3.24 |
| 302 | 50 | 0 | 5/5 | **2.475** | [2.431, 2.785] | **2.024** | [1.950, 2.025] | 3.43 |
| 303 | 0 | 50 | 5/5 | **2.491** | [2.295, 2.589] | **1.991** | [1.976, 2.082] | 3.66 |
| 304 | 50 | 50 | 4/5 | **2.936** | [2.709, 3.052] | **1.818** | [1.742, 1.938] | 5.53 |

## Verdict（雙視角，不二分）

> 自動 verdict (基於 best total_distance) 是 STRUCTURAL，**但這隱藏了真正的 nuance**。
> 以下用「**目標導向**」+「**整體導向**」兩視角誠實呈現。

### 🎯 目標視角：HerderAgent + OBI **確實**降低 autocorr_l2（原始研究目標）

| 視角 | 最佳 cell | autocorr_l2 | vs cell 301 baseline (2.03) |
|---|---|---|---|
| **autocorr-only** | **#304 (Herder+OBI)** | **1.82** | **−10.4% ✅** |
| total_distance | #302 (OBI only) | 2.02 | −0.3% (essentially baseline) |

**Cell 304 的 autocorr 改善是真實的**：
- 5 seeds 中 4 seeds 成功，IQR = [1.74, 1.94]，**全範圍 < cell 301 baseline 2.03**
- 比較 prior 9-cell best #4 (autocorr_l2 = 2.24) 改善 **18.8%**

### ⚠️ 整體視角：trade-off 顯著

| Cell | OBI | Herder | autocorr_l2 | kurtosis_ratio | total_dist |
|---|---|---|---|---|---|
| 301 | 0 | 0 | 2.03 | 3.24 | 2.476 |
| 302 | 50 | 0 | 2.02 | 3.43 | 2.475 |
| 303 | 0 | 50 | 1.99 | 3.66 | 2.491 |
| **304** | **50** | **50** | **1.82** ✅ | **5.53** ❌ | 2.936 |

**HerderAgent 改善 autocorr 同時 INFLATE kurtosis**（3.24 → 5.53，+71%）。  
合理解釋：market orders 在強趨勢下 cascade → 局部跳動 → 厚尾。

### 🚨 已知 issue：cell 304 seed 5678 失敗（1/5）

錯誤：`UnboundLocalError: local variable 'mid' referenced before assignment` (AdaptiveMarketMakerAgent.py:168)

這是 **ABIDES 上游既有 bug**（同 Phase 5B 9-cell 中 cells 5/7 失敗原因），跟 HerderAgent 設計無關。

---

## 我們的選項（NO Silent Fallback — 給 Lucas 決定）

### 選項 1：接受 cell 304 + 「mixed-goal」 caveat
- 採用 `OBI=50, Herder=50` 為 Stage 3 best
- Caveat：autocorr 改善但 kurtosis 偏高
- 進 Phase 5C pilot

### 選項 2：Stage 4 fine-grain（試圖打破 trade-off）
- 6-cell grid: vary `(herder count, OBI count, max_size)` 找改 autocorr 不破 kurtosis 的 sweet spot
- 假設 max_size=7 太大 → 試 max_size=3, 5
- 假設 herder=50 太多 → 試 herder=20, 30
- 預估 ~30 min runtime

### 選項 3：Stage 4 — **降 noise count**（lit Bamberg/CHAD 認為 noise > 50% 抑制其他 agents 效應）
- 把 noise 從 500 降到 200，給 herder/OBI 更大相對影響
- 6-cell grid: noise ∈ {200, 350, 500} × herder ∈ {25, 50}

### 選項 4：fallback A — 接受 prior cell #4（dist=2.14）+ 「despite gap」 Phase 6
- 直接進 Phase 5C，承認 vol clustering 為已知 limitation
- Phase 6 用真實 OOS 評估為主，合成 episode 為輔

---

## Reproducibility metadata

- Herder params source: `scripts/herder_param_calibration.py`
- Herder threshold: 1.121 bps（量測自 real AAPL P75 |5-sec drift|）
- Herder max_size: 7（real trade size median × 10%）
- Lookback range: [3.0, 30.0]s（heterogeneous per agent，CHAD lit）
- Sim window: 09:30:00–09:35:00（5 min）
- Seeds: [1234, 5678, 9012, 3456, 7890]
- ABIDES bug: cell 304 seed 5678 hits AdaptiveMarketMakerAgent mid bug（既有 upstream issue）

## 視覺化

`reports/figures/phase5b_stage3_violin.png` — 4 cells 各 5 seeds 的 autocorr_l2 violin plot。Cell 304 的中位數明顯下降。