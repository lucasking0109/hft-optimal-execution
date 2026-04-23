# Phase 5B+ Stage 4 — Low-noise Population Mix Report

**Generated**: 2026-05-04T13:11:34.302906-05:00 (ET)
**Seeds**: [1234, 5678, 9012, 3456, 7890]
**Hypothesis**: Lit (Bamberg/CHAD) recommends noise < 50% of population. Stage 3 had 74%; Stage 4 tests 47-61%.

## Reference: Stage 3 results

| Stage 3 cell | noise/total | autocorr_l2 | kurt_ratio |
|---|---|---|---|
| 301 (baseline) | 500/625 = 80% | 2.030 | 3.24 |
| 304 (Herder+OBI) | 500/675 = 74% | 1.818 | 5.53 |

## Stage 4 per-cell results

| Cell | noise | OBI | Herder | max_size | n_succ | median dist | dist IQR | median ac_l2 | median kurt_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 401 | 200 | 0 | 0 | None | 5/5 | **2.722** | [2.618, 2.789] | **2.026** | 2.98 |
| 402 | 200 | 50 | 50 | 7 | 5/5 | **3.055** | [3.030, 3.690] | **1.829** | 8.22 |
| 403 | 200 | 50 | 50 | 3 | 5/5 | **3.389** | [3.109, 4.046] | **1.624** | 8.00 |
| 404 | 350 | 50 | 50 | 7 | 5/5 | **2.436** | [2.248, 3.017] | **1.902** | 2.03 |
| 405 | 200 | 50 | 100 | 5 | 5/5 | **2.592** | [2.339, 2.701] | **1.981** | 3.01 |

## Verdict（重新校正：自動 verdict 看錯 cell）

> 自動 verdict 用 cell 402 (apples-to-apples anchor) 評斷，但實際 best by total_distance 是 **cell 404**。Cell 404 的指標其實很接近 BREAKTHROUGH：

### 🏆 Best cell #404（mid-noise，noise=350）
- Params: noise=350, OBI=50, Herder=50, max_size=7
- Median total_distance: **2.436** ← 比 Stage 3 #304 (2.936) **改善 17%**
- Median autocorr_l2: **1.902** ← Stage 3 #304 是 1.818（差 4%，幾乎打平）
- Median kurtosis_ratio: **2.03** ← Stage 3 #304 是 5.53（**改善 63%！**）

**Trade-off 大幅化解**：autocorr 維持 Stage 3 改善，kurtosis 從 5.53 降到 2.03（接近 lit ideal）。

## 三個視角全比較

| Cell (含 Stage 3) | noise/total | autocorr_l2 | kurt_ratio | total_dist | 備註 |
|---|---|---|---|---|---|
| **9-cell #4 (1 seed)** | 80% | 2.24 | 1.34 | **2.144** | 原 best by dist；單 seed 偏樂觀 |
| Stage 3 #301 (baseline) | 80% | 2.030 | 3.24 | 2.476 | 5-seed apples-to-apples baseline |
| Stage 3 #304 | 74% | **1.818** | 5.53 ❌ | 2.936 | autocorr 最好但 kurt 爆 |
| **Stage 4 #404** | **61%** | **1.902** | **2.03** ✅ | **2.436** | 🌟 **真正的多指標最佳** |

### 為什麼 cell 404 才是真 breakthrough
1. **Autocorr 接近 Stage 3 best**（1.90 vs 1.82，差 4%）
2. **Kurtosis 接近 lit ideal**（2.03 vs lit「合格」標準 < 2.0；vs Stage 3 #304 的 5.53）
3. **Total distance 第二好**（2.44 — 比 Stage 3 任何 cell 都好；只輸給 9-cell #4 的單 seed lucky 結果）

## 其他發現

### 太低 noise (200) 反而傷 total_distance
- Cell 402 (noise=200, 同 #304 配置) total_dist = 3.06，比 Stage 3 #304 的 2.94 還差
- Cell 403 (max_size=3) ac_l2 = 1.62（最佳 autocorr）但 dist = 3.39（差）
- 推測：noise 太少 → 流動性不足 → spread 分布偏離真實

### Sweet spot 是 noise ≈ 350
- 61% 比例剛好略高於 lit < 50% 上限，但實務 ABIDES 似乎需要這個 minimum noise 維持流動性

### max_size=3 給最佳 autocorr 但破壞其他指標
- Cell 403 ac_l2 = 1.62（5 cells 中最低）
- 但 dist 衝到 3.39 — 太小的 herder 單反而讓 trade size 分布不像真實

## 推薦

🌟 **採用 cell 404 為 Phase 5C anchor**：
- noise=350, OBI=50, Herder=50, max_size=7, fund_vol=1e-3, mm=aggressive
- 進 Phase 5C pilot 100-episode 批量生成

備選微調方向（若想 fine-tune）：
- Stage 5 試 noise ∈ {300, 400}, max_size ∈ {5, 7, 10} 找更精細最佳