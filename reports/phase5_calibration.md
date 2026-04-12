# Phase 5B Calibration Report

**Real benchmark**: AAPL 2020-01-13 (full RTH)

- Real spread bps median: **0.64**
- Real spread bps P95: 1.28
- Real trade size median: 71
- Real excess kurtosis: 16.4
- Real Hill tail index: 3.85
- Real vol_autocorr lag 1 / 50: 0.269 / 0.137

## Calibration grid (9 cells)

| Cell | Noise | Fund vol | MM | Success | Total dist | Kurtosis ratio | Hill ratio | Spread KS | Trade KS | Autocorr L2 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 1e-05 | timid | ✅ | **2.870** | 2.90 | 0.23 | 0.853 | 0.538 | 1.918 | 2.3 |
| 2 | 500 | 1e-05 | aggressive | ✅ | **2.198** | 1.04 | 0.34 | 0.525 | 0.575 | 2.232 | 2.0 |
| 3 | 500 | 1e-03 | timid | ✅ | **2.963** | 5.62 | 0.20 | 0.558 | 0.525 | 1.777 | 1.7 |
| 4 | 500 | 1e-03 | aggressive | ✅ | **2.144** | 1.34 | 0.45 | 0.574 | 0.574 | 2.240 | 2.2 |
| 5 | 25000 | 1e-05 | timid | ❌ | — | — | — | — | — | — | 54.6 |
| 6 | 25000 | 1e-05 | aggressive | ✅ | **2.210** | 0.43 | 0.56 | 0.506 | 0.580 | 2.410 | 56.0 |
| 7 | 25000 | 1e-03 | timid | ❌ | — | — | — | — | — | — | 68.7 |
| 8 | 25000 | 1e-03 | aggressive | ✅ | **2.723** | 0.15 | 1.13 | 0.567 | 0.581 | 2.169 | 76.6 |
| 9 | 5000 | 1e-04 | balanced | ✅ | **3.714** | 7.28 | 0.08 | 0.627 | 0.580 | 1.697 | 5.2 |

## Best cell

- **Cell #4**: noise=500, fund_vol=1e-03, mm=aggressive
- Total calibration distance: **2.144**
- Distances: {'kurtosis_ratio': 1.3382138831471, 'hill_ratio': 0.4545625110712045, 'spread_ks_dist': 0.5742308269115275, 'spread_ks_pvalue': 0.0, 'spread_wass': 8.37965683314914, 'trade_size_ks_dist': 0.5741611271426535, 'trade_size_ks_pvalue': 0.0, 'autocorr_l2': 2.2400402091845724}

🟡 **MARGINAL** — total distance 1.5-2.5 → 用但 Phase 6 報告需 sim-to-real gap caveat

## Notes

- 每個 cell 跑 2-min abides simulation（速度優先；Phase 5C 換成 30-min real episodes）
- 真實 AAPL 也是 2-min slice (09:30-09:32) for fair comparison
- 距離公式：sqrt of weighted squares of [|log kurt ratio|, |log hill ratio|, 2×spread KS, 2×trade KS, 0.5×autocorr L2]