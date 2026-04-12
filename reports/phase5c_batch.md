# Phase 5C — Batch 100-Episode Sanity Report

**Generated**: 2026-05-04T16:18:39.634944-05:00 (ET)
**Anchor**: Stage 4 cell 404 (noise=350, OBI=50, herder=50, max_size=7, fund_vol=1e-3)
**Sim window**: 5 min (09:30:00 – 09:35:00)
**Episodes**: 89/100 successful
**Total runtime**: 34.8 min

## Stylized facts distribution across 100 synthetic 5-min episodes

| Metric | Median | IQR (25-75%) | Min – Max |
|---|---|---|---|
| excess_kurtosis | 58.224 | [45.662, 114.899] | [15.971, 298.978] |
| hill_tail_index | 1.148 | [0.892, 1.739] | [0.327, 57.113] |
| vol_autocorr_lag_1 | 0.024 | [0.001, 0.103] | [-0.020, 0.676] |
| vol_autocorr_lag_10 | 0.686 | [0.372, 0.830] | [-0.015, 0.938] |
| vol_autocorr_lag_50 | 0.223 | [0.038, 0.394] | [-0.008, 0.829] |
| spread_bps_median | 1.305 | [0.561, 2.964] | [0.044, 18.338] |
| spread_bps_p95 | 221.745 | [49.847, 417.745] | [1.680, 1060.870] |
| trade_size_median | 5.000 | [5.000, 6.000] | [3.000, 7.000] |
| trade_size_p95 | 31.000 | [26.000, 34.000] | [9.000, 39.000] |
| n_returns | 299.000 | [299.000, 299.000] | [299.000, 299.000] |
| n_trades | 16552.000 | [15722.000, 18210.000] | [15068.000, 25244.000] |

## Reference: real AAPL 5-min benchmark IQR (from Step 0 bootstrap)

| Metric | Median | IQR | Synth match? |
|---|---|---|---|
| excess_kurtosis | real=5.672 | real IQR=[4.279, 7.355] synth median=58.224 | ⚠️ |
| vol_autocorr_lag_1 | real=0.065 | real IQR=[0.033, 0.112] synth median=0.024 | ⚠️ |
| vol_autocorr_lag_10 | real=0.038 | real IQR=[-0.020, 0.097] synth median=0.686 | ⚠️ |
| spread_bps_median | real=0.639 | real IQR=[0.635, 0.962] synth median=1.305 | ⚠️ |
| trade_size_median | real=54.000 | real IQR=[50.000, 68.250] synth median=5.000 | ⚠️ |

## Disk

- 89 parquet files, total 0.1 MB

## Next: Phase 5D
Strict KS + Wasserstein + Bonferroni vs real 5-min slices → pass means合成 episodes 統計上接近真實，可用於 Phase 6 RL 訓練。