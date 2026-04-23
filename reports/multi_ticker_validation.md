# Multi-Ticker Robustness Validation Report

**Generated**: 2026-05-04T19:12:35.802565-05:00 (ET)
**Date sampled**: 20200113
**Architecture (shared across tickers)**: cell 404 — noise=350, fund_vol=1e-3, mm=aggressive, mom=25, OBI=50, herder=50
**Per-ticker scaling**: herder_threshold_bps + herder_max_size auto-derived from real 20200113 data

## Per-ticker results

| Ticker | Rationale | Threshold (bps) | Max size | Episodes pass | Metrics in real IQR |
|---|---|---|---|---|---|
| AMZN | high-price tech ($1800) | 0.818 | 1 | 24/25 | **0/5** |
| MSFT | mid-price low-vol ($160) | 1.231 | 9 | 23/25 | **0/5** |
| NVDA | mid-price high-vol HFT favorite | 1.604 | 3 | 24/25 | **1/5** |
| TSLA | high-vol stress test | 3.183 | 3 | 23/25 | **0/5** |

## Per-ticker per-metric match (synth median vs real 5-min IQR)


### AMZN

| Metric | Synth median | Real median | Real 5-min IQR | Match |
|---|---|---|---|---|
| excess_kurtosis | 56.634 | 6.059 | [4.551, 7.935] | ⚠️ |
| vol_autocorr_lag_1 | 0.013 | 0.148 | [0.116, 0.187] | ⚠️ |
| vol_autocorr_lag_10 | 0.752 | 0.044 | [-0.023, 0.090] | ⚠️ |
| spread_bps_median | 0.754 | 2.245 | [2.093, 2.891] | ⚠️ |
| trade_size_median | 3.000 | 6.500 | [5.000, 9.250] | ⚠️ |

### MSFT

| Metric | Synth median | Real median | Real 5-min IQR | Match |
|---|---|---|---|---|
| excess_kurtosis | 64.321 | 5.358 | [3.967, 6.645] | ⚠️ |
| vol_autocorr_lag_1 | 0.004 | 0.068 | [0.018, 0.106] | ⚠️ |
| vol_autocorr_lag_10 | 0.761 | 0.024 | [-0.035, 0.045] | ⚠️ |
| spread_bps_median | 0.565 | 0.617 | [0.616, 0.618] | ⚠️ |
| trade_size_median | 6.000 | 80.500 | [67.500, 95.750] | ⚠️ |

### NVDA

| Metric | Synth median | Real median | Real 5-min IQR | Match |
|---|---|---|---|---|
| excess_kurtosis | 54.252 | 8.778 | [6.697, 11.728] | ⚠️ |
| vol_autocorr_lag_1 | 0.024 | 0.057 | [0.015, 0.106] | ✅ |
| vol_autocorr_lag_10 | 0.749 | 0.029 | [-0.034, 0.055] | ⚠️ |
| spread_bps_median | 0.929 | 1.590 | [1.485, 2.397] | ⚠️ |
| trade_size_median | 4.000 | 23.750 | [20.000, 25.250] | ⚠️ |

### TSLA

| Metric | Synth median | Real median | Real 5-min IQR | Match |
|---|---|---|---|---|
| excess_kurtosis | 42.072 | 8.458 | [5.826, 12.298] | ⚠️ |
| vol_autocorr_lag_1 | 0.047 | 0.123 | [0.061, 0.136] | ⚠️ |
| vol_autocorr_lag_10 | 0.696 | -0.010 | [-0.037, 0.065] | ⚠️ |
| spread_bps_median | 0.989 | 3.212 | [2.703, 3.700] | ⚠️ |
| trade_size_median | 4.000 | 25.500 | [25.000, 36.250] | ⚠️ |

## Total runtime: 36.2 min