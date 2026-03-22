# Phase 3 Calibration Report

**Method**: η estimated from 5-day large-trade impact regression; 
γ estimated as 10-min permanent-drift slope and **compared to literature priors**, NOT used directly.

All values in **bps per 1% of ADV**.


## Per-ticker results

| Ticker | ADV (M shares) | ADV source | η prior (used) | η 5-day est. | η R² | η consistent | γ prior | γ 5-day est. | γ consistent | σ bps/√s |
|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 33.8 | daily_ohlc | 8.0 | -1.73 | 0.001 | ⚠️ | 3.0 | -18.67 | ⚠️ | 0.50 |
| AMZN | 2.8 | daily_ohlc | 10.0 | +0.14 | 0.000 | ⚠️ | 4.0 | +9.66 | ✅ | 0.55 |
| AMD | 42.9 | daily_ohlc | 12.0 | -1.13 | 0.000 | ⚠️ | 5.5 | +4.65 | ✅ | 1.00 |
| TSLA | 20.4 | tick_files_5day_avg | 15.0 | +7.31 | 0.000 | ✅ | 6.0 | +139.06 | ⚠️ | 1.62 |
| NVDA | 7.7 | daily_ohlc | 12.0 | -1.28 | 0.000 | ⚠️ | 5.0 | -19.79 | ✅ | 0.83 |

## Notes (NO Silent Fallback)

- **AC parameters used downstream**: literature priors for both η and γ.
  - 5 days of tick gives R² ≈ 0 for OLS impact regression — not enough for reliable estimate.
  - This was the explicit Phase 3 plan: estimate-and-verify, use prior when 5 days too few.
- 5-day estimates are surfaced as sanity check, NOT silently substituted.
- ⚠️ markers indicate disagreement with prior by > 4× → 5 days too few for that parameter; prior used.
- TSLA missing from Jan 2020 NDX-100 daily OHLC universe → ADV computed from 5-day tick (`adv_source: tick_files_5day_avg`). Explicit, not silent.
- η range we treat as plausible: 1–50 bps per 1% ADV.
- σ estimated from intraday mid log-returns sampled every 60s.

These results feed AlmgrenChrissStrategy in scripts/run_phase3_baseline.py.