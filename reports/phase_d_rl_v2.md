# Phase D — Multi-Hour RL with v2 Microstructure Observation

**Generated**: 2026-05-12T22:46:50.158494-05:00 (ET)
**Sweep**: 5 tickers × 5 days × 6 strategies = 150 backtests  (150 succeeded / 0 failed)
**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing
**RL model**: 60-min episodes, 30-sec steps, 13-dim obs, 5-ticker × 4-day train pool, 500k timesteps, action cap 5%

## Sign convention

- IS / VWAP slip / markout: **negative = better for sell** (sold above arrival)
- Win rate vs TWAP: fraction of (ticker, date) pairs where IS_strategy < IS_TWAP

## Median across 5 tickers × 5 days, ranked by IS (best first)

| Strategy | n | IS (bps) | IS std | VWAP slip | Eff spread | Markout 60s | Win vs TWAP |
|---|---|---|---|---|---|---|---|
| rl_v2 | 25 | **-13.69** | 31.6 | +0.88 | 2.60 | +2.49 | 56% |
| ac_ra | 25 | **-13.24** | 36.6 | +0.61 | 2.51 | +2.16 | 60% |
| twap | 25 | **-5.77** | 48.1 | +1.78 | 2.28 | +1.28 | — |
| pov_5pct_cap | 25 | **-4.51** | 44.6 | +0.96 | 2.10 | +2.14 | 52% |
| vwap_following | 25 | **-4.51** | 44.6 | +0.96 | 2.10 | +2.14 | 52% |
| toth | 25 | **-4.51** | 44.6 | +0.96 | 2.10 | +2.14 | 52% |

## Day 5 OOS only (the honest verdict)

| Strategy | n | Median IS (Day 5) | Std |
|---|---|---|---|
| rl_v2 | 5 | **-8.52** | 22.9 |
| ac_ra | 5 | **-8.52** | 26.1 |
| twap | 5 | **-5.77** | 31.3 |
| pov_5pct_cap | 5 | **-4.51** | 29.5 |
| vwap_following | 5 | **-4.51** | 29.5 |
| toth | 5 | **-4.51** | 29.5 |

## Caveats

- Single 1-hour window 10:00–11:00 ET. Results may differ near open/close.
- 5 days of data; statistical claims should be read with that in mind.
- σ default 1 bps/√sec; not per-ticker calibrated.
- Fills at mid (no spread cost). Real fills should pay half-spread.
