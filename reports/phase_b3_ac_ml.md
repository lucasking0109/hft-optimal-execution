# Phase B.3 — ACML (AC + ML-predicted η) Eval Report

**Generated**: 2026-05-12T22:49:44.870746-05:00 (ET)
**Sweep**: 210 done / 15 failed across 5 tickers × 5 days × 9 strategies
**Parent**: 10,000 shares sell 10:00-11:00 ET, NBBO routing

## Aggregate (median across 5 tickers × 5 days), ranked by IS

| Strategy | n | Median IS | IS std | Median VWAP slip | Median eff spread |
|---|---|---|---|---|---|
| ac_ra | 25 | **-13.24** | 36.6 | +0.61 | 2.51 |
| ac_ml | 25 | **-9.38** | 43.2 | -0.79 | 2.41 |
| ac_rn | 25 | **-5.77** | 48.1 | +1.78 | 2.28 |
| twap | 25 | **-5.77** | 48.1 | +1.78 | 2.28 |
| vwap_following | 25 | **-4.51** | 44.6 | +0.96 | 2.10 |
| pov_auto | 25 | **-4.51** | 44.6 | +0.96 | 2.10 |
| pov_5pct_cap | 25 | **-4.51** | 44.6 | +0.96 | 2.10 |
| toth | 25 | **-4.51** | 44.6 | +0.96 | 2.10 |
| cvxpy_constrained | 10 | **-0.13** | 8.4 | -6.24 | 3.21 |

## ACML vs static AC-RA (key comparison)

- **Median IS Δ**: +3.860 bps
- **Std change**: +17.8%
- **Verdict**: 🔴 LOSE — ACML worse; static η preferred

## Caveats

- 5-day training data is small for ML; OOS R² known to be marginal.
- ACML predicts η once at parent.start_ns (not per child); per-child
  prediction would be closer to ideal but adds compute.
- σ default 1 bps/√sec; not per-ticker calibrated.
- Lambda_risk=1e-3 same for ACML and static AC-RA for fair comparison.
