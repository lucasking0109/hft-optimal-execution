# Phase B.7 — Strategy Zoo Benchmark Report

**Generated**: 2026-05-05T14:21:17.029338-05:00 (ET)
**Sweep**: 5 tickers × 5 days × 8 strategies = 200 backtests
  (185 succeeded / 15 failed)
**Parent**: 10,000 shares sell, 10:00–11:00 ET
**Routing**: NBBO (Phase C established static SOR underperforms)

## Sign convention

- IS / VWAP slip / markout / reversion: **negative = better for sell** (sold above arrival)
- price_var, eff_spread, hit_ratio: lower is better (except hit_ratio: higher is better)

## Median across 5 tickers × 5 days, ranked by IS (best first)

| Strategy | n | IS (bps) | IS std | VWAP slip | Eff spread | Markout 60s | Reversion 5m | Price var | POV % | Hit NBBO % |
|---|---|---|---|---|---|---|---|---|---|---|
| ac_ra | 25 | **-13.24** | 36.6 | +0.61 | 2.51 | +2.16 | -2.20 | 0.520 | 0.00 | 100 |
| toth | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | -2.02 | 0.512 | 0.00 | 100 |
| pov_auto | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | -2.02 | 0.512 | 0.00 | 100 |
| vwap_following | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | -2.02 | 0.512 | 0.00 | 100 |
| pov_5pct_cap | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | -2.02 | 0.512 | 0.00 | 100 |
| ac_rn | 25 | **-5.77** | 48.1 | +1.78 | 2.28 | +1.28 | -2.20 | 0.520 | 0.00 | 100 |
| twap | 25 | **-5.77** | 48.1 | +1.78 | 2.28 | +1.28 | -2.20 | 0.520 | 0.00 | 100 |
| cvxpy_constrained | 10 | **-0.40** | 5.5 | -7.33 | 3.20 | +2.48 | -14.08 | 0.332 | 0.01 | 100 |

## Reading the table

- **IS (Implementation Shortfall)**: cost vs arrival mid. Most negative = best execution.
- **IS std**: variability across (ticker, date). Lower = more consistent.
- **VWAP slip**: deviation from market VWAP. Closer to 0 = tracks VWAP better.
- **Eff/realized spread**: how much spread you paid/earned. Eff lower better.
- **Markout / Reversion**: post-trade price drift. Higher absolute = more info leakage.
- **Price var**: execution-window mid variance. Lower = traded in calmer regime.

## Notes

- AC-RN ≡ TWAP analytically (linear impact, λ=0); rows should match.
- POV / Tóth / CVXPY use volume profile; on quiet buckets, sizes adapt.
- 10k-share parent on liquid large-caps means impact coefficient regime is small;
  differences between sophisticated strategies (Tóth/CVXPY) and TWAP are 1-5 bps.
- All strategies fill at NBBO (best across 14 venues at each timestamp).

## Caveats

- Single 1-hour window 10:00–11:00 ET; results may differ near open/close.
- σ default 1 bps/√sec; not per-ticker calibrated (could be added in B.3 ML).
- AC-RA uses lambda_risk=1e-3 (Phase 3 default); not optimized per ticker.
- B.5 Cartea-Jaimungal and B.6 Obizhaeva-Wang deferred — high implementation cost, low marginal ROI.
