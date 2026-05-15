# Phase C — SOR Routing Comparison

**Generated**: 2026-05-12T22:51:19.942446-05:00 (ET)
**Sweep**: 5 tickers × 5 dates × 4 strategies × 4 routings = 400 backtests
**Parent order**: 10,000 shares, sell, 10:00–11:00 ET
**Runs done / skipped / failed**: 400 / 0 / 0

## Aggregate median (across 5 tickers × 5 days)

| base_strategy | routing | median IS (bps) | std IS | median VWAP slip | median eff spread |
|---|---|---|---|---|---|
| ac_ra | naive | -6.186 | 46.06 | +8.457 | 6.899 |
| ac_ra | nbbo | -13.238 | 36.65 | +0.611 | 2.514 |
| ac_ra | top1 | -13.124 | 39.22 | +2.120 | 4.175 |
| ac_ra | weighted | -13.132 | 41.92 | +2.694 | 5.799 |
| ac_rn | naive | -1.055 | 52.25 | +4.842 | 5.978 |
| ac_rn | nbbo | -5.768 | 48.09 | +1.777 | 2.279 |
| ac_rn | top1 | -4.809 | 50.70 | +2.370 | 4.002 |
| ac_rn | weighted | -3.976 | 52.01 | +2.510 | 5.168 |
| twap | naive | -1.055 | 52.25 | +4.842 | 5.978 |
| twap | nbbo | -5.768 | 48.09 | +1.777 | 2.279 |
| twap | top1 | -4.809 | 50.70 | +2.370 | 4.002 |
| twap | weighted | -3.976 | 52.01 | +2.510 | 5.168 |
| vwap_following | naive | -1.729 | 55.35 | +7.098 | 7.405 |
| vwap_following | nbbo | -4.511 | 44.63 | +0.959 | 2.102 |
| vwap_following | top1 | -4.260 | 47.28 | +2.215 | 3.357 |
| vwap_following | weighted | -4.035 | 51.05 | +4.397 | 5.017 |

## SOR improvement vs NBBO baseline (median IS Δbps)

Positive = SOR routing better (less negative median IS = less slippage)

| base_strategy | naive | top1 | weighted |
|---|---|---|---|
| twap |  +4.713 |  +0.958 |  +1.791 |
| vwap_following |  +2.782 |  +0.251 |  +0.476 |
| ac_rn |  +4.713 |  +0.958 |  +1.791 |
| ac_ra |  +7.052 |  +0.113 |  +0.105 |

## Caveats

- Small parent (10k shares) on liquid large-caps: SOR effects expected to be small (<1 bps).
- 'top1' uses single highest-score venue per (ticker,date); could exhaust depth at large size.
- Venue fallback to NBBO occurs when target venue has stale quote.