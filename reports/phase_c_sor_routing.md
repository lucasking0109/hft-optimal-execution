# Phase C — SOR Routing Comparison

**Generated**: 2026-05-05 (ET)
**Sweep**: 5 tickers × 5 dates × 4 strategies × 4 routings = **400 backtests**
**Parent order**: 10,000 shares, sell, 10:00–11:00 ET
**Runs done / skipped / failed**: 400 / 0 / 0
**Sweep wall time**: 2.5 min

## IS sign convention (read me first)

For SELL orders, `IS = -(executed_avg - arrival_mid) / arrival × 1e4`:
- **Negative IS = sold above arrival = profit / good execution**
- Positive IS = sold below arrival = slippage / bad execution

So **more negative is better** in the tables below.

---

## Aggregate median (across 5 tickers × 5 days = 25 cells per row)

| base_strategy | routing | median IS (bps) | std IS | median VWAP slip | median eff spread |
|---|---|---|---|---|---|
| TWAP | **nbbo** | **-5.77** | 48.09 | +1.78 | 2.28 |
| TWAP | naive | -1.06 | 52.25 | +4.84 | 5.98 |
| TWAP | top1 | -4.81 | 50.70 | +2.37 | 4.00 |
| TWAP | weighted | -3.98 | 52.01 | +2.51 | 5.17 |
| VWAP-following | **nbbo** | **-13.22** | 46.39 | +1.40 | 2.14 |
| VWAP-following | naive | -4.02 | 59.98 | +3.54 | 7.50 |
| VWAP-following | top1 | -12.63 | 49.11 | +2.07 | 3.43 |
| VWAP-following | weighted | -10.83 | 52.38 | +2.59 | 4.64 |
| AC-RN | **nbbo** | **-5.77** | 48.09 | +1.78 | 2.28 |
| AC-RN | naive | -1.06 | 52.25 | +4.84 | 5.98 |
| AC-RN | top1 | -4.81 | 50.70 | +2.37 | 4.00 |
| AC-RN | weighted | -3.98 | 52.01 | +2.51 | 5.17 |
| AC-RA | **nbbo** | **-13.24** | 36.65 | +0.61 | 2.51 |
| AC-RA | naive | -6.19 | 46.06 | +8.46 | 6.90 |
| AC-RA | top1 | -13.12 | 39.22 | +2.12 | 4.18 |
| AC-RA | weighted | -13.13 | 41.92 | +2.69 | 5.80 |

(AC-RN ≡ TWAP confirmed by identical results, as theory predicts when λ=0.)

---

## Static SOR vs NBBO baseline — IS degradation

| base_strategy | naive Δ | top1 Δ | weighted Δ |
|---|---|---|---|
| TWAP | **+4.71** | +0.96 | +1.79 |
| VWAP-following | **+9.20** | +0.59 | +2.39 |
| AC-RN | **+4.71** | +0.96 | +1.79 |
| AC-RA | **+7.05** | +0.11 | +0.10 |

**Δ = mode_IS − nbbo_IS** (positive = SOR worse than NBBO).

---

## 🔬 Key finding: **Static venue allocation underperforms NBBO**

**Counter-intuitive result, but mathematically expected**:

NBBO = "fill at whichever venue has the best price at this exact moment".
By definition, no other single venue can do better — NBBO is the upper bound.

Static SOR allocation (pre-committed venue choice based on historical scores)
necessarily picks venues that may *not* be NBBO leader at the child timestamp,
giving up some price quality.

| SOR mode | What it does | Cost vs NBBO |
|---|---|---|
| **naive** | Random by historical volume share | **+4.7 to +9.2 bps** (worst) |
| **weighted** | Random by composite score softmax | **+0.1 to +2.4 bps** |
| **top1** | All children to single highest-scored venue | **+0.1 to +1.0 bps** (best of SOR) |

`top1` is closest to NBBO because the top-scored venue (typically NASDAQ for
AAPL/AMZN/MSFT) frequently coincides with the NBBO leader. `naive` is worst
because it spreads orders across venues regardless of who's currently competitive.

**Spread / VWAP slip both worse for SOR** — confirms SOR routes to less-competitive
quotes. `naive` median effective spread ≈ 6 bps vs NBBO ≈ 2 bps.

---

## Implication: production SORs are NOT static

Real production smart-order-routers do **not** pre-commit to venue allocations.
At each child order timestamp they query venue books **right then** and route to
whoever shows the best quote. Static allocation is the wrong design abstraction.

**Phase C contribution**: quantifies the cost of static SOR vs NBBO across 400
backtests, confirming the industry practice of dynamic per-order routing.
The Phase 4 venue analysis (NBBO share / depth / adverse selection) remains
informative for *which* venues to consider, but committing to them ahead of
time is suboptimal.

---

## Caveats

- Parent size 10k shares on liquid large-caps: SOR effects are small
  (1-9 bps) — could differ for thinly-traded names or block-size orders.
- Venue fallback to NBBO triggered when target venue had no quote at
  timestamp; this acts as partial NBBO mixing for SOR modes.
- Tested on 5 days only (2020-01-13–17); may not generalize to other
  market regimes (e.g., volatile / news-heavy days).
- Adverse selection / fill rate analysis would refine this finding —
  static SOR could plausibly reduce post-fill markout even at small IS
  cost (not measured here).

---

## Files

- `reports/phase_c_sor_routing.csv` — raw 400-row results
- `reports/figures/phase_c_routing_violin.png` — IS distribution per
  (base_strategy, routing) cell

## Next: Phase B (Strategy Zoo)

Phase B will benchmark 5+ additional strategies (POV, Tóth, AC+ML, CVXPY,
RL-v1) all under NBBO routing — since Phase C established NBBO is the
correct execution benchmark for our 10k-share parent on large-caps.
