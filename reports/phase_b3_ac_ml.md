# Phase B.3 — ACML (AC + ML-predicted η) Eval Report

**Generated**: 2026-05-05T17:07:56.608881-05:00 (ET)
**Sweep**: 210 done / 15 failed across 5 tickers × 5 days × 9 strategies
**Parent**: 10,000 shares sell 10:00-11:00 ET, NBBO routing

## Aggregate (median across 5 tickers × 5 days), ranked by IS

| Strategy | n | Median IS | IS std | Median VWAP slip | Median eff spread |
|---|---|---|---|---|---|
| ac_ra | 25 | **-13.24** | 36.6 | +0.61 | 2.51 |
| toth | 25 | **-13.22** | 46.4 | +1.40 | 2.14 |
| pov_auto | 25 | **-13.22** | 46.4 | +1.40 | 2.14 |
| vwap_following | 25 | **-13.22** | 46.4 | +1.40 | 2.14 |
| pov_5pct_cap | 25 | **-13.22** | 46.4 | +1.40 | 2.14 |
| ac_ml | 25 | **-9.38** | 43.2 | -0.79 | 2.41 |
| ac_rn | 25 | **-5.77** | 48.1 | +1.78 | 2.28 |
| twap | 25 | **-5.77** | 48.1 | +1.78 | 2.28 |
| cvxpy_constrained | 10 | **-0.40** | 5.5 | -7.33 | 3.20 |

## ACML vs static AC-RA (key comparison)

- **Median IS Δ**: +3.860 bps
- **Std change**: +17.8%
- **Verdict**: 🔴 LOSE — ACML worse; static η preferred

## Root cause analysis (post-audit)

The ML model under-performs static η for two compounding reasons identified
in the end-to-end audit. These are negative findings worth recording, not
regression bugs.

### 1. Severe overfit (train→OOS gap of 9×)

`rl/checkpoints/eta_ml_v0/eval_metrics.json`:

| Metric | Train | OOS (Day 5) |
|---|---|---|
| R² | 0.158 | 0.018 |
| RMSE | 4.86 | 3.08 |
| MAE | 2.98 | 1.94 |

The 9× gap between train and OOS R² is a textbook overfit signal. The
xgboost setup uses 200 trees / max_depth=6 with `subsample=0.8`,
`colsample_bytree=0.8` — defaults that are not aggressive regularisation.
**`early_stopping_rounds` is NOT set**, so trees keep being added even after
validation R² stops improving. The script has a TODO note for adding it on
the next retrain; this run is preserved as an honest "first attempt"
baseline.

### 2. Distributional shift between training and inference

The training pipeline filters trades to **top-decile by size**
(`scripts/build_eta_ml_dataset.py` L48: `PERCENTILE_THRESHOLD = 0.90`),
yielding 762,207 large-trade events. Typical training-row `size_pct_adv`
ranges roughly 0.5–2%.

At inference time, `ACMLStrategy.predict_eta` queries the model with
`per_child_qty = parent_qty / num_slices` (e.g., 10 000 / 60 ≈ 167
shares ≈ `size_pct_adv` ≈ 0.025% on AAPL). **The model is being asked to
extrapolate to a regime ~20× smaller than anything in its training
distribution.**

Feature importance confirms the model never internalised the size→impact
relationship: `size_pct_adv` ranks **lowest** at 0.097, while
`recent_volume_15m_pct_adv` (0.181) and `ticker_idx` (0.172) dominate.
A model that ignores the most theoretically relevant feature cannot
plausibly improve upon a static literature-calibrated η for tiny child
orders. This is the proximate reason ACML loses to AC-RA by 3.86 bps
median IS with 18% higher variance.

### Implication

The negative result is preserved as the production verdict: AC-RA with
literature η stays the baseline. A follow-up retrain that (a) adds
`early_stopping_rounds=20`, (b) trains on a wider size distribution
(P50+ rather than P90+), and (c) introduces walk-forward CV is the
natural next step but is out of scope here.

## Caveats

- 5-day training data is small for ML; OOS R² known to be marginal.
- **OOS evaluation uses Day 5 only (124 144 events).** Statistical power
  is weak; results could differ on Day 6 (not in dataset). Walk-forward
  CV across multiple OOS days would improve confidence but requires
  more historical data.
- ACML predicts η once at parent.start_ns (not per child); per-child
  prediction would be closer to ideal but adds compute.
- σ default 1 bps/√sec; not per-ticker calibrated.
- Lambda_risk=1e-3 same for ACML and static AC-RA for fair comparison.
- Training distribution restricted to top-decile (P90+) trade events;
  any future inference must respect this distribution or the model will
  extrapolate (see Root cause section).
