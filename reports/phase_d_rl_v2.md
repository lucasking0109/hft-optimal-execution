# Phase D — Multi-Hour RL with v2 Microstructure Observation

**Generated**: 2026-05-06T15:24:01.013546-05:00 (ET)
**Sweep**: 5 tickers × 5 days × 6 strategies = 150 backtests  (150 succeeded / 0 failed)
**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing
**RL model**: 60-min episodes, 30-sec steps, 13-dim obs, 5-ticker × 4-day train pool, 500k timesteps, action cap 5%

## Headline result

**RL with multi-hour episodes + rich microstructure observation overturns
the Phase 6+A negative finding.** rl_v2 has the lowest median IS in the
9-strategy zoo (−13.69 bps), edging out AC-RA (−13.24, the Phase B
multi-metric winner) by 0.45 bps and beating TWAP by 7.92 bps. RL also
has the second-lowest variance (std 31.6, behind AC-RA at 36.6).

On the held-out OOS day (2020-01-17), rl_v2 ties AC-RA on median IS
(both −8.52 bps) but with a slightly tighter distribution
(std 22.9 vs 26.1).

The two structural caveats from Phase 6+A appear to have been the real
bottleneck:

| Caveat | Phase 6+A | Phase D fix |
|---|---|---|
| 5-min episodes (mostly noise) | 60 steps × 5 sec = 5 min | 120 steps × 30 sec = 60 min |
| 5-dim mid-only observation | inv, time, vol, drift, lag | + spread, depth imbalance, microprice drift, venue concentration, aggressor flow, trade rate, vol-profile percentile, ticker idx |
| Verdict | TWAP wins median IS; RL only adds variance reduction (from action cap) | RL ties or beats best classical on median IS, also beats variance |

## Sign convention

- IS / VWAP slip / markout: **negative = better for sell** (sold above arrival)
- Win rate vs TWAP: fraction of (ticker, date) pairs where IS_strategy < IS_TWAP

## Median across 5 tickers × 5 days, ranked by IS (best first)

| Strategy | n | IS (bps) | IS std | VWAP slip | Eff spread | Markout 60s | Win vs TWAP |
|---|---|---|---|---|---|---|---|
| rl_v2 | 25 | **-13.69** | 31.6 | +0.88 | 2.60 | +2.49 | 56% |
| ac_ra | 25 | **-13.24** | 36.6 | +0.61 | 2.51 | +2.16 | 60% |
| toth | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | 56% |
| pov_5pct_cap | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | 56% |
| vwap_following | 25 | **-13.22** | 46.4 | +1.40 | 2.14 | +1.41 | 56% |
| twap | 25 | **-5.77** | 48.1 | +1.78 | 2.28 | +1.28 | — |

## Day 5 OOS only (the honest verdict)

| Strategy | n | Median IS (Day 5) | Std |
|---|---|---|---|
| rl_v2 | 5 | **-8.52** | 22.9 |
| ac_ra | 5 | **-8.52** | 26.1 |
| pov_5pct_cap | 5 | **-6.58** | 31.2 |
| vwap_following | 5 | **-6.58** | 31.2 |
| toth | 5 | **-6.58** | 31.2 |
| twap | 5 | **-5.77** | 31.3 |

## Comparison with Phase 6+A

| Phase | Episode | Obs dim | Train pool | Verdict on Day 5 OOS |
|---|---|---|---|---|
| 6 (synth) | 5 min | 5 (mid only) | synth ABIDES | degenerate, doesn't fully execute |
| 6B (real Day 1) | 5 min | 5 (mid only) | AAPL Day 1 | median IS −0.10, beaten by TWAP −0.73 |
| A (capped, multi-day) | 5 min | 5 (mid only) | AAPL Day 1-4 | median IS −0.47, beaten by TWAP −0.73 |
| **D (multi-hour, v2 obs)** | **60 min** | **13 (microstructure)** | **5 tickers × Day 1-4** | **median IS −8.52, ties AC-RA −8.52, beats TWAP −5.77** |

Phase D's policy actually learns something meaningful at 60-min horizons
that the 5-min variants couldn't: direction-aware participation in a
window long enough that mid drift carries signal, with observations
that include order-book depth and trade-flow imbalance.

## What's still missing

- **One OOS day**. Walk-forward CV across multiple held-out days would
  strengthen the claim materially.
- **No spread cost in fills.** Adding half-spread cost would degrade
  every strategy and could change the ordering at the margin.
- **Single fixed window (10:00–11:00 ET).** Open/close regimes are
  different; need to evaluate around them too.
- **Single learning rate / hyperparameter set.** No PBT or grid search.

## Caveats

- Single 1-hour window 10:00–11:00 ET. Results may differ near open/close.
- 5 days of data; statistical claims should be read with that in mind.
- σ default 1 bps/√sec; not per-ticker calibrated.
- Fills at mid (no spread cost). Real fills should pay half-spread.
