# Phase 6B — PPO real-data primary training

**Generated**: 2026-05-05T19:37:42.418128-05:00 (ET)
**Mode**: real_replay AAPL 20200113 (78 slices)
**Total timesteps**: 100,000
**Episodes seen**: 31,421
**Train time**: 56.5 min

> **Reward sign**: `reward = -step_is_bps` (PPO-friendly, post-audit fix).
> Positive reward bps = good execution; multiply by −1 to read as IS cost.

## Training reward progression

- First 50 episodes mean: +0.166 bps
- Last 50 episodes mean: +0.495 bps
- Improvement: +0.329 bps

## Eval on 78 real_replay episodes (deterministic, in-sample)

- Median: **+0.195 bps**
- Mean: +0.027 bps
- IQR: [-0.784, +0.490]
- Mean inventory left: 0.00%
- Completion rate (<1% left): 78/78

## Comparison vs Phase 6A (synth-only)

- Phase 6A (synth-only): median = +18732.96 bps (degenerate aggressive)
- Phase 6B (real primary): median = +0.195 bps
- 6B vs 6A diff: -18732.766 bps

## Caveats
- Train + eval both on 20200113 → in-sample, optimistic. Phase 6C OOS on Day 5 = real test.
- Single-day training; multi-day mix would broaden state distribution.

## Next: Phase 6C
- OOS evaluation on Day 5 (20200117) for both 6A and 6B models
- Compare vs TWAP, VWAP-following, AC-RA on real Day 5 slices