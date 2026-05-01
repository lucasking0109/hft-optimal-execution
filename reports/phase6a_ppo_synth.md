# Phase 6A — PPO baseline on synthetic episodes

**Generated**: 2026-05-05T18:40:45.392468-05:00 (ET)
**Total timesteps**: 100,000
**Episodes seen**: 3,079
**Train time**: 0.4 min
**Reward clip**: ±50.0 bps (training stability vs synth outliers)

> **Reward sign**: `reward = -step_is_bps` (PPO-friendly, post-audit fix).
> Positive reward bps = good execution; multiply by −1 to read as IS cost.

## Training reward progression

- First 100 episodes mean: +3457.67 bps
- Last 100 episodes mean: +25941.01 bps
- Improvement: +22483.34 bps

## Eval on 100 fresh synthetic episodes (deterministic policy)

- Median: **+18732.96 bps**
- Mean: +53390.84 bps
- IQR: [+2867.51, +79881.08]

## Caveats
- Synthetic episodes have known calibration gaps (kurt/autocorr/trade_size)
- Reward clip ±50.0 bps applied during training (stability)
- True quality determined by Phase 6C OOS eval on real Day 5

## Next: Phase 6B
- Train PPO on real_replay mode (real-data primary)
- Compare vs this synth-trained agent on real OOS
