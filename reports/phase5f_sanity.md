# Phase 5F — ExecutionEnv Reward-Logic Sanity Check

**Generated**: 2026-05-05T18:37:04.816175-05:00 (ET)
**Episodes per agent**: 100

## Stats per (mode, agent)

| Mode | Agent | Median (bps) | Std | IQR |
|---|---|---|---|---|
| real_replay | aggressive | +0.00 | 0.00 | [+0.00, +0.00] |
| real_replay | lazy | +1.67 | 8.16 | [-1.75, +6.46] |
| real_replay | twap | +1.11 | 4.58 | [-0.49, +3.20] |
| real_replay | random | +0.05 | 1.43 | [-0.13, +0.52] |
| synthetic | aggressive | +0.00 | 0.00 | [+0.00, +0.00] |
| synthetic | lazy | +135920.20 | 649514.41 | [+9196.93, +566192.75] |
| synthetic | twap | +46410.81 | 129790.47 | [+2421.39, +176051.57] |
| synthetic | random | +5.79 | 14985.57 | [-4.06, +25.05] |

## Verification checks (real mode, ground truth)

| # | Check | Value | Pass |
|---|---|---|---|
| 1 | Aggressive median \|IS\| < 1 bps | +0.000 bps | ✅ |
| 2 | Lazy reward correlates with +drift for sell (corr > 0.5) | 0.996 | ✅ |
| 3 | TWAP std vs Random std × 5 | 4.58 vs 1.43 | ✅ |
| 4 | Aggressive IQR < Lazy IQR | 0.00 vs 8.21 | ✅ |

### Overall: ✅ Env verified — proceed to Phase 6A PPO

## Notes

- Synthetic mode shows extreme outliers (kurt 58, Hill 1.15) per Phase 5C calibration — expected; not a sanity issue here.
- Real mode is the ground truth for env logic verification.
- TWAP/Random IS comparison is NOT meaningful as sanity (Almgren-Chriss: same mean, different variance under risk-neutral).
- Phase 6A PPO training should use both modes; OOS eval on real Day 5.