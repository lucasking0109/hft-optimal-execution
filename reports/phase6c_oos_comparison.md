# Phase 6C — Out-of-Sample Evaluation (Day 5 = 2020-01-17)

**Generated**: 2026-05-05T19:59:18.871807-05:00 (ET)
**OOS Date**: 20200117 (NEVER seen during 6A/6B training)
**Episodes per agent**: 78 (all RTH 5-min slices)

## All-strategy comparison (median + IQR + win rate vs TWAP)

| Agent | Median IS (bps) | Mean | Std | IQR | Completion | vs TWAP win |
|---|---|---|---|---|---|---|
| aggressive | **+0.000** | +0.000 | 0.00 | [+0.000, +0.000] | 100% | 44% |
| lazy | **-0.947** | -1.085 | 6.61 | [-5.055, +2.804] | 100% | 53% |
| twap | **-0.728** | -0.925 | 3.78 | [-3.309, +1.463] | 100% | — |
| random | **-0.103** | -0.188 | 1.02 | [-0.501, +0.153] | 100% | 44% |
| ppo_synth | **+0.000** | +0.374 | 1.66 | [+0.000, +0.000] | 0% | 38% |
| ppo_real | **-0.097** | -0.171 | 0.97 | [-0.581, +0.194] | 100% | 42% |
| ppo_v1_capped | **-0.465** | -0.518 | 1.80 | [-1.608, +0.552] | 100% | 44% |

## Key findings

Stats report **IS in bps; lower is better** (positive = cost vs arrival).

- **PPO-6B (real-trained) vs TWAP**: median diff = **+0.632 bps** (negative = RL lower IS), win rate 42% — 🔴 TWAP beats RL
- **PPO-6A (synth-trained) vs TWAP**: median diff = +0.728 bps
- **PPO-6B vs PPO-6A**: median diff = **-0.097 bps** (negative = real-trained better than synth-trained)

## Caveats

- Single OOS day (20200117); larger sample needed for statistical claims.
- 5-min episodes; conclusions don't extrapolate to multi-hour parent orders.
- AC-RA / VWAP-following baselines from Phase 3 evaluated separately on different infrastructure.
- No spread modelling — fills at mid (caveat from env design).

## Reference: Phase 3 backtest results (different infra, multi-day)

- AC-RA beats TWAP by **5.3 bps median IS** on 5-day OOS (Phase 3 backtest engine)
- This eval uses ExecutionEnv on Day 5 only — direct comparison limited.