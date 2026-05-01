# Phase A — RL with action cap + multi-day pool (post-audit honest verdict)

**Generated**: 2026-05-05 (post-audit re-evaluation)

## Context

Phase 6A/6B PPO training revealed that synth-only RL collapses to a
degenerate "fire all at step 0" policy due to outlier rewards in synthetic
episodes. Phase 6B (real-data primary, single-day Day 1 training) improved
on this but still failed to beat TWAP on Day 5 OOS.

Phase A applied two fixes:
1. **Action cap (`max_action_per_step=0.1`)** — prevents step-0 dump
   (forces episode length ≥ 10 steps), removing the trivial "lock-in
   arrival" exploit.
2. **Multi-day training pool (`date_pool=[Day1..Day4]`)** — broadens state
   distribution.
3. **200k timesteps** (vs 6B's 100k) for the larger pool.

## Audit correction (2026-05-05)

The original Phase A run was conducted with a **reward-sign bug** in
`ExecutionEnv` that caused PPO to *maximise* cost rather than minimise it
(see audit plan; `src/hft/simulators/execution_env.py` L256). All numbers
in the prior version of this report are therefore unreliable. The bug is
now fixed, all three RL models (Phase 6A, Phase 6B, Phase A v1_capped)
have been **retrained with correct reward**, and Day 5 OOS has been
re-evaluated. **The re-evaluation overturns the prior "RL beats TWAP"
claim.** The honest verdict is reported below.

## Configuration

| Parameter | Value |
|---|---|
| `max_action_per_step` | 0.1 (per step ≤ 10% of remaining inventory) |
| `date_pool` | `["20200113", "20200114", "20200115", "20200116"]` |
| Total timesteps | 200,000 |
| Episodes seen | 3,345 |
| Train time | 7.7 min (Mac CPU) |
| Avg episode length | 60 steps (full-horizon, no early-exit) |

## In-sample eval (Day 1-4 pool, 312 episodes; new sign convention)

| Metric | Value (bps; positive reward = good execution) |
|---|---|
| Median reward | -0.411 |
| Mean reward | -0.322 |
| Std | 3.70 |
| Mean episode length | 60.0 steps |
| Mean inventory left | 0.31% |

(In IS_bps cost-positive convention, this is roughly +0.41 bps median IS
in-sample — i.e., the policy executes at slight cost vs arrival on
training data.)

## OOS Day 5 (20200117) — 7 strategies × 78 episodes

All values in **IS bps, cost-positive convention** (lower is better).

| Agent | Median IS | Std | Win rate vs TWAP |
|---|---|---|---|
| Aggressive (step 0 dump) | +0.000 | 0.00 | 44% |
| Lazy (step 59 dump) | -0.947 | 6.61 | 53% |
| **TWAP** (uniform) | **-0.728** | **3.78** | — |
| Random | -0.103 | 1.02 | 44% |
| PPO-6A (synth-only) | +0.000 | 1.66 | 38% (collapses to hold-and-pay-penalty) |
| PPO-6B (real Day 1 only) | -0.097 | **0.97** | 42% |
| **PPO-v1_capped (Phase A)** | **-0.465** | **1.80** | **44%** |

## Honest verdict (post-audit)

**TWAP beats every RL agent on Day 5 OOS median IS.**

| Metric | TWAP | PPO-6B | PPO-v1_capped | Δ vs TWAP |
|---|---|---|---|---|
| Median IS (bps) | **-0.728** | -0.097 | -0.465 | +0.26 to +0.63 (RL worse) |
| Std | 3.78 | **0.97** | 1.80 | -52% to -74% (RL better) |
| Win rate vs TWAP | — | 42% | 44% | RL loses majority |

**Two competing effects:**
- **Median IS**: TWAP wins. RL agents under-execute relative to TWAP and
  end up paying more cost on the median episode. PPO cannot reliably
  predict 5-second-ahead mid drift, so its policy drifts toward
  defensive holding rather than capturing the small price-impact
  opportunities.
- **Variance**: RL with action cap wins decisively — std drops
  52% (v1_capped) to 74% (ppo_real, single-day). This is a *real*
  benefit but it is **driven by the action cap**, not by directional
  learning. The cap forces every step to fire ≤10% inventory, which
  produces a 60-step uniform-ish schedule similar to TWAP but with
  slightly tighter dispersion across episodes.

## Why the original run looked like RL won

Under the inverted reward sign, PPO was *maximising* IS_bps cost. This
should have produced terrible execution, but:

1. **PPO can't predict mid drift on 5-sec windows** — directional signal
   is too weak for the policy to consistently sell low.
2. **Action cap=0.1 forces 60-step distribution** — the episode is
   essentially TWAP-like regardless of policy direction, because no
   single step can move much inventory.
3. So both the reward sign and its negation produced similar
   action-cap-uniform behaviour. The 52% variance reduction was always
   real, but the median-IS edge was an artifact of training on a
   small sample with a backwards objective.

After retraining with the **correct** reward sign, PPO is now genuinely
trying to minimise IS, and the verdict on Day 5 is: variance reduction
yes, median IS improvement no.

## Key takeaways

1. **Action cap is the load-bearing component**, not the RL policy. A
   plain TWAP-with-cap (no learning) would likely match the variance
   reduction without the median-IS regression.
2. **PPO directional learning does not add value on 5-min episodes** with
   the current observation set (5-dim, mid-derived only). Insufficient
   signal-to-noise.
3. **The negative result is the honest finding** and is more valuable
   than the prior "RL beats TWAP" claim, which was an artifact.
4. To revisit RL meaningfully would require: longer episodes (multi-hour
   parents), richer observation (LOB depth, queue-position features), or
   a different reward shape (per-step variance penalty rather than
   IS-only).

## Caveats (NO Silent Fallback)

- **Single OOS day** (Day 5 / 20200117). Larger statistical claims need
  multi-day eval.
- **5-min episodes**: directional alpha is dwarfed by noise; longer
  horizons may give RL a chance to learn impact-aware behaviour.
- **No spread modelling** in env (fills at mid). Real fills should pay
  half-spread.
- **Action cap 0.1 is hand-picked**; could grid-search for optimal cap.
- **Same hyperparameters as pre-audit run**, only the reward sign and
  resulting policy weights differ.

## Files produced

- `rl/checkpoints/ppo_real_v1_capped/model.zip` — retrained policy
- `rl/checkpoints/ppo_real_v1_capped/training_log.json` — training curve
- `rl/checkpoints/ppo_real_v1_capped/eval_in_sample.json` — in-sample eval
- `reports/figures/phase_a_training_curve.png` — training curve
- `reports/phase6c_oos_comparison.md` — 7-agent OOS table
