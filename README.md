# Optimal Execution on Real Tick Data

You have to dump 10k shares of AAPL and don't want to leave a crater in
the order book. Which child-order schedule actually works? This repo
takes every classical and RL execution approach I could find, runs them
on 5 days of full TAQ + 14-venue NBBO data, and writes down what worked
(and what looked good but didn't).

## TL;DR what worked

- **Volume-aware schedules** (VWAP-follow / POV / Tóth) beat TWAP by
  ~7.5 bps median IS for 10k-share parents on liquid large-caps.
- **Almgren-Chriss risk-averse** wins on *both* median IS and variance
  vs every other classical strategy. Best multi-metric classical strategy
  across 9 strategies × 200 backtests.
- **PPO RL with 5-min episodes + mid-only obs (Phase 6 + A) doesn't beat
  TWAP.** First version of the RL story is a negative finding: the
  variance reduction is from the action cap, not the policy.
- **PPO RL with 60-min episodes + 13-dim microstructure obs (Phase D)
  *does* tie or beat the best classical**. Median IS −13.69 bps across
  150 backtests (vs AC-RA −13.24, TWAP −5.77); on the held-out OOS day,
  rl_v2 ties AC-RA at −8.52 bps median IS. Episode horizon and
  observation richness were the bottleneck — not RL itself.
- **Static venue allocation** loses to NBBO routing by 0.1–9 bps —
  validates dynamic per-order routing.
- **Trying ML for the η coefficient** (xgboost on 762k trade events)
  overfits hard (9× train→OOS R² gap) and loses to the static-η
  baseline. Kept as a negative finding.

## Data

- 5 days of full TAQ tick data (2020-01-13 to 2020-01-17)
- 208 equities × 14 venues, NBBO reconstructed from raw quotes
- NQ futures L2 + 1-min aggregates

## Repo layout

```
src/hft/
├── data/         # parquet cache + NBBO reconstruction
├── strategies/   # twap, vwap_following, almgren_chriss, pov, toth, cvxpy
├── backtest/     # tick-replay engine + 10 metrics
├── simulators/   # ABIDES loader + RL gym env
└── analysis/     # impact regression, SOR, metrics

scripts/          # one driver per phase
tests/            # 135 unit tests
reports/          # per-phase writeups + figures
dashboards/       # streamlit explorers
vendor/abides-sim # vendored ABIDES (BSD-3) with patch notes
```

## Run

```bash
uv sync && uv pip install -e .
uv run pytest tests/                              # 135 tests
uv run python scripts/eval_ppo_oos.py             # phase 6/A 5-min OOS comparison
uv run python scripts/run_phase_d_eval.py         # phase D 60-min RL vs 5 baselines
uv run streamlit run dashboards/01_data_explorer.py
```

## What I tried, and the verdict

| Phase | What | Verdict |
|---|---|---|
| 2 | TWAP / VWAP-follow / AC baselines + 10-metric backtester | volume-aware wins by ~7.5 bps; AC-RA wins both IS and variance |
| 3 | AC calibration: η/γ from real impact regression | risk-averse improves *both* median IS and variance — no trade-off |
| 4 | 14-venue SOR analysis | NBBO is the upper bound; static routing loses 0.1–9 bps |
| 5 | ABIDES multi-agent calibration to AAPL | structural gaps in stylized facts; a custom HerderAgent (Lux 1998) cuts vol-autocorr distance ~19% |
| 6 + A | PPO RL on synthetic + real 5-min episodes, mid-only obs, with action cap | doesn't beat TWAP on median IS; variance reduction comes from the cap, not the policy |
| B | Strategy zoo (POV, Tóth, CVXPY-constrained AC) + 8-strategy sweep | adds nothing material on top of AC-RA |
| B.3 | xgboost-predicted state-conditional η | overfits, loses to static η, kept as negative finding |
| **D** | **PPO RL on 60-min episodes, 13-dim microstructure obs, multi-ticker train pool** | **median IS −13.69 bps — best of all 9 strategies, beats AC-RA by 0.45 bps and TWAP by 7.92 bps; ties AC-RA on Day-5 OOS (both −8.52)** |

## Caveats

- 5 days isn't a lot of data. Read statistical claims with that in mind.
- RL uses 5-min episodes. Conclusions don't necessarily extend to
  multi-hour parent orders.
- Fills happen at mid (no spread cost modelled). Real fills should pay
  half-spread.
- One OOS day across the board. Walk-forward CV is on the to-do list.

## Why bother

Most "RL beats TWAP" papers I read either trained on something that
doesn't look like real markets or compared against a strawman baseline.
Wanted to do this on real ticks with a fair fight.

First answer (Phase 6 + A): the volume-aware classical schedules are
hard to beat — RL with 5-min episodes and mid-only observation doesn't
add anything beyond what the action cap gets you. **Negative finding,
preserved.**

Second answer (Phase D): give PPO 60-min episodes and a richer
13-dim observation (NBBO depth, microprice drift, per-venue
concentration, trade-flow imbalance, intraday volume-profile percentile)
and it actually does match or slightly beat AC-RA. Same parent spec,
same backtest engine, same OOS day. **The earlier negative finding was
about horizon and observation, not about RL.**
