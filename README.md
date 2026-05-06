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
  vs every other strategy in the zoo. Best multi-metric strategy across
  9 strategies × 200 backtests.
- **PPO RL does *not* beat TWAP** out-of-sample. It cuts variance by
  52–74%, but that's the action cap forcing a roughly uniform 60-step
  schedule, not the policy actually learning anything useful.
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
tests/            # 122 unit tests
reports/          # per-phase writeups + figures
dashboards/       # streamlit explorers
vendor/abides-sim # vendored ABIDES (BSD-3) with patch notes
```

## Run

```bash
uv sync && uv pip install -e .
uv run pytest tests/                       # 122 tests
uv run python scripts/eval_ppo_oos.py      # day-5 OOS comparison
uv run streamlit run dashboards/01_data_explorer.py
```

## What I tried, and the verdict

| Phase | What | Verdict |
|---|---|---|
| 2 | TWAP / VWAP-follow / AC baselines + 10-metric backtester | volume-aware wins by ~7.5 bps; AC-RA wins both IS and variance |
| 3 | AC calibration: η/γ from real impact regression | risk-averse improves *both* median IS and variance — no trade-off |
| 4 | 14-venue SOR analysis | NBBO is the upper bound; static routing loses 0.1–9 bps |
| 5 | ABIDES multi-agent calibration to AAPL | structural gaps in stylized facts; a custom HerderAgent (Lux 1998) cuts vol-autocorr distance ~19% |
| 6 + A | PPO RL on synthetic + real 5-min episodes, with action cap | doesn't beat TWAP on median IS; variance reduction comes from the cap, not the policy |
| B | Strategy zoo (POV, Tóth, CVXPY-constrained AC) + 8-strategy sweep | adds nothing material on top of AC-RA |
| B.3 | xgboost-predicted state-conditional η | overfits, loses to static η, kept as negative finding |

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
Wanted to do this on real ticks with a fair fight. The honest answer
turned out to be: **the volume-aware classical schedules are pretty
hard to beat.**
