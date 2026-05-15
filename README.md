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
  vs every other classical strategy. Best multi-metric classical strategy.
- **PPO RL with 5-min episodes + mid-only obs (Phase 6 + A) doesn't beat
  TWAP.** First version of the RL story is a negative finding: the
  variance reduction is from the action cap, not the policy.
- **PPO RL with 60-min episodes + microstructure obs (Phase D) ties
  AC-RA on 5 tickers** — first sign the bottleneck was horizon/obs,
  not RL itself.
- **PPO RL ticker-agnostic + spread cost in training (Phase E) generalises
  to a 104-ticker universe** for **mid-morning execution (10:00–11:00 ET)**:
  trained on 20 diverse tickers from Day 1-4, evaluated on all 104 on
  Day 5 OOS. Beats VWAP-following on **60% of tickers**, lowest median IS
  across all 6 strategies (+1.55 vs +5.78 bps, both paying half-spread).
  4-fold walk-forward: wins 3 of 4 folds.
- **Time-of-day generalization (Phase G)**: tested across 5 RTH windows.
  RL wins open/early-mid/late-mid, **loses in mid and close** (close
  Δ +10 bps vs VWAP, win-rate 25%). v4 retrain with overlapping windows
  didn't fix close — root cause is **action cap (0.05) saturation** under
  tight regimes (both v3 and v4 collapse to TWAP-cap-uniform). Honest
  scope-limited finding; action-cap redesign is Phase H candidate.
- **Parent size sensitivity (Phase G.2)**: tested 0.1% / 1% / 5% ADV.
  Strategy ranking effectively unchanged across sizes — but that's a
  **limitation, not a robustness claim**: we lack a self-impact model,
  so big parents don't push price in our backtest. Real 1%+ ADV
  execution would behave differently. Phase H candidate.
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
tests/            # 155 unit tests
reports/          # per-phase writeups + figures
dashboards/       # streamlit explorers
vendor/abides-sim # vendored ABIDES (BSD-3) with patch notes
```

## Run

```bash
uv sync && uv pip install -e .
uv run pytest tests/                              # 155 tests
uv run python scripts/eval_ppo_oos.py             # phase 6/A 5-min OOS comparison
uv run python scripts/run_phase_d_eval.py         # phase D 60-min RL vs 5 baselines
uv run python scripts/run_phase_e_xticker_eval.py # phase E 104-ticker OOS sweep
uv run python scripts/run_phase_g_multi_window.py # phase G multi-window eval
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
| **D** | **PPO RL on 60-min episodes, 13-dim microstructure obs, 5-ticker train pool** | **median IS −13.69 bps — best of all 9 strategies, beats AC-RA by 0.45 bps and TWAP by 7.92 bps; ties AC-RA on Day-5 OOS (both −8.52)** |
| **E** | **Ticker-agnostic v3 obs (log_adv_norm replaces ticker_idx) + spread cost in training; 20-ticker train pool, 104-ticker OOS sweep, 4-fold walk-forward CV** | **RL beats VWAP-following on 60% of 104 tickers in the 10:00–11:00 ET window; lowest median IS of all 6 strategies (+1.55 vs +5.78 VWAP-follow). Generalises to ~84 unseen tickers. Walk-forward: wins on 3 of 4 folds; edge grows monotonically with more training data.** |
| **F** | Volume-profile **data-leakage audit + fix** — `compute_lookback_volume_profile()` (leave-one-out across other days), regression test guards; Phase 3 σ leak also fixed (constant σ instead of same-day-mid estimate) | Volume-aware classical strategies' apparent "−13.22 bps beats TWAP by 7.5 bps" was a same-day-leakage artifact; under clean lookback they're ~−4.51 bps (essentially tied with TWAP). RL / AC-RA unaffected (don't use volume_profile). Phase E.3 headline numbers unchanged. |
| **G** | **Multi-window OOS** (5 RTH windows × 104 tickers) + **parent size sweep** (0.1% / 1% / 5% ADV); attempted **v4 retrain** with overlapping training windows | **Time-of-day mixed**: RL beats VWAP-follow in open/early-mid/late-mid (Δ −1 to −5 bps, win 49–67%), loses in mid and close (close Δ +10 bps, win 25%). v4 retrain (overlapping 60-min windows) didn't fix close — **action cap (0.05) saturates**, both v3 and v4 collapse to TWAP-cap-uniform in tight regimes. Parent-size sensitivity ≈ 0 across sizes — but that's a **limitation** (no self-impact model), not a feature. |

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

**First answer (Phase 6 + A)**: the volume-aware classical schedules are
hard to beat — RL with 5-min episodes and mid-only observation doesn't
add anything beyond what the action cap gets you. Negative finding,
preserved.

**Second answer (Phase D)**: give PPO 60-min episodes and a richer
13-dim observation (NBBO depth, microprice drift, per-venue concentration,
trade-flow imbalance, intraday volume-profile percentile) and it actually
matches AC-RA on a 5-ticker eval.

**Third answer (Phase E)**: drop the categorical ticker index, add
log_adv_norm so the model is ticker-agnostic, add spread cost to training
so the policy isn't biased aggressive. Train on 20 diverse tickers from
Day 1-4, evaluate on all 104 tickers on Day 5 OOS. PPO ends up with the
lowest median IS of any strategy tested and beats VWAP-following on 62%
of tickers — including ~84 tickers the policy never saw during training.
