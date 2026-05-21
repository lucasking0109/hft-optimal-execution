# Phase G.1 — Multi-Window OOS Eval (104 ticker × 5 RTH windows)

**Generated**: 2026-05-14T21:01:00.464476-05:00 (ET)
**OOS date**: 20200117  |  **Tickers**: 104
**Parent**: 10,000 shares sell, 1-hour window, NBBO routing
**Lookback volume profile**: Day 1-4 average (causal, no leak)

## Regression sanity

rl_v4 early_mid median IS = +3.626 (within plausible range)

## Per-(window, strategy) median IS

| Window | Strategy | n | Median IS (bps) | Std | Win-rate vs VWAP-follow |
|---|---|---|---|---|---|
| close | pov_5pct_cap | 103 | **-6.17** | 19.5 | 8% |
| close | toth | 103 | **-6.17** | 19.5 | 2% |
| close | vwap_following | 103 | **-6.17** | 19.5 | — |
| close | twap | 103 | **+0.65** | 14.6 | 17% |
| close | rl_v4 | 103 | **+3.87** | 6.9 | 25% |
| close | ac_ra | 103 | **+4.12** | 8.4 | 22% |
| early_mid | ac_ra | 104 | **+3.17** | 20.8 | 59% |
| early_mid | rl_v4 | 104 | **+3.63** | 19.5 | 55% |
| early_mid | twap | 104 | **+4.93** | 27.6 | 71% |
| early_mid | toth | 104 | **+6.19** | 26.5 | 32% |
| early_mid | pov_5pct_cap | 104 | **+6.28** | 27.3 | 20% |
| early_mid | vwap_following | 104 | **+6.28** | 27.3 | — |
| late_mid | rl_v4 | 104 | **+1.62** | 6.3 | 66% |
| late_mid | ac_ra | 104 | **+2.44** | 7.4 | 68% |
| late_mid | toth | 104 | **+5.58** | 12.0 | 25% |
| late_mid | pov_5pct_cap | 104 | **+6.21** | 11.7 | 19% |
| late_mid | vwap_following | 104 | **+6.21** | 11.7 | — |
| late_mid | twap | 104 | **+6.39** | 11.7 | 51% |
| mid | vwap_following | 104 | **-3.78** | 15.4 | — |
| mid | pov_5pct_cap | 104 | **-3.78** | 15.4 | 25% |
| mid | twap | 104 | **-3.50** | 16.0 | 46% |
| mid | toth | 104 | **-1.02** | 15.5 | 32% |
| mid | ac_ra | 104 | **-0.33** | 10.9 | 36% |
| mid | rl_v4 | 104 | **+0.41** | 9.5 | 37% |
| open | twap | 104 | **+18.60** | 75.5 | 55% |
| open | ac_ra | 104 | **+19.62** | 67.0 | 52% |
| open | rl_v4 | 104 | **+20.59** | 63.6 | 49% |
| open | vwap_following | 104 | **+21.69** | 71.3 | — |
| open | pov_5pct_cap | 104 | **+21.70** | 71.3 | 14% |
| open | toth | 104 | **+22.23** | 72.0 | 39% |

## RL v3 across windows (key generalization check)

| Window | RL median IS | VWAP-follow median IS | Δ | Win-rate vs VWAP |
|---|---|---|---|---|
| open | +20.59 | +21.69 | **-1.10** | 49% |
| early_mid | +3.63 | +6.28 | **-2.66** | 55% |
| mid | +0.41 | -3.78 | **+4.19** | 37% |
| late_mid | +1.62 | +6.21 | **-4.59** | 66% |
| close | +3.87 | -6.17 | **+10.04** | 25% |

## G.4 retrain trigger check

- RL win-rate vs VWAP-follow in close = 25% < 45% — generalization failed, retrain v4 (G.4)

## Caveats

- Single OOS day (20200117).
- 1-hour parent window across 5 time-of-day points.
- Spread cost in BacktestEngine (NBB/NBO fills).
- RL trained on 20 tickers × Day 1-4 across 6 non-overlapping 60-min windows starting 09:30–14:30. The close window (15:00–16:00) was NOT in the training pool — this eval tests generalization to unseen time-of-day.
