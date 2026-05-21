# Phase G.2 — Parent Size Sweep (% of ADV)

**Generated**: 2026-05-14T21:31:09.943895-05:00 (ET)
**OOS date**: 20200117  |  **Tickers**: 104
**Sizes**: 0_1pct_adv (0.1% ADV), 1pct_adv (1.0% ADV), 5pct_adv (5.0% ADV)
**Windows**: open, early_mid, mid, late_mid, close
**Total backtests**: 9343 done / 17 failed / 0 skipped

## Caveats

- 0.1% ADV: no-impact assumption safe, results directly comparable
- 1% ADV: borderline; POV 5% cap occasionally binds on illiquid names
- 5% ADV × 1-hour: POV 5% cap heavily violated (per-bucket participation
  ~20-30%); force_completion water-fill makes every child ~4-5× POV cap.
  Treat this row as stress test, not production scenario.
- Self-impact not modelled — all results assume price-taker. At 1%+ ADV,
  real fills would push the market. Phase H candidate.

## RL vs VWAP-following per (size, window) — key signal

| Size | Window | RL median IS | VWAP-follow median IS | Δ (bps) | RL win-rate vs VWAP |
|---|---|---|---|---|---|
| 0_1pct_adv | open | +20.65 | +21.69 | **-1.04** | 49% |
| 0_1pct_adv | early_mid | +3.62 | +6.28 | **-2.66** | 56% |
| 0_1pct_adv | mid | +0.42 | -3.78 | **+4.20** | 37% |
| 0_1pct_adv | late_mid | +1.62 | +6.20 | **-4.57** | 66% |
| 0_1pct_adv | close | +3.89 | -6.17 | **+10.06** | 25% |
| 1pct_adv | open | +20.59 | +21.69 | **-1.10** | 49% |
| 1pct_adv | early_mid | +3.63 | +6.28 | **-2.65** | 55% |
| 1pct_adv | mid | +0.41 | -3.78 | **+4.19** | 37% |
| 1pct_adv | late_mid | +1.62 | +6.21 | **-4.59** | 67% |
| 1pct_adv | close | +3.88 | -6.17 | **+10.05** | 25% |
| 5pct_adv | open | +20.58 | +21.70 | **-1.11** | 49% |
| 5pct_adv | early_mid | +3.63 | +6.28 | **-2.65** | 55% |
| 5pct_adv | mid | +0.41 | -3.78 | **+4.19** | 37% |
| 5pct_adv | late_mid | +1.62 | +6.21 | **-4.59** | 67% |
| 5pct_adv | close | +3.87 | -6.17 | **+10.04** | 25% |

## Per-(size, window, strategy) median IS — full breakdown

| Size | Window | Strategy | n | Median IS (bps) | Std |
|---|---|---|---|---|---|
| 0_1pct_adv | close | pov_5pct_cap | 103 | **-6.17** | 19.5 |
| 0_1pct_adv | close | vwap_following | 103 | **-6.17** | 19.5 |
| 0_1pct_adv | close | toth | 103 | **-6.17** | 19.5 |
| 0_1pct_adv | close | twap | 103 | **+0.68** | 14.6 |
| 0_1pct_adv | close | rl_v4 | 103 | **+3.89** | 6.9 |
| 0_1pct_adv | close | ac_ra | 104 | **+4.29** | 8.4 |
| 0_1pct_adv | early_mid | ac_ra | 104 | **+3.17** | 20.8 |
| 0_1pct_adv | early_mid | rl_v4 | 104 | **+3.62** | 19.5 |
| 0_1pct_adv | early_mid | twap | 104 | **+4.98** | 27.6 |
| 0_1pct_adv | early_mid | toth | 104 | **+6.28** | 27.3 |
| 0_1pct_adv | early_mid | vwap_following | 104 | **+6.28** | 27.3 |
| 0_1pct_adv | early_mid | pov_5pct_cap | 104 | **+6.28** | 27.3 |
| 0_1pct_adv | late_mid | rl_v4 | 104 | **+1.62** | 6.3 |
| 0_1pct_adv | late_mid | ac_ra | 104 | **+2.43** | 7.4 |
| 0_1pct_adv | late_mid | vwap_following | 104 | **+6.20** | 11.7 |
| 0_1pct_adv | late_mid | pov_5pct_cap | 104 | **+6.20** | 11.7 |
| 0_1pct_adv | late_mid | toth | 104 | **+6.21** | 11.7 |
| 0_1pct_adv | late_mid | twap | 104 | **+6.32** | 11.7 |
| 0_1pct_adv | mid | pov_5pct_cap | 104 | **-3.78** | 15.4 |
| 0_1pct_adv | mid | vwap_following | 104 | **-3.78** | 15.4 |
| 0_1pct_adv | mid | toth | 104 | **-3.78** | 15.4 |
| 0_1pct_adv | mid | twap | 104 | **-3.45** | 15.9 |
| 0_1pct_adv | mid | ac_ra | 104 | **-0.33** | 10.9 |
| 0_1pct_adv | mid | rl_v4 | 104 | **+0.42** | 9.5 |
| 0_1pct_adv | open | twap | 104 | **+18.51** | 75.4 |
| 0_1pct_adv | open | ac_ra | 104 | **+19.63** | 67.0 |
| 0_1pct_adv | open | rl_v4 | 104 | **+20.65** | 63.6 |
| 0_1pct_adv | open | toth | 104 | **+21.67** | 71.3 |
| 0_1pct_adv | open | pov_5pct_cap | 104 | **+21.69** | 71.3 |
| 0_1pct_adv | open | vwap_following | 104 | **+21.69** | 71.3 |
| 1pct_adv | close | pov_5pct_cap | 103 | **-6.17** | 19.5 |
| 1pct_adv | close | toth | 103 | **-6.17** | 19.5 |
| 1pct_adv | close | vwap_following | 103 | **-6.17** | 19.5 |
| 1pct_adv | close | twap | 103 | **+0.63** | 14.6 |
| 1pct_adv | close | rl_v4 | 103 | **+3.88** | 6.9 |
| 1pct_adv | close | ac_ra | 103 | **+4.12** | 8.4 |
| 1pct_adv | early_mid | ac_ra | 104 | **+3.17** | 20.8 |
| 1pct_adv | early_mid | rl_v4 | 104 | **+3.63** | 19.5 |
| 1pct_adv | early_mid | twap | 104 | **+4.92** | 27.7 |
| 1pct_adv | early_mid | toth | 104 | **+5.76** | 27.2 |
| 1pct_adv | early_mid | vwap_following | 104 | **+6.28** | 27.3 |
| 1pct_adv | early_mid | pov_5pct_cap | 104 | **+6.28** | 27.3 |
| 1pct_adv | late_mid | rl_v4 | 104 | **+1.62** | 6.3 |
| 1pct_adv | late_mid | ac_ra | 104 | **+2.43** | 7.4 |
| 1pct_adv | late_mid | toth | 104 | **+5.95** | 11.9 |
| 1pct_adv | late_mid | pov_5pct_cap | 104 | **+6.21** | 11.7 |
| 1pct_adv | late_mid | vwap_following | 104 | **+6.21** | 11.7 |
| 1pct_adv | late_mid | twap | 104 | **+6.40** | 11.7 |
| 1pct_adv | mid | vwap_following | 104 | **-3.78** | 15.4 |
| 1pct_adv | mid | pov_5pct_cap | 104 | **-3.78** | 15.4 |
| 1pct_adv | mid | twap | 104 | **-3.50** | 16.0 |
| 1pct_adv | mid | toth | 104 | **-2.98** | 15.2 |
| 1pct_adv | mid | ac_ra | 104 | **-0.33** | 10.9 |
| 1pct_adv | mid | rl_v4 | 104 | **+0.41** | 9.5 |
| 1pct_adv | open | twap | 104 | **+18.60** | 75.5 |
| 1pct_adv | open | ac_ra | 104 | **+19.62** | 67.0 |
| 1pct_adv | open | rl_v4 | 104 | **+20.59** | 63.6 |
| 1pct_adv | open | toth | 104 | **+21.69** | 71.4 |
| 1pct_adv | open | vwap_following | 104 | **+21.69** | 71.3 |
| 1pct_adv | open | pov_5pct_cap | 104 | **+21.70** | 71.3 |
| 5pct_adv | close | toth | 103 | **-13.55** | 25.3 |
| 5pct_adv | close | pov_5pct_cap | 103 | **-6.17** | 19.5 |
| 5pct_adv | close | vwap_following | 103 | **-6.17** | 19.5 |
| 5pct_adv | close | twap | 103 | **+0.63** | 14.6 |
| 5pct_adv | close | rl_v4 | 103 | **+3.87** | 6.9 |
| 5pct_adv | close | ac_ra | 103 | **+4.13** | 8.4 |
| 5pct_adv | early_mid | ac_ra | 104 | **+3.17** | 20.8 |
| 5pct_adv | early_mid | rl_v4 | 104 | **+3.63** | 19.5 |
| 5pct_adv | early_mid | toth | 104 | **+4.30** | 29.4 |
| 5pct_adv | early_mid | twap | 104 | **+4.92** | 27.7 |
| 5pct_adv | early_mid | vwap_following | 104 | **+6.28** | 27.3 |
| 5pct_adv | early_mid | pov_5pct_cap | 104 | **+6.28** | 27.3 |
| 5pct_adv | late_mid | rl_v4 | 104 | **+1.62** | 6.3 |
| 5pct_adv | late_mid | ac_ra | 104 | **+2.44** | 7.4 |
| 5pct_adv | late_mid | toth | 104 | **+3.25** | 14.3 |
| 5pct_adv | late_mid | pov_5pct_cap | 104 | **+6.21** | 11.7 |
| 5pct_adv | late_mid | vwap_following | 104 | **+6.21** | 11.7 |
| 5pct_adv | late_mid | twap | 104 | **+6.41** | 11.7 |
| 5pct_adv | mid | vwap_following | 104 | **-3.78** | 15.4 |
| 5pct_adv | mid | pov_5pct_cap | 104 | **-3.78** | 15.4 |
| 5pct_adv | mid | twap | 104 | **-3.50** | 16.0 |
| 5pct_adv | mid | ac_ra | 104 | **-0.33** | 10.9 |
| 5pct_adv | mid | toth | 104 | **+0.36** | 14.9 |
| 5pct_adv | mid | rl_v4 | 104 | **+0.41** | 9.5 |
| 5pct_adv | open | toth | 104 | **+14.82** | 65.8 |
| 5pct_adv | open | twap | 104 | **+18.61** | 75.5 |
| 5pct_adv | open | ac_ra | 104 | **+19.62** | 67.0 |
| 5pct_adv | open | rl_v4 | 104 | **+20.58** | 63.6 |
| 5pct_adv | open | vwap_following | 104 | **+21.70** | 71.3 |
| 5pct_adv | open | pov_5pct_cap | 104 | **+21.70** | 71.3 |