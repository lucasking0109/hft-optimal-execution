# Phase E — Cross-Ticker OOS Eval (97 tickers × Day 5)

**Generated**: 2026-05-12T22:48:35.476928-05:00 (ET)
**OOS date**: 20200117  |  **Tickers with full data**: 104
**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing
**RL training pool**: 20 tickers sampled across ADV deciles (Day 1-4); OOS evaluates on ALL 104 tickers (most never seen during training)
**Total backtests**: 624 succeeded / 0 failed

## Per-strategy aggregates (sorted by median IS, best first)

| Strategy | n | Median IS (bps) | Mean IS | Std IS | Median VWAP slip | Win-rate vs VWAP-follow |
|---|---|---|---|---|---|---|
| rl_v3 | 104 | **+1.55** | +3.51 | 19.8 | +2.49 | 60% |
| ac_ra | 104 | **+3.17** | +4.20 | 20.8 | +2.43 | 59% |
| twap | 104 | **+4.93** | +3.84 | 27.6 | +3.52 | 71% |
| toth | 104 | **+6.19** | +4.19 | 26.5 | +4.00 | 32% |
| pov_5pct_cap | 104 | **+6.28** | +4.72 | 27.3 | +4.56 | 20% |
| vwap_following | 104 | **+6.28** | +4.72 | 27.3 | +4.56 | — |

## RL vs VWAP-following per ticker (delta in bps, negative = RL better for sell)

- RL beats VWAP-following on **62/104** tickers (60%)
- RL loses to VWAP-following on 42 tickers, ties on 0
- Median delta (RL − VWAP-follow): **-2.230 bps**
- Mean delta: -1.207 bps

**Top 10 RL wins:**
| Ticker | RL IS | VWAP-follow IS | Δ |
|---|---|---|---|
| KLAC | +5.01 | +50.93 | **-45.92** |
| VRTX | +7.09 | +42.29 | **-35.20** |
| INCY | -19.59 | +9.27 | **-28.86** |
| ULTA | +38.78 | +59.57 | **-20.79** |
| XLNX | -20.70 | -1.11 | **-19.58** |
| MNST | +4.94 | +24.43 | **-19.49** |
| CDNS | +39.82 | +59.05 | **-19.23** |
| AMAT | -19.82 | -0.85 | **-18.97** |
| IDXX | +70.91 | +89.81 | **-18.89** |
| NXPI | -2.08 | +16.60 | **-18.68** |

**Top 10 RL losses:**
| Ticker | RL IS | VWAP-follow IS | Δ |
|---|---|---|---|
| ILMN | +20.19 | +5.36 | **+14.84** |
| CMCSA | -0.65 | -17.24 | **+16.58** |
| SWKS | -18.29 | -37.12 | **+18.83** |
| WBA | -15.07 | -34.57 | **+19.50** |
| CSGP | +56.43 | +36.90 | **+19.53** |
| NTES | +44.60 | +24.20 | **+20.40** |
| SGEN | -3.60 | -25.63 | **+22.03** |
| MCHP | -8.26 | -32.96 | **+24.70** |
| AAL | -30.11 | -69.90 | **+39.79** |
| QCOM | -28.79 | -95.02 | **+66.23** |

## Caveats

- Single OOS day (Day 5 / 20200117). 5 days isn't a lot.
- Parent fixed at 10k shares, 10:00–11:00 ET window. Open/close regimes may differ.
- Spread cost in BacktestEngine (NBB / NBO fills), so all strategies pay half-spread realistically.
- RL trained on 20 tickers sampled across ADV deciles from the 97; ~77 tickers in OOS are unseen during training.
