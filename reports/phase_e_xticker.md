# Phase E — Cross-Ticker OOS Eval (104 tickers × Day 5)

**Generated**: 2026-05-10T17:47:47.639120-05:00 (ET)
**OOS date**: 20200117  |  **Tickers with full data**: 104
**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing
**RL training pool**: 20 tickers sampled across ADV deciles (Day 1-4); OOS evaluates on ALL 104 tickers (most never seen during training)
**Total backtests**: 624 succeeded / 0 failed
**Spread cost**: enabled (RL trained with NBB / NBO fills to match BacktestEngine — Phase D's training/eval asymmetry fixed)

## Headline

RL (Phase E v3, ticker-agnostic) has the **lowest median IS across 104
tickers** in the zoo and beats VWAP-following on **65 / 104 tickers (62%)**:

| Strategy | Median IS Day-5 OOS | vs VWAP-follow win-rate |
|---|---|---|
| **rl_v3 (Phase E)** | **+1.55 bps** ⭐ | **62%** |
| ac_ra | +3.17 | 56% |
| twap | +4.93 | 57% |
| toth | +5.57 | 42% |
| pov_5pct_cap | +5.78 | 24% |
| vwap_following | +5.78 | — |

All IS values are now in the half-spread-cost regime (BacktestEngine
fills at NBB for sells), so they're ~10–15 bps higher than Phase D's
mid-fill numbers — but the **relative ordering** is what matters.

The ticker-agnostic v3 policy (trained on 20 diverse tickers, evaluated
on 84 unseen tickers) generalises: RL is the best of the six strategies
tested, by 1.6 bps median IS vs AC-RA and by 4.2 bps vs VWAP-following.

## Per-strategy aggregates (sorted by median IS, best first)

| Strategy | n | Median IS (bps) | Mean IS | Std IS | Median VWAP slip | Win-rate vs VWAP-follow |
|---|---|---|---|---|---|---|
| rl_v3 | 104 | **+1.55** | +3.51 | 19.8 | +2.49 | 62% |
| ac_ra | 104 | **+3.17** | +4.20 | 20.8 | +2.43 | 56% |
| twap | 104 | **+4.93** | +3.84 | 27.6 | +3.52 | 57% |
| toth | 104 | **+5.57** | +4.42 | 29.5 | +3.34 | 42% |
| pov_5pct_cap | 104 | **+5.78** | +4.70 | 29.4 | +3.75 | 24% |
| vwap_following | 104 | **+5.78** | +4.70 | 29.4 | +3.75 | — |

## RL vs VWAP-following per ticker (delta in bps, negative = RL better for sell)

- RL beats VWAP-following on **65/104** tickers (62%)
- RL loses to VWAP-following on 39 tickers, ties on 0
- Median delta (RL − VWAP-follow): **-1.556 bps**
- Mean delta: -1.187 bps

**Top 10 RL wins:**
| Ticker | RL IS | VWAP-follow IS | Δ |
|---|---|---|---|
| VRTX | +7.09 | +52.10 | **-45.01** |
| INCY | -19.59 | +23.09 | **-42.68** |
| MXIM | +4.38 | +38.32 | **-33.94** |
| FAST | +8.27 | +41.59 | **-33.31** |
| KLAC | +5.01 | +33.48 | **-28.47** |
| IDXX | +70.91 | +95.03 | **-24.12** |
| NXPI | -2.08 | +21.06 | **-23.14** |
| ALGN | +29.16 | +51.02 | **-21.86** |
| MNST | +4.94 | +26.60 | **-21.65** |
| BIDU | -5.90 | +14.85 | **-20.75** |

**Top 10 RL losses:**
| Ticker | RL IS | VWAP-follow IS | Δ |
|---|---|---|---|
| FOX | -7.20 | -23.20 | **+16.00** |
| EXC | -3.89 | -20.20 | **+16.31** |
| UAL | -12.77 | -30.38 | **+17.61** |
| MCHP | -8.26 | -26.12 | **+17.86** |
| CMCSA | -0.65 | -18.84 | **+18.19** |
| SWKS | -18.29 | -39.62 | **+21.33** |
| WBA | -15.07 | -37.57 | **+22.50** |
| SGEN | -3.60 | -28.01 | **+24.41** |
| AAL | -30.11 | -86.58 | **+56.47** |
| QCOM | -28.79 | -114.67 | **+85.88** |

## Caveats

- Single OOS day (Day 5 / 20200117). 5 days isn't a lot.
- Parent fixed at 10k shares, 10:00–11:00 ET window. Open/close regimes may differ.
- Spread cost in BacktestEngine (NBB / NBO fills), so all strategies pay half-spread realistically.
- RL trained on 20 tickers sampled across ADV deciles from the 97; ~77 tickers in OOS are unseen during training.
