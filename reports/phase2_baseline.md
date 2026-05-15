# Phase 2 Baseline Report

**Setup**: sell 10,000 shares between 10:00 and 11:00 ET, 60 child orders.
**Tickers**: AAPL, AMZN, MSFT, NVDA, TSLA, AMD, NFLX, ADBE, AVGO, INTC
**Dates**: 20200113, 20200114, 20200115, 20200116, 20200117

**Total runs**: 100 (100 success / 0 failure)

## Per-strategy summary (median across all runs)

| Strategy | VWAP slip (bps) | IS (bps) | Eff spread (bps) | Markout 60s (bps) | Reversion 5m (bps) | POV | Hit ratio |
|---|---|---|---|---|---|---|---|
| **vwap_following** | +1.95 | -4.39 | 2.54 | +1.98 | -4.26 | 0.59% | 100.0% |
| **twap** | +1.72 | -3.76 | 2.70 | +1.36 | -4.37 | 0.59% | 100.0% |

## VWAP slippage per (ticker, date, strategy)

| Ticker | Date | TWAP slip (bps) | VWAP-follow slip (bps) | TWAP wins? |
|---|---|---|---|---|
| AAPL | 20200113 | +1.00 | -0.73 | ❌ |
| AAPL | 20200114 | -2.18 | +0.88 | ✅ |
| AAPL | 20200115 | +2.13 | +0.96 | ❌ |
| AAPL | 20200116 | -0.53 | -3.72 | ❌ |
| AAPL | 20200117 | +1.78 | +0.84 | ❌ |
| AMZN | 20200113 | +0.77 | +0.27 | ❌ |
| AMZN | 20200114 | -0.27 | -0.21 | ✅ |
| AMZN | 20200115 | +3.19 | +3.82 | ✅ |
| AMZN | 20200116 | +0.94 | -0.65 | ❌ |
| AMZN | 20200117 | +0.19 | -0.28 | ❌ |
| MSFT | 20200113 | +0.63 | -1.42 | ❌ |
| MSFT | 20200114 | +0.41 | -0.34 | ❌ |
| MSFT | 20200115 | +0.53 | +3.14 | ✅ |
| MSFT | 20200116 | +0.06 | +0.95 | ✅ |
| MSFT | 20200117 | +1.67 | +4.05 | ✅ |
| NVDA | 20200113 | +1.16 | +1.57 | ✅ |
| NVDA | 20200114 | +0.60 | -1.62 | ❌ |
| NVDA | 20200115 | +3.46 | +5.18 | ✅ |
| NVDA | 20200116 | +0.01 | -2.73 | ❌ |
| NVDA | 20200117 | +3.63 | +4.89 | ✅ |
| TSLA | 20200113 | +3.88 | +5.37 | ✅ |
| TSLA | 20200114 | -1.00 | +15.71 | ✅ |
| TSLA | 20200115 | +5.80 | +3.67 | ❌ |
| TSLA | 20200116 | +5.45 | +11.65 | ✅ |
| TSLA | 20200117 | +3.66 | +6.92 | ✅ |
| AMD | 20200113 | +3.16 | +0.26 | ❌ |
| AMD | 20200114 | -3.84 | -7.37 | ❌ |
| AMD | 20200115 | +4.50 | +13.65 | ✅ |
| AMD | 20200116 | +15.64 | +5.70 | ❌ |
| AMD | 20200117 | +3.77 | +6.63 | ✅ |
| NFLX | 20200113 | +9.09 | +6.83 | ❌ |
| NFLX | 20200114 | +5.16 | +6.11 | ✅ |
| NFLX | 20200115 | +4.61 | +8.13 | ✅ |
| NFLX | 20200116 | -0.08 | -1.20 | ❌ |
| NFLX | 20200117 | +3.82 | +7.59 | ✅ |
| ADBE | 20200113 | +2.89 | +3.89 | ✅ |
| ADBE | 20200114 | -5.54 | -7.02 | ❌ |
| ADBE | 20200115 | +4.26 | +5.10 | ✅ |
| ADBE | 20200116 | -0.26 | -0.70 | ❌ |
| ADBE | 20200117 | +2.61 | +4.87 | ✅ |
| AVGO | 20200113 | +5.81 | +6.92 | ✅ |
| AVGO | 20200114 | +1.26 | +1.06 | ❌ |
| AVGO | 20200115 | +4.55 | +2.49 | ❌ |
| AVGO | 20200116 | -1.59 | -7.19 | ❌ |
| AVGO | 20200117 | +2.61 | +3.36 | ✅ |
| INTC | 20200113 | +6.12 | +4.20 | ❌ |
| INTC | 20200114 | +0.81 | -1.04 | ❌ |
| INTC | 20200115 | -0.33 | -1.70 | ❌ |
| INTC | 20200116 | -0.13 | +0.37 | ✅ |
| INTC | 20200117 | +2.15 | +2.33 | ✅ |

## Notes

- All metrics: positive = cost (worse than benchmark), negative = better.
- Backtester known limitations (per NO Silent Fallback transparency):
  1. Marketable child orders fill at NBBO with **no self-impact modelling**.
  2. **Oversize fills** at best price (size > NBBO size at that level) are recorded in `oversize_count` not silently truncated.
  3. VWAP-following uses **same-day volume profile** (look-ahead bias). Phase 3+ will use rolling estimate from prior days.
  4. Both strategies share the same simulator → relative comparison is fair even with these biases.