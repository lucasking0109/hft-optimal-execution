# Phase 2 Baseline Report

**Setup**: sell 10,000 shares between 10:00 and 11:00 ET, 60 child orders.
**Tickers**: AAPL, AMZN, MSFT, NVDA, TSLA, AMD, NFLX, ADBE, AVGO, INTC
**Dates**: 20200113, 20200114, 20200115, 20200116, 20200117

**Total runs**: 100 (100 success / 0 failure)

## Per-strategy summary (median across all runs)

| Strategy | VWAP slip (bps) | IS (bps) | Eff spread (bps) | Markout 60s (bps) | Reversion 5m (bps) | POV | Hit ratio |
|---|---|---|---|---|---|---|---|
| **vwap_following** | +1.55 | -5.43 | 2.52 | +1.41 | -4.26 | 0.59% | 100.0% |
| **twap** | +1.72 | -3.76 | 2.70 | +1.36 | -4.37 | 0.59% | 100.0% |

## VWAP slippage per (ticker, date, strategy)

| Ticker | Date | TWAP slip (bps) | VWAP-follow slip (bps) | TWAP wins? |
|---|---|---|---|---|
| AAPL | 20200113 | +1.00 | +0.19 | ❌ |
| AAPL | 20200114 | -2.18 | +0.10 | ✅ |
| AAPL | 20200115 | +2.13 | +0.90 | ❌ |
| AAPL | 20200116 | -0.53 | -0.36 | ✅ |
| AAPL | 20200117 | +1.78 | +0.58 | ❌ |
| AMZN | 20200113 | +0.77 | +2.33 | ✅ |
| AMZN | 20200114 | -0.27 | +1.40 | ✅ |
| AMZN | 20200115 | +3.19 | +3.00 | ❌ |
| AMZN | 20200116 | +0.94 | +3.14 | ✅ |
| AMZN | 20200117 | +0.19 | +2.18 | ✅ |
| MSFT | 20200113 | +0.63 | +0.13 | ❌ |
| MSFT | 20200114 | +0.41 | +0.61 | ✅ |
| MSFT | 20200115 | +0.53 | +1.12 | ✅ |
| MSFT | 20200116 | +0.06 | +0.49 | ✅ |
| MSFT | 20200117 | +1.67 | +3.71 | ✅ |
| NVDA | 20200113 | +1.16 | +0.13 | ❌ |
| NVDA | 20200114 | +0.60 | -1.36 | ❌ |
| NVDA | 20200115 | +3.46 | +5.23 | ✅ |
| NVDA | 20200116 | +0.01 | +0.48 | ✅ |
| NVDA | 20200117 | +3.63 | +2.82 | ❌ |
| TSLA | 20200113 | +3.88 | +3.45 | ❌ |
| TSLA | 20200114 | -1.00 | +8.54 | ✅ |
| TSLA | 20200115 | +5.80 | +3.13 | ❌ |
| TSLA | 20200116 | +5.45 | +13.87 | ✅ |
| TSLA | 20200117 | +3.66 | +5.41 | ✅ |
| AMD | 20200113 | +3.16 | +0.34 | ❌ |
| AMD | 20200114 | -3.84 | -5.27 | ❌ |
| AMD | 20200115 | +4.50 | +1.28 | ❌ |
| AMD | 20200116 | +15.64 | -6.48 | ❌ |
| AMD | 20200117 | +3.77 | +1.79 | ❌ |
| NFLX | 20200113 | +9.09 | -0.40 | ❌ |
| NFLX | 20200114 | +5.16 | +6.52 | ✅ |
| NFLX | 20200115 | +4.61 | +6.72 | ✅ |
| NFLX | 20200116 | -0.08 | +2.45 | ✅ |
| NFLX | 20200117 | +3.82 | +6.35 | ✅ |
| ADBE | 20200113 | +2.89 | +3.68 | ✅ |
| ADBE | 20200114 | -5.54 | -5.84 | ❌ |
| ADBE | 20200115 | +4.26 | +3.28 | ❌ |
| ADBE | 20200116 | -0.26 | +2.09 | ✅ |
| ADBE | 20200117 | +2.61 | +4.43 | ✅ |
| AVGO | 20200113 | +5.81 | +3.46 | ❌ |
| AVGO | 20200114 | +1.26 | +2.06 | ✅ |
| AVGO | 20200115 | +4.55 | +1.70 | ❌ |
| AVGO | 20200116 | -1.59 | +0.11 | ✅ |
| AVGO | 20200117 | +2.61 | +2.39 | ❌ |
| INTC | 20200113 | +6.12 | -0.40 | ❌ |
| INTC | 20200114 | +0.81 | -1.34 | ❌ |
| INTC | 20200115 | -0.33 | -0.40 | ❌ |
| INTC | 20200116 | -0.13 | +1.03 | ✅ |
| INTC | 20200117 | +2.15 | +1.18 | ❌ |

## Notes

- All metrics: positive = cost (worse than benchmark), negative = better.
- Backtester known limitations (per NO Silent Fallback transparency):
  1. Marketable child orders fill at NBBO with **no self-impact modelling**.
  2. **Oversize fills** at best price (size > NBBO size at that level) are recorded in `oversize_count` not silently truncated.
  3. VWAP-following uses **same-day volume profile** (look-ahead bias). Phase 3+ will use rolling estimate from prior days.
  4. Both strategies share the same simulator → relative comparison is fair even with these biases.