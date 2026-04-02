# Phase 4: Multi-Venue SOR Routing Analysis

**Tickers**: AAPL, AMZN, MSFT, NVDA, AMD; **Dates**: 20200113, 20200114, 20200115, 20200116, 20200117; **Total venue-day rows**: 350


## Cross-ticker venue ranking (averaged over 5 tickers × 5 days)

| Venue | Routable | Volume share | NBBO share (bid) | NBBO share (ask) | Avg depth | Adverse selection (bps) | Composite score |
|---|---|---|---|---|---|---|---|
| NASDAQ | ✅ | 24.88% | 43.43% | 44.85% | 668 | -1.23 | +1.61 |
| CSE | ✅ | 1.61% | 1.55% | 1.58% | 2803 | -8.03 | +1.43 |
| BATS | ✅ | 10.16% | 28.33% | 26.91% | 455 | +0.09 | +0.32 |
| ARCA | ✅ | 6.70% | 9.09% | 9.28% | 347 | -0.09 | +0.04 |
| EDGX | ✅ | 6.43% | 5.56% | 4.69% | 347 | +0.08 | +0.01 |
| FINRA | ❌ | 42.04% | 0.00% | 0.00% | 0 | -0.58 | +0.00 |
| NYSE | ✅ | 1.56% | 4.11% | 4.25% | 240 | -0.18 | -0.37 |
| IEX | ✅ | 1.92% | 0.26% | 0.27% | 162 | -0.15 | -0.38 |
| NASDAQ PSX | ✅ | 0.94% | 4.47% | 4.62% | 298 | -0.09 | -0.40 |
| EDGA | ✅ | 1.35% | 1.84% | 2.09% | 180 | -0.28 | -0.40 |
| BATS Y | ✅ | 1.10% | 0.55% | 0.45% | 156 | -0.41 | -0.42 |
| NASDAQ BX | ✅ | 0.80% | 0.38% | 0.57% | 141 | -0.30 | -0.45 |
| AMEX | ✅ | 0.13% | 0.14% | 0.13% | 245 | +0.09 | -0.50 |
| NSE | ✅ | 0.39% | 0.29% | 0.31% | 123 | -0.22 | -0.50 |

## Top-5 routable allocation comparison

| Venue | Naive (volume %) | SOR (score %) | SOR − Naive |
|---|---|---|---|
| NASDAQ | 49.67% | 59.25% | +9.58pp |
| BATS | 20.28% | 16.41% | -3.87pp |
| ARCA | 13.38% | 12.38% | -0.99pp |
| EDGX | 12.84% | 11.95% | -0.89pp |
| IEX | 3.83% | 0.00% | -3.83pp |

## Notes (NO Silent Fallback)

- **Non-routable venues** (excluded from SOR): ['FINRA'].
  - FINRA isn't an exchange — it reports off-exchange (dark pool / internalizer) trades. SOR can't physically route there.
- Adverse selection uses Lee-Ready signed markout @ 60 sec. Lower (more negative) = better.
- Composite score weights: 0.5 volume + 0.3 depth − 0.2 adverse selection (z-scored across routable venues).
- Lookback is just 5 days — production SOR rules need months of data to stabilise. Use this analysis as methodology demo, not as live rule.
- Composite score is *relative* (z-scored), so absolute values aren't comparable across stocks.