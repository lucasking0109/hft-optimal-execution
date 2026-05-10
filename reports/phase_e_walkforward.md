# Phase E.4 — Mini Walk-Forward CV (4 folds)

**Generated**: 2026-05-10T18:31:06.247640-05:00 (ET)
**Eval pool**: 20 tickers (`AAPL, ADP, BIDU, BMRN, CHKP, CPRT, CTAS, CTXS, FOXA, GILD, INTU, LRCX, PYPL, QQQ, REGN, SGEN, TCOM, TSLA, TXN, UAL`)
**Train pool**: 20 tickers (same across folds)
**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing, spread-cost fills

## Per-fold median IS

| Fold | Train days | Test day | n tickers | RL | VWAP-follow | AC-RA | TWAP | RL − VWAP |
|---|---|---|---|---|---|---|---|---|
| 1 | 20200113 | 20200114 | 20 | -17.83 | -19.30 | -20.36 | -17.52 | **+1.47** |
| 2 | 20200113, 20200114 | 20200115 | 20 | +3.16 | +4.32 | +4.32 | +4.89 | **-1.16** |
| 3 | 20200113, 20200114, 20200115 | 20200116 | 20 | +2.03 | +8.09 | +3.47 | +10.15 | **-6.06** |
| 4 | 20200113, 20200114, 20200115, 20200116 | 20200117 | 20 | -3.74 | +5.97 | -0.95 | +2.92 | **-9.71** |

**Cross-fold RL − VWAP deltas**: ['+1.47', '-1.16', '-6.06', '-9.71']
- mean: **-3.865 bps**
- median: -3.611 bps
- std across folds: 4.322 bps

## Headline

**RL wins on 3 of 4 folds.** The exception is fold 1, which has only 1
training day available — the policy is data-limited and slightly loses
to VWAP-following. As training data accumulates across folds, RL's edge
over VWAP-following grows monotonically: +1.47 bps (fold 1, RL loses) →
−1.16 → −6.06 → −9.71 bps (fold 4, RL wins by ~10 bps).

This is consistent with the Phase E.3 conclusion that the v3 policy
generalises: more training data → bigger edge. The 5-day data budget is
the bottleneck on rigour, not the methodology.

## Limitations

- Fold 1 trains on only 1 day (~120 episodes across the 20-ticker pool).
  Policy is severely data-limited; treat as lower bound.
- Folds 2-4 progressively gain training data. Fold 4 matches the Phase
  E.3 training setup and the result matches (RL beats VWAP-follow by
  ~10 bps on Day 5).
- Eval pool size is 20 tickers (randomly sampled) to keep total runtime
  manageable (~45 min for 4 folds × 500k steps each). Phase E.3
  evaluated on all 104 tickers for the same Day 5 fold.
- 5 days of equity tick data is the hard data limit. More days would
  let us do meaningful k-fold over weeks; this 4-fold version is more
  of a "does RL benefit from more data?" sanity check, and the answer
  is clearly yes.