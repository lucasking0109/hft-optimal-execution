# Phase E.4 — Mini Walk-Forward CV (4 folds)

**Generated**: 2026-05-12T23:10:01.857271-05:00 (ET)
**Eval pool**: 20 tickers (`AAPL, ADP, BIDU, BMRN, CHKP, CPRT, CTAS, CTXS, FOXA, GILD, INTU, LRCX, PYPL, QQQ, REGN, SGEN, TCOM, TSLA, TXN, UAL`)
**Train pool**: 20 tickers (same across folds)
**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing, spread-cost fills

## Per-fold median IS

| Fold | Train days | Test day | n tickers | RL | VWAP-follow | AC-RA | TWAP | RL − VWAP |
|---|---|---|---|---|---|---|---|---|
| 1 | 20200113 | 20200114 | 20 | -17.83 | -18.59 | -20.36 | -17.52 | **+0.76** |
| 2 | 20200113, 20200114 | 20200115 | 20 | +3.16 | +3.49 | +4.32 | +4.89 | **-0.34** |
| 3 | 20200113, 20200114, 20200115 | 20200116 | 20 | +2.03 | +9.05 | +3.47 | +10.15 | **-7.02** |
| 4 | 20200113, 20200114, 20200115, 20200116 | 20200117 | 20 | -3.74 | +3.16 | -0.95 | +2.92 | **-6.89** |

**Cross-fold RL − VWAP deltas**: ['+0.76', '-0.34', '-7.02', '-6.89']
- mean: **-3.372 bps**
- median: -3.616 bps
- std across folds: 3.607 bps

## Limitations

- Fold 1 trains on only 1 day. Policy is severely data-limited; treat as lower bound.
- Folds 2-4 progressively gain training data. Fold 4 is closest to Phase E.3 setup.
- Eval pool size is small to keep runtime manageable; not representative of full 97-ticker universe.