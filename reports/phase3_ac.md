# Phase 3: AC vs TWAP vs VWAP-following Report

**Setup**: sell 10,000 shares 10:00–11:00 ET, 60 child orders.
**Tickers**: AAPL, AMZN, AMD, TSLA, NVDA; **Dates**: 20200113, 20200114, 20200115, 20200116, 20200117
**AC parameters**: literature priors for η, γ; σ estimated from intraday mids; λ_risk_averse=0.001

**Total runs**: 100 (100 ok / 0 failed)

## Per-strategy summary (median across all (ticker, date))

| Strategy | VWAP slip (bps) | IS (bps) | Eff spread | Markout 60s | Reversion 5m | Price var | POV |
|---|---|---|---|---|---|---|---|
| **twap** | +1.78 | -5.77 | 2.28 | +1.28 | -2.20 | 0.5197 | 0.31% |
| **vwap_following** | +0.96 | -4.51 | 2.10 | +2.14 | -2.02 | 0.5125 | 0.31% |
| **ac_risk_neutral** | +1.78 | -5.77 | 2.28 | +1.28 | -2.20 | 0.5197 | 0.31% |
| **ac_risk_averse** | +0.61 | -13.24 | 2.51 | +2.16 | -2.20 | 0.5197 | 0.31% |

## Sanity check: AC risk-neutral should ≡ TWAP

- TWAP avg IS = -21.77 bps
- AC-RN avg IS = -21.77 bps
- Difference = +0.00 bps → ✅ AC-RN ≈ TWAP (as theory predicts)

## Risk-averse AC effect on price variance

- TWAP avg price variance = 0.9924
- AC risk-averse avg price variance = 0.9924
- Change = +0.0% — risk-averse should reduce variance by trading earlier

## IS per (ticker, date) — AC risk-averse vs TWAP

| Ticker | Date | TWAP IS | VWAP-follow IS | AC RN IS | AC risk-averse IS | Best |
|---|---|---|---|---|---|---|
| AAPL | 20200113 | +31.39 | +29.67 | +31.39 | +21.21 | AC-RA |
| AAPL | 20200114 | -71.42 | -68.34 | -71.42 | -49.88 | TWAP |
| AAPL | 20200115 | +26.01 | +24.84 | +26.01 | +16.34 | AC-RA |
| AAPL | 20200116 | +2.88 | -0.31 | +2.88 | -10.41 | AC-RA |
| AAPL | 20200117 | +14.24 | +13.31 | +14.24 | +6.63 | AC-RA |
| AMZN | 20200113 | +6.69 | +6.20 | +6.69 | -0.20 | AC-RA |
| AMZN | 20200114 | -45.98 | -45.92 | -45.98 | -39.10 | TWAP |
| AMZN | 20200115 | +26.00 | +26.63 | +26.00 | +14.86 | AC-RA |
| AMZN | 20200116 | +24.45 | +22.86 | +24.45 | +8.66 | AC-RA |
| AMZN | 20200117 | +24.11 | +23.64 | +24.11 | +14.19 | AC-RA |
| AMD | 20200113 | -1.36 | -4.26 | -1.36 | -13.24 | AC-RA |
| AMD | 20200114 | -54.33 | -57.88 | -54.33 | -47.79 | VWAP-f |
| AMD | 20200115 | -73.75 | -64.53 | -73.75 | -43.49 | TWAP |
| AMD | 20200116 | +7.38 | -2.56 | +7.38 | -28.85 | AC-RA |
| AMD | 20200117 | -22.39 | -19.53 | -22.39 | -17.88 | TWAP |
| TSLA | 20200113 | -49.19 | -47.69 | -49.19 | -44.30 | TWAP |
| TSLA | 20200114 | -177.40 | -160.40 | -177.40 | -138.86 | TWAP |
| TSLA | 20200115 | -63.69 | -65.84 | -63.69 | -66.85 | AC-RA |
| TSLA | 20200116 | -67.53 | -61.28 | -67.53 | -72.40 | AC-RA |
| TSLA | 20200117 | -54.79 | -51.51 | -54.79 | -52.52 | TWAP |
| NVDA | 20200113 | -12.18 | -11.77 | -12.18 | -6.96 | TWAP |
| NVDA | 20200114 | -35.21 | -37.44 | -35.21 | -30.33 | VWAP-f |
| NVDA | 20200115 | +21.25 | +22.97 | +21.25 | +11.69 | AC-RA |
| NVDA | 20200116 | +6.29 | +3.55 | +6.29 | -4.42 | AC-RA |
| NVDA | 20200117 | -5.77 | -4.51 | -5.77 | -8.52 | AC-RA |

## Notes (NO Silent Fallback)

- **AC theory caveat**: with linear-impact assumption, risk-neutral AC ≡ TWAP (any schedule has same expected cost). Differences here come only from finite-sample noise, not strategy logic.
- AC risk-averse trades sooner → exposes less inventory to price drift, but pays slightly more in immediate impact.
- Backtester does NOT model self-impact, so the marginal cost of AC risk-averse front-loading is understated. Real-world impact would be larger.
- η, γ are **literature priors** (5 days too few for reliable estimation; see calibration_notes.md).
- σ estimated from intraday mid log-returns of each (ticker, date) → variable input per run.
- All metrics: positive = cost (worse than benchmark).