# Phase 3: AC vs TWAP vs VWAP-following Report

**Setup**: sell 10,000 shares 10:00–11:00 ET, 60 child orders.
**Tickers**: AAPL, AMZN, AMD, TSLA, NVDA; **Dates**: 20200113, 20200114, 20200115, 20200116, 20200117
**AC parameters**: literature priors for η, γ; σ estimated from intraday mids; λ_risk_averse=0.001

**Total runs**: 100 (100 ok / 0 failed)

## Per-strategy summary (median across all (ticker, date))

| Strategy | VWAP slip (bps) | IS (bps) | Eff spread | Markout 60s | Reversion 5m | Price var | POV |
|---|---|---|---|---|---|---|---|
| **twap** | +1.78 | -5.77 | 2.28 | +1.28 | -2.20 | 0.5197 | 0.31% |
| **vwap_following** | +1.40 | -13.22 | 2.14 | +1.41 | -2.02 | 0.5125 | 0.31% |
| **ac_risk_neutral** | +1.78 | -5.77 | 2.28 | +1.28 | -2.20 | 0.5197 | 0.31% |
| **ac_risk_averse** | +1.34 | -11.09 | 2.48 | +1.82 | -1.22 | 0.5197 | 0.31% |

## Sanity check: AC risk-neutral should ≡ TWAP

- TWAP avg IS = -21.77 bps
- AC-RN avg IS = -21.77 bps
- Difference = +0.00 bps → ✅ AC-RN ≈ TWAP (as theory predicts)

## Risk-averse AC effect on price variance

- TWAP avg price variance = 0.9924
- AC risk-averse avg price variance = 1.0005
- Change = +0.8% — risk-averse should reduce variance by trading earlier

## IS per (ticker, date) — AC risk-averse vs TWAP

| Ticker | Date | TWAP IS | VWAP-follow IS | AC RN IS | AC risk-averse IS | Best |
|---|---|---|---|---|---|---|
| AAPL | 20200113 | +31.39 | +30.59 | +31.39 | +27.01 | AC-RA |
| AAPL | 20200114 | -71.42 | -69.12 | -71.42 | -52.01 | TWAP |
| AAPL | 20200115 | +26.01 | +24.78 | +26.01 | +18.72 | AC-RA |
| AAPL | 20200116 | +2.88 | +3.05 | +2.88 | -5.52 | AC-RA |
| AAPL | 20200117 | +14.24 | +13.05 | +14.24 | +11.06 | AC-RA |
| AMZN | 20200113 | +6.69 | +8.26 | +6.69 | +3.35 | AC-RA |
| AMZN | 20200114 | -45.98 | -44.30 | -45.98 | -42.29 | TWAP |
| AMZN | 20200115 | +26.00 | +25.81 | +26.00 | +19.63 | AC-RA |
| AMZN | 20200116 | +24.45 | +26.64 | +24.45 | +15.76 | AC-RA |
| AMZN | 20200117 | +24.11 | +26.09 | +24.11 | +19.13 | AC-RA |
| AMD | 20200113 | -1.36 | -4.19 | -1.36 | -11.09 | AC-RA |
| AMD | 20200114 | -54.33 | -55.77 | -54.33 | -47.45 | VWAP-f |
| AMD | 20200115 | -73.75 | -76.99 | -73.75 | -43.61 | VWAP-f |
| AMD | 20200116 | +7.38 | -14.75 | +7.38 | -31.09 | AC-RA |
| AMD | 20200117 | -22.39 | -24.38 | -22.39 | -19.31 | VWAP-f |
| TSLA | 20200113 | -49.19 | -49.62 | -49.19 | -40.16 | VWAP-f |
| TSLA | 20200114 | -177.40 | -167.70 | -177.40 | -92.20 | TWAP |
| TSLA | 20200115 | -63.69 | -66.37 | -63.69 | -62.72 | VWAP-f |
| TSLA | 20200116 | -67.53 | -59.05 | -67.53 | -71.62 | AC-RA |
| TSLA | 20200117 | -54.79 | -53.03 | -54.79 | -51.67 | TWAP |
| NVDA | 20200113 | -12.18 | -13.22 | -12.18 | -7.46 | VWAP-f |
| NVDA | 20200114 | -35.21 | -37.18 | -35.21 | -30.47 | VWAP-f |
| NVDA | 20200115 | +21.25 | +23.02 | +21.25 | +12.99 | AC-RA |
| NVDA | 20200116 | +6.29 | +6.75 | +6.29 | -0.54 | AC-RA |
| NVDA | 20200117 | -5.77 | -6.58 | -5.77 | -8.06 | AC-RA |

## Notes (NO Silent Fallback)

- **AC theory caveat**: with linear-impact assumption, risk-neutral AC ≡ TWAP (any schedule has same expected cost). Differences here come only from finite-sample noise, not strategy logic.
- AC risk-averse trades sooner → exposes less inventory to price drift, but pays slightly more in immediate impact.
- Backtester does NOT model self-impact, so the marginal cost of AC risk-averse front-loading is understated. Real-world impact would be larger.
- η, γ are **literature priors** (5 days too few for reliable estimation; see calibration_notes.md).
- σ estimated from intraday mid log-returns of each (ticker, date) → variable input per run.
- All metrics: positive = cost (worse than benchmark).