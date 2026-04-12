# Phase 5B+ Stage 1 — Momentum Sensitivity

**Goal**: test whether MomentumAgent count meaningfully reduces vol_autocorr_l2 distance.

**Anchor cell**: prior best (500 noise, fund_vol=1e-3, aggressive MM)
**Varying**: num_momentum ∈ {25, 100, 400}
**Green-light threshold**: high-momentum cell autocorr_l2 < 1.0

## Per-cell results

| Cell | num_momentum | success | total_dist | autocorr_l2 | kurtosis | hill | spread KS | trade KS | runtime |
|---|---|---|---|---|---|---|---|---|---|
| 101 | 25 | ✅ | **2.332** | **1.893** | 2.39 | 0.63 | 0.691 | 0.643 | 18.7s |
| 102 | 100 | ✅ | **2.631** | **2.074** | 3.37 | 0.43 | 0.579 | 0.762 | 76.3s |
| 103 | 400 | ❌ | — | — | — | — | — | — | 300.0s |

## Verdict

🔴 STAGE-1-FAIL — fewer than 3 cells succeeded

→ Per NO Silent Fallback: STOP and present Lucas the choice between:
  - **(A)** Accept Phase 5B initial best cell #4 (total_dist=2.144) and add Phase 6 caveat.
  - **(B-alt)** Try a different agent type (e.g. add OrderBookImbalanceAgent or HBL).
  - **(C)** Switch evaluation metric (drop autocorr from total_distance with explicit caveat).
