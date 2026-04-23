# Phase 5B+ Stage 5 — 15-min Episode Re-calibration

**Generated**: 2026-05-04T14:57:09.717532-05:00 (ET)
**Seeds**: [1234, 5678, 9012]
**Sim window**: 15 min (09:30:00 – 09:45:00)

## Reference values
- Stage 4 #404 (5-min × 5 seeds): autocorr_l2=1.902, kurt_ratio=2.03, total_dist=2.436
- 5C Pilot (15-min × 1 seed, same #404 params): autocorr_l2=1.67, kurt_ratio=7.705, total_dist=3.077

## Stage 5 per-cell results

| Cell | Herder | max_size | pos_cap | n_succ | median dist | dist IQR | median ac_l2 | median kurt_ratio |
|---|---|---|---|---|---|---|---|---|
| 501 | 50 | 7 | 50 | 3/3 | **3.077** | [2.991, 3.291] | **1.670** | **7.71** |
| 502 | 30 | 5 | 20 | 3/3 | **3.362** | [3.301, 3.379] | **1.823** | **8.93** |
| 503 | 50 | 3 | 15 | 3/3 | **3.227** | [3.143, 3.271] | **1.786** | **8.24** |
| 504 | 25 | 5 | 25 | 3/3 | **3.218** | [3.209, 3.432] | **1.702** | **9.23** |

## Verdict

**AUTOCORR_OK_KURT_HIGH**: 🟡 Autocorr 維持但 kurt 仍偏高 — 試更小 max_size

### Best cell: #501
- Herder=50, max_size=7, pos_cap=50
- Median total_distance: **3.077** (pilot baseline 3.077)
- Median autocorr_l2: **1.670**
- Median kurtosis_ratio: **7.71** (pilot baseline 7.71)