# PATCHES.md — abides-sim Vendoring Notes

## Source

Vendored from: https://github.com/abides-sim/abides
Commit SHA pinned: **c4bf157678928934417aba6073eb0651aeaf6d15**
Original LICENSE: BSD 3-Clause (see `LICENSE.txt`)
Vendored on: 2026-05-03

## Why vendored (not git submodule)

- Upstream `abides-sim/abides` repo has been inactive since 2020
- Risk of repo deletion would kill our Phase 5/6 research
- Pinning by commit + committing into our own repo guarantees reproducibility
- BSD 3-Clause licence permits redistribution

## Patches applied

All patches address **deprecated pandas import** (`pandas.io.json` namespace removed in pandas ≥ 1.4):

### Patch 1 — `util/OrderBook.py` (line 12)

```diff
-from pandas.io.json import json_normalize
+from pandas import json_normalize
```

**Why**: `pandas.io.json.json_normalize` was deprecated in pandas 1.0 and removed from the public namespace in 1.4. The function moved to `pd.json_normalize` (top-level). Behaviour is identical.

### Patch 2 — `util/simulation_run_stats.py` (line 3)

```diff
-from pandas.io.json import json_normalize
+from pandas import json_normalize
```

Same reason as Patch 1.

### Patch 3 — `util/formatting/convert_order_stream.py` (line 3)

```diff
-from pandas.io.json import json_normalize
+from pandas import json_normalize
```

Same reason as Patch 1.

## Verification of patches

After applying all 3 patches, the following smoke test passes:

```bash
PYTHONPATH=vendor/abides-sim .venv-abides/bin/python vendor/abides-sim/abides.py \
    -c rmsc03 -t ABM -d 20200113 -s 1234 \
    --start-time '09:30:00' --end-time '09:35:00' \
    --log_dir test_output/
```

Expected: 5-min simulation completes in < 30 seconds with all 5 agent types
(NoiseAgent, ValueAgent, MomentumAgent, AdaptivePOVMarketMakerAgent, ExecutionAgent)
producing trades and an order-book log.

## Re-applying patches if upstream changes

If we ever re-vendor a different SHA from upstream, run:

```bash
cd vendor/abides-sim
sed -i '' 's|from pandas.io.json import json_normalize|from pandas import json_normalize|' \
    util/OrderBook.py util/simulation_run_stats.py util/formatting/convert_order_stream.py
```

### Patch 4 — `agent/OrderBookImbalanceAgent.py` (lines 5-7) — Phase 5B+ (2026-05-04)

```diff
-import matplotlib
-matplotlib.use('TkAgg')
-import matplotlib.pyplot as plt
+# Phase 5B+ patch (2026-05-04): removed forced matplotlib backend (TkAgg breaks
+# headless calibration runs). Plot helper is unused at runtime; only triggered
+# if user invokes plotting helper externally.
```

**Why**: 強制 `matplotlib.use('TkAgg')` 讓任何 headless 環境（如 calibration script）載入 OBI agent 時都會掛掉。OBI 的 plotting code (line 190 `#plt.show()`) 已 commented out 不會執行。

### Patch 5 — `agent/ExchangeAgent.py` (line 310) — Phase 5B+ (2026-05-04)

```diff
-((orderbook_last_update - last_agent_update).delta >= freq)):
+((orderbook_last_update - last_agent_update).value >= freq)):
```

**Why**: `pd.Timedelta.delta` 在新 pandas (≥ 2.0) deprecated 且最終移除。`.value` 給出同樣的 nanoseconds 整數值，跟原邏輯（freq 也是 nanoseconds）相容。觸發路徑：當任何 agent (OBI / Herder / MarketMaker subscribe=True) 訂閱市場資料時，`publishOrderBookData()` 會走這行。Phase 5A smoke test 不打到（rmsc03 預設 MarketMaker subscribe=False），但 Phase 5B+ 加 OBI/Herder 後就掛。

## Known unfixed upstream bugs（我們知道但沒修）

### `agent/market_makers/AdaptiveMarketMakerAgent.py:168` — `mid` undefined

```python
self.placeOrders(mid)  # UnboundLocalError under certain seed/config combinations
```

**症狀**：某些 (seed × num_agents × MM aggressiveness) 組合會觸發。Phase 5B 9-cell 中 cells 5/7（25k noise + timid MM）必中；Phase 5B+ Stage 3 cell 304 seed 5678 偶發。

**為什麼沒修**：需要修 `placeOrders` 路徑邏輯，影響範圍大。我們的 calibration runner 把這視為「flaky cell」記錄錯誤但繼續，多 seed 平均吸收。

## Files created by us (NOT from upstream)

- `PATCHES.md` (this file)
- `config/calib_aapl.py` (Phase 5B/5B+ calibration config，含 OBI/Herder CLI args)
- `agent/examples/HerderAgent.py` (Phase 5B+ Stage 3，新 Lux 1998 chartist agent，~180 lines)
