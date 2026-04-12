# Phase 5A.6 — abides-sim Output Schema Report

**Generated**: 2026-05-03
**Source**: 5-minute rmsc03 simulation, AAPL-style symbol "ABM", seed 1234.
**Output dir**: `log/test_output/` (5 files, total ~530KB compressed)

---

## File overview

| File | Size | Type | What it contains |
|---|---|---|---|
| `EXCHANGE_AGENT.bz2` | 380KB | DataFrame | **THE goldmine** — all market events (orders, executions, BBO updates, tape) |
| `ORDERBOOK_ABM_FULL.bz2` | 100KB | DataFrame | L2 order book snapshots: time × price level → signed depth |
| `summary_log.bz2` | 32KB | DataFrame | Lifecycle events for ALL agents (NoiseAgent / ValueAgent / etc.) |
| `fundamental_ABM.bz2` | 16KB | DataFrame | "True value" series the value agents track |
| `POV_EXECUTION_AGENT.bz2` | 4KB | DataFrame | Per-agent lifecycle (e.g. STARTING_CASH, FINAL_HOLDINGS) |

All files: `pickle`-protocol DataFrames compressed with bzip2. Read with
`pd.read_pickle(path, compression='bz2')`.

---

## 1. `EXCHANGE_AGENT.bz2` — Primary trade & quote stream

```
shape: (67477 rows, 2 cols)
index: EventTime (datetime, nanosecond precision)
columns: EventType (str), Event (object — varies by EventType)
```

**Event distribution (5-min sim)**:

| EventType | Count | Event payload type | Purpose |
|---|---|---|---|
| `ORDER_EXECUTED` | 12,920 | dict | **Trade fills** — see schema below |
| `LIMIT_ORDER` | 8,248 | dict | Order placements |
| `BEST_ASK` | 8,243 | str `"SYM,price,qty"` | Top-of-book updates |
| `BEST_BID` | 8,201 | str `"SYM,price,qty"` | Top-of-book updates |
| `QUERY_SPREAD` | 7,804 | (varies) | Internal queries |
| `WHEN_MKT_OPEN` | 5,128 | int (ns since epoch) | Market open broadcast |
| `WHEN_MKT_CLOSE` | 5,128 | int | Market close broadcast |
| `ORDER_ACCEPTED` | 5,094 | dict | Order accepted by exchange |
| `LAST_TRADE` | 3,564 | str `"qty,$price"` | Tape print of each trade |
| `CANCEL_ORDER` | 1,555 | dict | Cancel request |
| `ORDER_CANCELLED` | 1,531 | dict | Cancel confirmed |
| `QUERY_TRANSACTED_VOLUME` | 60 | (varies) | Internal query |

### `ORDER_EXECUTED` schema (most important)

```python
{
    'agent_id':     int,    # who placed the order
    'time_placed':  str (ISO 8601 ns),  # when it was placed
    'symbol':       str,    # 'ABM' or 'AAPL' depending on config
    'quantity':     int,    # how many shares filled
    'is_buy_order': int,    # 1 = buy, 0 = sell  ← AGGRESSOR SIDE
    'order_id':     int,    # unique order identifier
    'fill_price':   int,    # in CENTS (99791 = $997.91)
    'tag':          str | None,
    'limit_price':  int,    # original limit (in cents)
}
```

**Critical for our research**: aggressor side comes for free via `is_buy_order`
(no Lee-Ready inference needed, unlike real equity TAQ which doesn't tag side).

### `BEST_BID` / `BEST_ASK` schema

String format: `"SYMBOL,price_cents,qty"` (e.g., `"ABM,99791,22"`).

### Price units (IMPORTANT)

`fill_price` and BBO prices are in **integer cents** (rmsc03 default scale).
Default fundamental is 100,000 cents = $1,000.00. For AAPL @ $315 we'll calibrate
fundamental to `31_500_00`; downstream loader divides by 10000 (or equivalent)
to convert to dollars.

---

## 2. `ORDERBOOK_ABM_FULL.bz2` — L2 depth snapshots

```
shape: (7146 rows, 265 cols)
index: QuoteTime (datetime, ns precision; range 09:30:00 → 09:35:00)
columns: integer price levels in cents (e.g., 99348, 99414, ...)
values: SIGNED quantity at that price
        positive = ask side
        negative = bid side
        zero     = no resting order at this level/time
```

Sample row (only nonzero cells):

```
99791: -1594   # bid 1594 shares at $997.91
99859:    24   # ask  24 shares at $998.59
99945:    38   # ask  38 shares at $999.45
100018:   47   # ask  47 shares at $1000.18
100080:   49   # ask  49 shares at $1000.80
```

This is the **complete L2 depth** at every event-driven update. Sparsity = 0%
because zero-value cells are stored too — handy for vectorized analysis.

---

## 3. `fundamental_ABM.bz2` — "True value" series

```
shape: (2123 rows, 1 col)
index: FundamentalTime (datetime, ns precision)
columns: ['FundamentalValue']  (float, in cents)
```

The mean-reverting signal value-investor agents track. Useful for understanding
the signal-to-noise level of the synthetic regime.

---

## 4. `summary_log.bz2` — Per-agent lifecycle

```
shape: (20484 rows, 4 cols)
columns: ['AgentID', 'AgentStrategy', 'EventType', 'Event']
```

Each row: one event per agent (e.g., STARTING_CASH, FINAL_HOLDINGS, ORDER_PLACED).
Useful to enumerate which agent types existed (NoiseAgent, ValueAgent, ...) and
their starting parameters.

---

## 5. `POV_EXECUTION_AGENT.bz2` — Custom execution agent log

Same `(EventTime, EventType, Event)` schema as EXCHANGE_AGENT but only for the
POV ExecutionAgent (in rmsc03, this agent doesn't actually trade; it's a placeholder).

For our research, we'll add our own `OurExecutionAgent` that we plug into the
ABIDES kernel and track via this file.

---

## Loader plan (for Phase 5C.2)

### Step 1: Extract trades

```python
def load_abides_trades(log_dir: Path) -> pl.DataFrame:
    df = pd.read_pickle(log_dir / 'EXCHANGE_AGENT.bz2', compression='bz2')
    executed = df[df['EventType'] == 'ORDER_EXECUTED'].copy()
    # Expand the dict column into multiple columns
    expanded = pd.json_normalize(executed['Event'])
    expanded['EventTime'] = executed.index.values
    return pl.from_pandas(expanded[[
        'EventTime', 'symbol', 'quantity', 'is_buy_order',
        'fill_price', 'agent_id', 'order_id'
    ]])
```

### Step 2: Reconstruct top-of-book over time

```python
def load_abides_bbo(log_dir: Path) -> pl.DataFrame:
    df = pd.read_pickle(log_dir / 'EXCHANGE_AGENT.bz2', compression='bz2')
    bbo_events = df[df['EventType'].isin(['BEST_BID', 'BEST_ASK'])].copy()
    # Parse "SYM,price,qty" string
    parsed = bbo_events['Event'].str.split(',', expand=True)
    parsed.columns = ['symbol', 'price', 'qty']
    parsed['side'] = bbo_events['EventType'].map({'BEST_BID': 'bid', 'BEST_ASK': 'ask'})
    parsed['price'] = parsed['price'].astype(int)  # cents
    parsed['qty'] = parsed['qty'].astype(int)
    parsed['EventTime'] = bbo_events.index
    return pl.from_pandas(parsed)
```

### Step 3: Load L2 depth (lazy / on demand)

```python
def load_abides_orderbook(log_dir: Path) -> pl.DataFrame:
    raw = pd.read_pickle(log_dir / 'ORDERBOOK_ABM_FULL.bz2', compression='bz2')
    # Convert wide → long format for parquet efficiency
    long = raw.reset_index().melt(
        id_vars=['QuoteTime'], var_name='price_cents', value_name='signed_qty'
    )
    long = long[long['signed_qty'] != 0]
    long['side'] = pl.Series((long['signed_qty'] > 0).map({True:'ask',False:'bid'}))
    long['qty'] = long['signed_qty'].abs()
    return pl.from_pandas(long[['QuoteTime', 'price_cents', 'side', 'qty']])
```

---

## Schema alignment with our real TAQ data

| ABIDES output | Maps to our real TAQ column |
|---|---|
| `ORDER_EXECUTED.fill_price` (cents) | `Price` (dollars) — divide by 100 |
| `ORDER_EXECUTED.quantity` | `Quantity` |
| `ORDER_EXECUTED.is_buy_order` | aggressor side (real eq TAQ doesn't have; futures does) |
| `ORDER_EXECUTED.EventTime` | `Timestamp` |
| `BEST_BID` / `BEST_ASK` | `QUOTE BID NB` / `QUOTE ASK NB` |
| symbol = `'ABM'` | rebrand to `'AAPL'` in calibration config |

After Phase 5C loader, synthetic episodes will have **same column schema** as real TAQ,
so Phase 5E ExecutionEnv can swap mode without changing observation logic.

---

## Status: ✅ Phase 5A complete

- ✅ 5A.1-5A.2: vendored + patched (commit SHA c4bf157678928934417aba6073eb0651aeaf6d15)
- ✅ 5A.3-5A.4: `.venv-abides` built, deps locked in `requirements-abides.txt`
- ✅ 5A.5: smoke test 24 sec total (5-min sim takes 5 sec; rest is venv startup)
- ✅ 5A.6: schema documented (this file)

Ready for Phase 5B calibration.
