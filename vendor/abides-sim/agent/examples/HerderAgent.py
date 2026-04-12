"""HerderAgent — Lux 1998 chartist / trend-follower for ABIDES Phase 5B+.

Designed to induce volatility clustering by reactive herding behavior:
  1. Subscribes to LOB market data → no warmup (fixes MomentumAgent's 16.7-min
     warmup issue with MA crossover).
  2. Computes mid-price drift over a per-agent random lookback window
     (heterogeneous lookback ∈ [3, 30] sec, CHAD framework recommendation —
     avoid artificial resonance from synchronous agents).
  3. Triggers when |drift_bps| exceeds a *data-calibrated* threshold
     (NOT magic number — measured from real AAPL P75 |5-sec drift|).
  4. Places **marketable limit orders** (price = far ask + tolerance ticks for
     buy; far bid - tolerance for sell). Per Megan ICAIF 2021 (Strategic
     Reasoning Group), marketable limit orders are MORE effective than pure
     market orders for triggering cascade behavior; also avoids book-walk
     crash if liquidity is thin.
  5. Position cap prevents runaway accumulation (real trader risk limit).
  6. computation_delay_ns models "speed of thought" — herder doesn't react
     to every tick (lit recommendation: prevents unrealistic high-frequency
     oscillations).
  7. Comprehensive diagnostic logging (signal_count / cap_blocked /
     orders_placed / block_rate) at sim end via logEvent.

References:
  - Lux 1998 (chartist switching → vol clustering)
  - Brock-Hommes 1998 (heterogeneous agent ecology)
  - Strategic Reasoning Group / Megan ICAIF 2021 (marketable limit > market)
  - CHAD framework (heterogeneous lookback windows)
  - 2507.06345 (tactical reactive trader)

Phase 5B+ Stage 3.
"""

import collections

import numpy as np
import pandas as pd

from agent.TradingAgent import TradingAgent


class HerderAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, starting_cash,
                 lookback_window_secs=None,        # required, from Step 0
                 entry_threshold_bps=None,         # required, from Step 0
                 max_size=None,                    # required, from Step 0
                 position_cap=50,
                 tolerance_ticks=5,
                 subscribe_freq=int(1e9),          # 1 Hz (1e9 nanoseconds)
                 computation_delay_ns=1_000_000,   # 1 ms reaction time
                 levels=1,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, random_state=random_state)
        if entry_threshold_bps is None or max_size is None or lookback_window_secs is None:
            raise ValueError(
                "HerderAgent: lookback_window_secs + entry_threshold_bps + max_size "
                "must come from Step 0 data calibration (don't pass None)."
            )
        self.symbol = symbol
        self.lookback_window_secs = float(lookback_window_secs)
        self.entry_threshold_bps = float(entry_threshold_bps)
        self.max_size = int(max_size)
        self.position_cap = int(position_cap)
        self.tolerance_ticks = int(tolerance_ticks)
        self.subscribe_freq = int(subscribe_freq)
        self.computation_delay_ns = int(computation_delay_ns)
        self.levels = int(levels)

        self.subscription_requested = False
        self.mid_buffer: "collections.deque[tuple[pd.Timestamp, float]]" = collections.deque()

        # Diagnostic counters
        self._signal_count = 0
        self._cap_blocked_count = 0
        self._order_placed_count = 0

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        # Apply lit-recommended computation delay (speed of thought).
        # Must happen AFTER kernel is bound (Agent.setComputationDelay calls self.kernel.*).
        self.setComputationDelay(self.computation_delay_ns)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if not self.subscription_requested:
            self.requestDataSubscription(self.symbol, levels=self.levels,
                                         freq=self.subscribe_freq)
            self.subscription_requested = True

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body.get('msg') != 'MARKET_DATA':
            return
        bids = msg.body.get('bids') or []
        asks = msg.body.get('asks') or []
        if not bids or not asks:
            return
        best_bid_price = bids[0][0]
        best_ask_price = asks[0][0]
        if best_bid_price <= 0 or best_ask_price <= 0 or best_ask_price <= best_bid_price:
            return
        mid = (best_bid_price + best_ask_price) / 2.0
        # 1) Append to buffer
        self.mid_buffer.append((currentTime, mid))
        # 2) Trim by lookback window
        cutoff = currentTime - pd.Timedelta(seconds=self.lookback_window_secs)
        while self.mid_buffer and self.mid_buffer[0][0] < cutoff:
            self.mid_buffer.popleft()
        # 3) Need ≥ 2 obs to compute drift
        if len(self.mid_buffer) < 2:
            return
        oldest_mid = self.mid_buffer[0][1]
        if oldest_mid <= 0:
            return
        drift_bps = (mid - oldest_mid) / oldest_mid * 1e4
        # 4) Trigger threshold
        if abs(drift_bps) < self.entry_threshold_bps:
            return
        self._signal_count += 1
        # 5) Direction + size scaling
        is_buy = drift_bps > 0
        intensity = min(1.0, abs(drift_bps) / (self.entry_threshold_bps * 5.0))
        size = max(1, int(round(self.max_size * intensity)))
        # 6) Position cap
        current_pos = self.holdings.get(self.symbol, 0)
        if (is_buy and current_pos >= self.position_cap) or \
           (not is_buy and current_pos <= -self.position_cap):
            self._cap_blocked_count += 1
            return
        # 7) Marketable limit (aggressive cross + tolerance)
        if is_buy:
            limit_price = int(best_ask_price + self.tolerance_ticks)
        else:
            limit_price = int(best_bid_price - self.tolerance_ticks)
        if limit_price <= 0:
            return
        self.placeLimitOrder(self.symbol, size, is_buy, limit_price)
        self._order_placed_count += 1

    def kernelStopping(self):
        # Log diagnostics for post-hoc analysis
        block_rate = (self._cap_blocked_count / self._signal_count) if self._signal_count else 0.0
        fill_logged = self._order_placed_count
        self.logEvent('HERDER_DIAGNOSTICS', {
            'signal_count': self._signal_count,
            'cap_blocked': self._cap_blocked_count,
            'orders_placed': fill_logged,
            'block_rate': float(block_rate),
            'lookback_secs': self.lookback_window_secs,
            'entry_threshold_bps': self.entry_threshold_bps,
            'max_size': self.max_size,
        })
        super().kernelStopping()

    def getWakeFrequency(self):
        # Subscription drives behavior; wake just to renew subscription if needed.
        return pd.Timedelta('60s')
