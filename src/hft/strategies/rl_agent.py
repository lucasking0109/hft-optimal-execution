"""Wrap a trained PPO policy as an offline ExecutionStrategy.

Bridges the offline-only `ExecutionStrategy` contract used by Phase B
backtests to a per-step RL policy. At schedule() time we spin up a
shadow ExecutionEnv pinned to the parent's exact (ticker, date,
start_ns), roll the policy through every step, and convert each step's
fill into a ChildOrder. The returned schedule is what the backtest
engine then replays against tick data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from hft.simulators.execution_env import ExecutionEnv
from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder


ObservationMode = Literal["v1", "v2"]


class RLAgentStrategy(ExecutionStrategy):
    """Run a trained PPO policy and emit the resulting child schedule."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        slice_minutes: int,
        step_seconds: int,
        n_steps: int,
        observation_mode: ObservationMode = "v2",
        max_action_per_step: float = 0.05,
        deterministic: bool = True,
    ):
        path = Path(model_path)
        if not path.exists():
            raise ValueError(
                f"RLAgentStrategy: model not found at {path}. "
                f"Train via scripts/train_ppo_v2_multihour.py first."
            )
        if step_seconds * n_steps != slice_minutes * 60:
            raise ValueError(
                f"step_seconds × n_steps must equal slice_minutes × 60 "
                f"({slice_minutes * 60})"
            )
        # Defer stable_baselines3 import — heavy.
        from stable_baselines3 import PPO
        self._model = PPO.load(str(path), device="cpu")
        self.slice_minutes = slice_minutes
        self.step_seconds = step_seconds
        self.n_steps = n_steps
        self.observation_mode = observation_mode
        self.max_action_per_step = max_action_per_step
        self.deterministic = deterministic
        self.name = f"rl_v2_{slice_minutes}m"

    def schedule(
        self,
        parent: ParentOrder,
        *,
        market_context: dict,
    ) -> list[ChildOrder]:
        # Build a shadow env pinned to this parent's exact anchor.
        env = ExecutionEnv(
            mode="real_replay",
            side=parent.side,
            total_qty=parent.quantity,
            ticker=parent.ticker,
            date=parent.date,
            slice_minutes=self.slice_minutes,
            step_seconds=self.step_seconds,
            n_steps=self.n_steps,
            observation_mode=self.observation_mode,
            max_action_per_step=self.max_action_per_step,
            seed=0,
        )
        obs, _ = env.reset(options={"force_anchor": {
            "ticker": parent.ticker,
            "date": parent.date,
            "start_ns": parent.start_ns,
        }})

        children: list[ChildOrder] = []
        terminated = truncated = False
        while not (terminated or truncated):
            # step_idx BEFORE step() advances it; child timestamp = current step's anchor
            step_anchor_ns = (
                parent.start_ns + env._step_idx * env.step_seconds * 1_000_000_000
            )
            action, _ = self._model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            qty_filled = int(round(info["qty_filled_this_step"]))
            if qty_filled >= 1:
                children.append(ChildOrder(
                    timestamp_ns=int(step_anchor_ns), quantity=qty_filled,
                ))

        # Reconcile residual: if policy under-executed, stuff the residual
        # into a child at the final step. If over-executed (shouldn't happen
        # under cap), trim the last child.
        total = sum(c.quantity for c in children)
        diff = parent.quantity - total
        if diff > 0:
            final_anchor_ns = (
                parent.start_ns + (self.n_steps - 1) * self.step_seconds * 1_000_000_000
            )
            if children and children[-1].timestamp_ns == final_anchor_ns:
                last = children[-1]
                children[-1] = ChildOrder(
                    timestamp_ns=last.timestamp_ns, quantity=last.quantity + diff,
                )
            else:
                children.append(ChildOrder(
                    timestamp_ns=final_anchor_ns, quantity=diff,
                ))
        elif diff < 0:
            # Trim from the last child (cap should prevent, but be defensive)
            over = -diff
            for i in range(len(children) - 1, -1, -1):
                c = children[i]
                if c.quantity > over:
                    children[i] = ChildOrder(
                        timestamp_ns=c.timestamp_ns,
                        quantity=c.quantity - over,
                    )
                    over = 0
                    break
                else:
                    over -= c.quantity
                    children.pop(i)
            if over > 0:
                raise RuntimeError(
                    f"RLAgentStrategy over-executed by {over} shares — strategy bug"
                )

        if not children:
            # Hard guard: backtest engine requires at least one child.
            mid_anchor = (parent.start_ns + parent.end_ns) // 2
            children = [ChildOrder(timestamp_ns=mid_anchor, quantity=parent.quantity)]

        return children
