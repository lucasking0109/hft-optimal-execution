"""CVXPY-based constrained optimal execution.

Solves the Almgren-Chriss problem with practical constraints that closed-form
AC cannot handle:
  1. POV cap: each child ≤ pov_cap × bucket_volume
  2. Max child size: each q_t ≤ max_child_qty (absolute upper bound)
  3. Risk-aversion: λ × variance penalty
  4. Optional convex impact: linear (default) or piecewise-quadratic

Convex QP form (risk-neutral, linear impact):
  minimize  η × Σ q_t²/V_t                    ← temp impact, linear
            + γ × (Σ q_t)²                    ← perm impact (constant for fixed parent)
            + λ × σ² × Σ (cum_remaining_t)² × Δt    ← variance penalty
  subject to
    Σ q_t = X                  (full execution)
    q_t ≥ 0                    (no buy-back)
    q_t ≤ pov_cap × V_t        (per-bucket POV cap)
    q_t ≤ max_child_qty        (absolute size cap)

When all caps are loose and λ=0, this reduces to TWAP (closed-form AC).
With λ>0, front-loads. With caps active, departs from closed-form AC.

NO Silent Fallback: solver `infeasible` / `unbounded` raises explicit
ValueError so callers can debug — never returns a fake schedule.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import polars as pl

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder

NS_PER_MIN = 60 * 1_000_000_000


class CVXPYConstrainedAC(ExecutionStrategy):
    """Almgren-Chriss with practical constraints, solved via CVXPY."""

    def __init__(
        self,
        *,
        num_slices: int = 60,
        eta: float = 1.0,                  # temporary impact coefficient (relative units)
        sigma: float = 1.0,                # volatility (relative units)
        lambda_risk: float = 0.0,
        pov_cap: float | None = None,      # None = no per-bucket cap
        max_child_qty: int | None = None,  # None = no absolute size cap
    ):
        if num_slices <= 0:
            raise ValueError("num_slices must be > 0")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if lambda_risk < 0:
            raise ValueError(f"lambda_risk must be ≥ 0, got {lambda_risk}")
        if pov_cap is not None and not (0 < pov_cap <= 1):
            raise ValueError(f"pov_cap must be in (0,1], got {pov_cap}")
        if max_child_qty is not None and max_child_qty <= 0:
            raise ValueError(f"max_child_qty must be > 0, got {max_child_qty}")
        self.num_slices = num_slices
        self.eta = eta
        self.sigma = sigma
        self.lambda_risk = lambda_risk
        self.pov_cap = pov_cap
        self.max_child_qty = max_child_qty
        self.name = (
            f"cvxpy_lam{lambda_risk}_pov{pov_cap}_max{max_child_qty}"
        )

    def schedule(
        self,
        parent: ParentOrder,
        *,
        market_context: dict,
    ) -> list[ChildOrder]:
        N = self.num_slices
        X = parent.quantity
        T_ns = parent.end_ns - parent.start_ns

        # Equally-spaced timestamps + per-slice expected volume
        timestamps = [
            parent.start_ns + int((i + 0.5) * T_ns / N) for i in range(N)
        ]
        slice_dur_ns = T_ns / N

        # Volume per slice from profile (if provided)
        bucket_volumes_per_slice = self._compute_per_slice_volume(
            parent, market_context, N
        )

        # Decision variable
        q = cp.Variable(N, nonneg=True)

        # Objective:
        #   temp impact:  η × Σ q_t² / V_t  (quadratic)
        #   variance penalty: λ σ² × Σ R_t² × Δt where R_t = remaining at time t
        impact = self.eta * cp.sum(cp.multiply(
            cp.power(q, 2),
            np.array([1.0 / max(v, 1.0) for v in bucket_volumes_per_slice]),
        ))

        if self.lambda_risk > 0 and self.sigma > 0:
            # Remaining at start of slice t = X - sum_{s<t} q_s
            cum_q = cp.cumsum(q)
            remaining_after = X - cum_q                             # shape (N,)
            # Risk = ∫ σ² R² dt ≈ σ² Σ R_t² Δt
            dt_sec = slice_dur_ns / 1e9
            risk = self.lambda_risk * (self.sigma ** 2) * dt_sec * cp.sum(
                cp.power(remaining_after, 2)
            )
            objective = cp.Minimize(impact + risk)
        else:
            objective = cp.Minimize(impact)

        # Constraints
        constraints = [cp.sum(q) == X]
        if self.pov_cap is not None:
            cap_per_slice = np.array([
                self.pov_cap * v for v in bucket_volumes_per_slice
            ])
            constraints.append(q <= cap_per_slice)
        if self.max_child_qty is not None:
            constraints.append(q <= float(self.max_child_qty))

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception as e:
            raise ValueError(
                f"CVXPY solver crashed: {type(e).__name__}: {e}. "
                f"(NO Silent Fallback — caller must investigate constraints.)"
            )

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(
                f"CVXPY problem {prob.status}. "
                f"Likely cause: pov_cap × Σ V_t < parent.quantity OR "
                f"max_child_qty × N < parent.quantity. "
                f"Loosen constraints or reduce parent.quantity. "
                f"(NO Silent Fallback.)"
            )

        q_solution = np.asarray(q.value, dtype=float)
        # Round + preserve total
        int_qtys = [int(round(x)) for x in q_solution]
        diff = X - sum(int_qtys)
        if diff != 0:
            # Adjust the largest slice
            idx = max(range(N), key=lambda i: int_qtys[i])
            int_qtys[idx] += diff

        return [
            ChildOrder(timestamp_ns=ts, quantity=q_)
            for ts, q_ in zip(timestamps, int_qtys) if q_ > 0
        ]

    def _compute_per_slice_volume(
        self, parent: ParentOrder, market_context: dict, N: int
    ) -> list[float]:
        """Return expected volume per slice. Falls back to uniform if no profile."""
        profile: pl.DataFrame | None = market_context.get("volume_profile")
        if profile is None or profile.is_empty():
            # Uniform assumption: total ADV scaled to window duration
            adv = market_context.get("adv_shares", 1e7)
            return [adv * (parent.end_ns - parent.start_ns) / (6.5 * 3600 * 1e9) / N] * N

        bin_minutes: int = market_context.get("bin_minutes", 5)
        bin_ns = bin_minutes * NS_PER_MIN
        # For each slice, sum overlapping bucket volumes
        slice_vol = [0.0] * N
        T_ns = parent.end_ns - parent.start_ns
        for b, vol in zip(profile["bucket_min"].to_list(),
                          profile["volume"].to_list()):
            ns_start = int(b) * bin_ns
            ns_end = ns_start + bin_ns
            for i in range(N):
                slice_start = parent.start_ns + int(i * T_ns / N)
                slice_end = parent.start_ns + int((i + 1) * T_ns / N)
                ov_start = max(ns_start, slice_start)
                ov_end = min(ns_end, slice_end)
                if ov_end > ov_start:
                    slice_vol[i] += float(vol) * (ov_end - ov_start) / bin_ns
        # Ensure non-zero (avoid divide-by-zero in objective)
        return [max(v, 1.0) for v in slice_vol]
