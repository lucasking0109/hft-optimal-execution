"""Almgren-Chriss optimal-execution strategy with closed-form schedule.

Two regimes:
    - Risk-neutral (lambda_risk=0): degenerates to a uniform schedule
      (linear holdings, equivalent to TWAP). This is the AC theorem for
      pure linear impact: any schedule has the same expected cost.
    - Risk-averse (lambda_risk>0): closed-form sinh schedule. Front-loads
      trading to limit price-uncertainty exposure.

Discrete-time formulation following Almgren & Chriss (2000), §4:
    Define κ̃ ≈ √(λ σ² / η) (small-τ approximation; we use the exact
    formula derived from the eigenvalue problem below).
    Holdings h_k = X · sinh(κ(T − t_k)) / sinh(κ T).

Sign convention: positive `quantity` means we want to liquidate (sell) X
or accumulate (buy) X. The schedule applies to either direction; backtester
attaches the side at fill time.
"""

from __future__ import annotations

import math

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder

NS_PER_SEC = 1_000_000_000


class AlmgrenChrissStrategy(ExecutionStrategy):
    name = "almgren_chriss"

    def __init__(
        self,
        *,
        num_slices: int = 60,
        eta_bps_per_pct_adv: float,
        gamma_bps_per_pct_adv: float,
        sigma_bps_per_sqrt_sec: float,
        lambda_risk: float = 0.0,
        adv_shares: float,
    ):
        """Construct an AC strategy.

        Args:
            num_slices: number of equally-spaced child orders.
            eta_bps_per_pct_adv: temporary impact coefficient.
            gamma_bps_per_pct_adv: permanent impact (used for completeness;
                doesn't change risk-neutral schedule, affects total cost).
            sigma_bps_per_sqrt_sec: price volatility in bps per √second.
            lambda_risk: risk-aversion (≥ 0). 0 → linear schedule.
            adv_shares: average daily volume (for unit conversion).
        """
        if num_slices <= 0:
            raise ValueError("num_slices must be > 0")
        if eta_bps_per_pct_adv <= 0:
            raise ValueError("eta must be positive")
        if sigma_bps_per_sqrt_sec < 0:
            raise ValueError("sigma must be non-negative")
        if lambda_risk < 0:
            raise ValueError("lambda_risk must be ≥ 0")
        if adv_shares <= 0:
            raise ValueError("adv_shares must be > 0")

        self.num_slices = num_slices
        self.eta = eta_bps_per_pct_adv
        self.gamma = gamma_bps_per_pct_adv
        self.sigma = sigma_bps_per_sqrt_sec
        self.lambda_risk = lambda_risk
        self.adv_shares = adv_shares

    @property
    def is_risk_neutral(self) -> bool:
        return self.lambda_risk == 0.0 or self.sigma == 0.0

    def _kappa(self) -> float:
        """κ = √(λ σ² / η̃) where η̃ is in compatible units.

        Both η and σ are expressed in bps; lambda_risk is dimensionless
        scaling. We treat κ as 1/time (per second).
        """
        eta_per_sec = self.eta * 60.0  # convert from per-minute to per-second equivalent
        if eta_per_sec <= 0 or self.sigma <= 0 or self.lambda_risk <= 0:
            return 0.0
        return math.sqrt(self.lambda_risk * self.sigma ** 2 / eta_per_sec)

    def schedule(self, parent: ParentOrder, *, market_context: dict) -> list[ChildOrder]:
        N = self.num_slices
        X = parent.quantity
        T_ns = parent.end_ns - parent.start_ns
        T_sec = T_ns / NS_PER_SEC

        # Equally-spaced timestamps (centre of each interval)
        timestamps = [
            parent.start_ns + int((i + 0.5) * T_ns / N) for i in range(N)
        ]

        if self.is_risk_neutral:
            # Linear schedule: same as TWAP
            base = X // N
            remainder = X - base * N
            qtys = [base + (1 if i < remainder else 0) for i in range(N)]
            return [ChildOrder(timestamp_ns=ts, quantity=q)
                    for ts, q in zip(timestamps, qtys) if q > 0]

        kappa = self._kappa()
        if kappa * T_sec < 1e-9:
            # κT too small → fall back to linear (but report it loudly)
            raise ValueError(
                f"κT = {kappa * T_sec:.2e} too small to use sinh schedule. "
                f"With lambda_risk={self.lambda_risk}, sigma={self.sigma}, "
                f"eta={self.eta}, T_sec={T_sec}, optimization is degenerate. "
                f"Lower num_slices or use risk-neutral mode."
            )

        # Holdings at the *boundary* of each interval k=0..N
        # h_k = X * sinh(κ(T - t_k)) / sinh(κ T), where t_k = k * T/N
        boundary_times = [k * T_sec / N for k in range(N + 1)]
        holdings = [
            X * math.sinh(kappa * (T_sec - t)) / math.sinh(kappa * T_sec)
            for t in boundary_times
        ]
        # Ensure exact 0 at end and X at start
        holdings[0] = float(X)
        holdings[-1] = 0.0

        # Trade qty per slice = decrease in holdings
        raw_qtys = [holdings[k] - holdings[k + 1] for k in range(N)]

        # Round to integer shares while preserving total
        int_qtys = [int(round(q)) for q in raw_qtys]
        diff = X - sum(int_qtys)
        # Adjust the largest slice (front, since front-loaded) for any rounding remainder
        if diff != 0:
            idx = max(range(N), key=lambda i: int_qtys[i])
            int_qtys[idx] += diff

        return [ChildOrder(timestamp_ns=ts, quantity=q)
                for ts, q in zip(timestamps, int_qtys) if q > 0]


def estimate_intraday_sigma_bps_per_sqrt_sec(market_df, *, sample_seconds: int = 60) -> float:
    """Estimate price volatility σ in bps per √second from intraday mid prices.

    Compute mid every `sample_seconds` and take stdev of log-returns,
    then normalise to per-√sec.
    """
    import polars as pl

    from hft.data.timeparse import add_eq_ns_of_day, filter_rth

    df = market_df
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    df = filter_rth(df, src="Timestamp")
    nbb = df.filter(pl.col("EventType") == "QUOTE BID NB").select("ns_of_day", pl.col("Price").alias("nbb"))
    nbo = df.filter(pl.col("EventType") == "QUOTE ASK NB").select("ns_of_day", pl.col("Price").alias("nbo"))
    if nbb.is_empty() or nbo.is_empty():
        raise ValueError("No NBBO rows to estimate σ")

    # Build mids at sampled timestamps
    from hft.analysis.nbbo_lookup import NBBOLookup
    lookup = NBBOLookup(df)
    start_ns = int(df["ns_of_day"].min())
    end_ns = int(df["ns_of_day"].max())
    step_ns = sample_seconds * NS_PER_SEC
    mids = []
    ns = start_ns
    while ns < end_ns:
        m = lookup.mid_at(ns)
        if m and m > 0:
            mids.append(m)
        ns += step_ns
    if len(mids) < 5:
        raise ValueError("Too few mid samples to estimate σ")
    log_returns = [math.log(mids[i] / mids[i - 1]) for i in range(1, len(mids))]
    import statistics
    sigma_per_sample = statistics.stdev(log_returns)
    # Convert to per-√sec, then to bps
    sigma_per_sqrt_sec = sigma_per_sample / math.sqrt(sample_seconds)
    return sigma_per_sqrt_sec * 10000
