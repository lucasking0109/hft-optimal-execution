"""ACMLStrategy — Almgren-Chriss with ML-predicted η.

At schedule time:
  1. Build feature vector at parent.start_ns (state-conditional).
  2. Use trained xgboost model to predict expected impact_bps for the
     parent's per-child size.
  3. Convert prediction into an effective η in bps-per-pct-ADV units:
        η_eff = predicted_impact_bps / size_pct_adv
  4. Run AC sinh schedule with this η.

Falls back loudly if model file is missing (NO Silent Fallback).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from hft.analysis.eta_features import compute_eta_features, FEATURE_NAMES
from hft.strategies.almgren_chriss import AlmgrenChrissStrategy
from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder


class ACMLStrategy(ExecutionStrategy):
    """AC scheduling with ML-predicted η. Requires trained xgboost model."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        adv_shares: float,
        sigma_bps_per_sqrt_sec: float,
        lambda_risk: float = 1e-3,
        num_slices: int = 60,
        gamma_bps_per_pct_adv: float = 5.0,
        eta_floor: float = 0.5,
        eta_ceiling: float = 50.0,
    ):
        path = Path(model_path)
        if not path.exists():
            raise ValueError(
                f"ACMLStrategy: model not found at {path}. "
                f"Train via scripts/train_eta_ml_model.py first. "
                f"(NO Silent Fallback.)"
            )
        if adv_shares <= 0:
            raise ValueError(f"adv_shares must be > 0, got {adv_shares}")
        if not (0 < eta_floor < eta_ceiling):
            raise ValueError("eta_floor must be in (0, eta_ceiling)")

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path))
        self.adv_shares = adv_shares
        self.sigma = sigma_bps_per_sqrt_sec
        self.lambda_risk = lambda_risk
        self.num_slices = num_slices
        self.gamma = gamma_bps_per_pct_adv
        self.eta_floor = eta_floor
        self.eta_ceiling = eta_ceiling
        self.name = "ac_ml"

    def predict_eta(
        self,
        *,
        ticker: str,
        df,
        anchor_ns: int,
        parent_qty: int,
    ) -> float:
        """Predict effective η in bps-per-pct-ADV units.

        Distributional shift caveat: the underlying xgboost was trained on
        top-decile (P90+) trade events. Calling this with `parent_qty /
        num_slices` for typical 10k-share parents produces `size_pct_adv`
        far below the training distribution; the model will extrapolate
        and feature importance shows `size_pct_adv` is the **least**
        weighted feature. See `reports/phase_b3_ac_ml.md` Root cause.

        Sign caveat: the predicted impact is treated as a magnitude
        (`abs(predicted_impact_bps)`) since AC's η must be positive by
        definition. A predicted *negative* impact (price moves in the
        wrong direction relative to the side, an anomalous market
        signal) collapses to the same magnitude — directional information
        is intentionally discarded here. An alternative would be to feed
        the predicted direction into `lambda_risk` instead of η.
        """
        # Use the per-child slice size for size_pct_adv (size at planning time)
        per_child_qty = parent_qty / self.num_slices
        feats = compute_eta_features(
            ticker=ticker, df=df, anchor_ns=anchor_ns,
            parent_qty=per_child_qty, adv=self.adv_shares,
        )
        x = feats.to_array()
        predicted_impact_bps = float(self._model.predict(x)[0])

        size_pct_adv = feats.feature_dict["size_pct_adv"]
        if size_pct_adv <= 0:
            return self.eta_floor   # degenerate — use floor
        # |impact| / size — take absolute value to ensure η > 0
        eta_raw = abs(predicted_impact_bps) / size_pct_adv
        # Clamp to literature range
        return float(np.clip(eta_raw, self.eta_floor, self.eta_ceiling))

    def schedule(
        self,
        parent: ParentOrder,
        *,
        market_context: dict,
    ) -> list[ChildOrder]:
        df = market_context.get("market_df")
        if df is None:
            raise ValueError(
                "ACMLStrategy requires `market_df` in market_context. "
                "Engine should pass `engine.df`."
            )
        eta_pred = self.predict_eta(
            ticker=parent.ticker, df=df, anchor_ns=parent.start_ns,
            parent_qty=parent.quantity,
        )
        # Build a one-shot AC strategy with the predicted η
        ac = AlmgrenChrissStrategy(
            num_slices=self.num_slices,
            eta_bps_per_pct_adv=eta_pred,
            gamma_bps_per_pct_adv=self.gamma,
            sigma_bps_per_sqrt_sec=self.sigma,
            lambda_risk=self.lambda_risk,
            adv_shares=self.adv_shares,
        )
        # Stash predicted η for downstream inspection
        self.last_predicted_eta = eta_pred
        return ac.schedule(parent, market_context=market_context)
