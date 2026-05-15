"""Phase G shared helpers — used by run_phase_g_multi_window.py and
run_phase_g_parent_size.py.

Defines:
- 5 evaluation windows across RTH(open / early-mid / mid / late-mid / close)
- Ticker pool discovery(104 NDX100-class with full 5-day data)
- Strategy factory (TWAP / AC-RA / VWAP-follow / POV / Tóth + RL v3)
"""

from __future__ import annotations

import os
from pathlib import Path

from hft.analysis.impact import (
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
)
from hft.simulators.adv_cache import get_adv
from hft.strategies.almgren_chriss import AlmgrenChrissStrategy
from hft.strategies.pov import POVStrategy
from hft.strategies.rl_agent import RLAgentStrategy
from hft.strategies.toth import TothStrategy
from hft.strategies.twap import TWAPStrategy
from hft.strategies.vwap_following import VWAPFollowingStrategy


ROOT = Path(__file__).resolve().parents[1]
TAQ_ROOT = ROOT / "data" / "processed" / "eq_taq"
# Phase G uses v4 (overlapping windows, covers full RTH inc. 15:00-16:00).
# v3 had close-window generalization failure (win-rate 25%).
RL_MODEL_PATH = ROOT / "rl" / "checkpoints" / "ppo_v4_overlapping" / "model.zip"
RL_MODEL_NAME = "rl_v4"
OOS_DATE = "20200117"
ALL_DATES = ["20200113", "20200114", "20200115", "20200116", OOS_DATE]
TRAIN_DATES = [d for d in ALL_DATES if d != OOS_DATE]   # Day 1-4

# 5 evaluation windows covering RTH 09:30 → 16:00.
# Hour boundaries in seconds-from-midnight.
WINDOWS: list[tuple[str, int, int]] = [
    # name           start_sec end_sec
    ("open",          9 * 3600 + 30 * 60, 10 * 3600 + 30 * 60),  # 09:30-10:30
    ("early_mid",    10 * 3600,           11 * 3600),            # 10:00-11:00  (Phase E.3 baseline)
    ("mid",          12 * 3600,           13 * 3600),            # 12:00-13:00
    ("late_mid",     14 * 3600,           15 * 3600),            # 14:00-15:00
    ("close",        15 * 3600,           16 * 3600),            # 15:00-16:00
]

NUM_SLICES = 60
LAMBDA_RISK = 1e-3
SIGMA_DEFAULT = 1.0
DEFAULT_PARENT_QTY = 10_000


def all_5_days(ticker: str) -> bool:
    """Ticker has parquet for all 5 days (train + OOS)."""
    return all((TAQ_ROOT / d / f"{ticker}.parquet").exists() for d in ALL_DATES)


def list_eval_tickers() -> list[str]:
    """All tickers in cache with full 5-day coverage."""
    if not (TAQ_ROOT / OOS_DATE).exists():
        return []
    candidates = sorted(
        f.replace(".parquet", "")
        for f in os.listdir(TAQ_ROOT / OOS_DATE) if f.endswith(".parquet")
    )
    return [t for t in candidates if all_5_days(t)]


def load_rl_strategy() -> RLAgentStrategy | None:
    """Load Phase E v3 RL strategy if checkpoint exists, else None."""
    if not RL_MODEL_PATH.exists():
        return None
    rl_strat = RLAgentStrategy(
        model_path=str(RL_MODEL_PATH),
        slice_minutes=60, step_seconds=30, n_steps=120,
        observation_mode="v3", max_action_per_step=0.05,
    )
    rl_strat.name = RL_MODEL_NAME
    return rl_strat


def make_strategies(ticker: str, adv: float, rl_strat: RLAgentStrategy | None):
    """Build the 6-strategy list. RL last so eval ordering is consistent."""
    eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
    gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
    out = [
        ("twap",           TWAPStrategy(num_slices=NUM_SLICES)),
        ("ac_ra",          AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
            lambda_risk=LAMBDA_RISK, adv_shares=adv,
        )),
        ("vwap_following", VWAPFollowingStrategy(num_slices=NUM_SLICES)),
        ("pov_5pct_cap",   POVStrategy(target_pov=None, cap_pov=0.05)),
        ("toth",           TothStrategy(participation_cap=0.10)),
    ]
    if rl_strat is not None:
        out.append((RL_MODEL_NAME, rl_strat))
    return out
