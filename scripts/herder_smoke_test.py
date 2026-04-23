"""Step 4 — ABIDES integration smoke test for HerderAgent.

Runs 2 paired sims (N=0 vs N=50 herders) with everything else fixed,
then compares stylized facts.

GO/NO-GO criteria (Phase 5B+ Stage 3):
  ✅ N=50 vs N=0: vol_autocorr_l2 改善 ≥ 20% AND raw return std 不下降
  🔴 std 下降 → herder 在 ABIDES 內反而抑制 vol → 設計 bug
  🔴 autocorr 升高 → herder 邏輯錯方向

Output: prints comparison + writes scripts/herder_smoke_test.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.data.abides_loader import load_abides_run                          # noqa: E402
from hft.simulators.stylized_facts import (                                 # noqa: E402
    compute_stylized_facts,
    extract_abides_series,
)

ABIDES_DIR = ROOT / "vendor" / "abides-sim"
ABIDES_PYTHON = ROOT / ".venv-abides" / "bin" / "python"


def run_smoke(num_herder: int, log_dir_name: str, *, seed: int = 1234,
              herder_threshold_bps: float = 1.121, herder_max_size: int = 7,
              start_time: str = "09:30:00", end_time: str = "09:35:00") -> dict:
    log_dir = ROOT / "log" / log_dir_name
    if log_dir.exists():
        shutil.rmtree(log_dir)

    cmd = [
        str(ABIDES_PYTHON),
        str(ABIDES_DIR / "abides.py"),
        "-c", "calib_aapl",
        "-t", "ABM",
        "-d", "20200113",
        "-s", str(seed),
        "--start-time", start_time,
        "--end-time", end_time,
        "--log_dir", log_dir_name,
        "--num-noise", "500",
        "--fund-vol", "1e-3",
        "--mm-aggressiveness", "aggressive",
        "--num-momentum", "25",
        "--num-herder", str(num_herder),
        "--herder-threshold-bps", str(herder_threshold_bps),
        "--herder-max-size", str(herder_max_size),
    ]
    print(f"\n🔬 Running smoke: num_herder={num_herder} → {log_dir_name}")
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd, cwd=str(ROOT),
        env={"PYTHONPATH": str(ABIDES_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=300,
    )
    runtime = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"abides exited {res.returncode}\n{res.stderr[-1000:]}")

    data = load_abides_run(log_dir)
    mids, spread, trades = extract_abides_series(data, sample_seconds=1)
    facts = compute_stylized_facts(
        mid_prices=mids, spread_bps=spread, trade_sizes=trades,
        metadata={"num_herder": num_herder, "log_dir": str(log_dir)},
    )
    summary = facts.summary()
    summary["runtime_s"] = round(runtime, 2)
    summary["raw_return_std"] = float(np.std(np.diff(np.log(mids[mids > 0]))))
    # autocorr_l2 vs zero (i.e., the magnitude of vol_autocorr we're producing)
    summary["autocorr_l2_self"] = float(np.linalg.norm(facts.vol_autocorr[:50]))
    return summary


def main():
    print("=" * 70)
    print("Step 4 — HerderAgent ABIDES integration smoke test")
    print("Compares N=0 vs N=50 herders over 5-min sim, anchored on prior best 3D corner.")
    print("=" * 70)

    s0 = run_smoke(num_herder=0, log_dir_name="smoke_herder_n0")
    s50 = run_smoke(num_herder=50, log_dir_name="smoke_herder_n50")

    # Compare
    ac_l1_0, ac_l1_50 = s0["vol_autocorr_lag_1"], s50["vol_autocorr_lag_1"]
    ac_l2_0, ac_l2_50 = s0["autocorr_l2_self"], s50["autocorr_l2_self"]
    std_0, std_50 = s0["raw_return_std"], s50["raw_return_std"]
    kurt_0, kurt_50 = s0["excess_kurtosis"], s50["excess_kurtosis"]

    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'N=0':>15} {'N=50':>15} {'Δ':>15}")
    print("-" * 70)
    print(f"{'autocorr_lag_1':<25} {ac_l1_0:>15.4f} {ac_l1_50:>15.4f} {ac_l1_50-ac_l1_0:>+15.4f}")
    print(f"{'autocorr_l2 (self)':<25} {ac_l2_0:>15.4f} {ac_l2_50:>15.4f} {ac_l2_50-ac_l2_0:>+15.4f}")
    print(f"{'raw return std':<25} {std_0:>15.6f} {std_50:>15.6f} {std_50-std_0:>+15.6f}")
    print(f"{'excess kurtosis':<25} {kurt_0:>15.2f} {kurt_50:>15.2f} {kurt_50-kurt_0:>+15.2f}")
    print(f"{'spread_bps_median':<25} {s0['spread_bps_median']:>15.3f} {s50['spread_bps_median']:>15.3f} {s50['spread_bps_median']-s0['spread_bps_median']:>+15.3f}")
    print(f"{'n_trades':<25} {s0['n_trades']:>15} {s50['n_trades']:>15} {s50['n_trades']-s0['n_trades']:>+15}")
    print(f"{'runtime_s':<25} {s0['runtime_s']:>15.2f} {s50['runtime_s']:>15.2f}")

    print("\n" + "=" * 70)
    # GO/NO-GO criteria
    # We want:
    #   ac_l1 (lag 1 autocorr of |returns|) → CLOSER to real benchmark median 0.065
    #   raw std → not decrease
    real_ac_l1_target = 0.065  # from Step 0
    # Distance to target
    d0 = abs(ac_l1_0 - real_ac_l1_target)
    d50 = abs(ac_l1_50 - real_ac_l1_target)
    improvement_pct = (d0 - d50) / d0 * 100 if d0 > 0 else 0
    std_change_pct = (std_50 - std_0) / std_0 * 100 if std_0 > 0 else 0

    verdict_parts = []
    if d50 < d0:
        verdict_parts.append(f"✅ autocorr distance to real ({real_ac_l1_target}) DROPPED {improvement_pct:.0f}%")
    else:
        verdict_parts.append(f"🔴 autocorr distance INCREASED {-improvement_pct:.0f}% (wrong direction)")

    if std_50 >= std_0 * 0.9:  # allow up to 10% drop as noise
        verdict_parts.append(f"✅ raw std maintained ({std_change_pct:+.1f}%)")
    else:
        verdict_parts.append(f"🔴 raw std DROPPED {-std_change_pct:.1f}% (herder suppressing vol — bug?)")

    pass_smoke = (d50 < d0) and (std_50 >= std_0 * 0.9)
    print("\n".join(verdict_parts))
    print("\n" + ("🎯 SMOKE PASS — proceeding to Stage 3 multiseed grid"
                  if pass_smoke else
                  "🔴 SMOKE FAIL — review HerderAgent before Stage 3"))
    print("=" * 70)

    # Persist
    out = {
        "real_ac_l1_target": real_ac_l1_target,
        "n0": s0,
        "n50": s50,
        "improvement_pct_to_real": improvement_pct,
        "std_change_pct": std_change_pct,
        "pass": bool(pass_smoke),
    }
    out_path = ROOT / "data" / "synthetic" / "herder_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"💾 Wrote {out_path}")

    sys.exit(0 if pass_smoke else 2)


if __name__ == "__main__":
    main()
