"""Phase 5B calibration runner.

Drives abides-sim through subprocess (using `.venv-abides/bin/python`) over a
Latin-hypercube of (num_noise, fund_vol, mm_aggressiveness, num_momentum),
computes stylized facts on each output, scores against real AAPL, and picks
the best cell.

Phase 5B initial pass used 3D (9 cells); Phase 5B+ adds num_momentum as the
4th dimension to attack the volatility-clustering gap (lit ref: arxiv
2507.06345 — tactical/momentum agents reproduce momentum-like cascades).

NO Silent Fallback: any cell that crashes is recorded as failed and surfaces
in the final report; we don't quietly drop them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hft.data.abides_loader import load_abides_run
from hft.simulators.stylized_facts import (
    StylizedFacts,
    compute_stylized_facts,
    extract_abides_series,
    extract_real_taq_series,
    facts_distance,
    total_calibration_distance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ABIDES_DIR = PROJECT_ROOT / "vendor" / "abides-sim"
ABIDES_PYTHON = PROJECT_ROOT / ".venv-abides" / "bin" / "python"


# ---------------------------------------------------------------------------
# 9-cell Latin hypercube (Phase 5B initial — 3D coverage)
# ---------------------------------------------------------------------------

CALIBRATION_CELLS: list[dict] = [
    # Corners + center; first 8 corners of 2×2×2 box, last 1 center
    {"id": 1, "num_noise":   500, "fund_vol": 1e-5, "mm": "timid",      "num_momentum": 25},
    {"id": 2, "num_noise":   500, "fund_vol": 1e-5, "mm": "aggressive", "num_momentum": 25},
    {"id": 3, "num_noise":   500, "fund_vol": 1e-3, "mm": "timid",      "num_momentum": 25},
    {"id": 4, "num_noise":   500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 25},
    {"id": 5, "num_noise": 25000, "fund_vol": 1e-5, "mm": "timid",      "num_momentum": 25},
    {"id": 6, "num_noise": 25000, "fund_vol": 1e-5, "mm": "aggressive", "num_momentum": 25},
    {"id": 7, "num_noise": 25000, "fund_vol": 1e-3, "mm": "timid",      "num_momentum": 25},
    {"id": 8, "num_noise": 25000, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 25},
    {"id": 9, "num_noise":  5000, "fund_vol": 1e-4, "mm": "balanced",   "num_momentum": 25},  # center
]


# ---------------------------------------------------------------------------
# Phase 5B+ Stage 1: 3-cell momentum sensitivity sweep
# Anchors at the best 3D corner (#4 from initial 9-cell run, total_dist=2.144)
# and varies only num_momentum to test if vol clustering improves with more
# tactical/momentum agents (per arxiv 2507.06345).
# ---------------------------------------------------------------------------

STAGE1_CELLS: list[dict] = [
    {"id": 101, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum":  25},  # baseline = prior best
    {"id": 102, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 100},
    {"id": 103, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 400},
]


# ---------------------------------------------------------------------------
# Phase 5B+ Stage 2: 12-cell 4D Latin hypercube (gated on Stage 1 success)
# All cells use mm=aggressive (dominated previous 9-cell results) except #212
# which sanity-checks balanced MM at high momentum.
# ---------------------------------------------------------------------------

STAGE2_CELLS: list[dict] = [
    {"id": 201, "num_noise":   500, "fund_vol": 1e-5, "mm": "aggressive", "num_momentum": 100},
    {"id": 202, "num_noise":   500, "fund_vol": 1e-5, "mm": "aggressive", "num_momentum": 400},
    {"id": 203, "num_noise":   500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 100},
    {"id": 204, "num_noise":   500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 400},
    {"id": 205, "num_noise":  5000, "fund_vol": 1e-4, "mm": "aggressive", "num_momentum": 100},
    {"id": 206, "num_noise":  5000, "fund_vol": 1e-4, "mm": "aggressive", "num_momentum": 400},
    {"id": 207, "num_noise": 25000, "fund_vol": 1e-5, "mm": "aggressive", "num_momentum": 100},
    {"id": 208, "num_noise": 25000, "fund_vol": 1e-5, "mm": "aggressive", "num_momentum": 400},
    {"id": 209, "num_noise": 25000, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 100},
    {"id": 210, "num_noise": 25000, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 400},
    {"id": 211, "num_noise":   500, "fund_vol": 1e-3, "mm": "aggressive", "num_momentum": 200},
    {"id": 212, "num_noise":   500, "fund_vol": 1e-3, "mm": "balanced",   "num_momentum": 200},
]


# ---------------------------------------------------------------------------
# Phase 5B+ Stage 3: 4-cell factor experiment for HerderAgent + OBI
# Anchored on prior best 3D corner (cell #4: 500 noise, fund_vol=1e-3,
# aggressive MM, num_momentum=25).
# ---------------------------------------------------------------------------

STAGE3_CELLS: list[dict] = [
    {"id": 301, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi":  0, "num_herder":  0},   # baseline
    {"id": 302, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50, "num_herder":  0},   # +OBI only
    {"id": 303, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi":  0, "num_herder": 50},   # +Herder only (主力)
    {"id": 304, "num_noise": 500, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50, "num_herder": 50},   # +Both (Brock-Hommes mix)
]

# 5 seeds (lit recommends 10-20; folded to 5 for budget)
STAGE3_SEEDS: list[int] = [1234, 5678, 9012, 3456, 7890]


# ---------------------------------------------------------------------------
# Phase 5B+ Stage 4: low-noise population mix (lit recommendation)
# Hypothesis: Stage 3 noise=500 → 74% of total population, far above
# Bamberg/CHAD's <50% recommendation. Lower noise should let herder/OBI
# effects dominate.
#
# Anchor: fund_vol=1e-3, mm=aggressive, num_momentum=25 (best from prior).
# Vary: num_noise, num_obi, num_herder, herder_max_size.
# ---------------------------------------------------------------------------

STAGE4_CELLS: list[dict] = [
    # 401: low-noise baseline — isolate the effect of dropping noise alone
    {"id": 401, "num_noise": 200, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi":  0, "num_herder":  0,
     "herder_max_size": None},
    # 402: Stage 3 winning config (cell 304) but with noise=200 (apples-to-apples)
    {"id": 402, "num_noise": 200, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50, "num_herder": 50,
     "herder_max_size": 7},
    # 403: smaller herder size — does it reduce kurtosis side-effect?
    {"id": 403, "num_noise": 200, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50, "num_herder": 50,
     "herder_max_size": 3},
    # 404: mid-noise sensitivity
    {"id": 404, "num_noise": 350, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50, "num_herder": 50,
     "herder_max_size": 7},
    # 405: more herders at low noise — test cap suppression
    {"id": 405, "num_noise": 200, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50, "num_herder": 100,
     "herder_max_size": 5},
]

# Same seeds as Stage 3 for direct apples-to-apples comparison
STAGE4_SEEDS: list[int] = [1234, 5678, 9012, 3456, 7890]


# ---------------------------------------------------------------------------
# Phase 5B+ Stage 5: re-calibrate for long (15-min) episodes
# Phase 5C pilot revealed: Stage 4 cell 404 (5-min calibration) extrapolates
# poorly to 15-min — kurt explodes 2.03 → 7.71 (autocorr improves 1.90 → 1.67).
# Hypothesis: herder cascades accumulate over time. Smaller herder count +
# tighter position_cap + smaller max_size should prevent buildup.
#
# Anchor: 15-min sim, num_noise=350 (Stage 4 sweet spot), OBI=50, mom=25.
# Vary: num_herder, herder_max_size, herder_position_cap.
# ---------------------------------------------------------------------------

STAGE5_CELLS: list[dict] = [
    # 501: pilot baseline (= 15-min Stage 4 #404 config) — reproduce known result
    {"id": 501, "num_noise": 350, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50,
     "num_herder": 50, "herder_max_size": 7, "herder_position_cap": 50},
    # 502: scale down all params (lit "less is more" for long sims)
    {"id": 502, "num_noise": 350, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50,
     "num_herder": 30, "herder_max_size": 5, "herder_position_cap": 20},
    # 503: tight individual orders + cap (test if cap is the cascade trigger)
    {"id": 503, "num_noise": 350, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50,
     "num_herder": 50, "herder_max_size": 3, "herder_position_cap": 15},
    # 504: balanced reduction
    {"id": 504, "num_noise": 350, "fund_vol": 1e-3, "mm": "aggressive",
     "num_momentum": 25, "num_obi": 50,
     "num_herder": 25, "herder_max_size": 5, "herder_position_cap": 25},
]

# 3 seeds (compromised from 5 due to 15-min sim cost; lit recommends 10-20)
STAGE5_SEEDS: list[int] = [1234, 5678, 9012]


@dataclass
class CellResult:
    cell_id: int
    params: dict
    success: bool
    facts: StylizedFacts | None
    distances: dict | None
    total_distance: float | None
    runtime_seconds: float
    error: str | None
    log_dir: Path


# ---------------------------------------------------------------------------
# Run a single cell
# ---------------------------------------------------------------------------

def run_one_cell(
    cell: dict,
    *,
    ticker: str = "ABM",
    date: str = "20200113",
    start_time: str = "09:30:00",
    end_time: str = "09:32:00",
    seed: int = 1234,
    output_root: Path | None = None,
) -> CellResult:
    """Run one abides-sim simulation with the cell's parameters."""
    if output_root is None:
        output_root = PROJECT_ROOT / "log"
    log_dir_name = f"calib_cell_{cell['id']:03d}_seed{seed}"
    log_dir = output_root / log_dir_name
    if log_dir.exists():
        shutil.rmtree(log_dir)

    cmd = [
        str(ABIDES_PYTHON),
        str(ABIDES_DIR / "abides.py"),
        "-c", "calib_aapl",
        "-t", ticker,
        "-d", date,
        "-s", str(seed),
        "--start-time", start_time,
        "--end-time", end_time,
        "--log_dir", log_dir_name,
        "--num-noise", str(cell["num_noise"]),
        "--fund-vol", str(cell["fund_vol"]),
        "--mm-aggressiveness", cell["mm"],
        "--num-momentum", str(cell.get("num_momentum", 25)),
        "--num-obi", str(cell.get("num_obi", 0)),
        "--num-herder", str(cell.get("num_herder", 0)),
    ]
    if cell.get("num_herder", 0) > 0:
        herder_threshold = cell.get("herder_threshold_bps")
        herder_max_size = cell.get("herder_max_size")
        if herder_threshold is None or herder_max_size is None:
            raise ValueError(
                f"Cell {cell['id']} has num_herder > 0 but missing herder_threshold_bps "
                f"or herder_max_size. Pass them via cell dict or set defaults from Step 0."
            )
        cmd += [
            "--herder-threshold-bps", str(herder_threshold),
            "--herder-max-size", str(herder_max_size),
        ]
        # Optional position_cap (Stage 5+); falls back to default 50 if not set
        herder_position_cap = cell.get("herder_position_cap")
        if herder_position_cap is not None:
            cmd += ["--herder-position-cap", str(herder_position_cap)]
    env_python_path = str(ABIDES_DIR)

    # Auto-scale subprocess timeout based on sim duration.
    # Empirical: 5-min sim ~25s, 15-min sim ~340s; use 30x sim duration as safety margin.
    end_h, end_m, _ = map(int, end_time.split(":"))
    start_h, start_m, _ = map(int, start_time.split(":"))
    sim_duration_min = max(1, (end_h * 60 + end_m) - (start_h * 60 + start_m))
    sim_timeout = max(300, sim_duration_min * 90)  # 90s per sim-min, min 300s

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={"PYTHONPATH": env_python_path, "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=sim_timeout,
        )
        runtime = time.perf_counter() - t0
        if result.returncode != 0:
            return CellResult(
                cell_id=cell["id"], params=cell, success=False,
                facts=None, distances=None, total_distance=None,
                runtime_seconds=runtime,
                error=f"abides exited {result.returncode}: {result.stderr[-500:]}",
                log_dir=log_dir,
            )
    except subprocess.TimeoutExpired:
        return CellResult(
            cell_id=cell["id"], params=cell, success=False,
            facts=None, distances=None, total_distance=None,
            runtime_seconds=float(sim_timeout),
            error=f"abides timeout > {sim_timeout}s",
            log_dir=log_dir,
        )

    # Load output + compute stylized facts
    try:
        data = load_abides_run(log_dir)
        mids, spread, trades = extract_abides_series(data, sample_seconds=1)
        facts = compute_stylized_facts(
            mid_prices=mids, spread_bps=spread, trade_sizes=trades,
            metadata={"cell_id": cell["id"], **cell},
        )
    except Exception as e:
        return CellResult(
            cell_id=cell["id"], params=cell, success=False,
            facts=None, distances=None, total_distance=None,
            runtime_seconds=time.perf_counter() - t0,
            error=f"stylized facts failed: {type(e).__name__}: {e}",
            log_dir=log_dir,
        )

    return CellResult(
        cell_id=cell["id"], params=cell, success=True,
        facts=facts, distances=None, total_distance=None,
        runtime_seconds=time.perf_counter() - t0,
        error=None,
        log_dir=log_dir,
    )


# ---------------------------------------------------------------------------
# Run full calibration grid
# ---------------------------------------------------------------------------

def run_calibration(
    *,
    real_ticker: str = "AAPL",
    real_date: str = "20200113",
    end_time: str = "09:32:00",  # 2-min sims for speed
    cells: list[dict] | None = None,
) -> dict:
    """Run cells, return dict with results + best cell.

    cells: defaults to CALIBRATION_CELLS (Phase 5B initial 9-cell 3D grid).
           Pass STAGE1_CELLS or STAGE2_CELLS for the 4D Phase 5B+ runs.
    """
    from hft.data import load_eq_taq

    if cells is None:
        cells = CALIBRATION_CELLS

    print(f"📥 Loading real {real_ticker} {real_date}...")
    real_df = load_eq_taq(real_ticker, real_date)
    real_mids, real_spread, real_trades = extract_real_taq_series(
        real_df, sample_seconds=1, rth_only=True,
    )
    real_facts = compute_stylized_facts(
        mid_prices=real_mids, spread_bps=real_spread, trade_sizes=real_trades,
        metadata={"source": f"real_{real_ticker}_{real_date}"},
    )
    print(f"   Real facts: kurtosis={real_facts.excess_kurtosis:.1f}, "
          f"hill={real_facts.hill_tail_index:.2f}, "
          f"spread_med={float(real_facts.spread_bps.mean()):.2f}bps, "
          f"autocorr_l1={real_facts.vol_autocorr[0]:.3f}")
    print()

    print(f"🔬 Running {len(cells)} calibration cells...")
    results: list[CellResult] = []
    for cell in cells:
        print(f"  Cell {cell['id']}: noise={cell['num_noise']:>5} · "
              f"fv={cell['fund_vol']:.0e} · mm={cell['mm']:<10} · "
              f"mom={cell.get('num_momentum', 25):>3}", end=" ", flush=True)
        cr = run_one_cell(cell, end_time=end_time)
        if cr.success:
            cr.distances = facts_distance(real_facts, cr.facts)
            cr.total_distance = total_calibration_distance(cr.distances)
            print(f"→ dist={cr.total_distance:.3f} ac_l2={cr.distances['autocorr_l2']:.2f} ({cr.runtime_seconds:.1f}s)")
        else:
            print(f"❌ FAILED ({cr.runtime_seconds:.1f}s) {cr.error[:80]}")
        results.append(cr)

    # Pick best (smallest total_distance among successful)
    successful = [r for r in results if r.success]
    if not successful:
        print("\n⚠️  All 9 cells failed.")
        return {"results": results, "best": None, "real_facts": real_facts}

    best = min(successful, key=lambda r: r.total_distance)
    print(f"\n🏆 Best cell: #{best.cell_id} total_dist={best.total_distance:.3f}")
    print(f"   Params: {best.params}")
    return {"results": results, "best": best, "real_facts": real_facts}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_calibration_results(results: dict, output_path: Path) -> None:
    """Persist calibration outcome to JSON for downstream Phase 5C."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "real_facts_summary": results["real_facts"].summary(),
        "cells": [
            {
                "cell_id": r.cell_id,
                "params": r.params,
                "success": r.success,
                "runtime_seconds": r.runtime_seconds,
                "error": r.error,
                "facts_summary": r.facts.summary() if r.facts else None,
                "distances": r.distances,
                "total_distance": r.total_distance,
            }
            for r in results["results"]
        ],
        "best_cell": (
            {
                "cell_id": results["best"].cell_id,
                "params": results["best"].params,
                "total_distance": results["best"].total_distance,
                "distances": results["best"].distances,
            }
            if results["best"] else None
        ),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Phase 5B+ Stage 3: multi-seed runner
# ---------------------------------------------------------------------------

def run_multiseed_calibration(
    *,
    cells: list[dict],
    seeds: list[int],
    real_facts,                                        # StylizedFacts of real benchmark
    end_time: str = "09:35:00",
    real_benchmark_dist: dict | None = None,
) -> dict:
    """Run each cell × each seed; aggregate median + IQR per cell.

    Args:
        cells: list of cell dicts (id, num_noise, fund_vol, mm, num_momentum,
               num_obi, num_herder, optionally herder_threshold_bps + herder_max_size).
        seeds: list of seeds to run per cell.
        real_facts: StylizedFacts computed from real benchmark (full or 5-min IQR).
        end_time: ABIDES sim end time.
        real_benchmark_dist: optional aggregate dict from
                             scripts/herder_param_calibration.py:aggregate_slice_distribution
                             — used to score against 5-min IQR (apples-to-apples).

    Returns:
        dict with:
          - cells: list of per-cell aggregations (median + IQR + all_seed_results)
          - best_cell: cell with min median(autocorr_l2)
          - real_facts: passed through
          - real_benchmark_dist: passed through
    """
    print(f"🔬 Running {len(cells)} cells × {len(seeds)} seeds = {len(cells) * len(seeds)} sims...")
    all_results = []
    for cell in cells:
        cell_runs: list[CellResult] = []
        print(f"  Cell {cell['id']}: noise={cell['num_noise']:>5} fv={cell['fund_vol']:.0e} "
              f"mm={cell['mm']:<10} mom={cell.get('num_momentum', 25):>3} "
              f"obi={cell.get('num_obi', 0):>3} herder={cell.get('num_herder', 0):>3}")
        for seed in seeds:
            cr = run_one_cell(cell, end_time=end_time, seed=seed)
            if cr.success:
                cr.distances = facts_distance(real_facts, cr.facts)
                cr.total_distance = total_calibration_distance(cr.distances)
                ac_l2 = cr.distances["autocorr_l2"]
                print(f"    seed={seed} → dist={cr.total_distance:.3f} ac_l2={ac_l2:.2f} ({cr.runtime_seconds:.1f}s)")
            else:
                print(f"    seed={seed} ❌ FAILED ({cr.runtime_seconds:.1f}s) {cr.error[:80]}")
            cell_runs.append(cr)

        successes = [r for r in cell_runs if r.success]
        if not successes:
            agg = {"cell_id": cell["id"], "params": cell, "all_seed_runs": cell_runs,
                   "n_success": 0, "median_total_distance": None,
                   "median_autocorr_l2": None, "iqr_autocorr_l2": None}
        else:
            tds = np.array([r.total_distance for r in successes])
            acs = np.array([r.distances["autocorr_l2"] for r in successes])
            kurts = np.array([r.distances["kurtosis_ratio"] for r in successes])
            spread_kss = np.array([r.distances["spread_ks_dist"] for r in successes])
            agg = {
                "cell_id": cell["id"],
                "params": cell,
                "n_success": len(successes),
                "n_total": len(seeds),
                "median_total_distance": float(np.median(tds)),
                "iqr_total_distance": [float(np.percentile(tds, 25)), float(np.percentile(tds, 75))],
                "median_autocorr_l2": float(np.median(acs)),
                "iqr_autocorr_l2": [float(np.percentile(acs, 25)), float(np.percentile(acs, 75))],
                "median_kurtosis_ratio": float(np.median(kurts)),
                "median_spread_ks": float(np.median(spread_kss)),
                "all_seed_results": [
                    {"seed": s, "success": r.success, "total_distance": r.total_distance,
                     "distances": r.distances, "facts_summary": r.facts.summary() if r.facts else None,
                     "runtime_s": r.runtime_seconds}
                    for s, r in zip(seeds, cell_runs)
                ],
            }
        all_results.append(agg)

    # Pick best cell by median total_distance
    successful_aggs = [a for a in all_results if a["median_total_distance"] is not None]
    if not successful_aggs:
        best = None
    else:
        best = min(successful_aggs, key=lambda a: a["median_total_distance"])
        print(f"\n🏆 Best cell: #{best['cell_id']} median_dist={best['median_total_distance']:.3f} "
              f"median_autocorr_l2={best['median_autocorr_l2']:.3f}")

    return {
        "cells": all_results,
        "best_cell": best,
        "real_facts_summary": real_facts.summary(),
        "real_benchmark_dist": real_benchmark_dist,
        "n_seeds": len(seeds),
    }


def cells_to_dataframe(results: dict) -> pl.DataFrame:
    """Flatten cell results into a polars DataFrame for the report."""
    rows = []
    for r in results["results"]:
        row = {
            "cell": r.cell_id,
            "num_noise": r.params["num_noise"],
            "fund_vol": r.params["fund_vol"],
            "mm": r.params["mm"],
            "success": r.success,
            "runtime_s": round(r.runtime_seconds, 1),
        }
        if r.success and r.distances:
            row.update({
                "total_dist": round(r.total_distance, 3),
                "kurtosis_ratio": round(r.distances["kurtosis_ratio"], 2),
                "hill_ratio": round(r.distances["hill_ratio"], 2),
                "spread_ks": round(r.distances["spread_ks_dist"], 3),
                "trade_size_ks": round(r.distances["trade_size_ks_dist"], 3),
                "autocorr_l2": round(r.distances["autocorr_l2"], 3),
            })
        else:
            row.update({"total_dist": None, "kurtosis_ratio": None, "hill_ratio": None,
                        "spread_ks": None, "trade_size_ks": None, "autocorr_l2": None})
        rows.append(row)
    return pl.DataFrame(rows)
