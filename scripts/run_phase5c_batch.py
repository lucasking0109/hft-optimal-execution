"""Phase 5C — Batch generate 100 × 5-min ABIDES episodes (Stage 4 cell 404 anchor).

Each episode:
  - Different seed (base 1000 + i, i = 0..99)
  - Same params (noise=350, OBI=50, herder=50, max_size=7, mom=25, fund_vol=1e-3, mm=aggressive)
  - 5-min sim window (09:30:00 – 09:35:00)
  - Saves stylized facts summary + raw mid/spread/trade arrays as parquet

Outputs:
  - data/synthetic/aapl/episode_{seed}.parquet  (one per episode)
  - data/synthetic/aapl/batch_summary.json      (100-episode facts IQR)
  - reports/phase5c_batch.md
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.data.abides_loader import load_abides_run                        # noqa: E402
from hft.simulators.stylized_facts import (                               # noqa: E402
    compute_stylized_facts,
    extract_abides_series,
)


ABIDES_DIR = ROOT / "vendor" / "abides-sim"
ABIDES_PYTHON = ROOT / ".venv-abides" / "bin" / "python"

# Stage 4 cell 404 — proven multi-metric best
CELL_404_PARAMS = {
    "num_noise": 350,
    "fund_vol": 1e-3,
    "mm": "aggressive",
    "num_momentum": 25,
    "num_obi": 50,
    "num_herder": 50,
    "herder_max_size": 7,
}

N_EPISODES = 100
SEED_BASE = 1000   # episode i uses seed SEED_BASE + i


def run_episode(seed: int, threshold_bps: float, episode_dir: Path) -> dict:
    log_dir_name = f"phase5c_ep_seed{seed}"
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
        "--start-time", "09:30:00",
        "--end-time", "09:35:00",
        "--log_dir", log_dir_name,
        "--num-noise", str(CELL_404_PARAMS["num_noise"]),
        "--fund-vol", str(CELL_404_PARAMS["fund_vol"]),
        "--mm-aggressiveness", CELL_404_PARAMS["mm"],
        "--num-momentum", str(CELL_404_PARAMS["num_momentum"]),
        "--num-obi", str(CELL_404_PARAMS["num_obi"]),
        "--num-herder", str(CELL_404_PARAMS["num_herder"]),
        "--herder-threshold-bps", str(threshold_bps),
        "--herder-max-size", str(CELL_404_PARAMS["herder_max_size"]),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd, cwd=str(ROOT),
        env={"PYTHONPATH": str(ABIDES_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=180,
    )
    runtime = time.perf_counter() - t0
    if res.returncode != 0:
        return {"success": False, "seed": seed, "error": res.stderr[-500:], "runtime_s": runtime}

    # Load + facts (tolerate stylized_facts edge cases — degenerate seeds)
    try:
        abides_data = load_abides_run(log_dir)
        mids, spread, trades = extract_abides_series(abides_data, sample_seconds=1)
        facts = compute_stylized_facts(
            mid_prices=mids, spread_bps=spread, trade_sizes=trades,
            metadata={"source": "phase5c_batch", "seed": seed, **CELL_404_PARAMS},
        )
    except Exception as e:
        shutil.rmtree(log_dir, ignore_errors=True)
        return {"success": False, "seed": seed,
                "error": f"facts_failed: {type(e).__name__}: {str(e)[:200]}",
                "runtime_s": runtime}

    parquet_path = episode_dir / f"episode_seed{seed}.parquet"
    df = pl.DataFrame({"mid_price_1hz": mids[: len(mids)]})
    df.write_parquet(parquet_path)

    facts_summary = facts.summary()
    facts_json_path = episode_dir / f"episode_seed{seed}.facts.json"
    facts_json_path.write_text(json.dumps(facts_summary, indent=2))

    shutil.rmtree(log_dir)

    return {
        "success": True,
        "seed": seed,
        "runtime_s": runtime,
        "facts_summary": facts_summary,
        "parquet_path": str(parquet_path),
    }


def aggregate(episodes: list[dict]) -> dict:
    """Compute IQR of stylized facts across all successful episodes."""
    successes = [e for e in episodes if e["success"]]
    if not successes:
        return {"n_success": 0}

    metrics = [
        "excess_kurtosis", "hill_tail_index",
        "vol_autocorr_lag_1", "vol_autocorr_lag_10", "vol_autocorr_lag_50",
        "spread_bps_median", "spread_bps_p95",
        "trade_size_median", "trade_size_p95",
        "n_returns", "n_spread_obs", "n_trades",
    ]
    agg = {"n_success": len(successes), "n_total": len(episodes)}
    for m in metrics:
        vals = [e["facts_summary"][m] for e in successes if e["facts_summary"].get(m) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        agg[m] = {
            "median": float(np.median(arr)),
            "iqr_low": float(np.percentile(arr, 25)),
            "iqr_high": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return agg


def main():
    out_dir = ROOT / "data" / "synthetic" / "aapl"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"

    print("=" * 72)
    print(f"Phase 5C — Batch generate {N_EPISODES} × 5-min episodes")
    print(f"Anchor: Stage 4 cell 404 — {CELL_404_PARAMS}")
    print(f"Seeds: {SEED_BASE}..{SEED_BASE + N_EPISODES - 1}")
    print("=" * 72)

    # Step 0 herder threshold
    herder_params = json.loads((ROOT / "data" / "synthetic" / "herder_params.json").read_text())
    threshold = herder_params["herder_params"]["entry_threshold_bps"]
    print(f"\n📥 Herder threshold = {threshold:.3f} bps (from Step 0)")

    # Run all episodes (resume-able: skip if parquet already exists for that seed)
    print(f"\n🔬 Running {N_EPISODES} sims sequentially (skipping completed)...")
    t_total_start = time.perf_counter()
    episodes = []
    n_failed = 0
    n_skipped = 0
    for i in range(N_EPISODES):
        seed = SEED_BASE + i
        parquet_path = out_dir / f"episode_seed{seed}.parquet"
        facts_json_path = out_dir / f"episode_seed{seed}.facts.json"
        if parquet_path.exists() and facts_json_path.exists():
            try:
                facts_summary = json.loads(facts_json_path.read_text())
                episodes.append({
                    "success": True, "seed": seed, "runtime_s": 0.0,
                    "facts_summary": facts_summary,
                    "parquet_path": str(parquet_path),
                    "skipped_existing": True,
                })
                n_skipped += 1
                if i % 10 == 0:
                    print(f"  [{i+1:>3}/{N_EPISODES}] seed={seed} ⏭ (resumed)")
                continue
            except Exception:
                pass  # fall through and re-run
        ep = run_episode(seed=seed, threshold_bps=threshold, episode_dir=out_dir)
        episodes.append(ep)
        if ep["success"]:
            kurt = ep["facts_summary"]["excess_kurtosis"]
            ac_l1 = ep["facts_summary"]["vol_autocorr_lag_1"]
            print(f"  [{i+1:>3}/{N_EPISODES}] seed={seed} ✅ ({ep['runtime_s']:.1f}s) "
                  f"kurt={kurt:>6.1f} ac_l1={ac_l1:>+.3f}")
        else:
            n_failed += 1
            print(f"  [{i+1:>3}/{N_EPISODES}] seed={seed} ❌ ({ep['runtime_s']:.1f}s) "
                  f"{ep.get('error', '')[:60]}")
    print(f"\n   Skipped {n_skipped} pre-existing parquet (resumed run)")

    t_total = time.perf_counter() - t_total_start
    n_success = N_EPISODES - n_failed
    print(f"\n📊 Done in {t_total/60:.1f} min — {n_success}/{N_EPISODES} successful")

    # Aggregate
    agg = aggregate(episodes)
    summary = {
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "n_episodes": N_EPISODES,
        "n_success": n_success,
        "n_failed": n_failed,
        "total_runtime_s": t_total,
        "params": CELL_404_PARAMS,
        "herder_threshold_bps": threshold,
        "seed_base": SEED_BASE,
        "aggregate_facts": agg,
        "per_episode": [
            {"seed": e["seed"], "success": e["success"],
             "runtime_s": e["runtime_s"],
             "facts_summary": e.get("facts_summary"),
             "error": e.get("error")}
            for e in episodes
        ],
    }
    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"💾 Wrote {summary_path}")

    # Markdown report
    md = ["# Phase 5C — Batch 100-Episode Sanity Report\n"]
    md.append(f"**Generated**: {summary['timestamp_et']} (ET)")
    md.append(f"**Anchor**: Stage 4 cell 404 (noise=350, OBI=50, herder=50, max_size=7, fund_vol=1e-3)")
    md.append(f"**Sim window**: 5 min (09:30:00 – 09:35:00)")
    md.append(f"**Episodes**: {n_success}/{N_EPISODES} successful")
    md.append(f"**Total runtime**: {t_total/60:.1f} min\n")

    if "excess_kurtosis" in agg:
        md.append("## Stylized facts distribution across 100 synthetic 5-min episodes\n")
        md.append("| Metric | Median | IQR (25-75%) | Min – Max |")
        md.append("|---|---|---|---|")
        for m in ["excess_kurtosis", "hill_tail_index",
                  "vol_autocorr_lag_1", "vol_autocorr_lag_10", "vol_autocorr_lag_50",
                  "spread_bps_median", "spread_bps_p95",
                  "trade_size_median", "trade_size_p95",
                  "n_returns", "n_trades"]:
            if m in agg:
                a = agg[m]
                md.append(f"| {m} | {a['median']:.3f} | [{a['iqr_low']:.3f}, {a['iqr_high']:.3f}] | "
                          f"[{a['min']:.3f}, {a['max']:.3f}] |")

    md.append("\n## Reference: real AAPL 5-min benchmark IQR (from Step 0 bootstrap)\n")
    md.append("| Metric | Median | IQR | Synth match? |")
    md.append("|---|---|---|---|")
    real_bench = herder_params["benchmark_distribution"]
    for synth_m, real_m in [("excess_kurtosis", "excess_kurtosis"),
                              ("vol_autocorr_lag_1", "vol_autocorr_lag_1"),
                              ("vol_autocorr_lag_10", "vol_autocorr_lag_10"),
                              ("spread_bps_median", "spread_bps_median"),
                              ("trade_size_median", "trade_size_median")]:
        real_a = real_bench.get(real_m, {})
        synth_a = agg.get(synth_m, {})
        if real_a and synth_a:
            in_iqr = real_a.get("iqr_low", 0) <= synth_a["median"] <= real_a.get("iqr_high", 0)
            md.append(f"| {synth_m} | real={real_a['median']:.3f} | "
                      f"real IQR=[{real_a['iqr_low']:.3f}, {real_a['iqr_high']:.3f}] "
                      f"synth median={synth_a['median']:.3f} | {'✅' if in_iqr else '⚠️'} |")

    md.append(f"\n## Disk\n")
    parquet_files = list(out_dir.glob("episode_seed*.parquet"))
    total_size_mb = sum(p.stat().st_size for p in parquet_files) / 1024 / 1024
    md.append(f"- {len(parquet_files)} parquet files, total {total_size_mb:.1f} MB\n")

    md.append("## Next: Phase 5D")
    md.append("Strict KS + Wasserstein + Bonferroni vs real 5-min slices "
              "→ pass means合成 episodes 統計上接近真實，可用於 Phase 6 RL 訓練。")

    md_path = reports_dir / "phase5c_batch.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    print("\n" + "=" * 72)
    print(f"✅ Batch complete: {n_success}/{N_EPISODES} episodes, {total_size_mb:.1f} MB on disk")
    print("=" * 72)


if __name__ == "__main__":
    main()
