"""Multi-ticker robustness validation of Stage 4 cell 404 architecture.

For each ticker (AMZN, MSFT, NVDA, TSLA):
  1. Step 0 calibration on that ticker's real data (P75 drift, max_size, 5-min IQR)
  2. Run N × 5-min synthetic episodes with cell 404 architecture +
     ticker-specific scaling (herder threshold + max_size from Step 0)
  3. Compute stylized facts distance synth-vs-real per ticker

Outputs:
  - data/synthetic/multi_ticker/{ticker}_step0.json
  - data/synthetic/multi_ticker/{ticker}_episodes/episode_seed*.parquet
  - data/synthetic/multi_ticker/{ticker}_episodes/episode_seed*.facts.json
  - data/synthetic/multi_ticker/{ticker}_summary.json
  - reports/multi_ticker_validation.md

This validates whether the cell 404 architecture (noise=350, OBI=50, herder=50,
mom=25, fund_vol=1e-3, mm=aggressive) generalizes across tickers when
herder scaling parameters are auto-derived per-ticker.
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

from hft.data import load_eq_taq                                          # noqa: E402
from hft.data.abides_loader import load_abides_run                        # noqa: E402
from hft.simulators.stylized_facts import (                               # noqa: E402
    compute_stylized_facts,
    extract_abides_series,
    extract_real_taq_series,
    facts_distance,
    total_calibration_distance,
)

# Reuse Step 0 helpers
sys.path.insert(0, str(ROOT / "scripts"))
from herder_param_calibration import (                                    # noqa: E402
    measure_5sec_drift_bps,
    bootstrap_5min_slices,
    aggregate_slice_distribution,
    SANE_THRESHOLD_BPS_RANGE,
    SANE_TRADE_SIZE_RANGE,
)


ABIDES_DIR = ROOT / "vendor" / "abides-sim"
ABIDES_PYTHON = ROOT / ".venv-abides" / "bin" / "python"

CELL_404_ARCHITECTURE = {
    "num_noise": 350,
    "fund_vol": 1e-3,
    "mm": "aggressive",
    "num_momentum": 25,
    "num_obi": 50,
    "num_herder": 50,
}

# 4 new tickers + their representativeness rationale
NEW_TICKERS = [
    ("AMZN", "high-price tech ($1800)"),
    ("MSFT", "mid-price low-vol ($160)"),
    ("NVDA", "mid-price high-vol HFT favorite"),
    ("TSLA", "high-vol stress test"),
]

DATE = "20200113"
N_EPISODES_PER_TICKER = 25
SEED_BASE = 2000


def step0_for_ticker(ticker: str, taq_df: pl.DataFrame) -> dict:
    """Auto-derive Step 0 herder parameters from a ticker's real data."""
    mids_1hz, _, trades = extract_real_taq_series(
        taq_df, sample_seconds=1, rth_only=True,
    )

    drift_stats = measure_5sec_drift_bps(mids_1hz)
    threshold_bps = drift_stats["p75"]
    if not (SANE_THRESHOLD_BPS_RANGE[0] <= threshold_bps <= SANE_THRESHOLD_BPS_RANGE[1]):
        raise ValueError(
            f"{ticker}: P75 drift = {threshold_bps:.3f} bps outside sane "
            f"range {SANE_THRESHOLD_BPS_RANGE}. (NO Silent Fallback)"
        )

    trade_size_median = float(np.median(trades))
    if not (SANE_TRADE_SIZE_RANGE[0] <= trade_size_median <= SANE_TRADE_SIZE_RANGE[1]):
        raise ValueError(
            f"{ticker}: trade_size median = {trade_size_median:.0f} outside "
            f"sane range {SANE_TRADE_SIZE_RANGE} (NO Silent Fallback)."
        )
    max_size = max(1, int(round(trade_size_median * 0.1)))

    # Bootstrap 20 random 5-min slices for benchmark
    summaries = bootstrap_5min_slices(taq_df, n_slices=20, slice_minutes=5,
                                       sample_seconds=1, seed=1234)
    benchmark = aggregate_slice_distribution(summaries)

    return {
        "ticker": ticker,
        "drift_stats": drift_stats,
        "trade_size_median": trade_size_median,
        "herder_threshold_bps": threshold_bps,
        "herder_max_size": max_size,
        "benchmark_5min": benchmark,
    }


def run_episode(seed: int, ticker_step0: dict, episode_dir: Path) -> dict:
    log_dir_name = f"multi_ticker_{ticker_step0['ticker']}_seed{seed}"
    log_dir = ROOT / "log" / log_dir_name
    if log_dir.exists():
        shutil.rmtree(log_dir)

    cmd = [
        str(ABIDES_PYTHON),
        str(ABIDES_DIR / "abides.py"),
        "-c", "calib_aapl",
        "-t", "ABM",                # ABIDES uses synthetic ticker symbol regardless
        "-d", DATE,
        "-s", str(seed),
        "--start-time", "09:30:00",
        "--end-time", "09:35:00",
        "--log_dir", log_dir_name,
        "--num-noise", str(CELL_404_ARCHITECTURE["num_noise"]),
        "--fund-vol", str(CELL_404_ARCHITECTURE["fund_vol"]),
        "--mm-aggressiveness", CELL_404_ARCHITECTURE["mm"],
        "--num-momentum", str(CELL_404_ARCHITECTURE["num_momentum"]),
        "--num-obi", str(CELL_404_ARCHITECTURE["num_obi"]),
        "--num-herder", str(CELL_404_ARCHITECTURE["num_herder"]),
        "--herder-threshold-bps", str(ticker_step0["herder_threshold_bps"]),
        "--herder-max-size", str(ticker_step0["herder_max_size"]),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd, cwd=str(ROOT),
        env={"PYTHONPATH": str(ABIDES_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=180,
    )
    runtime = time.perf_counter() - t0
    if res.returncode != 0:
        return {"success": False, "seed": seed, "error": res.stderr[-300:], "runtime_s": runtime}

    try:
        abides_data = load_abides_run(log_dir)
        mids, spread, trades = extract_abides_series(abides_data, sample_seconds=1)
        facts = compute_stylized_facts(
            mid_prices=mids, spread_bps=spread, trade_sizes=trades,
            metadata={"source": "multi_ticker", "ticker": ticker_step0["ticker"], "seed": seed},
        )
    except Exception as e:
        shutil.rmtree(log_dir, ignore_errors=True)
        return {"success": False, "seed": seed,
                "error": f"facts_failed: {type(e).__name__}: {str(e)[:200]}",
                "runtime_s": runtime}

    parquet_path = episode_dir / f"episode_seed{seed}.parquet"
    pl.DataFrame({"mid_price_1hz": mids[: len(mids)]}).write_parquet(parquet_path)
    facts_summary = facts.summary()
    (episode_dir / f"episode_seed{seed}.facts.json").write_text(json.dumps(facts_summary, indent=2))

    shutil.rmtree(log_dir)
    return {
        "success": True, "seed": seed, "runtime_s": runtime,
        "facts_summary": facts_summary,
        "parquet_path": str(parquet_path),
    }


def aggregate_episodes(episodes: list[dict]) -> dict:
    successes = [e for e in episodes if e["success"]]
    if not successes:
        return {"n_success": 0}
    metrics = ["excess_kurtosis", "hill_tail_index",
               "vol_autocorr_lag_1", "vol_autocorr_lag_10", "vol_autocorr_lag_50",
               "spread_bps_median", "spread_bps_p95",
               "trade_size_median", "trade_size_p95"]
    agg = {"n_success": len(successes), "n_total": len(episodes)}
    for m in metrics:
        vals = [e["facts_summary"][m] for e in successes
                if e["facts_summary"].get(m) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        agg[m] = {"median": float(np.median(arr)),
                  "iqr_low": float(np.percentile(arr, 25)),
                  "iqr_high": float(np.percentile(arr, 75)),
                  "min": float(np.min(arr)), "max": float(np.max(arr))}
    return agg


def benchmark_match_score(synth_agg: dict, real_bench: dict) -> dict:
    """For each metric, check if synth median is in real 5-min IQR."""
    out = {}
    for m in ["excess_kurtosis", "vol_autocorr_lag_1", "vol_autocorr_lag_10",
              "spread_bps_median", "trade_size_median"]:
        s = synth_agg.get(m)
        r = real_bench.get(m)
        if s is None or r is None:
            out[m] = None
            continue
        in_iqr = r["iqr_low"] <= s["median"] <= r["iqr_high"]
        out[m] = {
            "synth_median": s["median"],
            "real_median": r["median"],
            "real_iqr": [r["iqr_low"], r["iqr_high"]],
            "synth_in_real_iqr": bool(in_iqr),
        }
    return out


def main():
    out_dir = ROOT / "data" / "synthetic" / "multi_ticker"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"

    print("=" * 72)
    print("Multi-ticker robustness validation (Stage 4 cell 404 architecture)")
    print(f"Date: {DATE}")
    print(f"Episodes per ticker: {N_EPISODES_PER_TICKER}")
    print(f"Tickers: {[t for t, _ in NEW_TICKERS]}")
    print("=" * 72)

    all_results: dict = {}
    t_total_start = time.perf_counter()

    for ticker, rationale in NEW_TICKERS:
        print(f"\n{'─' * 72}")
        print(f"🎯 {ticker} — {rationale}")
        print(f"{'─' * 72}")

        ticker_dir = out_dir / f"{ticker}_episodes"
        ticker_dir.mkdir(exist_ok=True)

        # 1. Step 0 calibration (or load if cached)
        step0_path = out_dir / f"{ticker}_step0.json"
        if step0_path.exists():
            print(f"  📥 Loading cached Step 0 from {step0_path}")
            step0 = json.loads(step0_path.read_text())
        else:
            print(f"  📥 Loading {ticker} {DATE} TAQ...")
            try:
                taq_df = load_eq_taq(ticker, DATE)
            except Exception as e:
                print(f"  ❌ Failed to load {ticker}: {e}")
                all_results[ticker] = {"error": f"load failed: {e}", "rationale": rationale}
                continue
            print(f"     Rows: {len(taq_df):,}")
            print(f"  🔬 Step 0 calibration...")
            try:
                step0 = step0_for_ticker(ticker, taq_df)
            except Exception as e:
                print(f"  ❌ Step 0 failed: {e}")
                all_results[ticker] = {"error": f"step0 failed: {e}", "rationale": rationale}
                continue
            step0_path.write_text(json.dumps(step0, indent=2))
        print(f"  → threshold = {step0['herder_threshold_bps']:.3f} bps, "
              f"max_size = {step0['herder_max_size']}")

        # 2. Run N episodes
        print(f"  🔬 Running {N_EPISODES_PER_TICKER} synth episodes...")
        episodes = []
        n_failed = 0
        for i in range(N_EPISODES_PER_TICKER):
            seed = SEED_BASE + i
            facts_json_path = ticker_dir / f"episode_seed{seed}.facts.json"
            parquet_path = ticker_dir / f"episode_seed{seed}.parquet"
            if facts_json_path.exists() and parquet_path.exists():
                episodes.append({
                    "success": True, "seed": seed, "runtime_s": 0.0,
                    "facts_summary": json.loads(facts_json_path.read_text()),
                    "parquet_path": str(parquet_path),
                })
                continue
            ep = run_episode(seed=seed, ticker_step0=step0, episode_dir=ticker_dir)
            episodes.append(ep)
            if ep["success"]:
                kurt = ep["facts_summary"]["excess_kurtosis"]
                ac_l1 = ep["facts_summary"]["vol_autocorr_lag_1"]
                if i % 5 == 0 or i == N_EPISODES_PER_TICKER - 1:
                    print(f"    [{i+1:>2}/{N_EPISODES_PER_TICKER}] seed={seed} ✅ "
                          f"({ep['runtime_s']:.1f}s) kurt={kurt:>6.1f} ac_l1={ac_l1:>+.3f}")
            else:
                n_failed += 1
                print(f"    [{i+1:>2}/{N_EPISODES_PER_TICKER}] seed={seed} ❌ "
                      f"{ep.get('error', '')[:60]}")

        # 3. Aggregate + match score
        agg = aggregate_episodes(episodes)
        match = benchmark_match_score(agg, step0["benchmark_5min"])

        all_results[ticker] = {
            "rationale": rationale,
            "step0": step0,
            "n_success": agg.get("n_success", 0),
            "n_failed": n_failed,
            "synth_agg": agg,
            "match_score": match,
        }

        # Quick verdict per ticker
        passing = sum(1 for m in match.values() if m and m.get("synth_in_real_iqr"))
        total_metrics = sum(1 for m in match.values() if m is not None)
        print(f"  📊 {ticker} result: {passing}/{total_metrics} metrics in real 5-min IQR")

    t_total = time.perf_counter() - t_total_start

    # Save full results
    summary_path = out_dir / "multi_ticker_summary.json"
    summary_path.write_text(json.dumps({
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "architecture": CELL_404_ARCHITECTURE,
        "n_episodes_per_ticker": N_EPISODES_PER_TICKER,
        "total_runtime_s": t_total,
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n💾 Wrote {summary_path}")

    # Markdown report
    md = ["# Multi-Ticker Robustness Validation Report\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Date sampled**: {DATE}")
    md.append(f"**Architecture (shared across tickers)**: cell 404 — "
              f"noise=350, fund_vol=1e-3, mm=aggressive, mom=25, OBI=50, herder=50")
    md.append(f"**Per-ticker scaling**: herder_threshold_bps + herder_max_size auto-derived from real {DATE} data\n")

    md.append("## Per-ticker results\n")
    md.append("| Ticker | Rationale | Threshold (bps) | Max size | Episodes pass | Metrics in real IQR |")
    md.append("|---|---|---|---|---|---|")
    for ticker, _ in NEW_TICKERS:
        r = all_results.get(ticker)
        if not r:
            md.append(f"| {ticker} | — | — | — | — | (not run) |")
            continue
        if "error" in r:
            md.append(f"| {ticker} | {r['rationale']} | — | — | — | ❌ {r['error']} |")
            continue
        s0 = r["step0"]
        match = r["match_score"]
        passing = sum(1 for m in match.values() if m and m.get("synth_in_real_iqr"))
        total_m = sum(1 for m in match.values() if m is not None)
        md.append(f"| {ticker} | {r['rationale']} | "
                  f"{s0['herder_threshold_bps']:.3f} | {s0['herder_max_size']} | "
                  f"{r['n_success']}/{r['n_success'] + r['n_failed']} | "
                  f"**{passing}/{total_m}** |")

    md.append("\n## Per-ticker per-metric match (synth median vs real 5-min IQR)\n")
    for ticker, _ in NEW_TICKERS:
        r = all_results.get(ticker)
        if not r or "error" in r:
            continue
        md.append(f"\n### {ticker}\n")
        md.append("| Metric | Synth median | Real median | Real 5-min IQR | Match |")
        md.append("|---|---|---|---|---|")
        for m_name, m_val in r["match_score"].items():
            if not m_val:
                continue
            md.append(f"| {m_name} | {m_val['synth_median']:.3f} | {m_val['real_median']:.3f} | "
                      f"[{m_val['real_iqr'][0]:.3f}, {m_val['real_iqr'][1]:.3f}] | "
                      f"{'✅' if m_val['synth_in_real_iqr'] else '⚠️'} |")

    md.append(f"\n## Total runtime: {t_total/60:.1f} min")

    md_path = reports_dir / "multi_ticker_validation.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    print("\n" + "=" * 72)
    print(f"✅ Multi-ticker validation complete in {t_total/60:.1f} min")
    print(f"Tickers tested: {len(all_results)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
