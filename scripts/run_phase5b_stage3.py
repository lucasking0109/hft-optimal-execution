"""Phase 5B+ Stage 3 — 4-cell × 5-seed multi-seed factor experiment.

Tests whether HerderAgent + OBI integration improves vol_autocorr_l2 distance
to real AAPL. Compares 4 cells (baseline / +OBI / +Herder / +Both) with
5 seeds each = 20 sims.

Outputs:
  - data/synthetic/calib_stage3_results.json (incl. reproducibility metadata)
  - reports/phase5b_stage3.md
  - reports/figures/phase5b_stage3_violin.png

Per Stage 3 plan 6-bucket decision tree.
NO Silent Fallback: any failed cell surfaces explicitly in report.
"""

from __future__ import annotations

import json
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.data import load_eq_taq                                          # noqa: E402
from hft.simulators.abides_calibration import (                           # noqa: E402
    STAGE3_CELLS,
    STAGE3_SEEDS,
    run_multiseed_calibration,
)
from hft.simulators.stylized_facts import (                               # noqa: E402
    compute_stylized_facts,
    extract_real_taq_series,
)


def load_herder_params() -> dict:
    """Load Step 0 herder parameter calibration output."""
    p = ROOT / "data" / "synthetic" / "herder_params.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/herder_param_calibration.py first (Step 0)."
        )
    return json.loads(p.read_text())


def attach_herder_params(cells: list[dict], herder_params: dict) -> list[dict]:
    """Attach Step 0 calibrated herder threshold/size to cells with num_herder > 0."""
    out = []
    for c in cells:
        c = dict(c)
        if c.get("num_herder", 0) > 0:
            c["herder_threshold_bps"] = herder_params["herder_params"]["entry_threshold_bps"]
            c["herder_max_size"] = herder_params["herder_params"]["max_size"]
        out.append(c)
    return out


def classify_verdict(median_ac_l2: float, real_iqr: list[float]) -> tuple[str, str]:
    """6-bucket decision tree per Stage 3 plan."""
    if median_ac_l2 < 0.5:
        return "PASS_STRONG", "🌟 Stage 4 fine-tune (autocorr in real IQR low end)"
    if median_ac_l2 < 1.0:
        return "PASS", "🎯 Stage 4 fine-tune (autocorr near real range)"
    if median_ac_l2 < 1.5:
        return "IMPROVED", "🟢 Stage 4 fine-tune (significant improvement vs prior 2.14)"
    if median_ac_l2 < 1.8:
        return "MODERATE", "🟡 Stage 4 OR accept + Phase 6 caveat"
    if median_ac_l2 < 2.0:
        return "MARGINAL", "🟠 Re-examine herder threshold (Step 0)"
    return "STRUCTURAL", "🔴 Fallback option A — accept prior cell #4 + 'despite gap' Phase 6"


def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 5B+ Stage 3 — 4-cell × 5-seed factor experiment")
    print(f"Cells: {[c['id'] for c in STAGE3_CELLS]}")
    print(f"Seeds: {STAGE3_SEEDS}")
    print("=" * 72)

    # 1. Load Step 0 herder params + benchmark distribution
    print("\n📥 Loading Step 0 herder calibration...")
    herder_params = load_herder_params()
    real_benchmark_dist = herder_params["benchmark_distribution"]
    print(f"   herder threshold: {herder_params['herder_params']['entry_threshold_bps']:.3f} bps")
    print(f"   herder max_size:  {herder_params['herder_params']['max_size']}")
    print(f"   real 5-min benchmark IQR (autocorr_lag_1): "
          f"[{real_benchmark_dist['vol_autocorr_lag_1']['iqr_low']:.3f}, "
          f"{real_benchmark_dist['vol_autocorr_lag_1']['iqr_high']:.3f}]")

    # 2. Real AAPL stylized facts (full RTH, baseline reference)
    print("\n📥 Computing real AAPL stylized facts (full RTH benchmark)...")
    real_df = load_eq_taq("AAPL", "20200113")
    real_mids, real_spread, real_trades = extract_real_taq_series(
        real_df, sample_seconds=1, rth_only=True,
    )
    real_facts = compute_stylized_facts(
        mid_prices=real_mids, spread_bps=real_spread, trade_sizes=real_trades,
        metadata={"source": "real_AAPL_20200113"},
    )
    print(f"   Real (full RTH): kurt={real_facts.excess_kurtosis:.1f}, "
          f"hill={real_facts.hill_tail_index:.2f}, "
          f"autocorr_l1={real_facts.vol_autocorr[0]:.3f}")

    # 3. Attach Step 0 calibrated params to herder cells
    cells = attach_herder_params(STAGE3_CELLS, herder_params)

    # 4. Run multiseed calibration
    results = run_multiseed_calibration(
        cells=cells,
        seeds=STAGE3_SEEDS,
        real_facts=real_facts,
        end_time="09:35:00",
        real_benchmark_dist=real_benchmark_dist,
    )

    # 5. Verdict
    if results["best_cell"] is None:
        verdict_code, verdict_msg = "ALL_FAILED", "🔴 All cells failed; STOP and inspect"
        best_cell = None
    else:
        best_cell = results["best_cell"]
        median_ac = best_cell["median_autocorr_l2"]
        real_iqr = [real_benchmark_dist["vol_autocorr_lag_1"]["iqr_low"],
                    real_benchmark_dist["vol_autocorr_lag_1"]["iqr_high"]]
        verdict_code, verdict_msg = classify_verdict(median_ac, real_iqr)

    # 6. Reproducibility metadata
    metadata = {
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "seeds": STAGE3_SEEDS,
        "cells": [c["id"] for c in STAGE3_CELLS],
        "herder_params_source": "scripts/herder_param_calibration.py",
        "herder_params": herder_params["herder_params"],
        "real_benchmark_dist": real_benchmark_dist,
        "real_facts_summary": real_facts.summary(),
        "verdict_code": verdict_code,
        "verdict_msg": verdict_msg,
    }
    payload = {**results, "metadata": metadata}

    # Convert to JSON-safe form
    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if hasattr(obj, "__dict__"): return obj.__dict__
        return str(obj)

    json_path = out_dir / "calib_stage3_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_safe))
    print(f"\n💾 Wrote {json_path}")

    # 7. Markdown report
    md = ["# Phase 5B+ Stage 3 — HerderAgent + OBI Calibration Report\n"]
    md.append(f"**Generated**: {metadata['timestamp_et']} (ET)")
    md.append(f"**Seeds**: {STAGE3_SEEDS}")
    md.append(f"**Sim window**: 09:30:00 – 09:35:00 (5 min)\n")

    md.append("## Real AAPL benchmark\n")
    md.append(f"- Full RTH (23k returns): kurt={real_facts.excess_kurtosis:.1f}, "
              f"hill={real_facts.hill_tail_index:.2f}, autocorr_l1={real_facts.vol_autocorr[0]:.3f}")
    md.append(f"- 5-min IQR (apples-to-apples): autocorr_l1 ∈ "
              f"[{real_benchmark_dist['vol_autocorr_lag_1']['iqr_low']:.3f}, "
              f"{real_benchmark_dist['vol_autocorr_lag_1']['iqr_high']:.3f}]\n")

    md.append("## Per-cell results (median ± IQR over 5 seeds)\n")
    md.append("| Cell | OBI | Herder | n_success | median dist | dist IQR | median ac_l2 | ac_l2 IQR | median kurt_ratio |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for c in results["cells"]:
        p = c["params"]
        if c["median_total_distance"] is None:
            md.append(f"| {p['id']} | {p.get('num_obi', 0)} | {p.get('num_herder', 0)} | "
                      f"{c['n_success']}/{len(STAGE3_SEEDS)} | — | — | — | — | — |")
        else:
            md.append(
                f"| {p['id']} | {p.get('num_obi', 0)} | {p.get('num_herder', 0)} | "
                f"{c['n_success']}/{len(STAGE3_SEEDS)} | "
                f"**{c['median_total_distance']:.3f}** | "
                f"[{c['iqr_total_distance'][0]:.3f}, {c['iqr_total_distance'][1]:.3f}] | "
                f"**{c['median_autocorr_l2']:.3f}** | "
                f"[{c['iqr_autocorr_l2'][0]:.3f}, {c['iqr_autocorr_l2'][1]:.3f}] | "
                f"{c['median_kurtosis_ratio']:.2f} |"
            )

    md.append("\n## Verdict\n")
    md.append(f"**{verdict_code}**: {verdict_msg}\n")
    if best_cell:
        md.append(f"### Best cell: #{best_cell['cell_id']}")
        md.append(f"- Params: noise={best_cell['params']['num_noise']}, "
                  f"fund_vol={best_cell['params']['fund_vol']:.0e}, "
                  f"mm={best_cell['params']['mm']}, "
                  f"momentum={best_cell['params'].get('num_momentum', 25)}, "
                  f"OBI={best_cell['params'].get('num_obi', 0)}, "
                  f"Herder={best_cell['params'].get('num_herder', 0)}")
        md.append(f"- Median total_distance: **{best_cell['median_total_distance']:.3f}** "
                  f"(prior Stage 1 v2 #101 baseline = 2.332)")
        md.append(f"- Median autocorr_l2: **{best_cell['median_autocorr_l2']:.3f}**")
        improvement_vs_2_14 = (2.144 - best_cell['median_total_distance']) / 2.144 * 100
        md.append(f"- vs prior best (cell #4 = 2.144): **{improvement_vs_2_14:+.1f}%**")

    md.append("\n## Reproducibility metadata\n")
    md.append(f"- Herder params source: `{metadata['herder_params_source']}`")
    md.append(f"- Herder threshold: {metadata['herder_params']['entry_threshold_bps']:.3f} bps")
    md.append(f"- Herder max_size: {metadata['herder_params']['max_size']}")
    md.append(f"- Lookback range: [{metadata['herder_params']['lookback_window_secs_min']}, "
              f"{metadata['herder_params']['lookback_window_secs_max']}]s")

    md_path = reports_dir / "phase5b_stage3.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # 8. Violin plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        cell_labels = []
        ac_data = []
        for c in results["cells"]:
            if c["median_autocorr_l2"] is None:
                continue
            label = f"#{c['cell_id']}\nOBI={c['params'].get('num_obi', 0)}\nHerder={c['params'].get('num_herder', 0)}"
            cell_labels.append(label)
            seed_acs = [r["distances"]["autocorr_l2"] for r in c["all_seed_results"]
                        if r["success"]]
            ac_data.append(seed_acs)
        if ac_data:
            parts = ax.violinplot(ac_data, showmedians=True, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor('#4a90d9')
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(cell_labels) + 1))
            ax.set_xticklabels(cell_labels, fontsize=9)
            ax.set_ylabel("autocorr_l2 distance (synth vs real RTH)")
            ax.set_title("Stage 3 — autocorr_l2 across 5 seeds per cell")
            # Add real benchmark IQR band (this is autocorr_l1 not l2 but useful for context)
            ax.axhline(y=2.144, color='red', linestyle='--', alpha=0.5, label='prior best 2.14')
            ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='IMPROVED bar')
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='PASS bar')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig_path = figures_dir / "phase5b_stage3_violin.png"
            plt.savefig(fig_path, dpi=120)
            print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Violin plot failed: {e}")

    # 9. Final verdict to stdout
    print("\n" + "=" * 72)
    print(verdict_msg)
    if best_cell:
        print(f"Best cell #{best_cell['cell_id']}: median dist={best_cell['median_total_distance']:.3f}, "
              f"ac_l2={best_cell['median_autocorr_l2']:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
