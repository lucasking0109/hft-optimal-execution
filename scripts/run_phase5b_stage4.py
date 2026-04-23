"""Phase 5B+ Stage 4 — low-noise population mix (lit-driven).

Stage 3 found Herder+OBI improves autocorr (-19%) but inflates kurtosis (+71%).
Lit (Bamberg/CHAD) attributes this to noise traders dominating (74% of pop);
they recommend <50%.

Stage 4 tests: drop num_noise from 500 → 200 (or 350) and re-evaluate.

5 cells × 5 seeds = 25 sims, ~10-15 min.

Outputs:
  - data/synthetic/calib_stage4_results.json
  - reports/phase5b_stage4.md
  - reports/figures/phase5b_stage4_violin.png
  - reports/figures/phase5b_stage3_vs_stage4.png  (overlay comparison)
"""

from __future__ import annotations

import json
import sys
import datetime as dt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.data import load_eq_taq                                          # noqa: E402
from hft.simulators.abides_calibration import (                           # noqa: E402
    STAGE4_CELLS,
    STAGE4_SEEDS,
    run_multiseed_calibration,
)
from hft.simulators.stylized_facts import (                               # noqa: E402
    compute_stylized_facts,
    extract_real_taq_series,
)


def load_herder_params() -> dict:
    p = ROOT / "data" / "synthetic" / "herder_params.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/herder_param_calibration.py first (Step 0)."
        )
    return json.loads(p.read_text())


def attach_threshold_only(cells: list[dict], herder_params: dict) -> list[dict]:
    """Attach Step 0 threshold; preserve per-cell herder_max_size."""
    out = []
    for c in cells:
        c = dict(c)
        if c.get("num_herder", 0) > 0:
            c["herder_threshold_bps"] = herder_params["herder_params"]["entry_threshold_bps"]
            # herder_max_size already set per-cell in STAGE4_CELLS
            if c.get("herder_max_size") is None:
                # Fallback to global (shouldn't happen for Stage 4 but defensive)
                c["herder_max_size"] = herder_params["herder_params"]["max_size"]
        out.append(c)
    return out


def classify_verdict(median_ac_l2: float, median_kurt_ratio: float) -> tuple[str, str]:
    """Stage 4-specific verdict:
    Aim is to maintain Stage 3's autocorr improvement WHILE reducing kurt inflation."""
    # Both axes need to look good
    autocorr_good = median_ac_l2 < 1.9      # at least keep Stage 3 baseline (1.82)
    kurt_good = median_kurt_ratio < 4.5      # better than Stage 3 cell 304 (5.53)

    if autocorr_good and kurt_good:
        return "BREAKTHROUGH", "🌟 Stage 4 解決了 trade-off — 進 Phase 5C"
    if autocorr_good and not kurt_good:
        return "AUTOCORR_OK_KURT_BAD", "🟢 Autocorr 維持，但 kurt 仍偏高 — fine-tune max_size"
    if not autocorr_good and kurt_good:
        return "KURT_OK_AUTOCORR_LOST", "🟡 Kurt 改善但 autocorr 退化 — noise 降太多?"
    return "NO_IMPROVEMENT", "🔴 兩個指標都沒改善 — Fallback A"


def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 5B+ Stage 4 — Low-noise population mix experiment")
    print(f"Cells: {[c['id'] for c in STAGE4_CELLS]}")
    print(f"Seeds: {STAGE4_SEEDS}")
    print("=" * 72)
    print("\nHypothesis: lit (Bamberg/CHAD) recommends noise < 50% of pop.")
    print("Stage 3 had noise=500/675 = 74% (too high).")
    print("Stage 4 tests noise ∈ {200, 350} → 47-61%.\n")

    # 1. Load Step 0 + real benchmark
    herder_params = load_herder_params()
    real_benchmark_dist = herder_params["benchmark_distribution"]
    print(f"📥 Step 0 herder threshold = {herder_params['herder_params']['entry_threshold_bps']:.3f} bps")

    # 2. Real AAPL stylized facts
    print("📥 Loading real AAPL...")
    real_df = load_eq_taq("AAPL", "20200113")
    real_mids, real_spread, real_trades = extract_real_taq_series(
        real_df, sample_seconds=1, rth_only=True,
    )
    real_facts = compute_stylized_facts(
        mid_prices=real_mids, spread_bps=real_spread, trade_sizes=real_trades,
        metadata={"source": "real_AAPL_20200113"},
    )

    # 3. Attach threshold (preserve per-cell max_size)
    cells = attach_threshold_only(STAGE4_CELLS, herder_params)

    # Population mix preview
    print("\n📊 Population mix preview:")
    print(f"  {'Cell':<6}{'Noise%':<10}{'Speculative%':<15}")
    for c in cells:
        n = c["num_noise"]
        spec = c.get("num_obi", 0) + c.get("num_herder", 0)
        total = n + 100 + 25 + spec  # noise + value + momentum + speculative
        print(f"  {c['id']:<6}{n/total*100:>5.0f}%    {spec/total*100:>5.0f}%")
    print()

    # 4. Run multiseed
    results = run_multiseed_calibration(
        cells=cells,
        seeds=STAGE4_SEEDS,
        real_facts=real_facts,
        end_time="09:35:00",
        real_benchmark_dist=real_benchmark_dist,
    )

    # 5. Verdict (use cell 402 as primary comparison since it's apples-to-apples vs Stage 3 #304)
    if results["best_cell"] is None:
        verdict_code, verdict_msg = "ALL_FAILED", "🔴 All cells failed"
        best_cell = None
    else:
        # For Stage 4 verdict, look at cell 402 specifically (apples-to-apples)
        cell_402 = next((c for c in results["cells"] if c["cell_id"] == 402), None)
        if cell_402 and cell_402["median_total_distance"] is not None:
            verdict_code, verdict_msg = classify_verdict(
                cell_402["median_autocorr_l2"], cell_402["median_kurtosis_ratio"]
            )
        else:
            verdict_code, verdict_msg = "402_FAILED", "🔴 Anchor cell 402 failed"
        best_cell = results["best_cell"]

    # 6. Reproducibility metadata
    metadata = {
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "seeds": STAGE4_SEEDS,
        "cells": [c["id"] for c in STAGE4_CELLS],
        "stage": "Stage 4 (low-noise mix experiment)",
        "lit_basis": "Bamberg BERG_107 + CHAD framework — noise < 50% recommendation",
        "herder_params_source": "scripts/herder_param_calibration.py",
        "herder_threshold_bps": herder_params["herder_params"]["entry_threshold_bps"],
        "real_benchmark_dist": real_benchmark_dist,
        "real_facts_summary": real_facts.summary(),
        "verdict_code": verdict_code,
        "verdict_msg": verdict_msg,
    }

    payload = {**results, "metadata": metadata}

    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if hasattr(obj, "__dict__"): return obj.__dict__
        return str(obj)

    json_path = out_dir / "calib_stage4_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_safe))
    print(f"\n💾 Wrote {json_path}")

    # 7. Markdown report
    md = ["# Phase 5B+ Stage 4 — Low-noise Population Mix Report\n"]
    md.append(f"**Generated**: {metadata['timestamp_et']} (ET)")
    md.append(f"**Seeds**: {STAGE4_SEEDS}")
    md.append(f"**Hypothesis**: Lit (Bamberg/CHAD) recommends noise < 50% of population. "
              f"Stage 3 had 74%; Stage 4 tests 47-61%.\n")

    md.append("## Reference: Stage 3 results\n")
    md.append("| Stage 3 cell | noise/total | autocorr_l2 | kurt_ratio |")
    md.append("|---|---|---|---|")
    md.append("| 301 (baseline) | 500/625 = 80% | 2.030 | 3.24 |")
    md.append("| 304 (Herder+OBI) | 500/675 = 74% | 1.818 | 5.53 |\n")

    md.append("## Stage 4 per-cell results\n")
    md.append("| Cell | noise | OBI | Herder | max_size | n_succ | median dist | dist IQR | median ac_l2 | median kurt_ratio |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in results["cells"]:
        p = c["params"]
        if c["median_total_distance"] is None:
            md.append(f"| {p['id']} | {p['num_noise']} | {p.get('num_obi', 0)} | "
                      f"{p.get('num_herder', 0)} | {p.get('herder_max_size', '-')} | "
                      f"{c['n_success']}/{len(STAGE4_SEEDS)} | — | — | — | — |")
        else:
            md.append(
                f"| {p['id']} | {p['num_noise']} | {p.get('num_obi', 0)} | "
                f"{p.get('num_herder', 0)} | {p.get('herder_max_size', '-')} | "
                f"{c['n_success']}/{len(STAGE4_SEEDS)} | "
                f"**{c['median_total_distance']:.3f}** | "
                f"[{c['iqr_total_distance'][0]:.3f}, {c['iqr_total_distance'][1]:.3f}] | "
                f"**{c['median_autocorr_l2']:.3f}** | "
                f"{c['median_kurtosis_ratio']:.2f} |"
            )

    md.append(f"\n## Verdict\n\n**{verdict_code}**: {verdict_msg}\n")
    if best_cell:
        md.append(f"### Best cell: #{best_cell['cell_id']}")
        md.append(f"- Params: noise={best_cell['params']['num_noise']}, "
                  f"OBI={best_cell['params'].get('num_obi', 0)}, "
                  f"Herder={best_cell['params'].get('num_herder', 0)}, "
                  f"max_size={best_cell['params'].get('herder_max_size', '-')}")
        md.append(f"- Median total_distance: **{best_cell['median_total_distance']:.3f}**")
        md.append(f"- Median autocorr_l2: **{best_cell['median_autocorr_l2']:.3f}**")
        md.append(f"- Median kurtosis_ratio: **{best_cell['median_kurtosis_ratio']:.2f}**")

    md_path = reports_dir / "phase5b_stage4.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # 8. Violin plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        cell_labels, ac_data, kurt_data = [], [], []
        for c in results["cells"]:
            if c["median_autocorr_l2"] is None:
                continue
            label = (f"#{c['cell_id']}\nN={c['params']['num_noise']}"
                     f"\nO={c['params'].get('num_obi', 0)}/H={c['params'].get('num_herder', 0)}"
                     f"\nmax={c['params'].get('herder_max_size', '-')}")
            cell_labels.append(label)
            ac_data.append([r["distances"]["autocorr_l2"] for r in c["all_seed_results"] if r["success"]])
            kurt_data.append([r["distances"]["kurtosis_ratio"] for r in c["all_seed_results"] if r["success"]])

        if ac_data:
            for ax, data, ylabel, refs in [
                (ax1, ac_data, "autocorr_l2",
                 [(2.030, '301 baseline', 'gray'), (1.818, '304 Stage3', 'red'), (1.0, 'PASS', 'green')]),
                (ax2, kurt_data, "kurtosis_ratio",
                 [(3.24, '301 baseline', 'gray'), (5.53, '304 Stage3', 'red'), (2.0, 'lit ideal', 'green')]),
            ]:
                parts = ax.violinplot(data, showmedians=True, widths=0.7)
                for pc in parts['bodies']:
                    pc.set_facecolor('#4a90d9')
                    pc.set_alpha(0.6)
                ax.set_xticks(range(1, len(cell_labels) + 1))
                ax.set_xticklabels(cell_labels, fontsize=8)
                ax.set_ylabel(ylabel)
                for y, label, color in refs:
                    ax.axhline(y=y, color=color, linestyle='--', alpha=0.5, label=label)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(alpha=0.3)
            ax1.set_title("Stage 4 — autocorr_l2 (lower=better)")
            ax2.set_title("Stage 4 — kurtosis_ratio (closer to 1 = better)")
            plt.tight_layout()
            fig_path = figures_dir / "phase5b_stage4_violin.png"
            plt.savefig(fig_path, dpi=120)
            print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Violin plot failed: {e}")

    print("\n" + "=" * 72)
    print(verdict_msg)
    if best_cell:
        print(f"Best cell #{best_cell['cell_id']}: dist={best_cell['median_total_distance']:.3f}, "
              f"ac_l2={best_cell['median_autocorr_l2']:.3f}, "
              f"kurt={best_cell['median_kurtosis_ratio']:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
