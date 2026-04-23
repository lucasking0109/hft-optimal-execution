"""Phase 5B+ Stage 5 — Long-episode re-calibration (15-min sim window).

Phase 5C pilot found: Stage 4 cell 404 extrapolates poorly from 5-min to
15-min. Kurtosis ratio explodes 2.03 → 7.71 (autocorr improves 1.90 → 1.67).

Hypothesis: herder cascades accumulate over longer time. Smaller herder count
+ tighter position_cap + smaller max_size should prevent buildup.

4 cells × 3 seeds = 12 sims. With ~340s per 15-min sim → ~70 min wall.

Outputs:
  - data/synthetic/calib_stage5_results.json
  - reports/phase5b_stage5.md
  - reports/figures/phase5b_stage5_violin.png
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
    STAGE5_CELLS,
    STAGE5_SEEDS,
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
            f"{p} not found. Run scripts/herder_param_calibration.py first."
        )
    return json.loads(p.read_text())


def attach_threshold_only(cells: list[dict], herder_params: dict) -> list[dict]:
    """Attach Step 0 threshold; preserve per-cell herder_max_size + position_cap."""
    out = []
    for c in cells:
        c = dict(c)
        if c.get("num_herder", 0) > 0:
            c["herder_threshold_bps"] = herder_params["herder_params"]["entry_threshold_bps"]
        out.append(c)
    return out


# Stage 4 cell 404 reference (5-min × 5 seeds)
STAGE4_404_REF = {"autocorr_l2": 1.902, "kurtosis_ratio": 2.03, "total_distance": 2.436}
# Phase 5C pilot reference (15-min × 1 seed, same params as 404)
PILOT_REF = {"autocorr_l2": 1.670, "kurtosis_ratio": 7.705, "total_distance": 3.077}


def classify_verdict(median_ac_l2: float, median_kurt_ratio: float, median_dist: float) -> tuple[str, str]:
    """Stage 5 verdict: at 15-min, want kurt < 4 (vs pilot 7.7) AND ac_l2 < 2."""
    autocorr_ok = median_ac_l2 < 2.0   # at least pilot baseline
    kurt_ok = median_kurt_ratio < 4.0   # halve the pilot kurt
    dist_ok = median_dist < 3.0        # better than pilot 3.08

    if autocorr_ok and kurt_ok and dist_ok:
        return "BREAKTHROUGH_15MIN", "🌟 找到 15-min sweet spot — 進 Phase 5C 100 episodes"
    if autocorr_ok and not kurt_ok:
        return "AUTOCORR_OK_KURT_HIGH", "🟡 Autocorr 維持但 kurt 仍偏高 — 試更小 max_size"
    if not autocorr_ok and kurt_ok:
        return "KURT_OK_AUTOCORR_LOST", "🟡 Kurt 改善但 autocorr 退化 — herder 砍太多"
    return "NO_IMPROVEMENT", "🔴 兩指標都沒改善 — 退回 5-min episodes 路 1"


def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 5B+ Stage 5 — 15-min episode re-calibration")
    print(f"Cells: {[c['id'] for c in STAGE5_CELLS]}")
    print(f"Seeds: {STAGE5_SEEDS}")
    print(f"Sim window: 09:30:00 – 09:45:00 (15 min)")
    print("=" * 72)
    print(f"\nReferences:")
    print(f"  Stage 4 #404 (5-min × 5 seeds): {STAGE4_404_REF}")
    print(f"  Phase 5C pilot (15-min × 1 seed, same #404 params): {PILOT_REF}")
    print()

    herder_params = load_herder_params()
    threshold = herder_params["herder_params"]["entry_threshold_bps"]
    print(f"📥 Step 0 herder threshold = {threshold:.3f} bps")

    print("📥 Loading real AAPL...")
    real_df = load_eq_taq("AAPL", "20200113")
    real_mids, real_spread, real_trades = extract_real_taq_series(
        real_df, sample_seconds=1, rth_only=True,
    )
    real_facts = compute_stylized_facts(
        mid_prices=real_mids, spread_bps=real_spread, trade_sizes=real_trades,
        metadata={"source": "real_AAPL_20200113"},
    )

    cells = attach_threshold_only(STAGE5_CELLS, herder_params)

    print("\n📊 Stage 5 cell preview:")
    print(f"  {'Cell':<6}{'Herder':<10}{'max_size':<12}{'pos_cap':<10}")
    for c in cells:
        print(f"  {c['id']:<6}{c['num_herder']:<10}{c.get('herder_max_size', '-'):<12}"
              f"{c.get('herder_position_cap', '-'):<10}")
    print()

    # Run with 15-min end_time
    results = run_multiseed_calibration(
        cells=cells,
        seeds=STAGE5_SEEDS,
        real_facts=real_facts,
        end_time="09:45:00",   # 15 min
        real_benchmark_dist=herder_params["benchmark_distribution"],
    )

    # Verdict
    if results["best_cell"] is None:
        verdict_code, verdict_msg = "ALL_FAILED", "🔴 All cells failed"
        best_cell = None
    else:
        best_cell = results["best_cell"]
        verdict_code, verdict_msg = classify_verdict(
            best_cell["median_autocorr_l2"],
            best_cell["median_kurtosis_ratio"],
            best_cell["median_total_distance"],
        )

    metadata = {
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "seeds": STAGE5_SEEDS,
        "cells": [c["id"] for c in STAGE5_CELLS],
        "stage": "Stage 5 (long-episode re-calibration)",
        "sim_window_minutes": 15,
        "stage4_404_ref": STAGE4_404_REF,
        "pilot_ref": PILOT_REF,
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

    json_path = out_dir / "calib_stage5_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_safe))
    print(f"\n💾 Wrote {json_path}")

    # Markdown report
    md = ["# Phase 5B+ Stage 5 — 15-min Episode Re-calibration\n"]
    md.append(f"**Generated**: {metadata['timestamp_et']} (ET)")
    md.append(f"**Seeds**: {STAGE5_SEEDS}")
    md.append(f"**Sim window**: 15 min (09:30:00 – 09:45:00)")
    md.append(f"\n## Reference values")
    md.append(f"- Stage 4 #404 (5-min × 5 seeds): autocorr_l2={STAGE4_404_REF['autocorr_l2']}, "
              f"kurt_ratio={STAGE4_404_REF['kurtosis_ratio']}, total_dist={STAGE4_404_REF['total_distance']}")
    md.append(f"- 5C Pilot (15-min × 1 seed, same #404 params): autocorr_l2={PILOT_REF['autocorr_l2']}, "
              f"kurt_ratio={PILOT_REF['kurtosis_ratio']}, total_dist={PILOT_REF['total_distance']}\n")

    md.append("## Stage 5 per-cell results\n")
    md.append("| Cell | Herder | max_size | pos_cap | n_succ | median dist | dist IQR | median ac_l2 | median kurt_ratio |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for c in results["cells"]:
        p = c["params"]
        if c["median_total_distance"] is None:
            md.append(f"| {p['id']} | {p['num_herder']} | {p.get('herder_max_size', '-')} | "
                      f"{p.get('herder_position_cap', '-')} | "
                      f"{c['n_success']}/{len(STAGE5_SEEDS)} | — | — | — | — |")
        else:
            md.append(
                f"| {p['id']} | {p['num_herder']} | {p.get('herder_max_size', '-')} | "
                f"{p.get('herder_position_cap', '-')} | "
                f"{c['n_success']}/{len(STAGE5_SEEDS)} | "
                f"**{c['median_total_distance']:.3f}** | "
                f"[{c['iqr_total_distance'][0]:.3f}, {c['iqr_total_distance'][1]:.3f}] | "
                f"**{c['median_autocorr_l2']:.3f}** | "
                f"**{c['median_kurtosis_ratio']:.2f}** |"
            )

    md.append(f"\n## Verdict\n\n**{verdict_code}**: {verdict_msg}\n")
    if best_cell:
        md.append(f"### Best cell: #{best_cell['cell_id']}")
        md.append(f"- Herder={best_cell['params']['num_herder']}, "
                  f"max_size={best_cell['params'].get('herder_max_size', '-')}, "
                  f"pos_cap={best_cell['params'].get('herder_position_cap', '-')}")
        md.append(f"- Median total_distance: **{best_cell['median_total_distance']:.3f}** "
                  f"(pilot baseline 3.077)")
        md.append(f"- Median autocorr_l2: **{best_cell['median_autocorr_l2']:.3f}**")
        md.append(f"- Median kurtosis_ratio: **{best_cell['median_kurtosis_ratio']:.2f}** "
                  f"(pilot baseline 7.71)")

    md_path = reports_dir / "phase5b_stage5.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Violin plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        cell_labels, ac_data, kurt_data = [], [], []
        for c in results["cells"]:
            if c["median_autocorr_l2"] is None:
                continue
            label = (f"#{c['cell_id']}\nH={c['params']['num_herder']}\n"
                     f"max={c['params'].get('herder_max_size', '-')}\n"
                     f"cap={c['params'].get('herder_position_cap', '-')}")
            cell_labels.append(label)
            ac_data.append([r["distances"]["autocorr_l2"] for r in c["all_seed_results"] if r["success"]])
            kurt_data.append([r["distances"]["kurtosis_ratio"] for r in c["all_seed_results"] if r["success"]])
        if ac_data:
            for ax, data, ylabel, refs in [
                (ax1, ac_data, "autocorr_l2",
                 [(STAGE4_404_REF["autocorr_l2"], 'Stage4 5min', 'green'),
                  (PILOT_REF["autocorr_l2"], 'Pilot 15min', 'red')]),
                (ax2, kurt_data, "kurtosis_ratio",
                 [(STAGE4_404_REF["kurtosis_ratio"], 'Stage4 5min ideal', 'green'),
                  (PILOT_REF["kurtosis_ratio"], 'Pilot 15min', 'red'),
                  (4.0, 'kurt threshold', 'orange')]),
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
            ax1.set_title("Stage 5 (15-min) — autocorr_l2")
            ax2.set_title("Stage 5 (15-min) — kurtosis_ratio")
            plt.tight_layout()
            fig_path = figures_dir / "phase5b_stage5_violin.png"
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
