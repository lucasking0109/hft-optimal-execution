"""Phase 5B+ Stage 1 — momentum sensitivity sanity check (3 cells).

Anchors at the Phase 5B initial-best 3D corner (#4: 500 noise, fund_vol=1e-3,
aggressive MM, total_dist=2.144) and varies only `num_momentum` ∈ {25, 100, 400}.

Decision rule (NO Silent Fallback):
  - If S3 (num_momentum=400) drops autocorr_l2 below 1.0 (>50% improvement
    over baseline) → green-light Stage 2 (12-cell 4D run).
  - Otherwise → STOP and report. Do NOT silently pad the result; the user
    decides whether to fall back to option A (accept marginal cell #4) or
    pivot to a different agent type.

Outputs:
  - data/synthetic/calib_stage1_results.json
  - data/synthetic/calib_stage1_results.csv
  - reports/phase5b_stage1.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.simulators.abides_calibration import (         # noqa: E402
    STAGE1_CELLS,
    cells_to_dataframe,
    run_calibration,
    save_calibration_results,
)


# Decision threshold — Stage 1 is a green-light test, not the final answer.
# Real AAPL autocorr_l2 ≈ 0; previous 9 cells all ≥ 1.6.
# Target: at least 50% improvement over baseline cell (101).
STAGE2_GREENLIGHT_AC_L2 = 1.0


def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"

    print("=" * 70)
    print("Phase 5B+ Stage 1 — Momentum sensitivity (3 cells)")
    print(f"Cells: {[c['id'] for c in STAGE1_CELLS]}")
    print(f"Varying num_momentum ∈ {[c['num_momentum'] for c in STAGE1_CELLS]}")
    print("=" * 70)

    # 5-min sim window (was 2-min): MomentumAgent needs 50-tick warmup
    # at 1s wake → ~50s warmup, leaves ~4min trading window.
    results = run_calibration(
        real_ticker="AAPL",
        real_date="20200113",
        end_time="09:35:00",
        cells=STAGE1_CELLS,
    )

    json_path = out_dir / "calib_stage1_results.json"
    save_calibration_results(results, json_path)
    print(f"\n💾 Wrote {json_path}")

    df = cells_to_dataframe(results)
    csv_path = out_dir / "calib_stage1_results.csv"
    df.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # ── Decision logic ──
    successful = [r for r in results["results"] if r.success]
    if len(successful) < 3:
        verdict = "🔴 STAGE-1-FAIL — fewer than 3 cells succeeded"
        proceed = False
    else:
        baseline = next(r for r in successful if r.params["num_momentum"] == 25)
        high_mom = next(r for r in successful if r.params["num_momentum"] == 400)
        baseline_ac = baseline.distances["autocorr_l2"]
        high_ac = high_mom.distances["autocorr_l2"]
        improvement_pct = (baseline_ac - high_ac) / baseline_ac * 100

        if high_ac < STAGE2_GREENLIGHT_AC_L2:
            verdict = (f"✅ GREEN-LIGHT — high-momentum cell autocorr_l2 = {high_ac:.2f} "
                       f"< target {STAGE2_GREENLIGHT_AC_L2}. Improvement {improvement_pct:.0f}%.")
            proceed = True
        else:
            verdict = (f"🔴 STAGE-1-INSUFFICIENT — high-momentum autocorr_l2 = {high_ac:.2f} "
                       f"(target < {STAGE2_GREENLIGHT_AC_L2}). Improvement only {improvement_pct:.0f}%.")
            proceed = False

    # ── Markdown report ──
    md = ["# Phase 5B+ Stage 1 — Momentum Sensitivity\n"]
    md.append(f"**Goal**: test whether MomentumAgent count meaningfully reduces vol_autocorr_l2 distance.\n")
    md.append(f"**Anchor cell**: prior best (500 noise, fund_vol=1e-3, aggressive MM)")
    md.append(f"**Varying**: num_momentum ∈ {{25, 100, 400}}")
    md.append(f"**Green-light threshold**: high-momentum cell autocorr_l2 < {STAGE2_GREENLIGHT_AC_L2}\n")

    md.append("## Per-cell results\n")
    md.append("| Cell | num_momentum | success | total_dist | autocorr_l2 | kurtosis | hill | spread KS | trade KS | runtime |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results["results"]:
        if r.success:
            md.append(
                f"| {r.cell_id} | {r.params['num_momentum']} | ✅ | "
                f"**{r.total_distance:.3f}** | "
                f"**{r.distances['autocorr_l2']:.3f}** | "
                f"{r.distances['kurtosis_ratio']:.2f} | "
                f"{r.distances['hill_ratio']:.2f} | "
                f"{r.distances['spread_ks_dist']:.3f} | "
                f"{r.distances['trade_size_ks_dist']:.3f} | "
                f"{r.runtime_seconds:.1f}s |"
            )
        else:
            md.append(
                f"| {r.cell_id} | {r.params['num_momentum']} | ❌ | — | — | — | — | — | — | "
                f"{r.runtime_seconds:.1f}s |"
            )

    md.append(f"\n## Verdict\n\n{verdict}\n")
    if proceed:
        md.append("→ Proceeding to Stage 2 (12-cell 4D Latin hypercube).\n")
    else:
        md.append("→ Per NO Silent Fallback: STOP and present Lucas the choice between:")
        md.append("  - **(A)** Accept Phase 5B initial best cell #4 (total_dist=2.144) and add Phase 6 caveat.")
        md.append("  - **(B-alt)** Try a different agent type (e.g. add OrderBookImbalanceAgent or HBL).")
        md.append("  - **(C)** Switch evaluation metric (drop autocorr from total_distance with explicit caveat).\n")

    md_path = reports_dir / "phase5b_stage1.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    print("\n" + "=" * 70)
    print(verdict)
    print("=" * 70)

    # Exit code signals proceed/stop to any wrapping shell logic.
    sys.exit(0 if proceed else 2)


if __name__ == "__main__":
    main()
