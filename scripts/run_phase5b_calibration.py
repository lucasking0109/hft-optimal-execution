"""Phase 5B.4 — run 9-cell abides calibration vs AAPL.

Outputs:
  - data/synthetic/calib_results.json  (full payload with per-cell facts + distances)
  - reports/phase5_calibration.md      (narrative + comparison table)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.simulators.abides_calibration import (         # noqa: E402
    cells_to_dataframe,
    run_calibration,
    save_calibration_results,
)


def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"

    print("=" * 60)
    print("Phase 5B.4 — Calibration of abides to real AAPL")
    print("=" * 60)
    results = run_calibration(real_ticker="AAPL", real_date="20200113",
                              end_time="09:32:00")

    json_path = out_dir / "calib_results.json"
    save_calibration_results(results, json_path)
    print(f"\n💾 Wrote {json_path}")

    df = cells_to_dataframe(results)
    csv_path = out_dir / "calib_results.csv"
    df.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # ── Markdown report ──
    md = ["# Phase 5B Calibration Report\n"]
    md.append(f"**Real benchmark**: AAPL 2020-01-13 (full RTH)\n")
    real_summary = results["real_facts"].summary()
    md.append(f"- Real spread bps median: **{real_summary['spread_bps_median']:.2f}**")
    md.append(f"- Real spread bps P95: {real_summary['spread_bps_p95']:.2f}")
    md.append(f"- Real trade size median: {real_summary['trade_size_median']:.0f}")
    md.append(f"- Real excess kurtosis: {real_summary['excess_kurtosis']:.1f}")
    md.append(f"- Real Hill tail index: {real_summary['hill_tail_index']:.2f}")
    md.append(f"- Real vol_autocorr lag 1 / 50: {real_summary['vol_autocorr_lag_1']:.3f} / {real_summary['vol_autocorr_lag_50']:.3f}\n")

    md.append(f"## Calibration grid ({len(results['results'])} cells)\n")
    md.append("| Cell | Noise | Fund vol | MM | Success | Total dist | Kurtosis ratio | Hill ratio | Spread KS | Trade KS | Autocorr L2 | Runtime (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results["results"]:
        row = (
            f"| {r.cell_id} | {r.params['num_noise']} | {r.params['fund_vol']:.0e} "
            f"| {r.params['mm']} | {'✅' if r.success else '❌'} "
        )
        if r.success and r.distances:
            row += (
                f"| **{r.total_distance:.3f}** "
                f"| {r.distances['kurtosis_ratio']:.2f} "
                f"| {r.distances['hill_ratio']:.2f} "
                f"| {r.distances['spread_ks_dist']:.3f} "
                f"| {r.distances['trade_size_ks_dist']:.3f} "
                f"| {r.distances['autocorr_l2']:.3f} "
            )
        else:
            row += "| — | — | — | — | — | — "
        row += f"| {r.runtime_seconds:.1f} |"
        md.append(row)

    md.append("\n## Best cell\n")
    if results["best"] is not None:
        b = results["best"]
        md.append(f"- **Cell #{b.cell_id}**: noise={b.params['num_noise']}, "
                  f"fund_vol={b.params['fund_vol']:.0e}, mm={b.params['mm']}")
        md.append(f"- Total calibration distance: **{b.total_distance:.3f}**")
        md.append(f"- Distances: {b.distances}")
        if b.total_distance < 1.5:
            md.append("\n✅ **PASS** — total distance < 1.5 → 推進到 Phase 5C")
        elif b.total_distance < 2.5:
            md.append("\n🟡 **MARGINAL** — total distance 1.5-2.5 → 用但 Phase 6 報告需 sim-to-real gap caveat")
        else:
            md.append("\n🔴 **FAIL** — total distance > 2.5 → STOP，依 NO Silent Fallback 呈報三選項給用戶")
    else:
        md.append("⚠️ All cells failed. STOP per NO Silent Fallback principle.")

    md.append("\n## Notes\n")
    md.append("- 每個 cell 跑 2-min abides simulation（速度優先；Phase 5C 換成 30-min real episodes）")
    md.append("- 真實 AAPL 也是 2-min slice (09:30-09:32) for fair comparison")
    md.append("- 距離公式：sqrt of weighted squares of [|log kurt ratio|, |log hill ratio|, 2×spread KS, 2×trade KS, 0.5×autocorr L2]")

    md_path = reports_dir / "phase5_calibration.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")


if __name__ == "__main__":
    main()
