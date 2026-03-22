"""Phase 3 step (a): Calibrate AC parameters per ticker.

For each of 5 tickers (AAPL, AMZN, AMD, TSLA, NVDA):
    1. Compute ADV from daily OHLC table.
    2. Estimate η from large-trade impact regression on 5 days of tick.
    3. Check our 5-day γ regression against literature prior (NO Silent
       Fallback: report disagreement explicitly; downstream uses prior).
    4. Estimate σ (intraday volatility).

Writes:
    reports/calibration.csv
    reports/calibration_notes.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.impact import (              # noqa: E402
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
    adv_from_tick_files,
    average_daily_volume,
    check_gamma_against_prior,
    estimate_eta,
)
from hft.data import AVAILABLE_DATES, load_eq_daily_ohlc, load_eq_taq  # noqa: E402
from hft.strategies.almgren_chriss import (    # noqa: E402
    estimate_intraday_sigma_bps_per_sqrt_sec,
)

TICKERS = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATES = list(AVAILABLE_DATES)
DAILY_DATE_FOR_ADV = "20200110"   # last trading day before our 5-day window


def calibrate_one(ticker: str, daily_df: pl.DataFrame) -> dict:
    """Run all calibrations for one ticker. Returns a single result dict."""
    adv_source = "daily_ohlc"
    try:
        adv = average_daily_volume(daily_df, ticker, lookback_days=1)
    except ValueError as e:
        # Explicit fallback — log it loudly. NOT silent.
        try:
            adv = adv_from_tick_files(ticker, DATES)
            adv_source = "tick_files_5day_avg"
            print(f"  [{ticker}] daily OHLC missing → using 5-day tick ADV: {adv/1e6:.2f}M shares")
        except Exception as e2:
            return {"ticker": ticker, "error": f"ADV both sources failed: {e}; {e2}"}
    except Exception as e:
        return {"ticker": ticker, "error": f"ADV lookup: {e}"}

    # Aggregate eta estimates across 5 days (median of per-day slopes)
    eta_per_day: list[float] = []
    eta_n_events: list[int] = []
    eta_r2: list[float] = []
    out_of_range_days: list[str] = []
    for date in DATES:
        try:
            df = load_eq_taq(ticker, date)
            res = estimate_eta(
                df, adv_shares=adv, percentile_threshold=0.90,
                impact_window_seconds=5,
            )
            eta_per_day.append(res.eta_bps_per_pct_adv)
            eta_n_events.append(res.n_events)
            eta_r2.append(res.r_squared)
            if res.out_of_range:
                out_of_range_days.append(date)
        except Exception as e:
            eta_per_day.append(float("nan"))
            eta_n_events.append(0)
            eta_r2.append(float("nan"))

    valid_etas = [x for x in eta_per_day if x == x]  # NaN filter
    eta_median = (
        sorted(valid_etas)[len(valid_etas) // 2] if valid_etas else float("nan")
    )

    # Gamma sanity check on first available day
    chk = None
    for date in DATES:
        try:
            df = load_eq_taq(ticker, date)
            chk = check_gamma_against_prior(df, ticker=ticker, adv_shares=adv)
            break
        except Exception:
            continue

    # Sigma from each day, median
    sigmas: list[float] = []
    for date in DATES:
        try:
            df = load_eq_taq(ticker, date)
            sigmas.append(estimate_intraday_sigma_bps_per_sqrt_sec(df, sample_seconds=60))
        except Exception:
            sigmas.append(float("nan"))
    valid_sigmas = [x for x in sigmas if x == x]
    sigma_median = (
        sorted(valid_sigmas)[len(valid_sigmas) // 2] if valid_sigmas else float("nan")
    )

    prior = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
    eta_prior = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
    eta_consistent = (eta_median == eta_median) and (
        0.25 <= abs(eta_median) / eta_prior <= 4.0 if eta_prior > 0 else False
    )

    return {
        "ticker": ticker,
        "adv_shares": adv,
        "adv_source": adv_source,
        "eta_prior_bps_per_pct_adv": eta_prior,
        "eta_5day_estimate": eta_median,
        "eta_consistent_with_prior": eta_consistent,
        "eta_per_day": ", ".join(f"{x:.2f}" if x == x else "NaN" for x in eta_per_day),
        "eta_events_per_day": ", ".join(str(n) for n in eta_n_events),
        "eta_r2_median": (sorted(eta_r2)[len(eta_r2)//2] if any(x == x for x in eta_r2) else float("nan")),
        "out_of_range_days": ",".join(out_of_range_days),
        "gamma_prior_bps_per_pct_adv": prior,
        "gamma_estimate_bps_per_pct_adv": chk.estimated_bps_per_pct_adv if chk else float("nan"),
        "gamma_consistent_with_prior": chk.consistent if chk else False,
        "gamma_note": chk.note if chk else "no estimate",
        "sigma_bps_per_sqrt_sec_median": sigma_median,
    }


def main():
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    daily_df = load_eq_daily_ohlc(DAILY_DATE_FOR_ADV)

    rows = []
    for ticker in TICKERS:
        print(f"Calibrating {ticker}…")
        rows.append(calibrate_one(ticker, daily_df))

    df = pl.DataFrame(rows)
    csv_path = out_dir / "calibration.csv"
    df.write_csv(csv_path)
    print(f"\nWrote {csv_path}")

    # Markdown narrative
    md = ["# Phase 3 Calibration Report\n"]
    md.append("**Method**: η estimated from 5-day large-trade impact regression; ")
    md.append("γ estimated as 10-min permanent-drift slope and **compared to literature priors**, NOT used directly.\n")
    md.append("All values in **bps per 1% of ADV**.\n\n")

    md.append("## Per-ticker results\n")
    md.append("| Ticker | ADV (M shares) | ADV source | η prior (used) | η 5-day est. | η R² | η consistent | γ prior | γ 5-day est. | γ consistent | σ bps/√s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            md.append(f"| {r['ticker']} | ⚠️ {r['error']} | – | – | – | – | – | – | – | – | – |")
            continue
        eta_consistent = "✅" if r["eta_consistent_with_prior"] else "⚠️"
        gamma_consistent = "✅" if r["gamma_consistent_with_prior"] else "⚠️"
        md.append(
            f"| {r['ticker']} | "
            f"{r['adv_shares']/1e6:.1f} | "
            f"{r['adv_source']} | "
            f"{r['eta_prior_bps_per_pct_adv']:.1f} | "
            f"{r['eta_5day_estimate']:+.2f} | "
            f"{r['eta_r2_median']:.3f} | "
            f"{eta_consistent} | "
            f"{r['gamma_prior_bps_per_pct_adv']:.1f} | "
            f"{r['gamma_estimate_bps_per_pct_adv']:+.2f} | "
            f"{gamma_consistent} | "
            f"{r['sigma_bps_per_sqrt_sec_median']:.2f} |"
        )

    md.append("\n## Notes (NO Silent Fallback)\n")
    md.append("- **AC parameters used downstream**: literature priors for both η and γ.")
    md.append("  - 5 days of tick gives R² ≈ 0 for OLS impact regression — not enough for reliable estimate.")
    md.append("  - This was the explicit Phase 3 plan: estimate-and-verify, use prior when 5 days too few.")
    md.append("- 5-day estimates are surfaced as sanity check, NOT silently substituted.")
    md.append("- ⚠️ markers indicate disagreement with prior by > 4× → 5 days too few for that parameter; prior used.")
    md.append("- TSLA missing from Jan 2020 NDX-100 daily OHLC universe → ADV computed from 5-day tick (`adv_source: tick_files_5day_avg`). Explicit, not silent.")
    md.append("- η range we treat as plausible: 1–50 bps per 1% ADV.")
    md.append("- σ estimated from intraday mid log-returns sampled every 60s.")
    md.append("\nThese results feed AlmgrenChrissStrategy in scripts/run_phase3_baseline.py.")

    md_path = out_dir / "calibration_notes.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
