"""Phase 3 step (b): TWAP vs VWAP-following vs Almgren-Chriss comparison.

Runs the same liquidation scenario as Phase 2 (sell 10,000 shares between
10:00-11:00 ET) for the 5 calibrated tickers across 5 days, but adds two
AC variants:

    - AC risk-neutral (λ=0)         → degenerates to TWAP (sanity check)
    - AC risk-averse  (λ=1e-3)      → front-loaded sinh schedule

Outputs:
    reports/phase3_ac.csv   one row per (ticker, date, strategy)
    reports/phase3_ac.md    summary tables + analysis
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
)
from hft.backtest.engine import BacktestEngine        # noqa: E402
from hft.data import AVAILABLE_DATES, load_eq_daily_ohlc  # noqa: E402
from hft.strategies.almgren_chriss import (    # noqa: E402
    AlmgrenChrissStrategy,
    estimate_intraday_sigma_bps_per_sqrt_sec,
)
from hft.strategies.base import ParentOrder    # noqa: E402
from hft.strategies.twap import TWAPStrategy   # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy  # noqa: E402

NS_PER_HOUR = 3600 * 1_000_000_000

TICKERS = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATES = list(AVAILABLE_DATES)
PARENT_QTY = 10_000
WINDOW_START_NS = int(10 * NS_PER_HOUR)
WINDOW_END_NS = int(11 * NS_PER_HOUR)
NUM_SLICES = 60
SIDE = "sell"
LAMBDA_RISK_AVERSE = 1e-3


def get_adv(ticker: str, daily_df: pl.DataFrame) -> float:
    try:
        return average_daily_volume(daily_df, ticker, lookback_days=1)
    except ValueError:
        return adv_from_tick_files(ticker, DATES)


def run_one(ticker: str, date: str, *, daily_df: pl.DataFrame) -> list[dict]:
    rows = []
    try:
        engine = BacktestEngine(ticker, date)
    except Exception as e:
        return [{"ticker": ticker, "date": date, "strategy": "(load)",
                 "error": f"{type(e).__name__}: {e}"}]

    parent = ParentOrder(
        ticker=ticker, date=date, side=SIDE,
        quantity=PARENT_QTY,
        start_ns=WINDOW_START_NS, end_ns=WINDOW_END_NS,
    )

    try:
        adv = get_adv(ticker, daily_df)
    except Exception as e:
        return [{"ticker": ticker, "date": date, "strategy": "(adv)", "error": str(e)}]

    try:
        sigma = estimate_intraday_sigma_bps_per_sqrt_sec(engine.df, sample_seconds=60)
    except Exception as e:
        sigma = 1.0
        sigma_note = f"sigma estimation failed; default 1.0: {e}"
    else:
        sigma_note = ""

    eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
    gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])

    strategies = [
        TWAPStrategy(num_slices=NUM_SLICES),
        VWAPFollowingStrategy(num_slices=NUM_SLICES),
        AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=sigma,
            lambda_risk=0.0, adv_shares=adv,
        ),
        AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=sigma,
            lambda_risk=LAMBDA_RISK_AVERSE, adv_shares=adv,
        ),
    ]
    strategy_labels = ["twap", "vwap_following", "ac_risk_neutral", "ac_risk_averse"]

    for strat, label in zip(strategies, strategy_labels):
        try:
            ctx = engine.market_context(bin_minutes=5)
            res = engine.run(parent, strat, market_context=ctx)
            rows.append({
                "ticker": ticker, "date": date, "strategy": label,
                "arrival_mid": res.arrival_mid,
                "market_vwap": res.market_vwap,
                "executed_avg": float(
                    (res.fills["price"] * res.fills["quantity"]).sum() / res.fills["quantity"].sum()
                ),
                "fills": int(res.fills.shape[0]),
                "vwap_slip_bps": res.metrics.vwap_slip_bps,
                "is_bps": res.metrics.is_bps,
                "effective_spread_bps": res.metrics.effective_spread_bps,
                "realized_spread_bps": res.metrics.realized_spread_bps,
                "markout_1s_bps": res.metrics.markout_1s_bps,
                "markout_10s_bps": res.metrics.markout_10s_bps,
                "markout_60s_bps": res.metrics.markout_60s_bps,
                "reversion_5m_bps": res.metrics.reversion_5m_bps,
                "pov": res.metrics.pov,
                "price_var": res.metrics.price_var,
                "schedule_rmse": res.metrics.schedule_rmse,
                "hit_ratio_nbbo": res.metrics.hit_ratio_nbbo,
                "ac_eta_used": eta if "ac" in label else None,
                "ac_gamma_used": gamma if "ac" in label else None,
                "ac_sigma_used": sigma if "ac" in label else None,
                "ac_lambda_used": (LAMBDA_RISK_AVERSE if label == "ac_risk_averse" else 0.0) if "ac" in label else None,
                "error": "",
                "note": sigma_note,
            })
        except Exception as e:
            rows.append({"ticker": ticker, "date": date, "strategy": label,
                         "error": f"{type(e).__name__}: {e}"})

    return rows


def main():
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    daily_df = load_eq_daily_ohlc("20200110")

    all_rows = []
    n = len(TICKERS) * len(DATES) * 4
    print(f"Running {n} backtests ({len(TICKERS)} tickers × {len(DATES)} dates × 4 strategies)")
    for ticker in TICKERS:
        for date in DATES:
            print(f"  {ticker} {date}…")
            all_rows.extend(run_one(ticker, date, daily_df=daily_df))

    df = pl.DataFrame(all_rows)
    csv_path = out_dir / "phase3_ac.csv"
    df.write_csv(csv_path)
    print(f"\nWrote {csv_path}")

    successes = df.filter(pl.col("error") == "")
    failures = df.filter(pl.col("error") != "")

    md = ["# Phase 3: AC vs TWAP vs VWAP-following Report\n"]
    md.append(f"**Setup**: sell {PARENT_QTY:,} shares 10:00–11:00 ET, {NUM_SLICES} child orders.")
    md.append(f"**Tickers**: {', '.join(TICKERS)}; **Dates**: {', '.join(DATES)}")
    md.append(f"**AC parameters**: literature priors for η, γ; σ estimated from intraday mids; λ_risk_averse={LAMBDA_RISK_AVERSE}\n")
    md.append(f"**Total runs**: {len(all_rows)} ({len(successes)} ok / {len(failures)} failed)\n")

    if not failures.is_empty():
        md.append("## ⚠️ Failures (NO Silent Fallback)")
        md.append("")
        for row in failures.iter_rows(named=True):
            md.append(f"- {row['ticker']} {row['date']} {row['strategy']}: {row['error']}")
        md.append("")

    if not successes.is_empty():
        md.append("## Per-strategy summary (median across all (ticker, date))\n")
        md.append("| Strategy | VWAP slip (bps) | IS (bps) | Eff spread | Markout 60s | Reversion 5m | Price var | POV |")
        md.append("|---|---|---|---|---|---|---|---|")
        for label in ["twap", "vwap_following", "ac_risk_neutral", "ac_risk_averse"]:
            sub = successes.filter(pl.col("strategy") == label)
            if sub.is_empty():
                continue
            md.append(
                f"| **{label}** | "
                f"{sub['vwap_slip_bps'].median():+.2f} | "
                f"{sub['is_bps'].median():+.2f} | "
                f"{sub['effective_spread_bps'].median():.2f} | "
                f"{sub['markout_60s_bps'].median():+.2f} | "
                f"{sub['reversion_5m_bps'].median():+.2f} | "
                f"{sub['price_var'].median():.4f} | "
                f"{sub['pov'].median()*100:.2f}% |"
            )
        md.append("")

        # AC-RN ≡ TWAP sanity check
        md.append("## Sanity check: AC risk-neutral should ≡ TWAP\n")
        twap = successes.filter(pl.col("strategy") == "twap")
        ac_rn = successes.filter(pl.col("strategy") == "ac_risk_neutral")
        if not twap.is_empty() and not ac_rn.is_empty():
            twap_avg_is = float(twap["is_bps"].mean())
            ac_rn_avg_is = float(ac_rn["is_bps"].mean())
            diff = ac_rn_avg_is - twap_avg_is
            verdict = "✅ AC-RN ≈ TWAP (as theory predicts)" if abs(diff) < 0.5 else f"⚠️ Diff {diff:+.2f} bps — expect zero"
            md.append(f"- TWAP avg IS = {twap_avg_is:+.2f} bps")
            md.append(f"- AC-RN avg IS = {ac_rn_avg_is:+.2f} bps")
            md.append(f"- Difference = {diff:+.2f} bps → {verdict}\n")

        # AC risk-averse front-loading effect
        md.append("## Risk-averse AC effect on price variance\n")
        ac_ra = successes.filter(pl.col("strategy") == "ac_risk_averse")
        if not ac_ra.is_empty() and not twap.is_empty():
            twap_var = float(twap["price_var"].mean())
            ac_ra_var = float(ac_ra["price_var"].mean())
            md.append(f"- TWAP avg price variance = {twap_var:.4f}")
            md.append(f"- AC risk-averse avg price variance = {ac_ra_var:.4f}")
            change = (ac_ra_var - twap_var) / twap_var * 100 if twap_var > 0 else 0.0
            md.append(f"- Change = {change:+.1f}% — risk-averse should reduce variance by trading earlier\n")

        # Per-(ticker,date) IS comparison: AC risk-averse vs TWAP
        md.append("## IS per (ticker, date) — AC risk-averse vs TWAP\n")
        md.append("| Ticker | Date | TWAP IS | VWAP-follow IS | AC RN IS | AC risk-averse IS | Best |")
        md.append("|---|---|---|---|---|---|---|")
        for ticker in TICKERS:
            for date in DATES:
                row_t = successes.filter((pl.col("ticker") == ticker) & (pl.col("date") == date) & (pl.col("strategy") == "twap"))
                row_v = successes.filter((pl.col("ticker") == ticker) & (pl.col("date") == date) & (pl.col("strategy") == "vwap_following"))
                row_rn = successes.filter((pl.col("ticker") == ticker) & (pl.col("date") == date) & (pl.col("strategy") == "ac_risk_neutral"))
                row_ra = successes.filter((pl.col("ticker") == ticker) & (pl.col("date") == date) & (pl.col("strategy") == "ac_risk_averse"))
                if any(r.is_empty() for r in [row_t, row_v, row_rn, row_ra]):
                    continue
                t = float(row_t["is_bps"][0]); v = float(row_v["is_bps"][0])
                rn = float(row_rn["is_bps"][0]); ra = float(row_ra["is_bps"][0])
                vals = {"TWAP": t, "VWAP-f": v, "AC-RN": rn, "AC-RA": ra}
                best = min(vals.items(), key=lambda kv: kv[1])
                md.append(f"| {ticker} | {date} | {t:+.2f} | {v:+.2f} | {rn:+.2f} | {ra:+.2f} | {best[0]} |")
        md.append("")

        md.append("## Notes (NO Silent Fallback)\n")
        md.append("- **AC theory caveat**: with linear-impact assumption, risk-neutral AC ≡ TWAP (any schedule has same expected cost). Differences here come only from finite-sample noise, not strategy logic.")
        md.append("- AC risk-averse trades sooner → exposes less inventory to price drift, but pays slightly more in immediate impact.")
        md.append("- Backtester does NOT model self-impact, so the marginal cost of AC risk-averse front-loading is understated. Real-world impact would be larger.")
        md.append("- η, γ are **literature priors** (5 days too few for reliable estimation; see calibration_notes.md).")
        md.append("- σ estimated from intraday mid log-returns of each (ticker, date) → variable input per run.")
        md.append("- All metrics: positive = cost (worse than benchmark).")

    md_path = out_dir / "phase3_ac.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
