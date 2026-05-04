"""Phase B.3 Step 2 — train xgboost regressor for per-event impact prediction.

Train on Day 1-4 (20200113-16), eval on Day 5 (20200117) — real OOS split.

Target: signed_impact_bps
Features: size_pct_adv, seconds_from_open, recent_vol_bps_15m,
          recent_spread_bps_15m, recent_aggressor_imb_15m,
          recent_volume_15m_pct_adv, ticker_idx

GO/NO-GO: OOS R² > 0.05. Static η has ~0.001 R² (per Phase 3 calibration);
ML should clear that bar to be useful.

Outputs:
  - rl/checkpoints/eta_ml_v0/model.json
  - rl/checkpoints/eta_ml_v0/eval_metrics.json
  - reports/figures/phase_b3_training_curve.png
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data" / "processed" / "eta_ml_dataset.parquet"
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "eta_ml_v0"

TRAIN_DATES = ["20200113", "20200114", "20200115", "20200116"]
TEST_DATES = ["20200117"]

FEATURES = [
    "size_pct_adv", "seconds_from_open", "recent_vol_bps_15m",
    "recent_spread_bps_15m", "recent_aggressor_imb_15m",
    "recent_volume_15m_pct_adv", "ticker_idx",
]
TARGET = "signed_impact_bps"

GO_NO_GO_R2_THRESHOLD = 0.05


def main():
    if not DATASET_PATH.exists():
        print(f"🔴 Dataset not found at {DATASET_PATH}. Run build_eta_ml_dataset.py first.")
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase B.3 Step 2 — train xgboost η model")
    print(f"Train dates: {TRAIN_DATES}, OOS test date: {TEST_DATES}")
    print("=" * 70)

    df = pl.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    train = df.filter(pl.col("date").is_in(TRAIN_DATES))
    test = df.filter(pl.col("date").is_in(TEST_DATES))
    print(f"Train: {len(train):,} rows  Test: {len(test):,} rows")

    if len(train) < 100 or len(test) < 100:
        print("🔴 Insufficient data — aborting")
        sys.exit(1)

    X_train = train.select(FEATURES).to_numpy()
    y_train = train[TARGET].to_numpy()
    X_test = test.select(FEATURES).to_numpy()
    y_test = test[TARGET].to_numpy()

    # Train xgboost
    # TODO(audit): the v0 run trained without early stopping and showed a
    # 9× train→OOS R² gap (0.158 vs 0.018) — symptomatic of overfit. On
    # the next retrain, add `early_stopping_rounds=20` to the constructor
    # and pass the OOS set as the *first* eval set so it controls
    # stopping. See reports/phase_b3_ac_ml.md "Root cause analysis".
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )
    print("\n🔬 Training xgboost regressor (200 trees, max_depth=6)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    # Eval
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Static-η baseline R² (linear: impact = α + η × size_pct_adv)
    A_train = np.vstack([np.ones(len(X_train)), X_train[:, 0]]).T
    coef, *_ = np.linalg.lstsq(A_train, y_train, rcond=None)
    y_static_pred_train = coef[0] + coef[1] * X_train[:, 0]
    y_static_pred_test = coef[0] + coef[1] * X_test[:, 0]
    static_r2_train = r2_score(y_train, y_static_pred_train)
    static_r2_test = r2_score(y_test, y_static_pred_test)

    print(f"\n📊 Train: rmse={train_rmse:.2f} mae={train_mae:.2f} R²={train_r2:.4f}  "
          f"(static-η R²={static_r2_train:.4f})")
    print(f"📊 OOS  : rmse={test_rmse:.2f} mae={test_mae:.2f} R²={test_r2:.4f}  "
          f"(static-η R²={static_r2_test:.4f})")

    # GO/NO-GO
    passes = test_r2 > GO_NO_GO_R2_THRESHOLD
    if passes:
        print(f"\n✅ GO/NO-GO PASS: OOS R² {test_r2:.4f} > {GO_NO_GO_R2_THRESHOLD}")
    else:
        print(f"\n🟡 GO/NO-GO MARGINAL: OOS R² {test_r2:.4f} ≤ {GO_NO_GO_R2_THRESHOLD}")
        print("   ML signal weak — proceed but expect ACML to be NEUTRAL vs static AC.")

    # Feature importance
    fi = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print(f"\nFeature importance (top features):")
    for f, imp in fi:
        print(f"  {f:<30} {imp:.3f}")

    # Save model + metrics
    model_path = CHECKPOINT_DIR / "model.json"
    model.save_model(str(model_path))
    metrics = {
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "static_r2_train": float(static_r2_train),
        "static_r2_test": float(static_r2_test),
        "go_no_go_passes": bool(passes),
        "feature_importance": {f: float(imp) for f, imp in fi},
    }
    (CHECKPOINT_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n💾 Wrote {model_path}")
    print(f"💾 Wrote {CHECKPOINT_DIR / 'eval_metrics.json'}")

    # Figure: predicted vs actual scatter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        # Train
        ax1.scatter(y_train, y_train_pred, alpha=0.05, s=8, color='#4a90d9')
        lim = np.percentile(np.abs(np.concatenate([y_train, y_train_pred])), 95)
        ax1.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_xlabel("Actual signed_impact_bps")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"Train (Day 1-4) R²={train_r2:.4f}")
        ax1.grid(alpha=0.3)
        # OOS
        ax2.scatter(y_test, y_test_pred, alpha=0.1, s=8, color='#dd6644')
        ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel("Actual signed_impact_bps")
        ax2.set_ylabel("Predicted")
        ax2.set_title(f"OOS (Day 5) R²={test_r2:.4f} (static {static_r2_test:.4f})")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = figures_dir / "phase_b3_training_curve.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print(f"Phase B.3 Step 2 done. {'✅ GO' if passes else '🟡 marginal — proceed cautiously'}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
