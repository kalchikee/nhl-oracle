#!/usr/bin/env python3
"""
NHL Playoff Model Trainer — trains on playoff_data.csv (no regular season CSV exists separately).
Output: models/playoff_coefficients.json

Usage: python train_playoff_model.py
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

PLAYOFF_CSV = DATA_DIR / "playoff_data.csv"

FEATURE_NAMES = [
    "elo_diff", "win_pct_diff", "gpg_diff", "gapg_diff", "pt_pct_diff",
    "series_game_num", "series_deficit", "is_elimination_game",
]


def main():
    print("NHL Playoff Model Trainer")
    print("=" * 40)

    if not PLAYOFF_CSV.exists():
        print("No playoff data — run fetch_playoff_data.py first.")
        sys.exit(1)

    df = pd.read_csv(PLAYOFF_CSV)
    print(f"Loaded {len(df)} playoff games")
    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
    seasons = sorted(df["season"].unique())
    print(f"Seasons: {seasons}\n")

    print("Leave-one-season-out CV:")
    print(f"  {'Season':>6}  {'N':>4}  {'LR':>6}  {'XGB':>6}  {'Ens':>6}")
    lr_accs, xgb_accs, ens_accs = [], [], []

    for i, ts in enumerate(seasons[1:], 1):
        train = df[df["season"].isin(seasons[:i])]
        test  = df[df["season"] == ts]
        if len(train) < 10 or len(test) < 4: continue

        X_tr = train[feat_cols].fillna(0).values; y_tr = train["label"].values
        X_te = test[feat_cols].fillna(0).values;  y_te = test["label"].values
        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr_s, y_tr)
        lr_p = np.clip(lr.predict_proba(X_te_s)[:, 1], 0.01, 0.99)
        lr_acc = accuracy_score(y_te, lr_p >= 0.5); lr_accs.append(lr_acc)

        xgb_str = ens_str = "  N/A"
        if HAS_XGB and len(train) >= 20:
            xgb = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                 min_child_weight=3, verbosity=0, eval_metric='logloss')
            xgb.fit(X_tr, y_tr)
            xp = np.clip(xgb.predict_proba(X_te)[:, 1], 0.01, 0.99)
            ep = (lr_p + xp) / 2
            xa = accuracy_score(y_te, xp >= 0.5); ea = accuracy_score(y_te, ep >= 0.5)
            xgb_accs.append(xa); ens_accs.append(ea)
            xgb_str = f"{xa:.3f}"; ens_str = f"{ea:.3f}"

        print(f"  {ts:>6}  {len(test):>4}  {lr_acc:.3f}  {xgb_str}  {ens_str}")

    print(f"\nSummary:")
    if lr_accs:  print(f"  LR:       {np.mean(lr_accs):.4f}")
    if xgb_accs: print(f"  XGBoost:  {np.mean(xgb_accs):.4f}")
    if ens_accs: print(f"  Ensemble: {np.mean(ens_accs):.4f}")

    X_all = df[feat_cols].fillna(0).values; y_all = df["label"].values
    sc_f = StandardScaler(); X_all_s = sc_f.fit_transform(X_all)
    lr_f = LogisticRegression(C=1.0, max_iter=1000); lr_f.fit(X_all_s, y_all)

    (MODEL_DIR / "playoff_coefficients.json").write_text(json.dumps({
        "intercept": float(lr_f.intercept_[0]), "coefficients": lr_f.coef_[0].tolist(), "feature_names": feat_cols,
    }, indent=2))
    (MODEL_DIR / "playoff_scaler.json").write_text(json.dumps({
        "mean": sc_f.mean_.tolist(), "scale": sc_f.scale_.tolist(), "feature_names": feat_cols,
    }, indent=2))
    (MODEL_DIR / "playoff_metadata.json").write_text(json.dumps({
        "sport": "NHL", "cv_accuracy_lr": float(np.mean(lr_accs)) if lr_accs else None,
        "feature_names": feat_cols, "playoff_seasons": [int(s) for s in seasons],
    }, indent=2))
    print(f"\nSaved to {MODEL_DIR}/playoff_*.json")


if __name__ == "__main__":
    main()
