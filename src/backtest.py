#!/usr/bin/env python3
"""
NHL Oracle -- Walk-Forward Backtest
Uses cached game data in data/cache/ and the same feature engineering as
the live predictor. Evaluates the trained model season by season with
no lookahead bias.

Usage:
  python src/backtest.py           # uses cached seasons in data/cache/
  python src/backtest.py --seasons 2022 2023 2024  # specific seasons only

Requirements:
  pip install scikit-learn joblib numpy pandas
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# Add src/ to path so we can import the existing modules
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
MODELS_DIR = ROOT_DIR / "models"

# Import the project's own feature engineering and Elo modules
try:
    import features as feat_eng
    import elo_system
    MODULES_OK = True
except ImportError as e:
    print(f"[WARN] Could not import project modules: {e}")
    print("  Make sure to run from the repo root: python src/backtest.py")
    MODULES_OK = False


# -- Elo constants (must match elo_system.py) ---------------------------------

K_FACTOR   = 6.0
OT_K_SCALE = 0.75
INITIAL_ELO = 1500.0


def _mov_mult(margin, went_ot):
    mult = math.log(1 + min(margin, 5))
    if went_ot:
        mult *= OT_K_SCALE
    return mult


def update_elo_local(ratings, home, away, home_won, margin, went_ot):
    """Local Elo update matching elo_system.py logic."""
    r_h = ratings.get(home, INITIAL_ELO)
    r_a = ratings.get(away, INITIAL_ELO)
    e_h = 1.0 / (1.0 + 10 ** ((r_a - r_h) / 400.0))
    s_h = 1.0 if home_won else 0.0
    delta = K_FACTOR * _mov_mult(margin, went_ot) * (s_h - e_h)
    ratings[home] = r_h + delta
    ratings[away] = r_a - delta
    return ratings


def offseason_regression(ratings):
    return {team: 0.30 * INITIAL_ELO + 0.70 * r for team, r in ratings.items()}


# -- Load cached games --------------------------------------------------------

def load_cached_games(seasons):
    all_games = []
    for season in seasons:
        cache_path = CACHE_DIR / f"games_{season}.json"
        if not cache_path.exists():
            print(f"  [WARN] Cache not found: {cache_path} -- skipping season {season}")
            continue
        with open(cache_path) as f:
            games = json.load(f)
        for g in games:
            g["season"] = season
        all_games.extend(games)
        print(f"  Loaded {len(games)} games for season {season}")
    return sorted(all_games, key=lambda g: g["date"])


# -- Build feature vectors game by game --------------------------------------

def build_features_chronological(all_games):
    """
    Recomputes feature vectors game by game with no lookahead.
    Returns list of (features, label, date, season) tuples.
    """
    if not MODULES_OK:
        print("[ERROR] Project modules not available — cannot build features.")
        return []

    records = []
    elo_ratings = {}
    last_game_date = {}
    season_standings = {}  # season -> {team -> cumulative stats}

    def _init_stats():
        return {
            "gp": 0, "pts": 0, "gf": 0, "ga": 0, "rw": 0,
            "l10": [],
            "home_gp": 0, "home_w": 0, "home_otl": 0,
            "away_gp": 0, "away_w": 0, "away_otl": 0,
            "streak_code": "", "streak_count": 0,
            "pp_pct": 0.18, "pk_pct": 0.80,
        }

    def make_standing(stats, team):
        gp = stats["gp"] or 1
        l10 = stats["l10"][-10:]
        return {
            "teamAbbrev": {"default": team},
            "gamesPlayed": gp,
            "points": stats["pts"],
            "pointPctg": stats["pts"] / (2 * gp),
            "goalFor": stats["gf"],
            "goalAgainst": stats["ga"],
            "regulationWins": stats["rw"],
            "powerPlayPctg": stats["pp_pct"],
            "penaltyKillPctg": stats["pk_pct"],
            "l10Wins":     sum(1 for r in l10 if r == "W"),
            "l10Losses":   sum(1 for r in l10 if r == "L"),
            "l10OtLosses": sum(1 for r in l10 if r == "OTL"),
            "homeWins":    stats["home_w"],
            "homeLosses":  max(0, stats["home_gp"] - stats["home_w"] - stats["home_otl"]),
            "homeOtLosses": stats["home_otl"],
            "roadWins":    stats["away_w"],
            "roadLosses":  max(0, stats["away_gp"] - stats["away_w"] - stats["away_otl"]),
            "roadOtLosses": stats["away_otl"],
            "streakCode":  stats["streak_code"],
            "streakCount": stats["streak_count"],
        }

    prev_season = None
    for g in all_games:
        season = g["season"]
        home   = g["home_team"]
        away   = g["away_team"]
        home_won = g["home_won"]
        went_ot  = g.get("went_ot", False)
        game_date = g["date"]
        margin = abs(g["home_score"] - g["away_score"])

        # Offseason regression between seasons
        if prev_season is not None and season != prev_season:
            elo_ratings = offseason_regression(elo_ratings)
        prev_season = season

        if season not in season_standings:
            season_standings[season] = {}
        ss = season_standings[season]
        for team in [home, away]:
            if team not in ss:
                ss[team] = _init_stats()

        standings_list = [make_standing(ss[home], home), make_standing(ss[away], away)]
        h_last = last_game_date.get(home)
        a_last = last_game_date.get(away)

        try:
            fv = feat_eng.compute_features(
                home, away, game_date,
                standings_list, elo_ratings,
                h_last, a_last,
                {}, {},   # no MoneyPuck xG in backtest
                0.0, 0.0, # no goalie GSAx
            )
            records.append({
                "features": fv,
                "label": 1 if home_won else 0,
                "date": game_date,
                "season": season,
            })
        except Exception as e:
            pass  # skip malformed games

        # Update Elo and standings AFTER feature computation (no lookahead)
        update_elo_local(elo_ratings, home, away, home_won, margin, went_ot)
        last_game_date[home] = game_date
        last_game_date[away] = game_date

        for team, won, scored, allowed, is_home in [
            (home, home_won,      g["home_score"], g["away_score"], True),
            (away, not home_won,  g["away_score"], g["home_score"], False),
        ]:
            s = ss[team]
            s["gp"] += 1
            s["gf"] += scored
            s["ga"] += allowed
            if is_home:
                s["home_gp"] += 1
            else:
                s["away_gp"] += 1
            if won:
                s["pts"] += 2
                if not went_ot:
                    s["rw"] += 1
                s["l10"].append("W")
                if is_home: s["home_w"] += 1
                else:       s["away_w"] += 1
                if s["streak_code"] == "W":
                    s["streak_count"] += 1
                else:
                    s["streak_code"] = "W"
                    s["streak_count"] = 1
            elif went_ot:
                s["pts"] += 1
                s["l10"].append("OTL")
                if is_home: s["home_otl"] += 1
                else:       s["away_otl"] += 1
                if s["streak_code"] == "OT":
                    s["streak_count"] += 1
                else:
                    s["streak_code"] = "OT"
                    s["streak_count"] = 1
            else:
                s["l10"].append("L")
                if s["streak_code"] == "L":
                    s["streak_count"] += 1
                else:
                    s["streak_code"] = "L"
                    s["streak_count"] = 1

    return records


# -- Simulated ROI ------------------------------------------------------------

def simulate_roi(preds, labels, min_conf=0.58, bet=10.0, bankroll=1000.0):
    w = l = 0
    bank = bankroll
    for p, y in zip(preds, labels):
        if min_conf <= p <= (1 - min_conf):
            continue
        pick = 1 if p >= 0.5 else 0
        edge_p = p if pick == 1 else (1 - p)
        fair_odds = 1.0 / max(edge_p, 0.01)
        offered = fair_odds * 0.95
        if pick == int(y):
            bank += bet * (offered - 1)
            w += 1
        else:
            bank -= bet
            l += 1
    roi = (bank - bankroll) / ((w + l) * bet) * 100 if (w + l) > 0 else 0.0
    return roi, w, l


# -- Walk-forward evaluation --------------------------------------------------

def walk_forward(records):
    all_seasons = sorted(set(r["season"] for r in records))
    if len(all_seasons) < 2:
        print("Need at least 2 seasons for walk-forward CV.")
        return {}, np.array([]), np.array([])

    all_preds, all_labels = [], []
    season_results = {}

    print("Walk-forward results:")
    print(f"  {'Season':>8}  {'N':>5}  {'Acc':>6}  {'Brier':>7}  {'HC%':>7}  {'HC N':>6}  {'ROI%':>7}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}")

    for i, test_season in enumerate(all_seasons[1:], 1):
        train_recs = [r for r in records if r["season"] in all_seasons[:i]]
        test_recs  = [r for r in records if r["season"] == test_season]
        if len(train_recs) < 50 or len(test_recs) < 20:
            continue

        X_tr = np.array([r["features"] for r in train_recs])
        y_tr = np.array([r["label"] for r in train_recs])
        X_te = np.array([r["features"] for r in test_recs])
        y_te = np.array([r["label"] for r in test_recs])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        model.fit(X_tr_s, y_tr)

        preds = np.clip(model.predict_proba(X_te_s)[:, 1], 0.01, 0.99)

        acc   = accuracy_score(y_te, preds >= 0.5)
        brier = brier_score_loss(y_te, preds)
        hc    = (preds >= 0.63) | (preds <= 0.37)
        hc_acc = accuracy_score(y_te[hc], preds[hc] >= 0.5) if hc.sum() > 0 else None
        roi, wins, losses = simulate_roi(preds, y_te)

        season_results[test_season] = {
            "n": len(test_recs),
            "accuracy": round(float(acc), 4),
            "brier": round(float(brier), 4),
            "hc_accuracy": round(float(hc_acc), 4) if hc_acc is not None else None,
            "hc_n": int(hc.sum()),
            "roi_pct": round(roi, 2),
        }

        hc_str = f"{hc_acc:.3f}" if hc_acc is not None else "   N/A"
        print(f"  {test_season:>8}  {len(test_recs):>5}  {acc:>6.3f}  {brier:>7.4f}  "
              f"{hc_str:>7}  {hc.sum():>6}  {roi:>+7.1f}%")

        all_preds.extend(preds.tolist())
        all_labels.extend(y_te.tolist())

    return season_results, np.array(all_preds), np.array(all_labels)


# -- Feature importance -------------------------------------------------------

def print_feature_importance(records):
    if not MODULES_OK or not records:
        return
    X = np.array([r["features"] for r in records])
    y = np.array([r["label"] for r in records])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_s, y)
    ranked = sorted(zip(feat_eng.FEATURE_NAMES, model.coef_[0]),
                    key=lambda x: abs(x[1]), reverse=True)
    print("\nFeature Importance (LR coefficient magnitude, full dataset):")
    for name, val in ranked[:15]:
        bar = "#" * int(abs(val) * 30)
        sign = "+" if val > 0 else "-"
        print(f"  {name:<25} {sign}{abs(val):>5.3f}  {bar}")


# -- Compare against trained model --------------------------------------------

def compare_trained_model(records):
    lr_path  = MODELS_DIR / "model.pkl"
    xgb_path = MODELS_DIR / "model_xgb.pkl"
    sc_path  = MODELS_DIR / "scaler.pkl"
    if not sc_path.exists() or (not lr_path.exists() and not xgb_path.exists()):
        return
    try:
        scaler = joblib.load(sc_path)
        model  = joblib.load(lr_path if lr_path.exists() else xgb_path)
        X = np.array([r["features"] for r in records])
        y = np.array([r["label"] for r in records])
        X_s = scaler.transform(X)
        preds = np.clip(model.predict_proba(X_s)[:, 1], 0.01, 0.99)
        print(f"\nTrained model (in-sample check):")
        print(f"  Accuracy:  {accuracy_score(y, preds >= 0.5):.4f}")
        print(f"  Brier:     {brier_score_loss(y, preds):.4f}")
    except Exception as e:
        print(f"[WARN] Could not load trained model: {e}")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NHL Oracle Walk-Forward Backtest")
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2020, 2021, 2022, 2023, 2024],
                        help="NHL season start years to include (default: 2020-2024)")
    args = parser.parse_args()

    print("NHL Oracle -- Walk-Forward Backtest")
    print("=" * 50)

    print(f"\nLoading cached games for seasons: {args.seasons}")
    all_games = load_cached_games(args.seasons)
    if not all_games:
        print("No cached game data found.")
        print("  Run the training workflow to populate data/cache/")
        sys.exit(1)

    print(f"\nTotal games: {len(all_games)}")
    print("Building feature vectors (chronological, no lookahead)...")
    records = build_features_chronological(all_games)
    print(f"Built {len(records)} feature records\n")

    if not records:
        print("No feature records generated.")
        sys.exit(1)

    season_results, all_preds, all_labels = walk_forward(records)

    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    print()
    print("Aggregate (all out-of-sample seasons):")
    print(f"  Total games:         {len(all_preds)}")
    print(f"  Accuracy:            {accuracy_score(all_labels, all_preds >= 0.5):.4f}")
    print(f"  Brier score:         {brier_score_loss(all_labels, all_preds):.4f}  (naive baseline ~0.247)")
    hc = (all_preds >= 0.63) | (all_preds <= 0.37)
    if hc.sum() > 0:
        print(f"  HC accuracy (>=63%): {accuracy_score(all_labels[hc], all_preds[hc] >= 0.5):.4f}  n={hc.sum()}")
    roi, w, l = simulate_roi(all_preds, all_labels)
    print(f"  Simulated ROI:       {roi:+.2f}%  (W={w}, L={l})")

    compare_trained_model(records)
    print_feature_importance(records)
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
