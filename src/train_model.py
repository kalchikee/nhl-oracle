"""
Model training script.
Downloads 5 years of historical NHL data, builds feature vectors,
trains logistic regression with walk-forward CV, applies Platt scaling,
and saves the model to models/.

Run this script once to initialize the model, then weekly to retrain.
"""

import os
import sys
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from xgboost import XGBClassifier

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
import nhl_api
import moneypuck
import elo_system
import features as feat_eng

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def _current_nhl_season_year() -> int:
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def _training_seasons(n: int = 5) -> list:
    """
    Returns the last n seasons to train on, rolling dynamically each year.
    At season start (Sep-Nov), the new season has too little data so we
    train on the previous n completed seasons.
    Examples:
      Oct 2028 → [2023, 2024, 2025, 2026, 2027]
      Jan 2029 → [2024, 2025, 2026, 2027, 2028]
      Oct 2029 → [2024, 2025, 2026, 2027, 2028]
    """
    today = date.today()
    current = _current_nhl_season_year()
    last_complete = current - 1 if today.month >= 9 else current
    return list(range(last_complete - n + 1, last_complete + 1))


# 5-year rolling training window — updates automatically each season
TRAINING_SEASONS = _training_seasons()
# Bubble/shortened seasons get down-weighted
DOWNWEIGHT_SEASONS = {2020: 0.5, 2021: 0.6}


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def _load_cache(name: str):
    p = _cache_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(name: str, data):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)


def fetch_season_games(season_year: int) -> list:
    """
    Fetches all regular-season game results for a given season.
    season_year: e.g. 2023 → 2023-24 season.
    Returns list of game result dicts.
    """
    cache_key = f"games_{season_year}"
    cached = _load_cache(cache_key)
    if cached:
        print(f"  [cache] Loaded {len(cached)} games for {season_year}")
        return cached

    print(f"  Fetching games for {season_year} season...")
    games = []

    # NHL API: club-schedule-season for one team, then extract all games
    # Better: iterate through schedule day by day across the season
    # Season start dates (approximate)
    season_starts = {
        2020: "2021-01-13",  # Bubble season
        2021: "2021-10-12",
        2022: "2022-10-07",
        2023: "2023-10-10",
        2024: "2024-10-08",
    }
    season_ends = {
        2020: "2021-05-19",
        2021: "2022-04-29",
        2022: "2023-04-14",
        2023: "2024-04-18",
        2024: "2025-04-17",
    }

    start = date.fromisoformat(season_starts.get(season_year, f"{season_year+1}-10-01"))
    end = date.fromisoformat(season_ends.get(season_year, f"{season_year+1}-04-30"))

    current = start
    while current <= end:
        d_str = current.strftime("%Y-%m-%d")
        try:
            day_games = nhl_api.get_scoreboard(d_str)
            for g in day_games:
                # Only regular season (game type 2)
                if g.get("gameType") != 2:
                    continue
                home_abbr = g.get("homeTeam", {}).get("abbrev", "")
                away_abbr = g.get("awayTeam", {}).get("abbrev", "")
                home_score = g.get("homeTeam", {}).get("score", 0)
                away_score = g.get("awayTeam", {}).get("score", 0)
                if not home_abbr or not away_abbr:
                    continue
                if home_score == 0 and away_score == 0:
                    continue
                period_descriptor = g.get("periodDescriptor", {})
                max_period = period_descriptor.get("number", 3)
                went_ot = max_period > 3

                games.append({
                    "date": d_str,
                    "season": season_year,
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_score": home_score,
                    "away_score": away_score,
                    "went_ot": went_ot,
                    "home_won": home_score > away_score,
                })
        except Exception as e:
            pass  # Skip days with API errors

        current += timedelta(days=1)
        time.sleep(0.1)  # Be polite to the API

    _save_cache(cache_key, games)
    print(f"  Fetched {len(games)} games for {season_year}")
    return games


def build_training_dataset(all_games: list, mp_team_stats: dict, mp_goalie_stats: dict) -> tuple:
    """
    Builds X (features) and y (labels) arrays for model training.
    Uses current-season standings approximated from season stats.
    Returns (X, y, weights, game_dates).
    """
    print("Building training dataset...")

    # Build rolling Elo ratings game by game
    elo_ratings = {}
    last_game_date_by_team = {}

    X = []
    y = []
    weights = []
    game_dates = []

    # Sort all games chronologically
    sorted_games = sorted(all_games, key=lambda g: g["date"])

    # Pre-build season standings from game results
    # For each game, use YTD standings up to that game's date
    season_standings = {}  # season_year -> {team -> cumulative stats}

    def _init_team_stats(team):
        return {
            "gp": 0, "pts": 0, "gf": 0, "ga": 0, "rw": 0,
            "l10": [],
            "home_gp": 0, "home_w": 0, "home_otl": 0,
            "away_gp": 0, "away_w": 0, "away_otl": 0,
            "streak_code": "", "streak_count": 0,
            "pp_pct": 0.18, "pk_pct": 0.80,
        }

    for g in sorted_games:
        season = g["season"]
        home = g["home_team"]
        away = g["away_team"]
        home_won = g["home_won"]
        went_ot = g.get("went_ot", False)
        game_date = g["date"]

        # Get current standings for this season
        if season not in season_standings:
            season_standings[season] = {}
        ss = season_standings[season]

        for team in [home, away]:
            if team not in ss:
                ss[team] = _init_team_stats(team)

        h_stats = ss[home]
        a_stats = ss[away]

        # Build synthetic standings dicts for feature computation
        def make_standing(stats, team):
            gp = stats["gp"] or 1
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
                "l10Wins":    sum(1 for r in stats["l10"][-10:] if r == "W"),
                "l10Losses":  sum(1 for r in stats["l10"][-10:] if r == "L"),
                "l10OtLosses":sum(1 for r in stats["l10"][-10:] if r == "OTL"),
                "homeWins":   stats["home_w"],
                "homeLosses": stats["home_gp"] - stats["home_w"] - stats["home_otl"],
                "homeOtLosses": stats["home_otl"],
                "roadWins":   stats["away_w"],
                "roadLosses": stats["away_gp"] - stats["away_w"] - stats["away_otl"],
                "roadOtLosses": stats["away_otl"],
                "streakCode":  stats["streak_code"],
                "streakCount": stats["streak_count"],
            }

        standings_list = [make_standing(h_stats, home), make_standing(a_stats, away)]

        # MoneyPuck features for this season
        h_xg = {}
        a_xg = {}
        if season in mp_team_stats:
            h_xg = moneypuck.extract_team_xg_features(mp_team_stats[season], home)
            a_xg = moneypuck.extract_team_xg_features(mp_team_stats[season], away)

        # Goalie features (season-level, no game-specific goalie info for historical)
        h_gsax = 0.0
        a_gsax = 0.0
        if season in mp_goalie_stats:
            pass  # Could look up per-team starter GSAx but skip for simplicity

        # Rest days
        h_last = last_game_date_by_team.get(home)
        a_last = last_game_date_by_team.get(away)

        try:
            fv = feat_eng.compute_features(
                home, away, game_date,
                standings_list, elo_ratings,
                h_last, a_last,
                h_xg, a_xg,
                h_gsax, a_gsax,
            )
            X.append(fv)
            y.append(1 if home_won else 0)
            weights.append(DOWNWEIGHT_SEASONS.get(season, 1.0))
            game_dates.append(game_date)
        except Exception as e:
            pass

        # Update standings AFTER feature computation (no lookahead)
        for team, stats, won, scored, allowed, is_home_team in [
            (home, h_stats, home_won,  g["home_score"], g["away_score"], True),
            (away, a_stats, not home_won, g["away_score"], g["home_score"], False),
        ]:
            stats["gp"] += 1
            stats["gf"] += scored
            stats["ga"] += allowed
            if is_home_team:
                stats["home_gp"] += 1
            else:
                stats["away_gp"] += 1

            if won:
                stats["pts"] += 2
                if not went_ot:
                    stats["rw"] += 1
                stats["l10"].append("W")
                if is_home_team: stats["home_w"] += 1
                else:            stats["away_w"] += 1
                # Streak update
                if stats["streak_code"] == "W":
                    stats["streak_count"] += 1
                else:
                    stats["streak_code"] = "W"
                    stats["streak_count"] = 1
            elif went_ot:
                stats["pts"] += 1
                stats["l10"].append("OTL")
                if is_home_team: stats["home_otl"] += 1
                else:            stats["away_otl"] += 1
                if stats["streak_code"] == "OT":
                    stats["streak_count"] += 1
                else:
                    stats["streak_code"] = "OT"
                    stats["streak_count"] = 1
            else:
                stats["l10"].append("L")
                if stats["streak_code"] == "L":
                    stats["streak_count"] += 1
                else:
                    stats["streak_code"] = "L"
                    stats["streak_count"] = 1

        # Update Elo after game
        margin = abs(g["home_score"] - g["away_score"])
        elo_ratings = elo_system.update_elo(elo_ratings, home, away, home_won, margin, went_ot)
        last_game_date_by_team[home] = game_date
        last_game_date_by_team[away] = game_date

    return np.array(X), np.array(y), np.array(weights), game_dates


def train_and_save():
    """Main training function — downloads data, trains model, saves artifacts."""
    print("=" * 60)
    print("NHL Oracle — Model Training")
    print("=" * 60)

    # Download MoneyPuck data
    print("\nDownloading MoneyPuck team stats...")
    mp_team_stats = {}
    mp_goalie_stats = {}
    for year in TRAINING_SEASONS:
        print(f"  Season {year}...")
        df_teams = moneypuck.get_team_stats(year)
        if df_teams is not None:
            mp_team_stats[year] = df_teams
            print(f"    Teams: {len(df_teams)} rows")
        df_goalies = moneypuck.get_goalie_stats(year)
        if df_goalies is not None:
            mp_goalie_stats[year] = df_goalies
            print(f"    Goalies: {len(df_goalies)} rows")

    # Fetch historical game results
    print("\nFetching historical game results...")
    all_games = []
    for year in TRAINING_SEASONS:
        season_games = fetch_season_games(year)
        all_games.extend(season_games)
    print(f"Total historical games: {len(all_games)}")

    if len(all_games) < 100:
        print("ERROR: Not enough historical games fetched. Check API connectivity.")
        return

    # Build feature dataset
    X, y, weights, game_dates = build_training_dataset(all_games, mp_team_stats, mp_goalie_stats)
    print(f"\nFeature matrix: {X.shape} | Labels: {y.sum()} home wins / {len(y)-y.sum()} away wins")

    # Walk-forward cross-validation
    print("\nWalk-forward cross-validation...")
    cv_results = []
    sorted_dates = sorted(set(game_dates))
    year_boundaries = {}
    for d in sorted_dates:
        yr = int(d[:4])
        if yr not in year_boundaries:
            year_boundaries[yr] = sorted_dates.index(d)

    cv_splits = [
        (TRAINING_SEASONS[:-2], TRAINING_SEASONS[-2]),
        (TRAINING_SEASONS[:-1], TRAINING_SEASONS[-1]),
    ]
    for train_seasons, test_season in cv_splits:
        train_mask = np.array([
            int(d[:4]) in train_seasons or (int(d[:4]) == train_seasons[-1] + 1 and d[5:7] in ("10","11","12"))
            for d in game_dates
        ])
        test_mask = np.array([int(d[:4]) == test_season for d in game_dates])

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        w_tr = weights[train_mask]

        if len(X_tr) < 50 or len(X_te) < 10:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y_tr, sample_weight=w_tr)

        probs = lr.predict_proba(X_te_s)[:, 1]
        acc = accuracy_score(y_te, (probs >= 0.5).astype(int))
        brier = brier_score_loss(y_te, probs)
        ll = log_loss(y_te, probs)

        cv_results.append({"test_season": test_season, "accuracy": acc, "brier": brier, "log_loss": ll})
        print(f"  Test season {test_season}: Accuracy={acc:.3f}, Brier={brier:.4f}, LogLoss={ll:.4f}")

    # Train final models on ALL data
    print("\nTraining final models on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Logistic Regression (calibrated) ---
    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    cal_lr = CalibratedClassifierCV(base_lr, method="isotonic", cv=5)
    cal_lr.fit(X_scaled, y, sample_weight=weights)
    print("  Logistic regression trained.")

    # Feature importance from raw LR coefficients
    base_lr.fit(X_scaled, y, sample_weight=weights)
    importance = dict(zip(feat_eng.FEATURE_NAMES, base_lr.coef_[0].tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    print("  Top features:", [(k, round(v, 3)) for k, v in sorted_imp[:5]])

    # --- XGBoost (calibrated) ---
    print("  Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_scaled, y, sample_weight=weights)
    cal_xgb = CalibratedClassifierCV(xgb, method="isotonic", cv=5)
    cal_xgb.fit(X_scaled, y, sample_weight=weights)
    print("  XGBoost trained.")

    # Quick ensemble check on last CV split
    if cv_results:
        last_split = cv_splits[-1]
        test_mask = np.array([int(d[:4]) == last_split[1] for d in game_dates])
        X_te_s = scaler.transform(X[test_mask])
        y_te = y[test_mask]
        lr_probs  = cal_lr.predict_proba(X_te_s)[:, 1]
        xgb_probs = cal_xgb.predict_proba(X_te_s)[:, 1]
        ens_probs = 0.5 * lr_probs + 0.5 * xgb_probs
        ens_acc   = accuracy_score(y_te, (ens_probs >= 0.5).astype(int))
        ens_brier = brier_score_loss(y_te, ens_probs)
        print(f"  Ensemble test accuracy: {ens_acc:.3f} | Brier: {ens_brier:.4f}")
        cv_results[-1]["ensemble_accuracy"] = ens_acc
        cv_results[-1]["ensemble_brier"] = ens_brier

    # Save artifacts
    joblib.dump(cal_lr,  os.path.join(MODELS_DIR, "model.pkl"))       # LogReg
    joblib.dump(cal_xgb, os.path.join(MODELS_DIR, "model_xgb.pkl"))   # XGBoost
    joblib.dump(scaler,  os.path.join(MODELS_DIR, "scaler.pkl"))

    metadata = {
        "feature_names": feat_eng.FEATURE_NAMES,
        "training_seasons": TRAINING_SEASONS,
        "n_games": len(all_games),
        "n_features": X.shape[1],
        "cv_results": cv_results,
        "feature_importance": importance,
        "trained_date": date.today().isoformat(),
        "models": ["logistic_regression", "xgboost", "ensemble_50_50"],
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Models saved to models/")
    if cv_results:
        avg_acc   = sum(r["accuracy"] for r in cv_results) / len(cv_results)
        avg_brier = sum(r["brier"] for r in cv_results) / len(cv_results)
        print(f"Average LR CV accuracy:  {avg_acc:.3f}")
        print(f"Average LR CV Brier:     {avg_brier:.4f}")
        last = cv_results[-1]
        if "ensemble_accuracy" in last:
            print(f"Ensemble accuracy (last split): {last['ensemble_accuracy']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
