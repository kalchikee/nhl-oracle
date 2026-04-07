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

# 5-year training window: seasons starting 2020 through 2024
TRAINING_SEASONS = [2020, 2021, 2022, 2023, 2024]
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
        return {"gp": 0, "pts": 0, "gf": 0, "ga": 0, "rw": 0, "l10": [], "pp_pct": 0.18, "pk_pct": 0.80}

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
                "l10Wins": sum(1 for r in stats["l10"][-10:] if r == "W"),
                "l10Losses": sum(1 for r in stats["l10"][-10:] if r == "L"),
                "l10OtLosses": sum(1 for r in stats["l10"][-10:] if r == "OTL"),
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
        for team, stats, won, scored, allowed in [
            (home, h_stats, home_won, g["home_score"], g["away_score"]),
            (away, a_stats, not home_won, g["away_score"], g["home_score"]),
        ]:
            stats["gp"] += 1
            stats["gf"] += scored
            stats["ga"] += allowed
            if won:
                stats["pts"] += 2
                if not went_ot:
                    stats["rw"] += 1
                stats["l10"].append("W")
            elif went_ot:
                stats["pts"] += 1
                stats["l10"].append("OTL")
            else:
                stats["l10"].append("L")

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

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # CalibratedClassifierCV with isotonic regression (Platt scaling)
    calibrated_model = CalibratedClassifierCV(base_lr, method="isotonic", cv=5)
    calibrated_model.fit(X_scaled, y, sample_weight=weights)

    # Feature importance (from base logistic regression coefficients)
    base_lr.fit(X_scaled, y, sample_weight=weights)
    importance = dict(zip(feat_eng.FEATURE_NAMES, base_lr.coef_[0].tolist()))

    # Save artifacts
    joblib.dump(calibrated_model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    metadata = {
        "feature_names": feat_eng.FEATURE_NAMES,
        "training_seasons": TRAINING_SEASONS,
        "n_games": len(all_games),
        "n_features": X.shape[1],
        "cv_results": cv_results,
        "feature_importance": importance,
        "trained_date": date.today().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Model saved to models/")
    print(f"Feature names: {feat_eng.FEATURE_NAMES}")
    if cv_results:
        avg_acc = sum(r["accuracy"] for r in cv_results) / len(cv_results)
        avg_brier = sum(r["brier"] for r in cv_results) / len(cv_results)
        print(f"Average CV accuracy: {avg_acc:.3f}")
        print(f"Average CV Brier: {avg_brier:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
