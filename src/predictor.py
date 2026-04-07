"""
Live prediction engine.
Loads trained model, fetches today's games, and produces calibrated predictions.
"""

import os
import sys
import json
import joblib
import numpy as np
from datetime import date, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
import nhl_api
import moneypuck
import elo_system
import features as feat_eng
import monte_carlo as mc

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CURRENT_SEASON_MP_YEAR = 2025   # MoneyPuck year for current season stats

# Odds API (optional — set ODDS_API_KEY env var)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"


def load_model():
    """Loads calibrated model + scaler from disk. Returns (model, scaler) or (None, None)."""
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        return joblib.load(model_path), joblib.load(scaler_path)
    except Exception as e:
        print(f"[predictor] Failed to load model: {e}")
        return None, None


def get_odds() -> dict:
    """
    Fetches moneyline odds from The-Odds-API.
    Returns dict: {home_team_name: {"home_ml": -145, "away_ml": +120}}
    Returns empty dict if no API key or request fails.
    """
    if not ODDS_API_KEY:
        return {}
    try:
        import requests
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        }
        r = requests.get(ODDS_API_URL, params=params, timeout=10)
        r.raise_for_status()
        games = r.json()
        odds_map = {}
        for g in games:
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            key = f"{home}|{away}"
            for bookmaker in g.get("bookmakers", []):
                if bookmaker.get("key") in ("fanduel", "draftkings", "betmgm"):
                    for market in bookmaker.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                            odds_map[key] = {
                                "home_ml": outcomes.get(home),
                                "away_ml": outcomes.get(away),
                                "home_team": home,
                                "away_team": away,
                            }
                            break
                    break
        return odds_map
    except Exception as e:
        print(f"[predictor] Odds API error: {e}")
        return {}


def american_to_implied(american_odds: Optional[int]) -> float:
    """Converts American moneyline odds to implied probability."""
    if american_odds is None:
        return 0.5
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def get_team_last_game(team_abbrev: str, before_date: str) -> Optional[str]:
    """Returns date string of the team's last game before the given date."""
    games = nhl_api.get_team_schedule(team_abbrev)
    completed = [
        g for g in games
        if g.get("gameDate", "9999")[:10] < before_date
        and g.get("gameState") in ("OFF", "FINAL", "CRIT")
    ]
    if not completed:
        return None
    return max(g["gameDate"][:10] for g in completed)


def predict_games(game_date: Optional[str] = None) -> list:
    """
    Main prediction function.
    Returns list of prediction dicts, one per game.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    print(f"[predictor] Predicting games for {game_date}")

    # Load model
    model, scaler = load_model()
    model_available = model is not None

    if not model_available:
        print("[predictor] WARNING: No trained model found. Using Monte Carlo only.")

    # Load Elo ratings
    elo_ratings = elo_system.load_ratings()

    # Fetch today's schedule
    games = nhl_api.get_schedule(game_date)
    if not games:
        print(f"[predictor] No games found for {game_date}")
        return []

    print(f"[predictor] Found {len(games)} games")

    # Fetch standings
    standings = nhl_api.get_standings()

    # Fetch MoneyPuck current season stats (cached)
    mp_teams = moneypuck.get_team_stats(CURRENT_SEASON_MP_YEAR)
    mp_goalies = moneypuck.get_goalie_stats(CURRENT_SEASON_MP_YEAR)

    # Fetch odds
    odds_map = get_odds()

    predictions = []
    for g in games:
        home_info = g.get("homeTeam", {})
        away_info = g.get("awayTeam", {})
        home_abbr = home_info.get("abbrev", "")
        away_abbr = away_info.get("abbrev", "")
        home_name = home_info.get("placeName", {}).get("default", home_abbr) + " " + home_info.get("commonName", {}).get("default", "")
        away_name = away_info.get("placeName", {}).get("default", away_abbr) + " " + away_info.get("commonName", {}).get("default", "")
        home_name = home_name.strip()
        away_name = away_name.strip()

        if not home_abbr or not away_abbr:
            continue

        game_time = g.get("startTimeUTC", "")

        # Last game dates (for rest/B2B detection)
        h_last = get_team_last_game(home_abbr, game_date)
        a_last = get_team_last_game(away_abbr, game_date)

        def days_since(last):
            if not last:
                return 3
            return (date.fromisoformat(game_date) - date.fromisoformat(last)).days

        h_b2b = days_since(h_last) == 1
        a_b2b = days_since(a_last) == 1

        # MoneyPuck xG features
        h_xg = moneypuck.extract_team_xg_features(mp_teams, home_abbr) if mp_teams is not None else {}
        a_xg = moneypuck.extract_team_xg_features(mp_teams, away_abbr) if mp_teams is not None else {}

        # Goalie GSAx (use team's leading goalie)
        h_gsax, a_gsax = 0.0, 0.0
        if mp_goalies is not None:
            # Find the team's most-used goalie
            h_goalies = nhl_api.get_goalie_stats_list(home_abbr)
            a_goalies = nhl_api.get_goalie_stats_list(away_abbr)

            for g_list, abbr, store_var in [(h_goalies, home_abbr, "h"), (a_goalies, away_abbr, "a")]:
                if g_list:
                    # Pick goalie with most games started
                    starter = max(g_list, key=lambda x: x.get("gamesStarted", x.get("gamesPlayed", 0)), default=None)
                    if starter:
                        name = starter.get("lastName", {})
                        if isinstance(name, dict):
                            name = name.get("default", "")
                        gsax = moneypuck.extract_goalie_gsax(mp_goalies, str(name))
                        if store_var == "h":
                            h_gsax = gsax
                        else:
                            a_gsax = gsax

        # Build feature vector
        try:
            fv = feat_eng.compute_features(
                home_abbr, away_abbr, game_date,
                standings, elo_ratings,
                h_last, a_last,
                h_xg, a_xg,
                h_gsax, a_gsax,
            )
        except Exception as e:
            print(f"[predictor] Feature error for {home_abbr} vs {away_abbr}: {e}")
            continue

        # Logistic regression prediction
        if model_available:
            fv_scaled = scaler.transform([fv])
            lr_prob = float(model.predict_proba(fv_scaled)[0][1])
        else:
            # Fallback: use Elo win probability
            lr_prob = elo_system.elo_win_probability(home_abbr, away_abbr, elo_ratings)

        # Monte Carlo simulation
        LEAGUE_AVG = 3.0
        h_xgf = h_xg.get("xgf_per60", LEAGUE_AVG)
        h_xga = h_xg.get("xga_per60", LEAGUE_AVG)
        a_xgf = a_xg.get("xgf_per60", LEAGUE_AVG)
        a_xga = a_xg.get("xga_per60", LEAGUE_AVG)

        mc_result = mc.simulate(
            home_abbr, away_abbr,
            home_xgf_per60=h_xgf, home_xga_per60=h_xga,
            away_xgf_per60=a_xgf, away_xga_per60=a_xga,
            home_goalie_gsax=h_gsax, away_goalie_gsax=a_gsax,
            home_b2b=h_b2b, away_b2b=a_b2b,
        )

        # Blend LR probability with Monte Carlo (weighted average)
        blended_prob = 0.65 * lr_prob + 0.35 * mc_result["home_win_pct"]

        # Confidence tier
        home_prob = blended_prob
        away_prob = 1.0 - blended_prob
        pick_team = home_abbr if home_prob >= away_prob else away_abbr
        pick_prob = max(home_prob, away_prob)

        if pick_prob >= 0.68:
            tier = "EXTREME CONVICTION"
            tier_emoji = "🔥"
        elif pick_prob >= 0.63:
            tier = "HIGH CONVICTION"
            tier_emoji = "⭐"
        elif pick_prob >= 0.60:
            tier = "STRONG"
            tier_emoji = "✅"
        elif pick_prob >= 0.55:
            tier = "LEAN"
            tier_emoji = "📊"
        else:
            tier = "COIN FLIP"
            tier_emoji = "🪙"

        # Market edge detection
        edge = 0.0
        recommend_bet = False
        odds_info = {}

        # Try to find odds for this game
        for key, o in odds_map.items():
            h_name = o.get("home_team", "")
            a_name = o.get("away_team", "")
            if home_abbr.lower() in h_name.lower() or home_name.lower() in h_name.lower():
                home_ml = o.get("home_ml")
                away_ml = o.get("away_ml")
                vegas_implied = american_to_implied(home_ml if pick_team == home_abbr else away_ml)
                edge = pick_prob - vegas_implied
                recommend_bet = edge >= 0.05 and pick_prob >= 0.58
                odds_info = {
                    "home_ml": home_ml,
                    "away_ml": away_ml,
                    "vegas_implied_home": round(american_to_implied(home_ml), 4),
                    "edge": round(edge, 4),
                }
                break

        pred = {
            "game_date": game_date,
            "game_time_utc": game_time,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_name": home_name,
            "away_name": away_name,
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "pick_team": pick_team,
            "pick_prob": round(pick_prob, 4),
            "tier": tier,
            "tier_emoji": tier_emoji,
            "mc": {k: round(v, 4) if isinstance(v, float) else v for k, v in mc_result.items()},
            "lr_prob": round(lr_prob, 4),
            "recommend_bet": recommend_bet,
            "edge": round(edge, 4),
            "odds": odds_info,
            "features": {feat_eng.FEATURE_NAMES[i]: round(fv[i], 4) for i in range(len(fv))},
            "b2b_home": h_b2b,
            "b2b_away": a_b2b,
            "model_used": "logistic_regression" if model_available else "elo_fallback",
        }
        predictions.append(pred)

    predictions.sort(key=lambda x: x["pick_prob"], reverse=True)
    return predictions
