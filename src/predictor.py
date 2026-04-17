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
import injury_tracker

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _current_nhl_season_year() -> int:
    """Returns the starting year of the current NHL season (e.g. 2028 for the 2028-29 season).
    NHL seasons start in October, so Oct-Dec belong to the current calendar year,
    Jan-Aug belong to the previous calendar year's season."""
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def _mp_season_year(standings: list) -> int:
    """
    Returns the MoneyPuck season year to use for live predictions.
    For the first 8 games of a new season, MoneyPuck won't have current-year data yet,
    so we fall back to the previous season's stats.
    """
    season_year = _current_nhl_season_year()
    avg_gp = (sum(s.get("gamesPlayed", 0) for s in standings) / len(standings)
              if standings else 0)
    if avg_gp < 8:
        return season_year - 1
    return season_year

# Odds API (optional — set ODDS_API_KEY env var)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"


def load_models():
    """
    Loads logistic regression + XGBoost + scaler.
    Returns (lr_model, xgb_model, scaler). Any can be None if not found.
    """
    def _load(path):
        try:
            return joblib.load(path) if os.path.exists(path) else None
        except Exception as e:
            print(f"[predictor] Failed to load {path}: {e}")
            return None

    return (
        _load(os.path.join(MODELS_DIR, "model.pkl")),
        _load(os.path.join(MODELS_DIR, "model_xgb.pkl")),
        _load(os.path.join(MODELS_DIR, "scaler.pkl")),
    )


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

    # Load models
    lr_model, xgb_model, scaler = load_models()
    model_available = scaler is not None and (lr_model is not None or xgb_model is not None)

    if not model_available:
        print("[predictor] WARNING: No trained model found. Using Elo fallback.")

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

    # Fetch MoneyPuck stats — use previous season for first 8 games of a new season
    mp_year = _mp_season_year(standings)
    print(f"[predictor] Using MoneyPuck year {mp_year} (avg GP: {sum(s.get('gamesPlayed',0) for s in standings)/max(len(standings),1):.1f})")
    mp_teams = moneypuck.get_team_stats(mp_year)
    mp_goalies = moneypuck.get_goalie_stats(mp_year)

    # Pre-fetch live PP%/PK% for all teams playing today (fixes the 0.0 bug)
    all_teams = list({g.get("homeTeam", {}).get("abbrev", "") for g in games} |
                     {g.get("awayTeam", {}).get("abbrev", "") for g in games})
    all_teams = [t for t in all_teams if t]
    print(f"[predictor] Fetching PP/PK for {len(all_teams)} teams...")
    special_teams = nhl_api.get_all_teams_special_teams(all_teams)

    # Fetch odds
    odds_map = get_odds()

    predictions = []
    for g in games:
        home_info = g.get("homeTeam", {})
        away_info = g.get("awayTeam", {})
        home_abbr = home_info.get("abbrev", "")
        away_abbr = away_info.get("abbrev", "")
        home_name = (home_info.get("placeName", {}).get("default", home_abbr) + " " +
                     home_info.get("commonName", {}).get("default", "")).strip()
        away_name = (away_info.get("placeName", {}).get("default", away_abbr) + " " +
                     away_info.get("commonName", {}).get("default", "")).strip()

        if not home_abbr or not away_abbr:
            continue

        game_time = g.get("startTimeUTC", "")

        # --- Rest / back-to-back ---
        h_last = get_team_last_game(home_abbr, game_date)
        a_last = get_team_last_game(away_abbr, game_date)

        def days_since(last):
            if not last:
                return 3
            return (date.fromisoformat(game_date) - date.fromisoformat(last)).days

        h_b2b = days_since(h_last) == 1
        a_b2b = days_since(a_last) == 1

        # --- MoneyPuck xG features ---
        h_xg = moneypuck.extract_team_xg_features(mp_teams, home_abbr) if mp_teams is not None else {}
        a_xg = moneypuck.extract_team_xg_features(mp_teams, away_abbr) if mp_teams is not None else {}

        # --- Confirmed starting goalie + GSAx ---
        h_gsax, a_gsax = 0.0, 0.0
        h_goalie_name, a_goalie_name = None, None

        # Try confirmed starter from game preview first
        confirmed_h = injury_tracker.get_confirmed_starter(home_abbr, game_date)
        confirmed_a = injury_tracker.get_confirmed_starter(away_abbr, game_date)

        if mp_goalies is not None:
            for abbr, confirmed, store in [(home_abbr, confirmed_h, "h"), (away_abbr, confirmed_a, "a")]:
                if confirmed:
                    gsax = moneypuck.extract_goalie_gsax(mp_goalies, confirmed)
                    name = confirmed
                else:
                    # Fall back to season leader by games started
                    goalie_list = nhl_api.get_goalie_stats_list(abbr)
                    starter = max(goalie_list,
                                  key=lambda x: x.get("gamesStarted", x.get("gamesPlayed", 0)),
                                  default=None) if goalie_list else None
                    if starter:
                        last = starter.get("lastName", {})
                        name = last.get("default", "") if isinstance(last, dict) else str(last)
                        gsax = moneypuck.extract_goalie_gsax(mp_goalies, name)
                    else:
                        name, gsax = None, 0.0
                if store == "h":
                    h_gsax, h_goalie_name = gsax, name
                else:
                    a_gsax, a_goalie_name = gsax, name

        # --- Injury impact ---
        h_injuries = injury_tracker.get_team_injury_impact(home_abbr)
        a_injuries = injury_tracker.get_team_injury_impact(away_abbr)

        # Convert injury impact to probability shift:
        # Each 0.10 goals/game of offensive loss ≈ -1.5% win probability
        h_inj_shift = -(h_injuries["offensive_impact"] * 0.15 + h_injuries["defensive_impact"] * 0.10)
        a_inj_shift = -(a_injuries["offensive_impact"] * 0.15 + a_injuries["defensive_impact"] * 0.10)
        # Net: positive = home team relatively healthier
        injury_prob_adj = (h_inj_shift - a_inj_shift)

        # --- Feature vector ---
        h_st = special_teams.get(home_abbr, {})
        a_st = special_teams.get(away_abbr, {})
        try:
            fv = feat_eng.compute_features(
                home_abbr, away_abbr, game_date,
                standings, elo_ratings,
                h_last, a_last,
                h_xg, a_xg,
                h_gsax, a_gsax,
                home_pp_pct=h_st.get("pp_pct"),
                away_pp_pct=a_st.get("pp_pct"),
                home_pk_pct=h_st.get("pk_pct"),
                away_pk_pct=a_st.get("pk_pct"),
            )
        except Exception as e:
            print(f"[predictor] Feature error for {home_abbr} vs {away_abbr}: {e}")
            continue

        # --- Logistic regression prediction ---
        if model_available:
            fv_scaled = scaler.transform([fv])
            probs = []
            if lr_model is not None:
                probs.append(float(lr_model.predict_proba(fv_scaled)[0][1]))
            if xgb_model is not None:
                probs.append(float(xgb_model.predict_proba(fv_scaled)[0][1]))
            # Equal-weight ensemble of whichever models are available
            lr_prob = sum(probs) / len(probs)
        else:
            lr_prob = elo_system.elo_win_probability(home_abbr, away_abbr, elo_ratings)

        # --- Monte Carlo simulation ---
        LEAGUE_AVG = 3.0
        mc_result = mc.simulate(
            home_abbr, away_abbr,
            home_xgf_per60=h_xg.get("xgf_per60", LEAGUE_AVG),
            home_xga_per60=h_xg.get("xga_per60", LEAGUE_AVG),
            away_xgf_per60=a_xg.get("xgf_per60", LEAGUE_AVG),
            away_xga_per60=a_xg.get("xga_per60", LEAGUE_AVG),
            home_goalie_gsax=h_gsax, away_goalie_gsax=a_gsax,
            home_b2b=h_b2b, away_b2b=a_b2b,
        )

        # --- Vegas odds lookup ---
        home_ml, away_ml = None, None
        vegas_implied_home = None
        for key, o in odds_map.items():
            h_name_odds = o.get("home_team", "")
            if home_abbr.lower() in h_name_odds.lower() or home_name.lower() in h_name_odds.lower():
                home_ml = o.get("home_ml")
                away_ml = o.get("away_ml")
                vegas_implied_home = american_to_implied(home_ml)
                break

        # --- Blend: Model + Monte Carlo + Vegas + Injury adjustment ---
        # Vegas encodes injuries, lineup news, travel — weight it heavily when available
        if vegas_implied_home is not None:
            # With Vegas: 50% LR model, 20% Monte Carlo, 30% Vegas implied
            blended_prob = (0.50 * lr_prob +
                            0.20 * mc_result["home_win_pct"] +
                            0.30 * vegas_implied_home)
        else:
            # Without Vegas: 65% LR model, 35% Monte Carlo
            blended_prob = 0.65 * lr_prob + 0.35 * mc_result["home_win_pct"]

        # Apply injury adjustment (capped at ±5%), then cap at 85%
        blended_prob = max(0.15, min(0.85, blended_prob + max(-0.05, min(0.05, injury_prob_adj))))

        home_prob = blended_prob
        away_prob = 1.0 - blended_prob
        pick_team = home_abbr if home_prob >= away_prob else away_abbr
        pick_prob = max(home_prob, away_prob)

        # --- Confidence tier ---
        if pick_prob >= 0.68:
            tier, tier_emoji = "EXTREME CONVICTION", "🔥"
        elif pick_prob >= 0.63:
            tier, tier_emoji = "HIGH CONVICTION", "⭐"
        elif pick_prob >= 0.60:
            tier, tier_emoji = "STRONG", "✅"
        elif pick_prob >= 0.55:
            tier, tier_emoji = "LEAN", "📊"
        else:
            tier, tier_emoji = "COIN FLIP", "🪙"

        # --- Edge vs Vegas ---
        edge = 0.0
        recommend_bet = False
        odds_info = {}
        if vegas_implied_home is not None:
            vegas_implied_pick = vegas_implied_home if pick_team == home_abbr else (1 - vegas_implied_home)
            edge = pick_prob - vegas_implied_pick
            # Recommend when model has a meaningful edge AND high confidence
            recommend_bet = edge >= 0.05 and pick_prob >= 0.58
            odds_info = {
                "home_ml": home_ml,
                "away_ml": away_ml,
                "vegas_implied_home": round(vegas_implied_home, 4),
                "edge": round(edge, 4),
            }

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
            "vegas_implied_home": round(vegas_implied_home, 4) if vegas_implied_home else None,
            "recommend_bet": recommend_bet,
            "edge": round(edge, 4),
            "odds": odds_info,
            "injuries": {
                "home": {
                    "n_injured": h_injuries["n_injured"],
                    "impact": h_injuries["total_impact"],
                    "players": [p["name"] for p in h_injuries["injured_players"]],
                },
                "away": {
                    "n_injured": a_injuries["n_injured"],
                    "impact": a_injuries["total_impact"],
                    "players": [p["name"] for p in a_injuries["injured_players"]],
                },
            },
            "goalies": {
                "home": h_goalie_name,
                "away": a_goalie_name,
                "home_gsax": round(h_gsax, 2),
                "away_gsax": round(a_gsax, 2),
            },
            "features": {feat_eng.FEATURE_NAMES[i]: round(fv[i], 4) for i in range(len(fv))},
            "b2b_home": h_b2b,
            "b2b_away": a_b2b,
            "model_used": ("ensemble_lr+xgb" if lr_model and xgb_model else
                           "logistic_regression" if lr_model else
                           "xgboost" if xgb_model else "elo_fallback"),
        }
        predictions.append(pred)

    predictions.sort(key=lambda x: x["pick_prob"], reverse=True)
    return predictions
