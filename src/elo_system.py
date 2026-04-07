"""Elo rating system for NHL teams with offseason regression."""

import json
import math
import os
from datetime import date
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ELO_FILE = os.path.join(DATA_DIR, "elo_ratings.json")

INITIAL_ELO = 1500.0
K_FACTOR = 6.0          # Per-game update factor (higher = faster adaptation)
OT_K_SCALE = 0.75       # OT wins update Elo less (near-coin-flip)
OFFSEASON_REGRESSION = 0.30  # Regress 30% toward mean each offseason
LEAGUE_MEAN = 1500.0


def load_ratings() -> dict:
    """Loads Elo ratings from disk. Returns dict of team_abbrev -> rating."""
    if os.path.exists(ELO_FILE):
        with open(ELO_FILE) as f:
            data = json.load(f)
            return data.get("ratings", {})
    return {}


def save_ratings(ratings: dict):
    """Saves Elo ratings to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(ELO_FILE):
        with open(ELO_FILE) as f:
            existing = json.load(f)
    existing["ratings"] = ratings
    existing["updated"] = date.today().isoformat()
    with open(ELO_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def get_rating(ratings: dict, team: str) -> float:
    return ratings.get(team, INITIAL_ELO)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic expected score for team A vs team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    ratings: dict,
    home_team: str,
    away_team: str,
    home_won: bool,
    margin: int = 1,
    went_ot: bool = False,
) -> dict:
    """
    Updates Elo ratings after a game.
    Returns updated ratings dict (does not save — caller must call save_ratings).
    """
    r_home = get_rating(ratings, home_team)
    r_away = get_rating(ratings, away_team)

    e_home = expected_score(r_home, r_away)
    s_home = 1.0 if home_won else 0.0

    # Margin of victory multiplier (log scale, capped at 5 goals)
    mov_mult = math.log(1 + min(margin, 5))
    if went_ot:
        mov_mult *= OT_K_SCALE

    delta = K_FACTOR * mov_mult * (s_home - e_home)

    ratings = dict(ratings)
    ratings[home_team] = r_home + delta
    ratings[away_team] = r_away - delta
    return ratings


def apply_offseason_regression(ratings: dict) -> dict:
    """
    Applies 30% regression toward mean (1500) for all teams.
    Call this once per season start.
    """
    return {
        team: OFFSEASON_REGRESSION * LEAGUE_MEAN + (1 - OFFSEASON_REGRESSION) * rating
        for team, rating in ratings.items()
    }


def elo_win_probability(home_team: str, away_team: str, ratings: dict) -> float:
    """Returns home team win probability based purely on Elo."""
    r_home = get_rating(ratings, home_team)
    r_away = get_rating(ratings, away_team)
    return expected_score(r_home, r_away)


def build_ratings_from_history(game_results: list) -> dict:
    """
    Builds Elo ratings from a list of game result dicts.
    Each dict: {home_team, away_team, home_score, away_score, went_ot, date}
    Results must be sorted chronologically.
    """
    ratings = {}
    last_season = None

    for g in sorted(game_results, key=lambda x: x.get("date", "")):
        season = g.get("season")
        if season and season != last_season and last_season is not None:
            ratings = apply_offseason_regression(ratings)
        last_season = season

        home = g["home_team"]
        away = g["away_team"]
        h_score = g.get("home_score", 0)
        a_score = g.get("away_score", 0)
        home_won = h_score > a_score
        margin = abs(h_score - a_score)
        went_ot = g.get("went_ot", False)

        ratings = update_elo(ratings, home, away, home_won, margin, went_ot)

    return ratings
