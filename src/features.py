"""
Feature engineering — builds a numeric feature vector per game.
All features are computed as (home - away) differences unless noted.
This file is used identically for training AND live prediction.
"""

import math
from datetime import date, timedelta
from typing import Optional

from elo_system import get_rating, elo_win_probability


# Ordered feature names — order must NEVER change after model is trained
FEATURE_NAMES = [
    "elo_diff",           # Elo rating difference (home - away)
    "pts_pct_diff",       # Points percentage difference
    "gf_pg_diff",         # Goals for per game difference
    "ga_pg_diff",         # Goals against per game difference (negative = home advantage)
    "gd_pg_diff",         # Goal differential per game
    "pp_pct_diff",        # Power play % difference
    "pk_pct_diff",        # Penalty kill % difference (higher = better PK)
    "l10_pts_diff",       # Last 10 games points percentage difference
    "l5_pts_diff",        # Last 5 games points percentage difference (tighter form window)
    "reg_win_pct_diff",   # Regulation win % difference
    "home_split_diff",    # Home team's home win% minus away team's away win% (venue-specific form)
    "streak_diff",        # Current streak: home (+W/-L) minus away (+W/-L)
    "xgf_pct_diff",       # xGF% difference (from MoneyPuck; 0 if unavailable)
    "cf_pct_diff",        # Corsi for % difference (0 if unavailable)
    "pdo_diff",           # PDO difference (luck indicator; 0 if unavailable)
    "goalie_gsax_diff",   # Starting goalie GSAx difference (0 if unavailable)
    "rest_days_diff",     # Days of rest: home minus away
    "b2b_home",           # Home team on back-to-back (0 or 1)
    "b2b_away",           # Away team on back-to-back (0 or 1)
    "is_home",            # Always 1.0 — model learns home ice coefficient
    "log5_prob",          # Log5 head-to-head win probability (from pts_pct)
]


def _safe(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _log5(p_a: float, p_b: float) -> float:
    """Log5 expected win probability for team A given both teams' win rates."""
    if p_a + p_b <= 0:
        return 0.5
    num = p_a * (1 - p_b)
    denom = p_a * (1 - p_b) + (1 - p_a) * p_b
    return num / denom if denom > 0 else 0.5


def _get_standing(standings_list: list, team_abbrev: str) -> dict:
    """Finds a team's standing dict from the NHL API standings list."""
    for s in standings_list:
        abbrev = s.get("teamAbbrev", {})
        if isinstance(abbrev, dict):
            abbrev = abbrev.get("default", "")
        if str(abbrev).upper() == team_abbrev.upper():
            return s
    return {}


def _pts_pct(standing: dict) -> float:
    """Extracts points percentage from a standing dict."""
    # NHL API field names vary slightly
    for key in ["pointPctg", "pointsPctg", "ptPctg"]:
        if key in standing:
            return _safe(standing[key])
    pts = _safe(standing.get("points", 0))
    gp = _safe(standing.get("gamesPlayed", 1)) or 1
    return pts / (2 * gp)


def _l10_pts_pct(standing: dict) -> float:
    """Computes last-10 points percentage from standing dict."""
    w = _safe(standing.get("l10Wins", 0))
    l = _safe(standing.get("l10Losses", 0))
    ot = _safe(standing.get("l10OtLosses", 0))
    total_games = w + l + ot
    if total_games == 0:
        return 0.5
    return (w + 0.5 * ot) / total_games


def _reg_win_pct(standing: dict) -> float:
    gp = _safe(standing.get("gamesPlayed", 1)) or 1
    rw = _safe(standing.get("regulationWins", 0))
    return rw / gp


def _gf_pg(standing: dict) -> float:
    gp = _safe(standing.get("gamesPlayed", 1)) or 1
    return _safe(standing.get("goalFor", standing.get("goalsFor", 0))) / gp


def _ga_pg(standing: dict) -> float:
    gp = _safe(standing.get("gamesPlayed", 1)) or 1
    return _safe(standing.get("goalAgainst", standing.get("goalsAgainst", 0))) / gp


def _l5_pts_pct(standing: dict) -> float:
    """Last 5 games points percentage — tighter form window than L10."""
    # NHL API only provides L10; derive L5 from streak if available
    # Fallback: use L10 as proxy (same directionally)
    streak_code = standing.get("streakCode", "")
    streak_count = _safe(standing.get("streakCount", 0))
    # If on a win streak of 5+, L5 is likely very good
    if streak_code == "W" and streak_count >= 5:
        return 0.90
    if streak_code == "L" and streak_count >= 5:
        return 0.10
    # Fall back to L10
    return _l10_pts_pct(standing)


def _home_win_pct(standing: dict) -> float:
    """Win percentage in home games specifically."""
    hw = _safe(standing.get("homeWins", 0))
    hl = _safe(standing.get("homeLosses", 0))
    hotl = _safe(standing.get("homeOtLosses", 0))
    total = hw + hl + hotl
    if total == 0:
        return 0.5
    return (hw + 0.5 * hotl) / total


def _away_win_pct(standing: dict) -> float:
    """Win percentage in road games specifically."""
    rw = _safe(standing.get("roadWins", standing.get("awayWins", 0)))
    rl = _safe(standing.get("roadLosses", standing.get("awayLosses", 0)))
    rotl = _safe(standing.get("roadOtLosses", standing.get("awayOtLosses", 0)))
    total = rw + rl + rotl
    if total == 0:
        return 0.5
    return (rw + 0.5 * rotl) / total


def _streak_value(standing: dict) -> float:
    """
    Converts current streak to a signed numeric value.
    W3 → +3, L2 → -2, OTL1 → -0.5
    """
    code = standing.get("streakCode", "")
    count = _safe(standing.get("streakCount", 0))
    if code == "W":
        return count
    elif code == "L":
        return -count
    elif code == "OT":
        return -count * 0.5
    return 0.0


def _pp_pct(standing: dict) -> float:
    """Power play % — tries several field names."""
    for key in ["powerPlayPctg", "ppPctg", "powerPlayPercentage"]:
        if key in standing:
            val = _safe(standing[key])
            return val / 100.0 if val > 1.5 else val
    return 0.18  # league average fallback


def _pk_pct(standing: dict) -> float:
    for key in ["penaltyKillPctg", "pkPctg", "penaltyKillPercentage"]:
        if key in standing:
            val = _safe(standing[key])
            return val / 100.0 if val > 1.5 else val
    return 0.80  # league average fallback


def compute_features(
    home_team: str,
    away_team: str,
    game_date: str,
    standings: list,
    elo_ratings: dict,
    home_last_game_date: Optional[str] = None,
    away_last_game_date: Optional[str] = None,
    home_xg: Optional[dict] = None,
    away_xg: Optional[dict] = None,
    home_goalie_gsax: float = 0.0,
    away_goalie_gsax: float = 0.0,
) -> list:
    """
    Builds and returns the ordered feature vector for a game.
    home_xg / away_xg: dicts from moneypuck.extract_team_xg_features (optional).
    Returns a list of floats in FEATURE_NAMES order.
    """
    h = _get_standing(standings, home_team)
    a = _get_standing(standings, away_team)

    # Elo
    elo_diff = _safe(get_rating(elo_ratings, home_team) - get_rating(elo_ratings, away_team))

    # Points percentage
    h_pts = _pts_pct(h)
    a_pts = _pts_pct(a)
    pts_pct_diff = h_pts - a_pts

    # Goals per game
    gf_pg_diff = _gf_pg(h) - _gf_pg(a)
    ga_pg_diff = _ga_pg(h) - _ga_pg(a)  # Negative = home allows fewer goals
    gd_pg_diff = (_gf_pg(h) - _ga_pg(h)) - (_gf_pg(a) - _ga_pg(a))

    # Special teams
    pp_pct_diff = _pp_pct(h) - _pp_pct(a)
    pk_pct_diff = _pk_pct(h) - _pk_pct(a)

    # Form (last 10 and last 5)
    l10_pts_diff = _l10_pts_pct(h) - _l10_pts_pct(a)
    l5_pts_diff = _l5_pts_pct(h) - _l5_pts_pct(a)

    # Regulation win %
    reg_win_pct_diff = _reg_win_pct(h) - _reg_win_pct(a)

    # Venue-specific splits: home team's home win% vs away team's road win%
    home_split_diff = _home_win_pct(h) - _away_win_pct(a)

    # Streak: signed value (+W, -L)
    streak_diff = _streak_value(h) - _streak_value(a)

    # xG features (MoneyPuck) — 0 if not available
    h_xg = home_xg or {}
    a_xg = away_xg or {}
    xgf_pct_diff = _safe(h_xg.get("xgf_pct", 0.5)) - _safe(a_xg.get("xgf_pct", 0.5))
    cf_pct_diff = _safe(h_xg.get("cf_pct", 0.5)) - _safe(a_xg.get("cf_pct", 0.5))

    # PDO (luck indicator) — league average = 1.000
    h_pdo = _safe(h_xg.get("pdo", 1.0))
    a_pdo = _safe(a_xg.get("pdo", 1.0))
    if h_pdo > 2:
        h_pdo /= 100.0
    if a_pdo > 2:
        a_pdo /= 100.0
    pdo_diff = h_pdo - a_pdo

    # Goalie GSAx difference
    goalie_gsax_diff = _safe(home_goalie_gsax - away_goalie_gsax)

    # Rest / fatigue
    today = date.fromisoformat(game_date)

    def days_rest(last_game: Optional[str]) -> int:
        if not last_game:
            return 3
        return (today - date.fromisoformat(last_game)).days

    h_rest = days_rest(home_last_game_date)
    a_rest = days_rest(away_last_game_date)
    rest_days_diff = float(h_rest - a_rest)
    b2b_home = 1.0 if h_rest == 1 else 0.0
    b2b_away = 1.0 if a_rest == 1 else 0.0

    # Log5 probability
    log5_prob = _log5(h_pts if h_pts > 0 else 0.5, a_pts if a_pts > 0 else 0.5)

    # Return in exact FEATURE_NAMES order
    return [
        elo_diff,
        pts_pct_diff,
        gf_pg_diff,
        ga_pg_diff,
        gd_pg_diff,
        pp_pct_diff,
        pk_pct_diff,
        l10_pts_diff,
        l5_pts_diff,        # NEW
        reg_win_pct_diff,
        home_split_diff,    # NEW
        streak_diff,        # NEW
        xgf_pct_diff,
        cf_pct_diff,
        pdo_diff,
        goalie_gsax_diff,
        rest_days_diff,
        b2b_home,
        b2b_away,
        1.0,                # is_home
        log5_prob,
    ]
