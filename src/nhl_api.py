"""NHL API client — wraps api-web.nhle.com endpoints."""

import time
import requests
from datetime import date, timedelta
from typing import Optional

BASE_URL = "https://api-web.nhle.com/v1"
STATS_URL = "https://api.nhle.com/stats/rest/en"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# Maps full team names to abbreviations (used for various lookups)
TEAM_ABBREVS = [
    "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH",
    "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG",
]


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"[NHL API] Failed {url}: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


def get_schedule(game_date: Optional[str] = None) -> list:
    """Returns list of game dicts for the given date (YYYY-MM-DD)."""
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")
    data = _get(f"{BASE_URL}/schedule/{game_date}")
    for week in data.get("gameWeek", []):
        if week.get("date") == game_date:
            return week.get("games", [])
    # Sometimes the API returns the week starting on a different day
    # Flatten all games from the week and filter by date
    all_games = []
    for week in data.get("gameWeek", []):
        for g in week.get("games", []):
            g_date = g.get("gameDate", "")[:10]
            if g_date == game_date:
                all_games.append(g)
    return all_games


def get_scoreboard(game_date: Optional[str] = None) -> list:
    """Returns completed game results for the given date."""
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")
    data = _get(f"{BASE_URL}/score/{game_date}")
    return data.get("games", [])


def get_standings() -> list:
    """Returns current NHL standings (list of team dicts)."""
    data = _get(f"{BASE_URL}/standings/now")
    return data.get("standings", [])


def get_club_stats(team_abbrev: str) -> dict:
    """Returns season stats for a team (skaters + goalies)."""
    return _get(f"{BASE_URL}/club-stats/{team_abbrev}/now")


def get_team_schedule(team_abbrev: str, season: str = "20252026") -> list:
    """Returns all games for a team in a season."""
    data = _get(f"{BASE_URL}/club-schedule-season/{team_abbrev}/{season}")
    return data.get("games", [])


def get_schedule_range(days_ahead: int = 7) -> list:
    """Returns all games scheduled in the next N days."""
    games = []
    today = date.today()
    for i in range(days_ahead):
        d = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        games.extend(get_schedule(d))
    return games


def get_last_game_date(team_abbrev: str, before_date: Optional[str] = None) -> Optional[str]:
    """Returns the date of the team's most recent completed game."""
    if before_date is None:
        before_date = date.today().strftime("%Y-%m-%d")
    games = get_team_schedule(team_abbrev)
    completed = [
        g for g in games
        if g.get("gameDate", "9999") < before_date
        and g.get("gameState") in ("OFF", "FINAL")
    ]
    if not completed:
        return None
    return max(g["gameDate"] for g in completed)


def get_goalie_stats_list(team_abbrev: str) -> list:
    """Returns list of goalie stat dicts for the team."""
    data = get_club_stats(team_abbrev)
    return data.get("goalies", [])


def get_all_teams_special_teams(team_abbrevs: list) -> dict:
    """
    Fetches PP%/PK% for all teams in one request from the NHL Stats REST API.
    Returns dict: {abbrev: {pp_pct, pk_pct}}
    """
    # Hardcoded fallbacks for teams that may not match by name
    _FULLNAME_OVERRIDES = {
        "New York Rangers": "NYR",
        "New York Islanders": "NYI",
        "Montréal Canadiens": "MTL",
        "Montreal Canadiens": "MTL",
        "Utah Hockey Club": "UTA",
        "Utah Mammoth": "UTA",
    }

    result = {}
    try:
        # Determine current season ID (e.g. 20252026 for 2025-26 season)
        today = date.today()
        start_year = today.year if today.month >= 9 else today.year - 1
        season_id = f"{start_year}{start_year + 1}"
        url = (f"{STATS_URL}/team/summary"
               f"?cayenneExp=seasonId={season_id}%20and%20gameTypeId=2&limit=40")
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Build name → abbrev from standings
        standings_r = requests.get(f"{BASE_URL}/standings/now", headers=HEADERS, timeout=15)
        name_to_abbrev = {}
        if standings_r.ok:
            for s in standings_r.json().get("standings", []):
                abbrev = s.get("teamAbbrev", {})
                abbrev = abbrev.get("default", "") if isinstance(abbrev, dict) else str(abbrev)
                place = s.get("placeName", {})
                place = place.get("default", "") if isinstance(place, dict) else str(place)
                common = s.get("teamCommonName", {})
                common = common.get("default", "") if isinstance(common, dict) else str(common)
                name_to_abbrev[f"{place} {common}"] = abbrev

        name_to_abbrev.update(_FULLNAME_OVERRIDES)

        for t in data.get("data", []):
            full = t.get("teamFullName", "")
            abbrev = name_to_abbrev.get(full)
            if abbrev:
                result[abbrev] = {
                    "pp_pct": float(t.get("powerPlayPct", 0.183)),
                    "pk_pct": float(t.get("penaltyKillPct", 0.799)),
                }
    except Exception as e:
        print(f"[NHL API] Special teams fetch error: {e}")

    # Fill missing teams with league averages
    for abbrev in team_abbrevs:
        if abbrev not in result:
            result[abbrev] = {"pp_pct": 0.183, "pk_pct": 0.799}
    return result
