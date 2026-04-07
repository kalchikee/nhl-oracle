"""NHL API client — wraps api-web.nhle.com endpoints."""

import time
import requests
from datetime import date, timedelta
from typing import Optional

BASE_URL = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "NHL-Oracle/4.0"}

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
