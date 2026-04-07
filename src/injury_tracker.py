"""
Injury tracker — pulls current injury reports and calculates lineup strength impact.

Sources:
  1. NHL API roster endpoint (injuryStatus field on each player)
  2. ESPN unofficial API as backup

Impact model:
  - Top-6 forward out   → -0.10 expected goals/game offense
  - Top-4 D out         → -0.07 expected goals/game defense
  - Backup goalie start → applied via goalie GSAx in predictor
  - Others              → -0.03
"""

import requests
import time
from typing import Optional

HEADERS = {"User-Agent": "NHL-Oracle/4.0"}

# NHL API injured player statuses
INJURED_STATUSES = {"IR", "IR-NR", "LTIR", "O", "DTD", "SUS", "INJURED"}

# ESPN team ID mapping (NHL abbrev → ESPN ID)
ESPN_TEAM_IDS = {
    "ANA": 25, "BOS": 1,  "BUF": 2,  "CGY": 3,  "CAR": 12,
    "CHI": 4,  "COL": 17, "CBJ": 29, "DAL": 14, "DET": 5,
    "EDM": 6,  "FLA": 13, "LAK": 8,  "MIN": 30, "MTL": 9,
    "NSH": 18, "NJD": 10, "NYI": 11, "NYR": 15, "OTT": 19,
    "PHI": 20, "PIT": 21, "SEA": 55, "SJS": 22, "STL": 16,
    "TBL": 24, "TOR": 26, "UTA": 53, "VAN": 23, "VGK": 54,
    "WSH": 27, "WPG": 52,
}


def _get(url: str, retries: int = 2) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return {}


def get_nhl_injuries(team_abbrev: str) -> list:
    """
    Pulls injured players from the NHL API roster.
    Returns list of dicts: {name, position, status, impact_score}
    impact_score = estimated goals/game impact of this player being out
    """
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
    data = _get(url)
    injured = []

    all_players = (
        data.get("forwards", []) +
        data.get("defensemen", []) +
        data.get("goalies", [])
    )

    for p in all_players:
        status = p.get("injuryStatus", "").upper().strip()
        if not status or status not in INJURED_STATUSES:
            continue

        pos = p.get("positionCode", "")
        first = p.get("firstName", {})
        last = p.get("lastName", {})
        if isinstance(first, dict):
            first = first.get("default", "")
        if isinstance(last, dict):
            last = last.get("default", "")
        name = f"{first} {last}".strip()

        # Estimate impact based on position and whether they're a regular
        sweater = p.get("sweaterNumber", 99)
        games_played = p.get("currentTeamRoster", {}).get("gamesPlayed", 20)

        if pos in ("L", "R", "C"):
            # Forwards: top-6 vs bottom-6 estimated by sweater number heuristic
            impact = 0.10 if int(sweater) <= 19 else 0.04
        elif pos == "D":
            impact = 0.07 if int(sweater) <= 29 else 0.03
        else:
            impact = 0.02  # Goalie handled separately

        if status == "DTD":
            impact *= 0.5  # Day-to-day = 50% chance of playing

        injured.append({
            "name": name,
            "position": pos,
            "status": status,
            "impact_score": impact,
        })

    return injured


def get_espn_injuries(team_abbrev: str) -> list:
    """
    Pulls injury report from ESPN unofficial API.
    Returns list of dicts: {name, position, status, impact_score}
    """
    espn_id = ESPN_TEAM_IDS.get(team_abbrev.upper())
    if not espn_id:
        return []

    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{espn_id}/injuries"
    data = _get(url)
    injured = []

    for item in data.get("injuries", []):
        athlete = item.get("athlete", {})
        name = athlete.get("displayName", "Unknown")
        pos = athlete.get("position", {}).get("abbreviation", "")
        status = item.get("status", "").upper()
        injury_type = item.get("type", {}).get("abbreviation", "")

        if status in ("OUT", "IR", "DOUBTFUL"):
            if pos in ("LW", "RW", "C", "F"):
                impact = 0.10
            elif pos == "D":
                impact = 0.07
            else:
                impact = 0.02
            if status == "DOUBTFUL":
                impact *= 0.6

            injured.append({
                "name": name,
                "position": pos,
                "status": status,
                "impact_score": impact,
            })

    return injured


def get_team_injury_impact(team_abbrev: str) -> dict:
    """
    Returns combined injury impact for a team.
    Tries NHL API first, ESPN as fallback.

    Returns:
        {
          "offensive_impact": float,  # Expected goals/game lost on offense
          "defensive_impact": float,  # Expected goals/game lost on defense
          "total_impact": float,      # Total combined impact
          "injured_players": list,    # Raw injury list
          "source": str,
        }
    """
    # Try NHL API first
    injured = get_nhl_injuries(team_abbrev)
    source = "nhl_api"

    # If NHL API returned nothing, try ESPN
    if not injured:
        injured = get_espn_injuries(team_abbrev)
        source = "espn"

    offensive_impact = sum(
        p["impact_score"] for p in injured
        if p["position"] in ("L", "R", "C", "LW", "RW", "F")
    )
    defensive_impact = sum(
        p["impact_score"] for p in injured
        if p["position"] == "D"
    )

    # Cap total impact at reasonable maximums
    offensive_impact = min(offensive_impact, 0.35)
    defensive_impact = min(defensive_impact, 0.25)

    return {
        "offensive_impact": round(offensive_impact, 3),
        "defensive_impact": round(defensive_impact, 3),
        "total_impact": round(offensive_impact + defensive_impact, 3),
        "injured_players": injured,
        "source": source,
        "n_injured": len(injured),
    }


def get_confirmed_starter(team_abbrev: str, game_date: str) -> Optional[str]:
    """
    Attempts to identify the confirmed starting goalie for tonight's game.
    Uses the NHL API game preview endpoint.
    Returns goalie last name or None if unconfirmed.
    """
    # Look up today's game for this team
    url = f"https://api-web.nhle.com/v1/schedule/{game_date}"
    data = _get(url)

    for week in data.get("gameWeek", []):
        for g in week.get("games", []):
            home = g.get("homeTeam", {}).get("abbrev", "")
            away = g.get("awayTeam", {}).get("abbrev", "")
            if team_abbrev not in (home, away):
                continue

            game_id = g.get("id")
            if not game_id:
                continue

            # Check game preview for confirmed starters
            preview = _get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing")
            matchup = preview.get("matchup", {})

            side = "homeTeam" if team_abbrev == home else "awayTeam"
            team_data = matchup.get(side, {})
            goalie_data = team_data.get("goalieStats", [])

            if goalie_data:
                g_info = goalie_data[0]
                name = g_info.get("name", {})
                if isinstance(name, dict):
                    return name.get("default", "")
                return str(name)

    return None
