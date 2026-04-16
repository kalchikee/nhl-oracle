#!/usr/bin/env python3
"""
NHL Playoff Data Fetcher — last 5 playoff seasons using NHLe API.
gameType=3 = playoffs. Output: data/playoff_data.csv

Usage: python fetch_playoff_data.py
"""
import sys, json, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent
DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = DATA_DIR / "playoff_data.csv"

BASE_URL = "https://api-web.nhle.com/v1"
HEADERS  = {"User-Agent": "NHL-Oracle/4.0"}

# NHL season IDs: 20232024 = 2023-24 season
# Skip 20192020: COVID bubble (neutral site in Edmonton/Toronto — no home ice)
PLAYOFF_SEASONS = [
    ("20162017", 2017), ("20172018", 2018), ("20182019", 2019),
    ("20202021", 2021), ("20212022", 2022), ("20222023", 2023),
    ("20232024", 2024), ("20242025", 2025),
]

K_FACTOR   = 20.0
HOME_ADV   = 60.0
LEAGUE_ELO = 1500.0


def nhle_get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                print(f"  Failed: {e}"); return {}
            time.sleep(2 ** i)
    return {}


def fetch_playoff_schedule(season_id: str) -> list:
    cache = CACHE_DIR / f"nhl_playoffs_{season_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    # Get playoff bracket / series schedule
    url = f"{BASE_URL}/playoff-bracket/{season_id}"
    data = nhle_get(url)
    games = []

    # Try schedule endpoint with gameType=3
    sched_url = f"{BASE_URL}/schedule/now"  # fallback
    series_list = data.get("series", [])

    if not series_list:
        # Use standings/schedule-based approach
        url2 = f"https://api.nhle.com/stats/rest/en/game?cayenneExp=season={season_id}%20and%20gameType=3"
        data2 = nhle_get(url2)
        game_data = data2.get("data", [])
        for g in game_data:
            if g.get("homeScore") is None:
                continue
            games.append({
                "game_id":    str(g.get("id", "")),
                "game_date":  str(g.get("gameDate", ""))[:10],
                "home_team":  g.get("homeTeamAbbrev", ""),
                "away_team":  g.get("visitingTeamAbbrev", ""),
                "home_score": int(g.get("homeScore", 0) or 0),
                "away_score": int(g.get("visitingScore", 0) or 0),
                "season":     int(str(season_id)[:4]) + 1,
            })
    else:
        for series in series_list:
            for game in series.get("games", []):
                if not game.get("gameOutcome"):
                    continue
                h_abbr = game.get("homeTeam", {}).get("abbrev", "")
                a_abbr = game.get("awayTeam", {}).get("abbrev", "")
                h_score = int(game.get("homeTeam", {}).get("score", 0) or 0)
                a_score = int(game.get("awayTeam", {}).get("score", 0) or 0)
                if not h_abbr or not a_abbr:
                    continue
                games.append({
                    "game_id":    str(game.get("id", "")),
                    "game_date":  str(game.get("gameDate", ""))[:10],
                    "home_team":  h_abbr,
                    "away_team":  a_abbr,
                    "home_score": h_score,
                    "away_score": a_score,
                    "season":     int(str(season_id)[:4]) + 1,
                })

    cache.write_text(json.dumps(games, indent=2))
    return games


# Approximate last day of each NHL regular season for standings lookup
_REG_SEASON_END = {
    "20162017": "2017-04-09",
    "20172018": "2018-04-08",
    "20182019": "2019-04-06",
    "20202021": "2021-05-19",  # shortened COVID season
    "20212022": "2022-05-01",
    "20222023": "2023-04-14",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
}


def fetch_reg_season_stats(season_id: str) -> dict:
    """Fetch end-of-regular-season team stats from NHLe standings API."""
    cache = CACHE_DIR / f"nhl_reg_stats_{season_id}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached:  # only use if non-empty
            return cached

    end_date = _REG_SEASON_END.get(season_id, f"{str(season_id)[4:]}-04-15")
    url = f"https://api-web.nhle.com/v1/standings/{end_date}"
    data = nhle_get(url)

    stats = {}
    for t in data.get("standings", []):
        # teamAbbrev is a localized dict: {"default": "WPG"}
        abbrev_raw = t.get("teamAbbrev", "")
        abbrev = abbrev_raw.get("default", "") if isinstance(abbrev_raw, dict) else abbrev_raw
        if not abbrev:
            continue
        gp   = t.get("gamesPlayed", 1) or 1
        wins = t.get("wins", 0) or 0
        stats[abbrev] = {
            "win_pct": wins / gp,
            "gpg":     (t.get("goalFor", 0) or 0) / gp,
            "gapg":    (t.get("goalAgainst", 0) or 0) / gp,
            "pt_pct":  t.get("pointPctg", 0.5) or 0.5,
        }

    cache.write_text(json.dumps(stats, indent=2))
    return stats


def add_series_context(games: list) -> list:
    """
    Add series context to NHL playoff games.
    NHL format: all rounds are best-of-7. No Play-In tournament.
    Series identified by frozenset of two teams (unique per season in NHL).
    """
    sorted_games = sorted(games, key=lambda g: g.get("game_date", ""))
    series_wins: dict = defaultdict(lambda: defaultdict(int))

    result = []
    for g in sorted_games:
        h, a = g["home_team"], g["away_team"]
        key = frozenset([h, a])

        h_wins = series_wins[key][h]
        a_wins = series_wins[key][a]
        game_num = h_wins + a_wins + 1

        series_deficit = h_wins - a_wins
        # Elimination: loser goes home (first to 4 wins in best-of-7)
        is_elimination = int((h_wins == 3) or (a_wins == 3))

        label = 1 if g["home_score"] > g["away_score"] else 0
        if label == 1:
            series_wins[key][h] += 1
        else:
            series_wins[key][a] += 1

        result.append({
            **g,
            "series_game_num":     game_num,
            "series_deficit":      series_deficit,
            "is_elimination_game": is_elimination,
        })
    return result


def main():
    print("NHL Playoff Data Fetcher")
    print("=" * 40)

    all_rows = []
    elo_map = defaultdict(lambda: LEAGUE_ELO)  # persist across seasons

    for season_id, year in PLAYOFF_SEASONS:
        print(f"\nSeason {season_id} ({year})")
        stats = fetch_reg_season_stats(season_id)
        games = fetch_playoff_schedule(season_id)
        games = add_series_context(games)
        print(f"  Fetched {len(games)} playoff games, {len(stats)} teams with stats")

        for g in sorted(games, key=lambda x: x.get("game_date", "")):
            h, a = g["home_team"], g["away_team"]
            h_elo = elo_map[h]; a_elo = elo_map[a]
            hs  = stats.get(h, {"win_pct": 0.5, "gpg": 3.0, "gapg": 3.0, "pt_pct": 0.5})
            as_ = stats.get(a, {"win_pct": 0.5, "gpg": 3.0, "gapg": 3.0, "pt_pct": 0.5})
            label = 1 if g["home_score"] > g["away_score"] else 0

            row = {
                "season":       year,
                "game_id":      g["game_id"],
                "game_date":    g["game_date"],
                "home_team":    h,
                "away_team":    a,
                "home_score":   g["home_score"],
                "away_score":   g["away_score"],
                "label":        label,
                "is_playoff":   1,
                "elo_diff":     h_elo - a_elo,
                "win_pct_diff": hs["win_pct"] - as_["win_pct"],
                "gpg_diff":     hs["gpg"]  - as_["gpg"],
                "gapg_diff":    hs["gapg"] - as_["gapg"],
                "pt_pct_diff":  hs["pt_pct"] - as_["pt_pct"],
                # Series context — all NHL rounds are best-of-7
                "series_game_num":     g["series_game_num"],
                "series_deficit":      g["series_deficit"],
                "is_elimination_game": g["is_elimination_game"],
            }
            all_rows.append(row)

            exp = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))
            elo_map[h] = h_elo + K_FACTOR * (label - exp)
            elo_map[a] = a_elo + K_FACTOR * ((1 - label) - (1 - exp))

        # Offseason regression
        for team in list(elo_map.keys()):
            elo_map[team] = 0.75 * elo_map[team] + 0.25 * LEAGUE_ELO

    if not all_rows:
        print("\nNo data fetched.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} playoff games to {OUT_CSV}")
    print(f"Home win rate: {df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
