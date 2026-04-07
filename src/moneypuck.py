"""MoneyPuck data downloader — xG, GSAx, team shot quality metrics."""

import io
import requests
import pandas as pd
from typing import Optional

HEADERS = {"User-Agent": "NHL-Oracle/4.0"}

# MoneyPuck CSV URL patterns
MP_TEAMS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv"
MP_GOALIES_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/goalies.csv"


def _fetch_csv(url: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"[MoneyPuck] Could not fetch {url}: {e}")
        return None


def get_team_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Downloads MoneyPuck team season summary for a given year.
    year = season start year (e.g., 2024 for 2024-25 season).
    Returns DataFrame with xGF%, CF%, GSAx columns per team.
    """
    df = _fetch_csv(MP_TEAMS_URL.format(year=year))
    if df is None:
        return None
    # Filter to all-situations or 5v5 as appropriate
    if "situation" in df.columns:
        df = df[df["situation"] == "all"].copy()
    return df


def get_goalie_stats(year: int) -> Optional[pd.DataFrame]:
    """
    Downloads MoneyPuck goalie season summary for a given year.
    Returns DataFrame with GSAx, xGSv% per goalie.
    """
    df = _fetch_csv(MP_GOALIES_URL.format(year=year))
    if df is None:
        return None
    if "situation" in df.columns:
        df = df[df["situation"] == "all"].copy()
    return df


def get_multi_year_team_stats(years: list) -> pd.DataFrame:
    """Downloads and combines team stats across multiple seasons."""
    frames = []
    for y in years:
        df = get_team_stats(y)
        if df is not None:
            df["season_year"] = y
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_multi_year_goalie_stats(years: list) -> pd.DataFrame:
    """Downloads and combines goalie stats across multiple seasons."""
    frames = []
    for y in years:
        df = get_goalie_stats(y)
        if df is not None:
            df["season_year"] = y
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def extract_team_xg_features(mp_df: pd.DataFrame, team_abbrev: str) -> dict:
    """
    Extracts xG-based features for a team from MoneyPuck DataFrame.
    Returns dict with xgf_pct, xgf_per60, xga_per60, cf_pct keys.
    """
    if mp_df is None or mp_df.empty:
        return {}

    # MoneyPuck uses different team abbreviation conventions in some years
    team_col = None
    for col in ["team", "teamCode", "name"]:
        if col in mp_df.columns:
            team_col = col
            break
    if team_col is None:
        return {}

    row = mp_df[mp_df[team_col].str.upper() == team_abbrev.upper()]
    if row.empty:
        # Try fuzzy match for teams that changed abbreviations (ARI -> UTA, etc.)
        abbrev_map = {"ARI": "UTA", "UTA": "ARI"}
        alt = abbrev_map.get(team_abbrev.upper())
        if alt:
            row = mp_df[mp_df[team_col].str.upper() == alt]
    if row.empty:
        return {}

    r = row.iloc[0]
    features = {}
    col_map = {
        "xgf_pct": ["xGoalsPercentage", "xGF%", "xGoalsForPercentage"],
        "xgf_per60": ["xGoalsFor", "xGF/60", "xGoalsForPer60"],
        "xga_per60": ["xGoalsAgainst", "xGA/60", "xGoalsAgainstPer60"],
        "cf_pct": ["corsiPercentage", "CF%", "corsiForPercentage"],
        "pdo": ["pdo", "PDO"],
    }
    for feature_key, possible_cols in col_map.items():
        for c in possible_cols:
            if c in r.index:
                try:
                    val = float(r[c])
                    # Normalize percentages to 0-1 if stored as 0-100
                    if feature_key.endswith("pct") and val > 1.5:
                        val = val / 100.0
                    features[feature_key] = val
                    break
                except (ValueError, TypeError):
                    pass
    return features


def extract_goalie_gsax(mp_goalie_df: pd.DataFrame, goalie_name: str) -> float:
    """
    Looks up a goalie's GSAx from MoneyPuck goalie DataFrame.
    Returns 0.0 if not found.
    """
    if mp_goalie_df is None or mp_goalie_df.empty:
        return 0.0

    name_col = None
    for col in ["name", "playerName", "player"]:
        if col in mp_goalie_df.columns:
            name_col = col
            break
    if name_col is None:
        return 0.0

    # Try to match last name
    last_name = goalie_name.split()[-1].lower()
    matches = mp_goalie_df[mp_goalie_df[name_col].str.lower().str.contains(last_name, na=False)]
    if matches.empty:
        return 0.0

    for col in ["gsax", "GSAx", "goalsAboveExpected", "goalsAllowedAboveExpected"]:
        if col in matches.columns:
            try:
                return float(matches.iloc[0][col])
            except (ValueError, TypeError):
                pass
    return 0.0
