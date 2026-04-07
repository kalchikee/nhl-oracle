"""MoneyPuck data downloader — xG, GSAx, team shot quality metrics."""

import io
import requests
import pandas as pd
from typing import Optional

HEADERS = {"User-Agent": "NHL-Oracle/4.0"}

MP_TEAMS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv"
MP_GOALIES_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/goalies.csv"

# NHL API abbrev → all MoneyPuck team code variants (dot notation used in older seasons)
ABBREV_VARIANTS = {
    "TBL": ["TBL", "T.B"],
    "NJD": ["NJD", "N.J"],
    "LAK": ["LAK", "L.A"],
    "SJS": ["SJS", "S.J"],
    "CBJ": ["CBJ", "CLS"],
    "UTA": ["UTA", "ARI", "PHX"],
    "ARI": ["ARI", "PHX", "UTA"],
    "NYR": ["NYR", "N.Y.R"],
    "NYI": ["NYI", "N.Y.I"],
}


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
    Downloads MoneyPuck team season summary. Falls back to prior year if current
    season CSV not yet published (happens mid-season).
    """
    df = _fetch_csv(MP_TEAMS_URL.format(year=year))
    if df is None or df.empty:
        # Try previous season as fallback
        df = _fetch_csv(MP_TEAMS_URL.format(year=year - 1))
        if df is not None:
            print(f"[MoneyPuck] Using {year-1} season data as fallback for {year}")
    if df is None:
        return None
    if "situation" in df.columns:
        df = df[df["situation"] == "all"].copy()
    # Normalise team column to uppercase for matching
    for col in ["team", "teamCode", "name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
            break
    return df


def get_goalie_stats(year: int) -> Optional[pd.DataFrame]:
    """Downloads MoneyPuck goalie season summary, falls back to prior year."""
    df = _fetch_csv(MP_GOALIES_URL.format(year=year))
    if df is None or df.empty:
        df = _fetch_csv(MP_GOALIES_URL.format(year=year - 1))
    if df is None:
        return None
    if "situation" in df.columns:
        df = df[df["situation"] == "all"].copy()
    return df


def get_multi_year_team_stats(years: list) -> pd.DataFrame:
    frames = []
    for y in years:
        df = get_team_stats(y)
        if df is not None:
            df["season_year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_multi_year_goalie_stats(years: list) -> pd.DataFrame:
    frames = []
    for y in years:
        df = get_goalie_stats(y)
        if df is not None:
            df["season_year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _find_team_row(mp_df: pd.DataFrame, team_abbrev: str, team_col: str) -> pd.DataFrame:
    """Finds a team row trying all known abbreviation variants."""
    abbrev_upper = team_abbrev.upper()
    variants = ABBREV_VARIANTS.get(abbrev_upper, [abbrev_upper])
    if abbrev_upper not in variants:
        variants = [abbrev_upper] + variants

    for v in variants:
        row = mp_df[mp_df[team_col] == v.upper()]
        if not row.empty:
            return row
    return pd.DataFrame()


def extract_team_xg_features(mp_df: pd.DataFrame, team_abbrev: str) -> dict:
    """
    Extracts xG-based features for a team from MoneyPuck DataFrame.
    Handles dot-notation abbreviations and computes PDO from raw columns if needed.
    """
    if mp_df is None or mp_df.empty:
        return {}

    team_col = next((c for c in ["team", "teamCode", "name"] if c in mp_df.columns), None)
    if team_col is None:
        return {}

    row = _find_team_row(mp_df, team_abbrev, team_col)
    if row.empty:
        return {}

    r = row.iloc[0]

    def _get(*cols, normalize_pct=False):
        for c in cols:
            if c in r.index:
                try:
                    v = float(r[c])
                    if normalize_pct and v > 1.5:
                        v /= 100.0
                    return v
                except (ValueError, TypeError):
                    pass
        return None

    features = {}

    # xGF%
    v = _get("xGoalsPercentage", "xGF%", "xGoalsForPercentage", normalize_pct=True)
    if v is not None:
        features["xgf_pct"] = v

    # xGF/60 and xGA/60
    v = _get("xGoalsFor", "xGF/60", "xGoalsForPer60")
    if v is not None:
        features["xgf_per60"] = v
    v = _get("xGoalsAgainst", "xGA/60", "xGoalsAgainstPer60")
    if v is not None:
        features["xga_per60"] = v

    # Corsi%
    v = _get("corsiPercentage", "CF%", "corsiForPercentage", normalize_pct=True)
    if v is not None:
        features["cf_pct"] = v

    # PDO — try direct column first, then calculate from shots/goals
    v = _get("pdo", "PDO")
    if v is not None:
        if v > 2:
            v /= 100.0
        features["pdo"] = v
    else:
        # Calculate: SH% + SV%
        # SH% = goalsFor / shotsOnGoalFor
        # SV% = 1 - goalsAgainst / shotsOnGoalAgainst
        gf = _get("goalsFor")
        sog_for = _get("shotsOnGoalFor")
        ga = _get("goalsAgainst")
        sog_against = _get("shotsOnGoalAgainst")
        if gf is not None and sog_for and sog_against:
            sh_pct = gf / sog_for if sog_for > 0 else 0.08
            sv_pct = 1.0 - (ga / sog_against) if sog_against > 0 else 0.91
            features["pdo"] = sh_pct + sv_pct  # ~1.0 at league average

    # PP% and PK% from MoneyPuck if available
    v = _get("powerPlayPct", "ppPct", "powerPlayPercentage", normalize_pct=True)
    if v is not None:
        features["pp_pct"] = v
    v = _get("penaltyKillPct", "pkPct", "penaltyKillPercentage", normalize_pct=True)
    if v is not None:
        features["pk_pct"] = v

    return features


def extract_goalie_gsax(mp_goalie_df: pd.DataFrame, goalie_name: str) -> float:
    """
    Looks up a goalie's GSAx from MoneyPuck goalie DataFrame.
    Tries full name, last name, and first-initial.last pattern.
    """
    if mp_goalie_df is None or mp_goalie_df.empty or not goalie_name:
        return 0.0

    name_col = next((c for c in ["name", "playerName", "player"] if c in mp_goalie_df.columns), None)
    if name_col is None:
        return 0.0

    def _gsax(matches):
        if matches.empty:
            return None
        for col in ["gsax", "GSAx", "goalsAboveExpected", "goalsAllowedAboveExpected",
                    "goalsAllowedAboveExpectedPer60"]:
            if col in matches.columns:
                try:
                    return float(matches.iloc[0][col])
                except (ValueError, TypeError):
                    pass
        return None

    name_lower = goalie_name.lower().strip()
    col_lower = mp_goalie_df[name_col].astype(str).str.lower()

    # 1. Full name exact match
    result = _gsax(mp_goalie_df[col_lower == name_lower])
    if result is not None:
        return result

    # 2. Last name match
    last = name_lower.split()[-1] if name_lower else ""
    result = _gsax(mp_goalie_df[col_lower.str.contains(last, na=False, regex=False)])
    if result is not None:
        return result

    # 3. First initial + last name (e.g. "F.Andersen")
    parts = name_lower.split()
    if len(parts) >= 2:
        initials = f"{parts[0][0]}.{parts[-1]}"
        result = _gsax(mp_goalie_df[col_lower.str.contains(initials, na=False, regex=False)])
        if result is not None:
            return result

    return 0.0
