"""
Writes today's NHL predictions to predictions/YYYY-MM-DD.json.
The kalshi-safety service fetches this file via GitHub raw URL to
decide which picks to back on Kalshi.
"""

import json
import os
from datetime import datetime, timezone

PREDICTIONS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "predictions"
)

MIN_PROB = float(os.environ.get("KALSHI_MIN_PROB", "0.58"))


def _confidence_tier(prob: float) -> str:
    p = max(prob, 1 - prob)
    if p >= 0.72:
        return "extreme"
    if p >= 0.67:
        return "high"
    if p >= 0.62:
        return "medium"
    if p >= 0.57:
        return "low"
    return "none"


def write_predictions_file(game_date: str, predictions: list) -> str:
    """
    Writes predictions for game_date (YYYY-MM-DD) to predictions/<date>.json.
    Each prediction dict is expected to have the shape produced by predictor.predict_games().
    Returns the absolute path to the written file.
    """
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = os.path.join(PREDICTIONS_DIR, f"{game_date}.json")

    picks = []
    for p in predictions:
        home = p.get("home_team", "")
        away = p.get("away_team", "")
        home_prob = float(p.get("home_prob", 0.0))
        away_prob = float(p.get("away_prob", 1.0 - home_prob))
        favored_home = home_prob >= away_prob
        model_prob = max(home_prob, away_prob)
        if model_prob < MIN_PROB:
            continue

        picked_team = p.get("pick_team") or (home if favored_home else away)
        picked_side = "home" if picked_team == home else "away"

        vegas_home = p.get("vegas_implied_home")
        vegas_prob = None
        if vegas_home is not None:
            vegas_prob = float(vegas_home) if picked_side == "home" else 1.0 - float(vegas_home)

        edge = p.get("edge")
        pick = {
            "gameId": f"nhl-{game_date}-{away}-{home}",
            "home": home,
            "away": away,
            "pickedTeam": picked_team,
            "pickedSide": picked_side,
            "modelProb": round(model_prob, 4),
            "confidenceTier": _confidence_tier(model_prob),
            "extra": {
                "homeName": p.get("home_name"),
                "awayName": p.get("away_name"),
                "tier": p.get("tier"),
                "modelUsed": p.get("model_used"),
                "recommendBet": p.get("recommend_bet", False),
                "gameTimeUtc": p.get("game_time_utc"),
            },
        }
        if p.get("game_time_utc"):
            pick["startTime"] = p["game_time_utc"]
        if vegas_prob is not None:
            pick["vegasProb"] = round(vegas_prob, 4)
        if edge is not None:
            pick["edge"] = round(float(edge), 4)
        picks.append(pick)

    out = {
        "sport": "NHL",
        "date": game_date,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "picks": picks,
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    return path
