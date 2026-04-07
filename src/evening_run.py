"""
Evening run — executes at 11:30pm CST via GitHub Actions (after all games end).
Fetches game results, compares to predictions, sends recap, updates Elo ratings.
"""

import os
import sys
import json
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))
import nhl_api
import discord_notifier as discord
import elo_system

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HISTORY_FILE = os.path.join(DATA_DIR, "prediction_history.json")


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {"predictions": [], "season_record": {
        "total": 0, "correct": 0,
        "high_conv_total": 0, "high_conv_correct": 0,
        "rec_total": 0, "rec_correct": 0,
    }}


def save_history(history: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def main():
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"[evening_run] Starting for {today_str}")

    history = load_history()

    # Find today's predictions
    today_entry = None
    for entry in history.get("predictions", []):
        if entry["date"] == today_str:
            today_entry = entry
            break

    if today_entry is None:
        print("[evening_run] No predictions found for today. Skipping recap.")
        return

    if today_entry.get("results_recorded"):
        print("[evening_run] Results already recorded for today. Skipping.")
        return

    # Fetch actual game results
    print("[evening_run] Fetching game results...")
    results = nhl_api.get_scoreboard(today_str)

    # Filter to completed games only
    completed = [
        r for r in results
        if r.get("gameState") in ("OFF", "FINAL", "CRIT")
    ]

    if not completed:
        print("[evening_run] No completed games found yet. Will retry next run.")
        return

    print(f"[evening_run] {len(completed)} completed games")

    # Match predictions to results and update accuracy
    result_map = {}
    for r in completed:
        home = r.get("homeTeam", {}).get("abbrev", "")
        away = r.get("awayTeam", {}).get("abbrev", "")
        if home and away:
            result_map[f"{home}|{away}"] = r

    season_record = history.get("season_record", {
        "total": 0, "correct": 0,
        "high_conv_total": 0, "high_conv_correct": 0,
        "rec_total": 0, "rec_correct": 0,
    })

    # Load Elo ratings for updates
    elo_ratings = elo_system.load_ratings()

    for pred in today_entry.get("predictions", []):
        key = f"{pred['home_team']}|{pred['away_team']}"
        res = result_map.get(key)
        if not res:
            continue

        h_score = res.get("homeTeam", {}).get("score", 0)
        a_score = res.get("awayTeam", {}).get("score", 0)
        actual_winner = pred["home_team"] if h_score > a_score else pred["away_team"]
        correct = actual_winner == pred["pick_team"]

        pred["actual_home_score"] = h_score
        pred["actual_away_score"] = a_score
        pred["actual_winner"] = actual_winner
        pred["correct"] = correct

        # Update season record
        season_record["total"] = season_record.get("total", 0) + 1
        if correct:
            season_record["correct"] = season_record.get("correct", 0) + 1

        if pred.get("pick_prob", 0) >= 0.63:
            season_record["high_conv_total"] = season_record.get("high_conv_total", 0) + 1
            if correct:
                season_record["high_conv_correct"] = season_record.get("high_conv_correct", 0) + 1

        if pred.get("recommend_bet"):
            season_record["rec_total"] = season_record.get("rec_total", 0) + 1
            if correct:
                season_record["rec_correct"] = season_record.get("rec_correct", 0) + 1

        # Update Elo after game
        period = res.get("periodDescriptor", {})
        went_ot = period.get("number", 3) > 3
        margin = abs(h_score - a_score)
        home_won = h_score > a_score
        elo_ratings = elo_system.update_elo(
            elo_ratings, pred["home_team"], pred["away_team"],
            home_won, margin, went_ot
        )

    today_entry["results_recorded"] = True
    history["season_record"] = season_record

    save_history(history)
    elo_system.save_ratings(elo_ratings)

    # Send Discord recap
    discord.send_evening_recap(
        today_entry.get("predictions", []),
        completed,
        season_record,
    )
    print("[evening_run] Evening recap sent and data saved.")


if __name__ == "__main__":
    main()
