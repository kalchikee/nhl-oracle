"""
Morning run — executes at 6am CST via GitHub Actions.
Checks if season is active, generates predictions, sends Discord briefing.
"""

import os
import sys
import json
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))
import nhl_api
import predictor
import discord_notifier as discord
import elo_system

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEASON_STATE_FILE = os.path.join(DATA_DIR, "season_state.json")
HISTORY_FILE = os.path.join(DATA_DIR, "prediction_history.json")
ELO_FILE = os.path.join(DATA_DIR, "elo_ratings.json")

os.makedirs(DATA_DIR, exist_ok=True)


def load_season_state() -> dict:
    if os.path.exists(SEASON_STATE_FILE):
        with open(SEASON_STATE_FILE) as f:
            return json.load(f)
    return {
        "was_active": False,
        "last_game_date": None,
        "season_over_notified": False,
        "season_start_notified": False,
    }


def save_season_state(state: dict):
    with open(SEASON_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {
        "predictions": [],
        "season_record": {
            "total": 0, "correct": 0,
            "high_conv_total": 0, "high_conv_correct": 0,
            "rec_total": 0, "rec_correct": 0,
        },
    }


def save_history(history: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def check_season_active() -> bool:
    """Returns True if there are NHL games scheduled in the next 3 days."""
    games = nhl_api.get_schedule_range(days_ahead=3)
    return len(games) > 0


def main():
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"[morning_run] Starting for {today_str}")

    state = load_season_state()
    history = load_history()

    # Check if season is active
    season_active = check_season_active()

    if season_active:
        # Season just started (after an inactive period)
        if not state.get("was_active") and not state.get("season_start_notified"):
            print("[morning_run] Season starting — sending notification")
            discord.send_season_start_message()
            state["season_start_notified"] = True
            state["season_over_notified"] = False  # Reset for next year

        state["was_active"] = True
        state["last_game_date"] = today_str
        state["season_start_notified"] = True
        save_season_state(state)

    else:
        # No games in the next 3 days
        last = state.get("last_game_date")
        days_since_last = 999
        if last:
            days_since_last = (date.today() - date.fromisoformat(last)).days

        if state.get("was_active") and days_since_last >= 3 and not state.get("season_over_notified"):
            # Season just ended
            print("[morning_run] Season appears to be over — sending notification")
            discord.send_season_over_message(history.get("season_record", {}))
            state["was_active"] = False
            state["season_over_notified"] = True
            state["season_start_notified"] = False
            save_season_state(state)

        print(f"[morning_run] No games in the next 3 days. Last game: {last}. Skipping.")
        return

    # Check if there are games specifically TODAY
    today_games = nhl_api.get_schedule(today_str)
    if not today_games:
        print("[morning_run] No games today specifically. Skipping briefing.")
        return

    print(f"[morning_run] {len(today_games)} games today — generating predictions")

    # Generate predictions
    predictions = predictor.predict_games(today_str)

    if not predictions:
        print("[morning_run] No predictions generated.")
        return

    # Store today's predictions in history (without results yet)
    today_entry = {
        "date": today_str,
        "predictions": predictions,
        "results_recorded": False,
    }

    # Remove any existing entry for today (in case of re-run)
    history["predictions"] = [
        p for p in history.get("predictions", []) if p["date"] != today_str
    ]
    history["predictions"].append(today_entry)
    save_history(history)

    # Send Discord morning briefing
    discord.send_morning_briefing(predictions, history.get("season_record", {}))
    print(f"[morning_run] Morning briefing sent with {len(predictions)} predictions")


if __name__ == "__main__":
    main()
