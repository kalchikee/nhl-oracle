"""Discord notification system — formats and sends messages via webhook embeds."""

import os
import requests
from datetime import datetime
from typing import Optional

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Discord embed colors
COLOR_BLUE   = 0x1E90FF  # morning briefing
COLOR_GREEN  = 0x2ECC71  # good recap day
COLOR_RED    = 0xE74C3C  # bad recap day
COLOR_GOLD   = 0xF1C40F  # season messages


def _send_embed(embeds: list) -> bool:
    """Posts one or more embed objects to the Discord webhook.
    Returns True on success, False on failure. Raises on missing webhook
    so GitHub Actions can catch configuration errors."""
    if not WEBHOOK_URL:
        # Previously silent — now raise so GH Actions marks the run as failed
        raise RuntimeError("DISCORD_WEBHOOK_URL not configured")
    try:
        r = requests.post(WEBHOOK_URL, json={"embeds": embeds}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as ex:
        # Print AND re-raise so workflow fails visibly
        print(f"[discord] Failed to send embed: {ex}", flush=True)
        raise


def _field(name: str, value: str, inline: bool = False) -> dict:
    return {"name": name, "value": value, "inline": inline}


def _american_odds_str(ml: Optional[int]) -> str:
    if ml is None:
        return "N/A"
    return f"+{ml}" if ml > 0 else str(ml)


def send_morning_briefing(predictions: list, season_record: dict):
    """Sends the morning picks briefing as a Discord embed."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    n_games = len(predictions)
    recommended = [p for p in predictions if p.get("recommend_bet")]
    high_conv = [p for p in predictions if p["pick_prob"] >= 0.63]

    total = season_record.get("total", 0)
    correct = season_record.get("correct", 0)
    season_str = (f"Season {correct}-{total - correct}"
                  if total > 0 else "Season starting")

    # --- All games field ---
    if predictions:
        game_lines = []
        for p in predictions:
            home, away = p["home_team"], p["away_team"]
            pick = p["pick_team"]
            pct = f"{p['pick_prob']*100:.1f}%"
            mc = p.get("mc", {})
            proj = mc.get("most_likely_score")
            score_str = f" *({proj[0]}-{proj[1]})*" if proj else ""

            flags = []
            if p.get("b2b_home") and pick == home:
                flags.append("⚠️B2B")
            if p.get("b2b_away") and pick == away:
                flags.append("⚠️B2B")
            inj = p.get("injuries", {})
            if inj.get("home", {}).get("n_injured", 0):
                flags.append(f"🏥{home}:{inj['home']['n_injured']}")
            if inj.get("away", {}).get("n_injured", 0):
                flags.append(f"🏥{away}:{inj['away']['n_injured']}")
            flag_str = " " + " ".join(flags) if flags else ""

            tier_emoji = p.get("tier_emoji", "")
            game_lines.append(
                f"**{away} @ {home}** → {pick} {pct}{score_str} {tier_emoji}{flag_str}"
            )
        games_value = "\n".join(game_lines)
    else:
        games_value = "_No games scheduled today._"

    # --- Recommended bets field ---
    if recommended:
        bet_lines = []
        for p in recommended:
            home, away = p["home_team"], p["away_team"]
            pick = p["pick_team"]
            odds = p.get("odds", {})
            ml_key = "home_ml" if pick == home else "away_ml"
            ml_str = f" ({_american_odds_str(odds[ml_key])})" if odds.get(ml_key) else ""
            edge_str = f" · edge +{p['edge']*100:.1f}%" if p.get("edge", 0) > 0 else ""
            goalies = p.get("goalies", {})
            goalie = goalies.get("home") if pick == home else goalies.get("away")
            goalie_str = f" · G: {goalie}" if goalie else ""
            bet_lines.append(
                f"💰 **{pick}** ML{ml_str} — {p['pick_prob']*100:.1f}%{edge_str}{goalie_str} · {p['tier_emoji']} {p['tier']}"
            )
        bets_value = "\n".join(bet_lines)
    else:
        bets_value = "No games cleared the 63% confidence threshold today"

    embed = {
        "title": f"🏒 NHL Oracle — {date_str}",
        "description": (
            f"**{n_games} game{'s' if n_games != 1 else ''}** · "
            f"**{len(high_conv)}** high-conviction (63%+) · {season_str}"
        ),
        "color": COLOR_BLUE,
        "fields": [
            _field(f"📋 All Games ({n_games} total)", games_value),
            _field("🎯 Recommended Bets Today", bets_value),
        ],
        "footer": {"text": "NHL Oracle v4.0 · Monte Carlo 10,000 simulations"},
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    _send_embed([embed])


def send_evening_recap(predictions: list, results: list, season_record: dict):
    """Sends the end-of-day recap as a Discord embed."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    result_by_teams = {}
    for r in results:
        home = r.get("homeTeam", {}).get("abbrev", "")
        away = r.get("awayTeam", {}).get("abbrev", "")
        h_score = r.get("homeTeam", {}).get("score", 0)
        a_score = r.get("awayTeam", {}).get("score", 0)
        if home and away:
            result_by_teams[f"{home}|{away}"] = {
                "home_score": h_score,
                "away_score": a_score,
                "actual_winner": home if h_score > a_score else away,
            }

    daily_correct = 0
    daily_total = 0
    daily_rec_correct = 0
    daily_rec_total = 0
    game_lines = []

    for p in predictions:
        key = f"{p['home_team']}|{p['away_team']}"
        res = result_by_teams.get(key)
        if not res:
            continue

        h_score = res["home_score"]
        a_score = res["away_score"]
        correct = res["actual_winner"] == p["pick_team"]
        emoji = "✅" if correct else "❌"

        game_lines.append(
            f"{emoji} **{p['away_team']} @ {p['home_team']}** "
            f"{h_score}-{a_score} · picked {p['pick_team']} ({p['pick_prob']*100:.0f}%)"
        )

        daily_total += 1
        if correct:
            daily_correct += 1
        if p.get("recommend_bet"):
            daily_rec_total += 1
            if correct:
                daily_rec_correct += 1

    day_acc_str = (f"{daily_correct}/{daily_total} ({daily_correct/daily_total*100:.0f}%)"
                   if daily_total else "0/0")

    total = season_record.get("total", 0)
    correct_s = season_record.get("correct", 0)
    hc_total = season_record.get("high_conv_total", 0)
    hc_correct = season_record.get("high_conv_correct", 0)
    rec_total = season_record.get("rec_total", 0)
    rec_correct = season_record.get("rec_correct", 0)

    color = COLOR_GREEN if daily_total > 0 and daily_correct / daily_total >= 0.5 else COLOR_RED

    season_lines = []
    if total > 0:
        season_lines.append(f"Overall: **{correct_s}-{total - correct_s}** ({correct_s/total*100:.1f}%) · {total} games")
    if hc_total > 0:
        season_lines.append(f"High Conviction: **{hc_correct}/{hc_total}** ({hc_correct/hc_total*100:.1f}%)")
    if rec_total > 0:
        season_lines.append(f"Recommended Bets: **{rec_correct}/{rec_total}** ({rec_correct/rec_total*100:.1f}%)")

    bets_desc = ""
    if daily_rec_total:
        bets_desc = f" · Bets: {daily_rec_correct}/{daily_rec_total}"

    embed = {
        "title": f"🏒 NHL Oracle — {date_str} Recap",
        "description": f"**Today: {day_acc_str} correct**{bets_desc}",
        "color": color,
        "fields": [
            _field(f"📋 Results ({daily_total} games)",
                   "\n".join(game_lines) if game_lines else "_No completed games found._"),
            _field("📈 Season Record",
                   "\n".join(season_lines) if season_lines else "_No season data yet._"),
        ],
        "footer": {"text": "NHL Oracle v4.0 · Monte Carlo 10,000 simulations"},
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    _send_embed([embed])


def send_season_over_message(season_record: dict):
    total = season_record.get("total", 0)
    correct = season_record.get("correct", 0)
    acc_str = f"{correct/total*100:.1f}%" if total > 0 else "N/A"

    embed = {
        "title": "🏆 NHL Oracle | Season Complete",
        "description": (
            "The Stanley Cup has been awarded! The NHL season is over.\n\n"
            "I'll be back with daily predictions when the new season begins in **October**.\n\n"
            f"**Final Record: {correct}/{total} ({acc_str})**\n"
            "See you in October! 🏒"
        ),
        "color": COLOR_GOLD,
        "footer": {"text": "NHL Oracle v4.0"},
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _send_embed([embed])


def send_season_start_message():
    embed = {
        "title": "🏒 NHL Oracle | New Season Starting!",
        "description": (
            "The NHL regular season is back! I'll be sending:\n"
            "• **6:00 AM CST** — Morning picks with predictions\n"
            "• **After games end** — Evening recap with results\n\n"
            "Let's have a great season! Good luck! 🍀"
        ),
        "color": COLOR_BLUE,
        "footer": {"text": "NHL Oracle v4.0"},
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _send_embed([embed])
