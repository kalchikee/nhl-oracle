"""Discord notification system — formats and sends messages via webhook."""

import os
import json
import requests
from datetime import datetime, timezone
from typing import Optional

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Discord message length limit
MAX_MSG_LEN = 2000


def _send(content: str):
    """Sends a message to the Discord webhook."""
    if not WEBHOOK_URL:
        print("[discord] No webhook URL configured (DISCORD_WEBHOOK_URL env var).")
        print("[discord] Message preview:")
        print(content)
        return

    # Split into chunks if needed
    chunks = []
    while len(content) > MAX_MSG_LEN:
        split_at = content.rfind("\n", 0, MAX_MSG_LEN)
        if split_at == -1:
            split_at = MAX_MSG_LEN
        chunks.append(content[:split_at])
        content = content[split_at:].lstrip("\n")
    chunks.append(content)

    for chunk in chunks:
        if not chunk.strip():
            continue
        payload = {"content": chunk}
        try:
            r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"[discord] Failed to send message: {e}")



def _american_odds_str(ml: Optional[int]) -> str:
    if ml is None:
        return "N/A"
    return f"+{ml}" if ml > 0 else str(ml)


def _prob_bar(prob: float, width: int = 10) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def send_morning_briefing(predictions: list, season_record: dict):
    """Sends the 6am morning picks briefing to Discord."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    now_str = datetime.now().strftime("%-I:%M %p")
    n_games = len(predictions)
    recommended = [p for p in predictions if p.get("recommend_bet")]
    high_conv = [p for p in predictions if p["pick_prob"] >= 0.63]

    lines = [
        f"🏒 **NHL Oracle** — {date_str}",
        f"**{n_games} game{'s' if n_games != 1 else ''}** · **{len(high_conv)}** high-conviction (63%+) · "
        + (f"Season {season_record.get('correct',0)}-{season_record.get('total',0)-season_record.get('correct',0)}" if season_record.get('total', 0) > 0 else "Season starting"),
        "",
        f"📋 **All Games ({n_games} total)**",
    ]

    if not predictions:
        lines.append("_No games scheduled today._")
    else:
        for p in predictions:
            home = p["home_team"]
            away = p["away_team"]
            mc = p.get("mc", {})
            proj_score = mc.get("most_likely_score")
            pick_prob_pct = f"{p['pick_prob']*100:.1f}%"
            pick = p["pick_team"]

            score_str = f"({proj_score[0]}-{proj_score[1]})" if proj_score else ""
            b2b = " ⚠️B2B" if (p.get("b2b_home") and pick == home) or (p.get("b2b_away") and pick == away) else ""
            lines.append(f"**{away} @ {home}** → {pick} {pick_prob_pct} *{score_str}*{b2b}")

    lines.append("")

    # Recommended bets section
    lines.append(f"🎯 **Recommended Bets Today**")
    if recommended:
        for p in recommended:
            home = p["home_team"]
            away = p["away_team"]
            pick = p["pick_team"]
            edge_str = f" · edge +{p['edge']*100:.1f}%" if p.get("edge", 0) > 0 else ""
            odds = p.get("odds", {})
            ml_key = "home_ml" if pick == home else "away_ml"
            ml_str = f" ({_american_odds_str(odds[ml_key])})" if odds.get(ml_key) else ""
            lines.append(f"💰 **{pick}** ML{ml_str} — {p['pick_prob']*100:.1f}%{edge_str} · {p['tier_emoji']} {p['tier']}")
    else:
        lines.append(f"No games cleared the 63% confidence threshold today")

    lines.append("")
    lines.append(f"NHL Oracle v4.0 | Monte Carlo 10,000 simulations · Today at {now_str}")

    _send("\n".join(lines))


def send_evening_recap(predictions: list, results: list, season_record: dict):
    """Sends the end-of-day recap to Discord after games are complete."""
    today = datetime.now().strftime("%A, %B %-d, %Y")

    # Match predictions to results
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

    lines = [
        f"🏒 **NHL Oracle | Evening Recap** — {today}",
        "",
        "**Today's Results**",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for p in predictions:
        key = f"{p['home_team']}|{p['away_team']}"
        res = result_by_teams.get(key)
        if not res:
            continue

        home_name = p["home_name"] or p["home_team"]
        away_name = p["away_name"] or p["away_team"]
        h_score = res["home_score"]
        a_score = res["away_score"]
        actual_winner = res["actual_winner"]
        predicted_winner = p["pick_team"]
        correct = actual_winner == predicted_winner

        emoji = "✅" if correct else "❌"
        pick_name = home_name if predicted_winner == p["home_team"] else away_name
        lines.append(
            f"{emoji} **{home_name} {h_score}–{a_score} {away_name}** "
            f"(Predicted: {pick_name.split()[-1]} {p['pick_prob']*100:.0f}%)"
        )

        daily_total += 1
        if correct:
            daily_correct += 1
        if p.get("recommend_bet"):
            daily_rec_total += 1
            if correct:
                daily_rec_correct += 1

    if daily_total == 0:
        lines.append("_No completed games with predictions today._")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if daily_total > 0:
        day_acc = daily_correct / daily_total
        lines.append(f"📊 **Today:** {daily_correct}/{daily_total} correct ({day_acc*100:.1f}%)")
    if daily_rec_total > 0:
        rec_acc = daily_rec_correct / daily_rec_total
        lines.append(f"💰 **Recommended Bets Today:** {daily_rec_correct}/{daily_rec_total} ({rec_acc*100:.1f}%)")

    lines.append("")
    total = season_record.get("total", 0)
    correct_s = season_record.get("correct", 0)
    hc_total = season_record.get("high_conv_total", 0)
    hc_correct = season_record.get("high_conv_correct", 0)
    rec_total = season_record.get("rec_total", 0)
    rec_correct = season_record.get("rec_correct", 0)

    lines.append("**Season Statistics**")
    if total > 0:
        lines.append(f"Overall: **{correct_s}/{total}** ({correct_s/total*100:.1f}%)")
    if hc_total > 0:
        lines.append(f"High Conviction: **{hc_correct}/{hc_total}** ({hc_correct/hc_total*100:.1f}%)")
    if rec_total > 0:
        lines.append(f"Recommended Bets: **{rec_correct}/{rec_total}** ({rec_correct/rec_total*100:.1f}%)")

    _send("\n".join(lines))


def send_season_over_message(season_record: dict):
    """Sends a notification that the NHL season has ended."""
    total = season_record.get("total", 0)
    correct = season_record.get("correct", 0)
    acc_str = f"{correct/total*100:.1f}%" if total > 0 else "N/A"

    msg = (
        "🏆 **NHL Oracle | Season Complete**\n\n"
        "The Stanley Cup has been awarded! The NHL season is over.\n\n"
        "I'll be back with daily predictions when the new season begins in **October**.\n\n"
        f"_Final Season Record: {correct}/{total} ({acc_str})_\n"
        "_See you in October! 🏒_"
    )
    _send(msg)


def send_season_start_message():
    """Sends a notification that the new NHL season is starting."""
    msg = (
        "🏒 **NHL Oracle | New Season Starting!**\n\n"
        "The NHL regular season is back! I'll be sending:\n"
        "• **6:00 AM CST** — Morning picks with predictions\n"
        "• **After games end** — Evening recap with accuracy tracking\n\n"
        "_Let's have a great season! Good luck! 🍀_"
    )
    _send(msg)
