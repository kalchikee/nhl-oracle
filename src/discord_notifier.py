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


def _format_game_time(utc_str: str) -> str:
    """Converts UTC ISO timestamp to ET time string."""
    if not utc_str:
        return "TBD"
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        # Convert UTC to ET (UTC-5 in winter, UTC-4 in summer)
        from datetime import timedelta
        # Simple offset — April is EDT (UTC-4)
        month = dt.month
        et_offset = -4 if 3 <= month <= 10 else -5
        et = dt + timedelta(hours=et_offset)
        return et.strftime("%I:%M %p ET").lstrip("0")
    except Exception:
        return utc_str[:16] + " UTC"


def _american_odds_str(ml: Optional[int]) -> str:
    if ml is None:
        return "N/A"
    return f"+{ml}" if ml > 0 else str(ml)


def _prob_bar(prob: float, width: int = 10) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def send_morning_briefing(predictions: list, season_record: dict):
    """Sends the 6am morning picks briefing to Discord."""
    today = datetime.now().strftime("%A, %B %-d, %Y")
    n_games = len(predictions)
    recommended = [p for p in predictions if p.get("recommend_bet")]
    high_conv = [p for p in predictions if p["pick_prob"] >= 0.63]

    lines = [
        f"🏒 **NHL Oracle | Morning Picks** — {today}",
        f"**{n_games} game{'s' if n_games != 1 else ''} today** | {len(recommended)} recommended bet{'s' if len(recommended) != 1 else ''} | {len(high_conv)} high conviction",
        "",
    ]

    if not predictions:
        lines.append("_No games scheduled today._")
    else:
        for p in predictions:
            home = p["home_name"] or p["home_team"]
            away = p["away_name"] or p["away_team"]
            game_time = _format_game_time(p.get("game_time_utc", ""))
            pick_team_name = home if p["pick_team"] == p["home_team"] else away
            pick_prob_pct = f"{p['pick_prob']*100:.1f}%"
            mc = p.get("mc", {})
            h_win_pct = mc.get("home_win_pct", p["home_prob"])
            a_win_pct = mc.get("away_win_pct", p["away_prob"])
            proj_score = mc.get("most_likely_score", None)
            avg_total = mc.get("avg_total_goals", None)

            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"**{home}** vs **{away}**")
            lines.append(f"⏰ {game_time}")
            if p.get("b2b_home"):
                lines.append("⚠️ Home team on BACK-TO-BACK")
            if p.get("b2b_away"):
                lines.append("⚠️ Away team on BACK-TO-BACK")
            lines.append("")
            lines.append(f"**Pick: {pick_team_name}** | {p['tier_emoji']} {p['tier']} ({pick_prob_pct})")
            lines.append(
                f"Monte Carlo: {home.split()[-1]} {h_win_pct*100:.1f}% — {away.split()[-1]} {a_win_pct*100:.1f}%"
            )
            if proj_score:
                lines.append(f"Projected: {home.split()[-1]} {proj_score[0]}–{proj_score[1]} {away.split()[-1]}")
            if avg_total:
                lines.append(f"Avg total goals: {avg_total}")

            # Odds / edge info
            odds = p.get("odds", {})
            if odds:
                home_ml_str = _american_odds_str(odds.get("home_ml"))
                away_ml_str = _american_odds_str(odds.get("away_ml"))
                vegas_implied = odds.get("vegas_implied_home", 0)
                edge = p.get("edge", 0)
                lines.append(f"Vegas: {home.split()[-1]} {home_ml_str} / {away.split()[-1]} {away_ml_str} (implied {vegas_implied*100:.1f}%)")
                if abs(edge) >= 0.03:
                    lines.append(f"Model edge: **{edge*100:+.1f}%**")

            if p.get("recommend_bet"):
                pick_name = home if p["pick_team"] == p["home_team"] else away
                lines.append(f"💰 **RECOMMENDED BET: {pick_name} ML**")
            lines.append("")

    # Season stats
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    total = season_record.get("total", 0)
    correct = season_record.get("correct", 0)
    hc_total = season_record.get("high_conv_total", 0)
    hc_correct = season_record.get("high_conv_correct", 0)
    rec_total = season_record.get("rec_total", 0)
    rec_correct = season_record.get("rec_correct", 0)

    if total > 0:
        acc = correct / total
        lines.append(f"📈 **Season:** {correct}/{total} ({acc*100:.1f}%) overall")
    if hc_total > 0:
        hc_acc = hc_correct / hc_total
        lines.append(f"⭐ **High Conviction:** {hc_correct}/{hc_total} ({hc_acc*100:.1f}%)")
    if rec_total > 0:
        rec_acc = rec_correct / rec_total
        lines.append(f"💰 **Recommended Bets:** {rec_correct}/{rec_total} ({rec_acc*100:.1f}%)")

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
