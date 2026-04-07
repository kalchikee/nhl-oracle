"""
Monte Carlo goal simulation.
Runs 10,000 Poisson-distributed game simulations to produce win probabilities,
regulation win %, projected totals, and shutout probability.
"""

import numpy as np
from typing import Optional

LEAGUE_AVG_GOALS = 3.0   # Per team per 60 min (approx NHL average)
OT_LAMBDA_SCALE = 0.30   # 3-on-3 OT has higher pace but is only 5 min
HOME_ADVANTAGE_MULT = 1.04   # ~4% home goal boost

N_SIMULATIONS = 10_000


def _estimate_lambda(
    team_xgf_per60: float,
    opp_xga_per60: float,
    goalie_quality_mult: float = 1.0,
    is_home: bool = False,
    b2b: bool = False,
) -> float:
    """
    Estimates expected goals (lambda) for Poisson simulation.
    Uses team offensive xGF/60 adjusted by opponent defense and goalie quality.
    """
    # Geometric mean of team offense and opponent defense
    league_adj = LEAGUE_AVG_GOALS
    if team_xgf_per60 <= 0:
        team_xgf_per60 = league_adj
    if opp_xga_per60 <= 0:
        opp_xga_per60 = league_adj

    lam = (team_xgf_per60 * opp_xga_per60 / league_adj)

    # Goalie quality (1.0 = average; 0.88 = elite; 1.12 = poor)
    lam *= goalie_quality_mult

    # Home advantage
    if is_home:
        lam *= HOME_ADVANTAGE_MULT

    # Back-to-back fatigue: reduce expected offense by 3%
    if b2b:
        lam *= 0.97

    # Clamp to reasonable range
    return max(0.5, min(lam, 6.0))


def _goalie_quality_multiplier(gsax: float) -> float:
    """
    Converts goalie GSAx to a goals-allowed multiplier.
    Elite: GSAx > 10 → 0.88 (allows 12% fewer goals than expected)
    Average: GSAx ≈ 0 → 1.00
    Poor: GSAx < -10 → 1.12
    """
    # Scale: 10 GSAx ≈ 12% reduction in goals
    mult = 1.0 - (gsax / 80.0)
    return max(0.80, min(mult, 1.20))


def simulate(
    home_team: str,
    away_team: str,
    home_xgf_per60: float = LEAGUE_AVG_GOALS,
    home_xga_per60: float = LEAGUE_AVG_GOALS,
    away_xgf_per60: float = LEAGUE_AVG_GOALS,
    away_xga_per60: float = LEAGUE_AVG_GOALS,
    home_goalie_gsax: float = 0.0,
    away_goalie_gsax: float = 0.0,
    home_b2b: bool = False,
    away_b2b: bool = False,
    n: int = N_SIMULATIONS,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Runs N Monte Carlo simulations and returns probability estimates.

    Returns dict with:
        home_win_pct        — total win probability (regulation + OT + SO)
        home_reg_win_pct    — regulation win probability only
        away_win_pct
        away_reg_win_pct
        avg_total_goals     — average total goals per game
        home_shutout_pct    — probability away team scores 0
        away_shutout_pct    — probability home team scores 0
        most_likely_score   — (home_goals, away_goals) modal outcome
    """
    if rng is None:
        rng = np.random.default_rng()

    h_goalie_mult = _goalie_quality_multiplier(away_goalie_gsax)  # Opponent's goalie defends
    a_goalie_mult = _goalie_quality_multiplier(home_goalie_gsax)

    lam_home = _estimate_lambda(
        home_xgf_per60, away_xga_per60, h_goalie_mult, is_home=True, b2b=home_b2b
    )
    lam_away = _estimate_lambda(
        away_xgf_per60, home_xga_per60, a_goalie_mult, is_home=False, b2b=away_b2b
    )

    # Regulation goals (60 min)
    home_goals = rng.poisson(lam_home, n)
    away_goals = rng.poisson(lam_away, n)

    reg_home_wins = (home_goals > away_goals).sum()
    reg_away_wins = (away_goals > home_goals).sum()
    ties = (home_goals == away_goals).sum()

    # Overtime (3-on-3, 5 min) for tied games
    ot_lam_home = lam_home * OT_LAMBDA_SCALE
    ot_lam_away = lam_away * OT_LAMBDA_SCALE

    ot_home = rng.poisson(ot_lam_home, n)
    ot_away = rng.poisson(ot_lam_away, n)

    # In tied games only
    tie_mask = home_goals == away_goals
    ot_home_wins = ((tie_mask) & (ot_home > ot_away)).sum()
    ot_away_wins = ((tie_mask) & (ot_away > ot_home)).sum()
    still_tied = ((tie_mask) & (ot_home == ot_away)).sum()

    # Shootout: slight home advantage (52%)
    so_home_wins = int(still_tied * 0.52)
    so_away_wins = still_tied - so_home_wins

    total_home_wins = reg_home_wins + ot_home_wins + so_home_wins
    total_away_wins = reg_away_wins + ot_away_wins + so_away_wins

    # Totals
    total_goals = home_goals + away_goals
    avg_total = float(np.mean(total_goals))

    # Most likely score (mode)
    from collections import Counter
    score_counts = Counter(zip(home_goals.tolist(), away_goals.tolist()))
    most_likely = score_counts.most_common(1)[0][0]

    return {
        "home_win_pct": total_home_wins / n,
        "home_reg_win_pct": reg_home_wins / n,
        "away_win_pct": total_away_wins / n,
        "away_reg_win_pct": reg_away_wins / n,
        "avg_total_goals": round(avg_total, 1),
        "home_shutout_pct": float((away_goals == 0).mean()),
        "away_shutout_pct": float((home_goals == 0).mean()),
        "most_likely_score": most_likely,
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
    }
