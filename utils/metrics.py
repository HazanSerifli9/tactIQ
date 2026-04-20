from __future__ import annotations

import pandas as pd
import numpy as np

from shared.constants import (
    ATTACKING_HALF_MIN,
    ATT_THIRD_MIN,
    OWN_HALF_MAX,
    PPDA_DEF_ACTION_MIN,
    PPDA_OPP_HALF_MAX,
    PROGRESSIVE_PASS_RATIO,
    XA_DANGER_RADIUS,
)

# ============================================================
# TACTICAL METRICS & CONSISTENCY SCORE
# ============================================================

_DEF_ACTIONS = ['Tackle', 'Interception', 'Challenge', 'Foul', 'Ball Recovery']
_DEF_ACTIONS_WITH_CLEARANCE = _DEF_ACTIONS + ['Clearance']
_SHOT_EVENTS = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']


def calculate_high_press_percent(df: pd.DataFrame, team_name: str) -> float:
    """
    Percentage of defensive actions that occur in the opponent's half (x > 50).
    Higher = more aggressive high pressing / counter-pressing.
    """
    df_team = df[(df['team_name'] == team_name) & (df['event'].isin(_DEF_ACTIONS))]

    total_def_acts = len(df_team)
    if total_def_acts == 0:
        return 0.0

    high_def_acts = len(df_team[df_team['x'] > ATTACKING_HALF_MIN])
    return round((high_def_acts / total_def_acts) * 100, 1)


def calculate_directness(df: pd.DataFrame, team_name: str) -> float:
    """
    Operational directness: forward distance / total distance of passes.
    Returns a value in [0, 1]; higher = more direct play.
    """
    passes = df[(df['team_name'] == team_name) & (df['event'] == 'Pass')]

    if passes.empty or 'Pass End X' not in passes.columns:
        return 0.0

    dx = passes['Pass End X'] - passes['x']
    dy = passes['Pass End Y'] - passes['y']

    fwd = np.maximum(dx, 0).sum()
    total = np.sqrt(dx**2 + dy**2).sum()

    if total == 0:
        return 0.0

    return round(fwd / total, 2)


def calculate_line_height(df: pd.DataFrame, team_name: str) -> float:
    """
    Average x-coordinate of defensive actions.
    Higher = deeper defensive line.
    """
    df_team = df[
        (df['team_name'] == team_name)
        & (df['event'].isin(_DEF_ACTIONS_WITH_CLEARANCE))
    ]

    if df_team.empty:
        return 50.0

    return round(df_team['x'].mean(), 1)


def calculate_tcs(
    df_match: pd.DataFrame,
    team_name: str,
    df_season: pd.DataFrame | None = None,
) -> dict:
    """
    Tactical Consistency Score (0-100).
    Compares match metrics to season averages.

    Returns:
        dict with keys 'score' (int) and 'details' (per-metric breakdown).
    """
    hp_pct = calculate_high_press_percent(df_match, team_name)
    directness = calculate_directness(df_match, team_name)
    line_h = calculate_line_height(df_match, team_name)

    if df_season is not None and not df_season.empty:
        baseline = {
            "high_press": calculate_high_press_percent(df_season, team_name),
            "directness": calculate_directness(df_season, team_name),
            "line_height": calculate_line_height(df_season, team_name),
        }
    else:
        baseline = {"high_press": 25.0, "directness": 0.35, "line_height": 45.0}

    score_hp = max(0, 100 - abs(hp_pct - baseline['high_press']) * 2.5)
    score_dir = max(0, 100 - abs(directness - baseline['directness']) * 300)
    score_line = max(0, 100 - abs(line_h - baseline['line_height']) * 3.0)

    final_score = (score_hp * 0.4) + (score_dir * 0.3) + (score_line * 0.3)

    return {
        "score": int(final_score),
        "details": {
            "high_press": {"value": hp_pct, "target": baseline['high_press'], "score": score_hp},
            "directness": {"value": directness, "target": baseline['directness'], "score": score_dir},
            "line_height": {"value": line_h, "target": baseline['line_height'], "score": score_line},
        },
    }


# ============================================================
# ADVANCED METRICS (xG, xA, PPDA, Field Tilt, etc.)
# ============================================================

def calculate_xg(df: pd.DataFrame, team_name: str) -> float:
    """
    Total Expected Goals (xG) from the pre-computed XGBoost 'xG' column.
    Only sums shot events to keep consistency across views.
    """
    if 'xG' not in df.columns:
        return 0.0

    team_df = df[(df['team_name'] == team_name) & (df['event'].isin(_SHOT_EVENTS))]
    return round(team_df['xG'].sum(), 2)


def calculate_xa(df: pd.DataFrame, team_name: str) -> float:
    """
    Basic Expected Assists (xA): assigns value to passes ending near goal.
    """
    passes = df[
        (df['team_name'] == team_name)
        & (df['event'] == 'Pass')
        & (df['outcome'] == 1)
    ].copy()

    if passes.empty or 'Pass End X' not in passes.columns:
        return 0.0

    passes['end_dist_to_goal'] = np.sqrt(
        (100 - passes['Pass End X']) ** 2 + (50 - passes['Pass End Y']) ** 2
    )

    dangerous = passes[passes['end_dist_to_goal'] < XA_DANGER_RADIUS].copy()
    if dangerous.empty:
        return 0.0

    dangerous['xa'] = np.exp(-0.2 * dangerous['end_dist_to_goal']) * 0.5
    return round(dangerous['xa'].sum(), 2)


def calculate_ppda(df: pd.DataFrame, team_name: str) -> float:
    """
    Passes Per Defensive Action — measures pressing intensity.
    Lower = more intense press.
    """
    opp_passes = df[
        (df['team_name'] != team_name)
        & (df['event'] == 'Pass')
        & (df['x'] < PPDA_OPP_HALF_MAX)
    ]
    def_actions = df[
        (df['team_name'] == team_name)
        & (df['event'].isin(['Tackle', 'Interception', 'Challenge', 'Foul', 'Ball recovery']))
        & (df['x'] > PPDA_DEF_ACTION_MIN)
    ]

    if len(def_actions) == 0:
        return np.nan

    return round(len(opp_passes) / len(def_actions), 1)


def calculate_field_tilt(df: pd.DataFrame, team_name: str) -> float:
    """
    Share of passes in the final third — proxy for territorial dominance.
    """
    all_f3_passes = df[(df['event'] == 'Pass') & (df['x'] >= ATT_THIRD_MIN)]
    if all_f3_passes.empty:
        return 50.0

    team_f3 = all_f3_passes[all_f3_passes['team_name'] == team_name]
    return round((len(team_f3) / len(all_f3_passes)) * 100, 1)


def calculate_xt(df: pd.DataFrame, team_name: str) -> float:
    """
    Expected Threat proxy: sum of distance-to-goal reductions per successful pass.
    """
    passes = df[
        (df['team_name'] == team_name)
        & (df['event'] == 'Pass')
        & (df['outcome'] == 1)
    ].copy()

    if passes.empty or 'Pass End X' not in passes.columns:
        return 0.0

    passes['start_dist'] = np.sqrt((100 - passes['x']) ** 2 + (50 - passes['y']) ** 2)
    passes['end_dist'] = np.sqrt(
        (100 - passes['Pass End X']) ** 2 + (50 - passes['Pass End Y']) ** 2
    )
    passes['xt_added'] = passes['start_dist'] - passes['end_dist']

    xt = passes[passes['xt_added'] > 0]['xt_added'].sum() / 100.0
    return round(xt, 2)


def calculate_progressive_passes(df: pd.DataFrame, team_name: str) -> int:
    """
    Count of passes that move the ball at least 25% closer to the opponent's goal.
    """
    passes = df[
        (df['team_name'] == team_name)
        & (df['event'] == 'Pass')
        & (df['outcome'] == 1)
    ].copy()

    if passes.empty or 'Pass End X' not in passes.columns:
        return 0

    passes['start_dist'] = np.sqrt((100 - passes['x']) ** 2 + (50 - passes['y']) ** 2)
    passes['end_dist'] = np.sqrt(
        (100 - passes['Pass End X']) ** 2 + (50 - passes['Pass End Y']) ** 2
    )

    progressive = passes[passes['end_dist'] <= passes['start_dist'] * PROGRESSIVE_PASS_RATIO]
    return len(progressive)


def calculate_bdp(df: pd.DataFrame, team_name: str) -> float:
    """
    Build-up Disruption Percentage: 100 - opponent pass completion rate in own half.
    Higher = more pressing disruption.
    """
    opp_passes = df[
        (df['team_name'] != team_name)
        & (df['event'] == 'Pass')
        & (df['x'] < OWN_HALF_MAX)
    ]

    if opp_passes.empty:
        return 0.0

    opp_succ = opp_passes[opp_passes['outcome'] == 1]
    comp_rate = (len(opp_succ) / len(opp_passes)) * 100
    return round(100 - comp_rate, 1)


# ============================================================
# SEPP — Shot-Ending Possession Passes
# ============================================================

def calculate_sepp(df: pd.DataFrame, team_name: str) -> dict:
    """
    Wrapper for SEPP metrics.
    Returns dict with sepp_total, sepp_per_shot, sepp_f3, sepp_prog, etc.
    """
    from utils.sepp_metrics import calculate_sepp as _calc_sepp
    return _calc_sepp(df, team_name)


# ============================================================
# BALL TRACE — Territorial Time Analysis
# ============================================================

def calculate_ball_trace(df: pd.DataFrame, team_name: str) -> dict:
    """
    Wrapper for Ball Trace (territorial time analysis).
    Returns dict with zone_grid, thirds, flanks, territorial_dominance, timeline.
    """
    from utils.ball_trace import calculate_ball_trace as _calc_bt
    return _calc_bt(df, team_name)
