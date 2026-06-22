"""
xG Chain & Shot Quality Analysis Module
========================================
Analyzes how opponents create chances:
- xG origin breakdown (open play, cross, set piece, fast break, through ball)
- Inside vs outside box distribution
- xGOT (Expected Goals on Target) from Goal Mouth coordinates
- Shot quality metrics (xG/shot, SoT%, conversion, finishing quality)
- Match-by-match and season-aggregate xG profiling
"""

import pandas as pd
import numpy as np
import os
import importlib.util

from utils.data import get_data_dir

# Import predict_xg from the MAIN app's utils (not the hub's utils)
# We use importlib to avoid namespace conflicts between hub/utils and main/utils
def _import_predict_xg():
    main_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xg_model_path = os.path.join(main_app_dir, 'utils', 'xg_model.py')
    if not os.path.exists(xg_model_path):
        return lambda df: df
    try:
        spec = importlib.util.spec_from_file_location('main_xg_model', xg_model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.predict_xg
    except Exception:
        return lambda df: df

predict_xg = _import_predict_xg()

GOZTEPE = 'Göztepe Spor Kulübü'

# Shot event types
SHOT_EVENTS = {'Goal', 'Miss', 'Saved Shot', 'Post'}
SHOT_TYPE_IDS = [13, 14, 15, 16]  # Miss, Post, Saved Shot, Goal
ON_TARGET_TYPE_IDS = [15, 16]      # Saved Shot, Goal

# Zone classification columns
INSIDE_BOX_COLS = [
    'Small box-centre', 'Small box-right', 'Small box-left',
    'Box-centre', 'Box-right', 'Box-left',
    'Box-deep right', 'Box-deep left',
]
OUTSIDE_BOX_COLS = [
    'Out of box-centre', 'Out of box-right', 'Out of box-left',
    'Out of box-deep right', 'Out of box-deep left',
    '35+ centre', '35+ right', '35+ left',
]

# Cache
_OPPONENT_MATCH_CACHE = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _safe_div(a, b, decimals=2):
    if b == 0:
        return 0
    return round(a / b, decimals)


def _safe_pct(a, b, decimals=1):
    if b == 0:
        return 0.0
    return round((a / b) * 100, decimals)


def _parse_opta_bool(val):
    """Parse Opta qualifier boolean: 'Si' = True, else False."""
    if pd.isna(val) or val == 0 or val == '0' or val is False:
        return False
    return str(val).strip().lower() in ('si', '1', 'true', 'yes')


def _load_opponent_matches(team_name):
    """Load all parquet files where team plays, EXCLUDING Göztepe matches."""
    if team_name in _OPPONENT_MATCH_CACHE:
        return _OPPONENT_MATCH_CACHE[team_name]

    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    match_dfs = []
    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' not in df.columns:
                continue
            teams = df['team_name'].unique().tolist()
            if team_name in teams and GOZTEPE not in teams:
                # Apply xG model at load time
                try:
                    df = predict_xg(df)
                except Exception:
                    pass
                match_dfs.append((filename, df))
        except Exception:
            continue

    _OPPONENT_MATCH_CACHE[team_name] = match_dfs
    return match_dfs


# ============================================================
# SHOT ZONE CLASSIFICATION
# ============================================================

def _classify_shot_zone(row):
    """
    Classify a shot as 'inside_box' or 'outside_box' using Opta zone qualifiers.
    Falls back to x-coordinate if no qualifier is set.
    """
    for col in INSIDE_BOX_COLS:
        if col in row.index and _parse_opta_bool(row.get(col)):
            return 'inside_box'

    for col in OUTSIDE_BOX_COLS:
        if col in row.index and _parse_opta_bool(row.get(col)):
            return 'outside_box'

    # Fallback: use x coordinate (penalty area starts at ~83 in 0-100 scale)
    x = row.get('x', 0)
    y = row.get('y', 50)
    if x >= 83 and 21.1 <= y <= 78.9:
        return 'inside_box'
    return 'outside_box'


# ============================================================
# SHOT ORIGIN CLASSIFICATION
# ============================================================

def _classify_shot_origin(shot_row, df):
    """
    Classify the origin of a shot:
    - 'from_cross': assist was a cross
    - 'set_piece': from corner, free kick, set piece, throw-in set piece
    - 'fast_break': counter-attack
    - 'through_ball': assisted by a through ball
    - 'open_play': regular play (default)
    
    Priority: set_piece > fast_break > from_cross > through_ball > open_play
    """
    # 1. Set piece check (highest priority)
    is_corner = _parse_opta_bool(shot_row.get('From corner'))
    is_fk = _parse_opta_bool(shot_row.get('Free kick'))
    is_sp = _parse_opta_bool(shot_row.get('Set piece'))
    is_throw = _parse_opta_bool(shot_row.get('Throw In set piece'))

    if is_corner or is_fk or is_sp or is_throw:
        return 'set_piece'

    # 2. Fast break
    is_fb = _parse_opta_bool(shot_row.get('Fast break'))
    if is_fb:
        return 'fast_break'

    # 3. Trace back to assist via Related event ID
    related_id = shot_row.get('Related event ID')
    if pd.notna(related_id):
        try:
            related_id = int(float(related_id))
            related_events = df[df['event_id'] == related_id]
            if not related_events.empty:
                assist_row = related_events.iloc[0]
                # Check if assist was a cross
                if _parse_opta_bool(assist_row.get('Cross')):
                    return 'from_cross'
                # Check if assist was a through ball
                if _parse_opta_bool(assist_row.get('Through ball')):
                    return 'through_ball'
        except (ValueError, TypeError):
            pass

    # 4. Default: open play
    return 'open_play'


# ============================================================
# xGOT CALCULATION (Geometric Model)
# ============================================================

def calculate_xgot(goal_mouth_y, goal_mouth_z):
    """
    Calculate Expected Goals on Target (xGOT) from Goal Mouth coordinates.
    
    Uses a geometric model based on the placement of the shot relative
    to the goal frame. Shots placed in the corners are harder to save
    and thus have higher xGOT.
    
    Goal Mouth Y: 0-100 (left post to right post) → 7.32m
    Goal Mouth Z: 0-100 (ground to crossbar) → 2.44m
    
    Returns xGOT value between 0 and 1.
    """
    if pd.isna(goal_mouth_y) or pd.isna(goal_mouth_z):
        return None

    try:
        gm_y = float(goal_mouth_y)
        gm_z = float(goal_mouth_z)
    except (ValueError, TypeError):
        return None

    # Convert to meters (goal is 7.32m wide, 2.44m tall)
    y_m = (gm_y / 100) * 7.32  # 0 = left post, 7.32 = right post
    z_m = (gm_z / 100) * 2.44  # 0 = ground, 2.44 = crossbar

    # Goalkeeper's optimal position (center of goal, slightly off ground)
    gk_y = 3.66   # center of goal
    gk_z = 0.8    # keeper's hands height when ready

    # Goalkeeper reach (approximate)
    gk_reach_y = 2.5   # horizontal dive reach
    gk_reach_z = 2.2   # vertical reach

    # Normalized distance from goalkeeper's optimal position
    dy = abs(y_m - gk_y) / gk_reach_y
    dz = abs(z_m - gk_z) / gk_reach_z

    # Combined difficulty for keeper (higher = harder to save = higher xGOT)
    difficulty = np.sqrt(dy**2 + dz**2)

    # Convert to probability using sigmoid-like function
    # Calibrated so center shots ≈ 0.15-0.25, corner shots ≈ 0.65-0.85
    xgot = 1 / (1 + np.exp(-2.5 * (difficulty - 0.8)))

    # Boost for very top corners (nearly unsaveable)
    if z_m > 2.0 and (y_m < 1.0 or y_m > 6.32):
        xgot = max(xgot, 0.85)

    # High shots near crossbar
    if z_m > 2.2:
        xgot = min(xgot * 1.15, 0.95)

    # Ground-level corners (low driven shots to posts)
    if z_m < 0.5 and (y_m < 0.8 or y_m > 6.52):
        xgot = max(xgot, 0.70)

    return round(min(max(xgot, 0.03), 0.95), 3)


# ============================================================
# SINGLE MATCH xG ANALYSIS
# ============================================================

def _analyze_match_shots(df, team_name):
    """
    Analyze all shots for a team in a single match dataframe.
    Returns a dict with shot-level and aggregate metrics.
    """
    # Apply xG model if not already present
    if 'xG' not in df.columns or df['xG'].sum() == 0:
        try:
            df = predict_xg(df)
        except Exception:
            pass

    team_df = df[df['team_name'] == team_name]

    # Get shots (exclude own goals)
    has_og = 'own goal' in team_df.columns
    shot_mask = team_df['event'].isin(SHOT_EVENTS)
    if has_og:
        shot_mask = shot_mask & (team_df['own goal'] != 'Si')

    shots = team_df[shot_mask].copy()

    if shots.empty:
        return None

    total_shots = len(shots)

    # On-target: Goal + Saved Shot
    on_target = shots[shots['event'].isin(['Goal', 'Saved Shot'])]
    total_on_target = len(on_target)

    # Goals
    goals = shots[shots['event'] == 'Goal']
    total_goals = len(goals)

    # xG
    total_xg = round(shots['xG'].sum(), 3) if 'xG' in shots.columns else 0

    # xGOT (only for on-target shots)
    xgot_values = []
    for _, row in on_target.iterrows():
        xgot = calculate_xgot(row.get('Goal Mouth Y Coordinate'), row.get('Goal Mouth Z Coordinate'))
        if xgot is not None:
            xgot_values.append(xgot)
    total_xgot = round(sum(xgot_values), 3) if xgot_values else 0

    # Shot origin classification
    origins = {'open_play': 0, 'from_cross': 0, 'set_piece': 0, 'fast_break': 0, 'through_ball': 0}
    origin_xg = {'open_play': 0.0, 'from_cross': 0.0, 'set_piece': 0.0, 'fast_break': 0.0, 'through_ball': 0.0}

    for _, row in shots.iterrows():
        origin = _classify_shot_origin(row, df)
        origins[origin] += 1
        if 'xG' in row.index:
            origin_xg[origin] += row.get('xG', 0)

    # Round origin xG values
    for k in origin_xg:
        origin_xg[k] = round(origin_xg[k], 3)

    # Inside vs outside box
    zones = {'inside_box': 0, 'outside_box': 0}
    zone_xg = {'inside_box': 0.0, 'outside_box': 0.0}
    zone_on_target = {'inside_box': 0, 'outside_box': 0}
    zone_goals = {'inside_box': 0, 'outside_box': 0}

    for _, row in shots.iterrows():
        zone = _classify_shot_zone(row)
        zones[zone] += 1
        if 'xG' in row.index:
            zone_xg[zone] += row.get('xG', 0)
        if row['event'] in ('Goal', 'Saved Shot'):
            zone_on_target[zone] += 1
        if row['event'] == 'Goal':
            zone_goals[zone] += 1

    for k in zone_xg:
        zone_xg[k] = round(zone_xg[k], 3)

    # Big chances
    big_chances = sum(1 for _, row in shots.iterrows() if _parse_opta_bool(row.get('Big Chance')))

    # Header shots
    header_shots = sum(1 for _, row in shots.iterrows() if _parse_opta_bool(row.get('Head')))

    # Shot coordinates for visualization
    shot_coords = []
    for _, row in shots.iterrows():
        xgot_val = calculate_xgot(row.get('Goal Mouth Y Coordinate'), row.get('Goal Mouth Z Coordinate'))
        shot_coords.append({
            'x': row['x'],
            'y': row['y'],
            'xG': round(row.get('xG', 0), 3),
            'xGOT': xgot_val,
            'event': row['event'],
            'player': row.get('player_name', ''),
            'minute': int(row.get('time_min', 0)),
            'origin': _classify_shot_origin(row, df),
            'zone': _classify_shot_zone(row),
            'is_header': _parse_opta_bool(row.get('Head')),
        })

    return {
        'total_shots': total_shots,
        'total_on_target': total_on_target,
        'total_goals': total_goals,
        'total_xg': total_xg,
        'total_xgot': total_xgot,
        'big_chances': big_chances,
        'header_shots': header_shots,
        'sot_pct': _safe_pct(total_on_target, total_shots),
        'conversion_pct': _safe_pct(total_goals, total_shots),
        'xg_per_shot': _safe_div(total_xg, total_shots, 3),
        'xgot_xg_ratio': _safe_div(total_xgot, total_xg, 2) if total_xg > 0 else 0,
        'origins': origins,
        'origin_xg': origin_xg,
        'origin_pcts': {k: _safe_pct(v, total_shots) for k, v in origins.items()},
        'zones': zones,
        'zone_xg': zone_xg,
        'zone_on_target': zone_on_target,
        'zone_goals': zone_goals,
        'zone_pcts': {k: _safe_pct(v, total_shots) for k, v in zones.items()},
        'zone_sot_pct': {
            'inside_box': _safe_pct(zone_on_target['inside_box'], zones['inside_box']),
            'outside_box': _safe_pct(zone_on_target['outside_box'], zones['outside_box']),
        },
        'zone_xg_per_shot': {
            'inside_box': _safe_div(zone_xg['inside_box'], zones['inside_box'], 3),
            'outside_box': _safe_div(zone_xg['outside_box'], zones['outside_box'], 3),
        },
        'shot_coords': shot_coords,
    }


# ============================================================
# FULL SEASON OPPONENT xG PROFILE
# ============================================================
