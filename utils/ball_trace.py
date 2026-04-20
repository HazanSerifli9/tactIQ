"""
Ball Trace — Territorial Time Analysis
=========================================
Instead of heat maps or RCS, track where the ball SPENT TIME on the pitch.
Which zone, which flank, how many minutes — because football = goals / time.

Duran top (set piece) süreleri ayrı tutulur.

Calculates:
- 9-zone grid time distribution (3x3: thirds × flanks)
- Third distribution (defensive / middle / attacking)
- Flank distribution (left / center / right)
- Territorial dominance (% time in opponent's half)
- Minute-by-minute timeline of ball position
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Events that signal dead ball / stoppage — exclude from time calculations
DEAD_BALL_EVENTS = {
    'Start', 'End', 'Start delay', 'End delay',
    'Injury Time Announcement', 'Card', 'Player Off', 'Player on',
    'Formation change', 'Collection End', 'Deleted event',
    'Referee Drop Ball', 'Contentious referee decision',
    'Team setp up', 'Referee stop', 'Referee delay',
    'Resume', 'Suspended', 'Game end', 'Post match complete',
}

# Set piece indicators — if event has these, the time gap before it is dead ball
SET_PIECE_QUALIFIERS = [
    'Corner taken', 'Free kick taken', 'Throw In',
    'Goal Kick', 'Throw In set piece',
]

# Zone classification
ZONE_NAMES_X = {0: 'Defensive', 1: 'Middle', 2: 'Attacking'}
ZONE_NAMES_Y = {0: 'Left', 1: 'Center', 2: 'Right'}


def _classify_x_third(x: float) -> int:
    """0 = Defensive, 1 = Middle, 2 = Attacking"""
    if x < 33.33:
        return 0
    elif x < 66.66:
        return 1
    return 2


def _classify_y_flank(y: float) -> int:
    """0 = Left, 1 = Center, 2 = Right"""
    if y < 33.33:
        return 0
    elif y < 66.66:
        return 1
    return 2


def _is_set_piece_event(ev_dict: dict) -> bool:
    """Check if event is a set piece restart."""
    for q in SET_PIECE_QUALIFIERS:
        val = ev_dict.get(q)
        if val is not None and str(val).strip().lower() in ('si', '1', 'true', 'yes'):
            return True
    return False


def _is_dead_ball_restart(event_name: str) -> bool:
    """Check if event signals a dead ball restart (Out, Foul, etc.)."""
    return event_name in {'Out', 'Foul', 'Offside Pass', 'Corner Awarded'}


def calculate_ball_trace(df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
    """
    Calculate ball trace (territorial time analysis) for a team in a single match.
    
    Time calculation: Between two consecutive events by this team,
    the ball is considered to be in the zone of the FIRST event.
    Dead ball periods (fouls → set pieces, out of play) are excluded.
    
    Args:
        df: Match dataframe
        team_name: Team to analyze
        
    Returns:
        dict with zone_grid, thirds, flanks, territorial_dominance, timeline
    """
    if df.empty:
        return _empty_trace()
    
    # Sort chronologically
    df_sorted = df.sort_values(
        by=['period_id', 'time_min', 'time_sec', 'event_id']
    ).reset_index(drop=True)
    
    # Filter only game periods (exclude setup period 16)
    df_sorted = df_sorted[df_sorted['period_id'].isin([1, 2, 3, 4, 5])].copy()
    
    if df_sorted.empty:
        return _empty_trace()
    
    # Initialize 3x3 zone grid (duration in seconds)
    zone_grid = np.zeros((3, 3))  # [x_third][y_flank]
    
    # Also track set piece time separately
    sp_zone_grid = np.zeros((3, 3))
    
    # Timeline: per-minute zone tracking
    # 0-based minute → {zone counts}
    timeline_minutes = {}
    
    # Process events sequentially
    team_events = df_sorted[
        (df_sorted['team_name'] == team_name) & 
        (~df_sorted['event'].isin(DEAD_BALL_EVENTS))
    ].copy()
    
    if team_events.empty:
        return _empty_trace()
    
    # Calculate time between consecutive team events
    team_events = team_events.reset_index(drop=True)
    
    # We need ALL events to detect dead ball gaps
    all_events = df_sorted[~df_sorted['event'].isin(DEAD_BALL_EVENTS)].reset_index(drop=True)
    
    # Build timeline of team possession segments
    total_ball_time = 0.0
    opp_half_time = 0.0
    
    for i in range(len(all_events) - 1):
        curr = all_events.iloc[i]
        next_ev = all_events.iloc[i + 1]
        
        # Only count time for our team's events
        if curr['team_name'] != team_name:
            continue
        
        # Same period check
        if curr['period_id'] != next_ev['period_id']:
            continue
        
        # Calculate time gap
        t1 = float(curr['time_min']) * 60 + float(curr['time_sec'])
        t2 = float(next_ev['time_min']) * 60 + float(next_ev['time_sec'])
        dt = t2 - t1
        
        # Skip if time gap is too large (> 30s = likely dead ball)
        # or negative
        if dt <= 0 or dt > 30:
            continue
        
        # Check if next event is a dead ball restart (exclude this gap)
        if _is_dead_ball_restart(next_ev.get('event', '')):
            continue
        
        # If next event is from opponent and is a set piece restart, skip
        if next_ev['team_name'] != team_name and _is_set_piece_event(next_ev.to_dict()):
            continue
        
        # Zone classification based on current event position
        x = float(curr.get('x', 50))
        y = float(curr.get('y', 50))
        x_third = _classify_x_third(x)
        y_flank = _classify_y_flank(y)
        
        # Check if this is a set piece event
        is_sp = _is_set_piece_event(curr.to_dict())
        
        if is_sp:
            sp_zone_grid[x_third][y_flank] += dt
        else:
            zone_grid[x_third][y_flank] += dt
        
        total_ball_time += dt
        
        # Opponent's half = x > 50 (attacking half)
        if x > 50:
            opp_half_time += dt
        
        # Timeline tracking (which minute)
        minute = int(curr['time_min'])
        if minute not in timeline_minutes:
            timeline_minutes[minute] = {'x_sum': 0, 'y_sum': 0, 'count': 0, 'time': 0}
        timeline_minutes[minute]['x_sum'] += x * dt
        timeline_minutes[minute]['y_sum'] += y * dt
        timeline_minutes[minute]['count'] += 1
        timeline_minutes[minute]['time'] += dt
    
    # Convert to minutes
    zone_grid_min = zone_grid / 60.0
    sp_zone_grid_min = sp_zone_grid / 60.0
    total_ball_time_min = total_ball_time / 60.0
    
    # Third distribution (sum across flanks)
    thirds = {
        'defensive': round(float(zone_grid_min[0].sum()), 2),
        'middle': round(float(zone_grid_min[1].sum()), 2),
        'attacking': round(float(zone_grid_min[2].sum()), 2),
    }
    
    # Flank distribution (sum across thirds)
    flanks = {
        'left': round(float(zone_grid_min[:, 0].sum()), 2),
        'center': round(float(zone_grid_min[:, 1].sum()), 2),
        'right': round(float(zone_grid_min[:, 2].sum()), 2),
    }
    
    # Percentages
    total_open = float(zone_grid_min.sum())
    if total_open > 0:
        thirds_pct = {k: round(v / total_open * 100, 1) for k, v in thirds.items()}
        flanks_pct = {k: round(v / total_open * 100, 1) for k, v in flanks.items()}
    else:
        thirds_pct = {k: 0 for k in thirds}
        flanks_pct = {k: 0 for k in flanks}
    
    # Territorial dominance
    territorial = round((opp_half_time / total_ball_time) * 100, 1) if total_ball_time > 0 else 50.0
    
    # Timeline: average x per minute
    timeline = []
    for minute in sorted(timeline_minutes.keys()):
        md = timeline_minutes[minute]
        if md['time'] > 0:
            avg_x = md['x_sum'] / md['time']
            avg_y = md['y_sum'] / md['time']
        else:
            avg_x, avg_y = 50, 50
        timeline.append({
            'minute': minute,
            'avg_x': round(avg_x, 1),
            'avg_y': round(avg_y, 1),
            'time_seconds': round(md['time'], 1),
            'third': ZONE_NAMES_X[_classify_x_third(avg_x)],
        })
    
    # 9-zone grid with labels for visualization
    zone_details = []
    for xi in range(3):
        for yi in range(3):
            val_min = round(float(zone_grid_min[xi][yi]), 2)
            pct = round(val_min / total_open * 100, 1) if total_open > 0 else 0
            zone_details.append({
                'third': ZONE_NAMES_X[xi],
                'flank': ZONE_NAMES_Y[yi],
                'x_idx': xi,
                'y_idx': yi,
                'minutes': val_min,
                'pct': pct,
            })
    
    return {
        'total_ball_time_min': round(total_ball_time_min, 2),
        'zone_grid': zone_grid_min.tolist(),  # 3x3 array
        'zone_details': zone_details,          # labeled list
        'thirds': thirds,
        'thirds_pct': thirds_pct,
        'flanks': flanks,
        'flanks_pct': flanks_pct,
        'territorial_dominance': territorial,
        'timeline': timeline,
        'set_piece_grid': sp_zone_grid_min.tolist(),
    }


def _empty_trace() -> Dict[str, Any]:
    return {
        'total_ball_time_min': 0,
        'zone_grid': [[0]*3]*3,
        'zone_details': [],
        'thirds': {'defensive': 0, 'middle': 0, 'attacking': 0},
        'thirds_pct': {'defensive': 0, 'middle': 0, 'attacking': 0},
        'flanks': {'left': 0, 'center': 0, 'right': 0},
        'flanks_pct': {'left': 0, 'center': 0, 'right': 0},
        'territorial_dominance': 50.0,
        'timeline': [],
        'set_piece_grid': [[0]*3]*3,
    }
