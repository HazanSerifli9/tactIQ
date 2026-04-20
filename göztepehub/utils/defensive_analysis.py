import pandas as pd
import numpy as np
import os

from utils.data import get_data_dir

GOZTEPE = 'Göztepe Spor Kulübü'
# Defending team goal is at x=0. Attacking team goal is at x=100.
# Penalty box boundaries for defending team:
DEF_BOX_X = 17.0
DEF_BOX_Y_MIN = 21.1
DEF_BOX_Y_MAX = 78.9

# Zone 14 (roughly outside the penalty box centrally)
Z14_X_MIN = 66.6
Z14_X_MAX = 83.3
Z14_Y_MIN = 33.3
Z14_Y_MAX = 66.6

DEFENSIVE_EVENTS = {'Tackle', 'Interception', 'Clearance', 'Blocked Pass', 'Ball touch', 'Duel', 'Challenge'} 

_OPPONENT_MATCHES_CACHE = {}

def _load_opponent_matches(team_name):
    if team_name in _OPPONENT_MATCHES_CACHE:
        return _OPPONENT_MATCHES_CACHE[team_name]

    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    match_dfs = []
    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' not in df.columns:
                continue
            teams_in_match = df['team_name'].unique().tolist()
            if team_name in teams_in_match and GOZTEPE not in teams_in_match:
                match_dfs.append((filename, df))
        except Exception:
            continue

    _OPPONENT_MATCHES_CACHE[team_name] = match_dfs
    return match_dfs

def _is_in_def_box(x, y):
    return x <= DEF_BOX_X and DEF_BOX_Y_MIN <= y <= DEF_BOX_Y_MAX

def _is_in_zone14(x, y):
    """Note: this is from tracking the OPPONENT's pass end locations"""
    return Z14_X_MIN <= x <= Z14_X_MAX and Z14_Y_MIN <= y <= Z14_Y_MAX

def _get_flank(y):
    if y < 33.3:
        return 'Left'
    elif y <= 66.6:
        return 'Center'
    else:
        return 'Right'

def analyze_defensive_match(df, team_name):
    teams = df['team_name'].unique().tolist()
    opponent = [t for t in teams if t != team_name]
    opp_name = opponent[0] if opponent else 'Unknown'

    # 1. Defansif Aksiyonlar ve Çizgi (Shape & Pressing)
    def_events = df[(df['team_name'] == team_name) & (df['event'].isin(DEFENSIVE_EVENTS))]
    def_actions = []
    aerial_duels_box = []
    
    for _, row in def_events.iterrows():
        x = row.get('x', 0)
        y = row.get('y', 0)
        e_type = row.get('event')
        
        # Valid defensive action (ignore 0,0 anomalies)
        if e_type in ('Tackle', 'Interception', 'Clearance', 'Challenge') and (x > 0 or y > 0):
            def_actions.append({'x': x, 'y': y, 'type': e_type, 'minute': row.get('time_min')})
            
        # Aerial Duels in Box
        if e_type == 'Aerial' and _is_in_def_box(x, y):
            # outcome 1 generally means won the aerial duel in this format
            aerial_duels_box.append(1 if row.get('outcome') == 1 else 0)

    # 2. Zaafiyetler (Vulnerability Map)
    # Rakibin Final 3rd'e başarılı girmesi (Sağ, Sol, Merkez)
    opp_events_pass = df[(df['team_name'] == opp_name) & (df['event'] == 'Pass') & (df['outcome'] == 1)]
    f3_entries = []
    zone14_passes = []
    
    for _, row in opp_events_pass.iterrows():
        start_x = row.get('x', 0)
        end_x = row.get('Pass End X', 0)
        end_y = row.get('Pass End Y', 0)
        
        # Rakip F3'e girdi mi?
        if start_x < 66.6 and end_x >= 66.6:
            f3_entries.append(_get_flank(end_y))
            
        # Zone 14 Control (Rakip Zone 14'e pas attı mı?)
        if _is_in_zone14(end_x, end_y):
            zone14_passes.append(1) # Basarili pas

    # Rakibin başarısız Zone 14 paslarını da sayalım (kontrol oranını bulmak için)
    opp_failed_passes = df[(df['team_name'] == opp_name) & (df['event'] == 'Pass') & (df['outcome'] == 0)]
    for _, row in opp_failed_passes.iterrows():
        if _is_in_zone14(row.get('Pass End X', 0), row.get('Pass End Y', 0)):
            zone14_passes.append(0)

    # 3. Yenen Gollerden Önceki 30 Saniye (Pre-Goal Structure)
    goals = df[(df['team_name'] == opp_name) & (df['event'] == 'Goal')]
    pre_goal_windows = []
    
    for _, goal in goals.iterrows():
        goal_time = goal['time_min'] * 60 + goal['time_sec']
        t_start = max(0, goal_time - 30)
        
        # O 30 saniye dilimindeki aksiyonlar
        window = df[(df['time_min'] * 60 + df['time_sec'] >= t_start) & 
                    (df['time_min'] * 60 + df['time_sec'] <= goal_time)]
        
        team_actions = window[window['team_name'] == team_name]
        opp_actions = window[window['team_name'] == opp_name]
        
        pre_goal_windows.append({
            'goal_min': goal['time_min'],
            'goal_sec': goal['time_sec'],
            'team_def_actions': len(team_actions[team_actions['event'].isin(DEFENSIVE_EVENTS)]),
            'opp_passes': len(opp_actions[opp_actions['event'] == 'Pass']),
            'failed_clearances': len(team_actions[(team_actions['event'] == 'Clearance') & (team_actions['outcome'] == 0)])
        })

    return {
        'def_actions': def_actions,
        'aerial_box': aerial_duels_box,
        'f3_entries': f3_entries,
        'zone14_passes': zone14_passes,
        'pre_goal_windows': pre_goal_windows
    }


def get_opponent_defensive_profile(team_name):
    match_dfs = _load_opponent_matches(team_name)
    if not match_dfs:
        return None

    all_actions = []
    all_aerials = []
    all_f3 = []
    all_zone14 = []
    all_pre_goals = []
    
    for fn, df in match_dfs:
        res = analyze_defensive_match(df, team_name)
        all_actions.extend(res['def_actions'])
        all_aerials.extend(res['aerial_box'])
        all_f3.extend(res['f3_entries'])
        all_zone14.extend(res['zone14_passes'])
        all_pre_goals.extend(res['pre_goal_windows'])

    profile = {}
    
    # Pressing / Line Height
    if all_actions:
        xs = [a['x'] for a in all_actions]
        profile['avg_def_line'] = sum(xs) / len(xs)
        profile['total_def_actions'] = len(all_actions)
        # Sample max 1000 points to keep UI fast
        import random
        profile['heat_coords'] = random.sample(all_actions, min(len(all_actions), 500))
    else:
        profile['avg_def_line'] = 0
        profile['total_def_actions'] = 0
        profile['heat_coords'] = []
        
    # Box Aerials
    if all_aerials:
        profile['box_aerial_win_pct'] = round((sum(all_aerials) / len(all_aerials)) * 100, 1)
        profile['box_aerial_total'] = len(all_aerials)
    else:
        profile['box_aerial_win_pct'] = 0
        profile['box_aerial_total'] = 0
        
    # Flank Vulnerability
    if all_f3:
        profile['f3_entries_total'] = len(all_f3)
        profile['f3_flanks'] = {
            'Left': round((all_f3.count('Left') / len(all_f3)) * 100, 1),
            'Center': round((all_f3.count('Center') / len(all_f3)) * 100, 1),
            'Right': round((all_f3.count('Right') / len(all_f3)) * 100, 1),
        }
    else:
        profile['f3_entries_total'] = 0
        profile['f3_flanks'] = {'Left': 0, 'Center': 0, 'Right': 0}
        
    # Zone 14 Control
    if all_zone14:
        profile['z14_passes_allowed'] = len(all_zone14)
        # Ratio of opponent's successful passes in Zone 14 -> high means BAD control by defending team
        profile['z14_success_allowed_pct'] = round((sum(all_zone14) / len(all_zone14)) * 100, 1)
    else:
        profile['z14_passes_allowed'] = 0
        profile['z14_success_allowed_pct'] = 0
        
    # Pre-Goal Scenarios
    profile['goals_conceded'] = len(all_pre_goals)
    if all_pre_goals:
        avg_actions = sum([g['team_def_actions'] for g in all_pre_goals]) / len(all_pre_goals)
        avg_opp_passes = sum([g['opp_passes'] for g in all_pre_goals]) / len(all_pre_goals)
        errors = sum([g['failed_clearances'] for g in all_pre_goals])
        
        profile['pre_goal_summary'] = {
            'avg_def_actions_30s': round(avg_actions, 1),
            'avg_opp_passes_30s': round(avg_opp_passes, 1),
            'total_failed_clearances': errors
        }
    else:
        profile['pre_goal_summary'] = {
            'avg_def_actions_30s': 0,
            'avg_opp_passes_30s': 0,
            'total_failed_clearances': 0
        }

    return profile
