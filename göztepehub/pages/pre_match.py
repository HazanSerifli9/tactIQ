import dash
import numpy as np
import pandas as pd
import os
import time
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, ALL, ctx
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data, calculate_standings, get_data_dir, TEAM_LOGOS

dash.register_page(__name__, path='/pre-match', title='Göztepe Hub | Pre-Match')

_PITCH_BG = "#0e1b0f"
_GOLD     = "#fbbf24"
_RED      = "#ef4444"
_BLUE     = "#3b82f6"
_PURPLE   = "#a855f7"
_GREEN    = "#22c55e"

GOZTEPE = 'Göztepe Spor Kulübü'

# Pitch & Goal Dimensions
PITCH_LENGTH = 105
PITCH_WIDTH  = 68

GOAL_WIDTH_M  = 7.32     # real goal width in meters
GOAL_DEPTH_M  = 2.0      # goal depth in meters
GOAL_MARGIN_X = 0.2      # separation from goal line

# Opta Coordinates for goal mouth
GOAL_MOUTH_Y_COL = "Goal Mouth Y Coordinate"
GOAL_MOUTH_LEFT_OPT        = 45.2
GOAL_MOUTH_RIGHT_OPT       = 54.8
GOAL_MOUTH_CENTER_OPT      = 50.0
GOAL_MOUTH_HALF_SPAN_OPT   = GOAL_MOUTH_RIGHT_OPT - GOAL_MOUTH_CENTER_OPT

PENALTY_SHOT_EVENTS = {'Goal', 'Saved Shot', 'Miss', 'Post', 'Penalty'}

IGNORED_MACRO_CATEGORIES = {"match_admin", "stoppage_restart", "feed_meta"}
IGNORED_EVENT_NAMES = {"deleted event"}


_RADAR_CACHE = {}
_LEAGUE_CACHE = {'df': None, 'timestamp': 0, 'building': False, 'error': None}
_LEAGUE_CACHE_LOCK = threading.Lock()
_SEASON_STATS_CACHE = {}
_PENALTY_RANK_CACHE = {'data': None, 'timestamp': 0}
_SET_PIECE_GOAL_RANK_CACHE = {'data': None, 'timestamp': 0}

def _load_team_matches(team_name):
    data_dir = get_data_dir()
    match_dfs = []
    try:
        files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    except Exception:
        return match_dfs

    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' in df.columns and team_name in df['team_name'].unique():
                match_dfs.append((filename, df))
        except Exception:
            continue
    return match_dfs

def _parse_opta_bool(val):
    if pd.isna(val) or val == 0 or val == '0' or val is False:
        return False
    return str(val).strip().lower() in ('si', '1', 'true', 'yes')

def _event_seconds(row):
    return float(row.get('time_min') or 0) * 60 + float(row.get('time_sec') or 0)

def _is_penalty_attempt(row):
    if not _parse_opta_bool(row.get('Penalty')):
        return False

    event = row.get('event')
    type_id = row.get('type_id')
    if event in PENALTY_SHOT_EVENTS:
        return True
    return type_id in [13, 14, 15, 16] and event not in {'Foul', 'Penalty faced', 'Save', 'Unknown'}

def _collect_penalty_attempts(records, team_name):
    attempts = []
    seen = set()
    for idx, row in enumerate(records):
        if row.get('team_name') != team_name or not _is_penalty_attempt(row):
            continue

        period = int(row.get('period_id') or 0)
        second_bucket = round(_event_seconds(row))
        player = row.get('player_name') or ''
        key = (period, second_bucket, player)
        if key in seen:
            continue
        seen.add(key)
        attempts.append((idx, row))
    return attempts

def _shot_side(goal_mouth_y):
    y = _safe_float(goal_mouth_y)
    if y is None:
        return 'Unknown'
    if y < 48:
        return 'Left'
    if y > 52:
        return 'Right'
    return 'Centre'

def _get_season_stats(team_name):
    """
    Loads all match data for team_name (excluding Göztepe), computes:
    - 9-zone transitions perf (regains & ball losses)
    - Set-pieces season averages (corners takers, targets splits, goalkicks, penalties)
    - Goal sequence breakdown (goals scored by origin)
    """
    if team_name in _SEASON_STATS_CACHE:
        return _SEASON_STATS_CACHE[team_name]
        
    match_dfs = _load_team_matches(team_name)
    
    all_actions = []
    all_aerials = []
    
    total_penalties = 0
    scored_penalties = 0
    saved_penalties = 0
    missed_penalties = 0
    penalty_placements = []
    penalty_takers = {}
    
    total_goalkicks = 0
    long_goalkicks = 0
    short_goalkicks = 0
    goalkick_destinations = []
    
    total_corners = 0
    left_corners = 0
    right_corners = 0
    inswinger_corners = 0
    outswinger_corners = 0
    straight_corners = 0
    corner_takers = {}
    corner_targets = {}
    total_corner_goals = 0
    
    total_freekicks = 0
    direct_fk_shots = 0
    direct_fk_on_target = 0
    direct_fk_goals = 0
    direct_fk_placements = []
    
    goal_sequences = []
    
    regains_by_zone = {}
    losses_by_zone = {}
    all_recoveries_list = []
    all_losses_list = []
    for r_i in range(3):
        for c_i in range(3):
            regains_by_zone[(r_i, c_i)] = []
            losses_by_zone[(r_i, c_i)] = []
            
    def get_zone_indices(x, y):
        if x < 33.3: r = 0
        elif x < 66.6: r = 1
        else: r = 2
        
        if y < 33.3: c = 0
        elif y <= 66.7: c = 1
        else: c = 2
        return r, c

    for fn, df in match_dfs:
        teams = df['team_name'].unique().tolist()
        opp = [t for t in teams if t != team_name]
        opp_name = opp[0] if opp else 'Unknown'
        week = int(df['week'].iloc[0]) if 'week' in df.columns else 0
        
        records = df.to_dict('records')
        penalty_attempt_indices = {
            idx for idx, penalty_row in _collect_penalty_attempts(records, team_name)
        }
        for i, r in enumerate(records):
            # Scored Goals origin
            if r['team_name'] == team_name and r['event'] == 'Goal':
                from göztepehub.utils.xg_chain_analysis import _classify_shot_origin
                origin = _classify_shot_origin(df.loc[df.index[i]], df)
                goal_sequences.append({
                    'player': r.get('player_name', 'Unknown'),
                    'opponent': _clean(opp_name),
                    'week': week,
                    'minute': int(r.get('time_min', 0)),
                    'origin': origin,
                    'x': r.get('x', 88.5),
                    'y': r.get('y', 50.0),
                    'filename': fn,
                    'event_id': r.get('event_id', r.get('id', i))
                })
                
            # Def profile
            if r['team_name'] == team_name and r['event'] in ('Tackle', 'Interception', 'Clearance', 'Challenge'):
                x, y = r.get('x', 0), r.get('y', 0)
                if x > 0 or y > 0:
                    all_actions.append({'x': x, 'y': y, 'type': r['event']})
            if r['team_name'] == team_name and r['event'] == 'Aerial':
                x, y = r.get('x', 0), r.get('y', 0)
                if x <= 17.0 and 21.1 <= y <= 78.9:
                    all_aerials.append(1 if r.get('outcome') == 1 else 0)
                    
            # Penalty attempts only. The feed also flags the foul, keeper
            # "Penalty faced", save, and VAR/admin rows with Penalty == Si.
            if i in penalty_attempt_indices:
                taker = r.get('player_name') or 'Unknown'
                event = r.get('event')
                opponent = _clean(opp_name)
                side = _shot_side(r.get(GOAL_MOUTH_Y_COL))
                if taker not in penalty_takers:
                    penalty_takers[taker] = {'total': 0, 'scored': 0, 'saved': 0, 'missed': 0}
                penalty_takers[taker]['total'] += 1

                total_penalties += 1
                penalty_placements.append({
                    'num': total_penalties,
                    'goal_mouth_y': r.get(GOAL_MOUTH_Y_COL),
                    'goal_mouth_z': r.get('Goal Mouth Z Coordinate'),
                    'event': event,
                    'player': taker,
                    'opponent': opponent,
                    'week': week,
                    'minute': int(r.get('time_min') or 0),
                    'side': side,
                })
                if event == 'Goal':
                    scored_penalties += 1
                    penalty_takers[taker]['scored'] += 1
                else:
                    if _parse_opta_bool(r.get('Saved')) or event == 'Saved Shot':
                        saved_penalties += 1
                        penalty_takers[taker]['saved'] += 1
                    else:
                        missed_penalties += 1
                        penalty_takers[taker]['missed'] += 1
                        
            # Goalkicks
            if r['team_name'] == team_name and (r['event'] == 'Goal Kick' or _parse_opta_bool(r.get('Goal Kick'))):
                total_goalkicks += 1
                end_x = r.get('Pass End X')
                end_y = r.get('Pass End Y')
                goalkick_destinations.append({'x': end_x, 'y': end_y})
                try:
                    if end_x is not None and float(end_x) >= 50:
                        long_goalkicks += 1
                    else:
                        short_goalkicks += 1
                except:
                    short_goalkicks += 1
                    
            # Corners taken details & receiver target tracing
            if r['team_name'] == team_name and (r['event'] == 'Corner' or _parse_opta_bool(r.get('Corner taken'))):
                total_corners += 1
                y_coord = r.get('y', 50)
                if y_coord >= 50:
                    right_corners += 1
                else:
                    left_corners += 1
                    
                if _parse_opta_bool(r.get('Inswinger')):
                    inswinger_corners += 1
                elif _parse_opta_bool(r.get('Outswinger')):
                    outswinger_corners += 1
                elif _parse_opta_bool(r.get('Straight')):
                    straight_corners += 1
                    
                taker = r.get('player_name', 'Unknown')
                if taker not in corner_takers:
                    corner_takers[taker] = {'total': 0, 'left': 0, 'right': 0, 'goals': 0}
                corner_takers[taker]['total'] += 1
                if y_coord >= 50:
                    corner_takers[taker]['right'] += 1
                else:
                    corner_takers[taker]['left'] += 1

                # Check if it led to a goal: look up to 6 events ahead (max 10 seconds)
                led_to_goal = False
                for offset in range(1, 7):
                    if i + offset >= len(records):
                        break
                    next_r = records[i + offset]
                    if (next_r['time_min']*60 + next_r['time_sec']) - (r['time_min']*60 + r['time_sec']) > 10:
                        break
                    if next_r['team_name'] == team_name and next_r['event'] == 'Goal':
                        led_to_goal = True
                        break
                
                if led_to_goal:
                    corner_takers[taker]['goals'] += 1
                    total_corner_goals += 1
                
                # Trace target: look up to 3 events ahead (max 6 seconds)
                target = None
                target_idx = None
                for offset in range(1, 4):
                    if i + offset >= len(records):
                        break
                    next_r = records[i + offset]
                    if (next_r['team_name'] == team_name and 
                        next_r.get('player_name') != taker and 
                        (next_r['time_min']*60 + next_r['time_sec']) - (r['time_min']*60 + r['time_sec']) <= 6):
                        target = next_r.get('player_name', 'Unknown')
                        target_idx = i + offset
                        break
                
                if target:
                    if target not in corner_targets:
                        corner_targets[target] = {'targeted': 0, 'shots': 0, 'goals': 0}
                    corner_targets[target]['targeted'] += 1
                    
                    # Track if this target receiver took a shot or scored within the corner delivery sequence
                    # (up to 6 events after corner, max 10s from corner)
                    for offset in range(target_idx - i, 7):
                        if i + offset >= len(records):
                            break
                        seq_r = records[i + offset]
                        if (seq_r['time_min']*60 + seq_r['time_sec']) - (r['time_min']*60 + r['time_sec']) > 10:
                            break
                        if seq_r['team_name'] == team_name and seq_r.get('player_name') == target:
                            if seq_r['event'] in {'Goal', 'Miss', 'Saved Shot', 'Post'}:
                                corner_targets[target]['shots'] += 1
                                if seq_r['event'] == 'Goal':
                                    corner_targets[target]['goals'] += 1
                                break # Count at most one shot/goal per target per corner sequence
                        
            # Free Kicks
            if r['team_name'] == team_name and (r['event'] == 'Free Kick' or _parse_opta_bool(r.get('Free kick taken'))):
                total_freekicks += 1
                
            if r['team_name'] == team_name and r.get('type_id') in [13, 14, 15, 16] and _parse_opta_bool(r.get('Free kick')):
                direct_fk_shots += 1
                direct_fk_placements.append({
                    'x': r.get('x'),
                    'y': r.get('y'),
                    'goal_mouth_y': r.get(GOAL_MOUTH_Y_COL),
                    'goal_mouth_z': r.get('Goal Mouth Z Coordinate'),
                    'event': r['event'],
                })
                if r['event'] == 'Goal':
                    direct_fk_goals += 1
                    direct_fk_on_target += 1
                elif r['event'] == 'Saved Shot':
                    direct_fk_on_target += 1
                    
        # Transitions by match
        from göztepehub.utils.transitions_analysis import extract_transitions_for_match
        att, deff = extract_transitions_for_match(df, team_name)
        for a in att:
            x, y = a['recovery_x'], a['recovery_y']
            row_idx, col_idx = get_zone_indices(x, y)
            regains_by_zone[(row_idx, col_idx)].append({
                'reached_f3': a['reached_f3'],
                'shot': a['shot'],
                'goal': a['goal']
            })
            all_recoveries_list.append({
                'x': x, 'y': y,
                'player': a.get('player', 'Unknown'),
                'passes': a.get('passes', 0),
                'carries': a.get('carries', 0),
                'reached_f3': a.get('reached_f3', False),
                'shot': a.get('shot', False),
                'goal': a.get('goal', False),
                'shot_coords': a.get('shot_coords', [])
            })
            
        for d in deff:
            x, y = d['loss_x'], d['loss_y']
            row_idx, col_idx = get_zone_indices(x, y)
            losses_by_zone[(row_idx, col_idx)].append({
                'reached_f3': d['reached_f3'],
                'shot': d['shot'],
                'goal': d['goal']
            })
            all_losses_list.append({
                'x': x, 'y': y,
                'player': d.get('player', 'Unknown'),
                'passes': d.get('passes', 0),
                'carries': d.get('carries', 0),
                'reached_f3': d.get('reached_f3', False),
                'shot': d.get('shot', False),
                'goal': d.get('goal', False),
                'shot_coords': d.get('shot_coords', [])
            })

    from göztepehub.utils.defensive_analysis import get_opponent_defensive_profile
    def_profile = get_opponent_defensive_profile(team_name) or {}
    
    # 9-zone transitions mapping
    transitions_map_att = {}
    transitions_map_def = {}
    for k, v in regains_by_zone.items():
        total_r = len(v)
        cc_rate = round(sum(1 for x in v if x['reached_f3'] or x['shot']) / max(total_r, 1) * 100, 1) if total_r > 0 else 0
        transitions_map_att[k] = {'count': total_r, 'chance_creation_rate': cc_rate}
        
    for k, v in losses_by_zone.items():
        total_l = len(v)
        danger_rate = round(sum(1 for x in v if x['reached_f3'] or x['shot']) / max(total_l, 1) * 100, 1) if total_l > 0 else 0
        transitions_map_def[k] = {'count': total_l, 'danger_conceded_rate': danger_rate}
        
    sorted_takers = sorted(corner_takers.items(), key=lambda x: x[1]['total'], reverse=True)[:3]
    sorted_targets = sorted(corner_targets.items(), key=lambda x: x[1]['targeted'], reverse=True)[:3]
    sorted_penalty_takers = sorted(
        penalty_takers.items(),
        key=lambda x: (-x[1]['total'], -x[1]['scored'], x[0])
    )
    
    from göztepehub.utils.buildup_analysis import get_opponent_buildup_analysis
    _, buildup_season = get_opponent_buildup_analysis(team_name)
    
    result_stats = {
        'def_profile': def_profile,
        'penalties': {
            'total': total_penalties, 'scored': scored_penalties,
            'saved': saved_penalties, 'missed': missed_penalties,
            'placements': penalty_placements,
            'takers': sorted_penalty_takers,
        },
        'goalkicks': {
            'total': total_goalkicks, 'long': long_goalkicks, 'short': short_goalkicks,
            'long_pct': round(long_goalkicks / max(total_goalkicks, 1) * 100, 1),
            'destinations': goalkick_destinations,
        },
        'corners': {
            'total': total_corners, 'left': left_corners, 'right': right_corners,
            'inswinger': inswinger_corners, 'outswinger': outswinger_corners, 'straight': straight_corners,
            'takers': sorted_takers, 'targets': sorted_targets,
            'total_goals': total_corner_goals
        },
        'freekicks': {
            'total': total_freekicks, 'direct_shots': direct_fk_shots, 'direct_goals': direct_fk_goals,
            'direct_on_target': direct_fk_on_target, 'placements': direct_fk_placements,
        },
        'set_piece_goals': total_corner_goals + direct_fk_goals + scored_penalties,
        'goal_sequences': goal_sequences,
        'transitions_att': transitions_map_att,
        'transitions_def': transitions_map_def,
        'all_recoveries': all_recoveries_list,
        'all_losses': all_losses_list,
        'buildup': buildup_season or {}
    }
    
    _SEASON_STATS_CACHE[team_name] = result_stats
    return result_stats


RADAR_METRICS = [
    ('goals_pg',        'Goals',          True),
    ('shots_pg',        'Shots',          True),
    ('interceptions_pg','Interceptions',  True),
    ('ball_rec_pg',     'Ball Recovery',  True),
    ('ball_lost_pg',    'Ball Lost',      False),
    ('passes_pg',       'Pass Volume',    True),
    ('pass_acc',        'Pass Accuracy',  True),
]

BENCH_METRICS = [
    ('goals_pg',    'Goals / Game',       True,  '{:.2f}'),
    ('shots_pg',    'Shots / Game',       True,  '{:.1f}'),
    ('interceptions_pg', 'Interceptions', True,  '{:.1f}'),
    ('ball_rec_pg', 'Ball Recoveries',    True,  '{:.1f}'),
    ('ball_lost_pg','Ball Lost',          False, '{:.1f}'),
    ('passes_pg',   'Passes / Game',      True,  '{:.0f}'),
    ('pass_acc',    'Pass Accuracy',      True,  '{:.1f}%'),
]

PHASE_BENCH_METRICS = {
    'offensive-tab': [
        ('goals_pg',         'Goals / Game',         True,  '{:.2f}'),
        ('xg_pg',            'xG / Game',            True,  '{:.2f}'),
        ('shots_pg',         'Shots / Game',         True,  '{:.1f}'),
        ('passes_pg',        'Passes / Game',        True,  '{:.0f}'),
        ('pass_acc',         'Pass Accuracy',        True,  '{:.1f}%'),
    ],
    'defensive-tab': [
        ('xga_pg',               'xGA / Game',          False, '{:.2f}'),
        ('tackles_pg',           'Tackles / Game',       True,  '{:.1f}'),
        ('ball_rec_pg',          'Ball Recoveries',      True,  '{:.1f}'),
        ('buildup_danger_pct',   'Danger Conceded %',    False, '{:.1f}%'),
        ('buildup_turnover_pct', 'Turnover Rate %',      False, '{:.1f}%'),
    ],
    'off-trans-tab': [
        ('press_rec_pg',     'High Turnovers',        True,  '{:.1f}'),
        ('ball_rec_pg',      'Ball Recoveries',       True,  '{:.1f}'),
        ('buildup_f3_pct',   'F3 Entry from Trans %', True,  '{:.1f}%'),
        ('shots_pg',         'Shots / Game',          True,  '{:.1f}'),
    ],
    'def-trans-tab': [
        ('xga_pg',               'xGA / Game',          False, '{:.2f}'),
        ('buildup_danger_pct',   'Danger Conceded %',    False, '{:.1f}%'),
        ('buildup_turnover_pct', 'Turnover Rate %',      False, '{:.1f}%'),
        ('tackles_pg',           'Tackles / Game',       True,  '{:.1f}'),
    ],
    'set-pieces-tab': [
        ('penalty_total',    'Penalties Won', True,  '{:.0f}'),
        ('penalty_conv_pct', 'Penalty Conv.', True,  '{:.1f}%'),
        ('xg_pg',           'xG / Game',      True,  '{:.2f}'),
        ('xga_pg',          'xGA / Game',     False, '{:.2f}'),
    ],
}

def _compute_league_benchmarks():
    """Build the expensive league benchmark dataset once in a background thread."""
    result = None
    error = None
    try:
        # Let the selected opponent render before scanning every match in the league.
        time.sleep(45)
        result = _build_league_benchmarks()
    except Exception as exc:
        error = str(exc)
        print(f"Error loading league benchmarks: {exc}")

    with _LEAGUE_CACHE_LOCK:
        if result is not None:
            _LEAGUE_CACHE['df'] = result
            _LEAGUE_CACHE['timestamp'] = time.time()
        _LEAGUE_CACHE['building'] = False
        _LEAGUE_CACHE['error'] = error


def _load_league_benchmarks():
    """Return cached benchmarks immediately and warm stale data in the background."""
    if time.time() - _LEAGUE_CACHE.get('timestamp', 0) < 3600:
        return _LEAGUE_CACHE['df']

    with _LEAGUE_CACHE_LOCK:
        if not _LEAGUE_CACHE['building']:
            _LEAGUE_CACHE['building'] = True
            _LEAGUE_CACHE['error'] = None
            threading.Thread(
                target=_compute_league_benchmarks,
                name='league-benchmark-warmup',
                daemon=True,
            ).start()

    cached_df = _LEAGUE_CACHE.get('df')
    return cached_df if cached_df is not None else pd.DataFrame()


def _build_league_benchmarks():
    try:
        from utils.xg_model import predict_xg
    except ImportError:
        predict_xg = lambda d: d
    from göztepehub.utils.buildup_analysis import (
        analyze_buildup_for_match,
        get_opponent_buildup_analysis,
    )

    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    acc = {}

    for fn in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, fn))
            df = predict_xg(df)
        except Exception:
            continue
        for team in df['team_name'].unique():
            tdf = df[df['team_name'] == team]
            odf = df[df['team_name'] != team]
            n_pass  = len(tdf[tdf['type_id'] == 1])
            n_succ  = len(tdf[(tdf['type_id'] == 1) & (tdf['outcome'] == 1)])
            n_shots = len(tdf[tdf['type_id'].isin([13, 14, 15, 16])])
            if 'own goal' in tdf.columns:
                n_goals = len(tdf[(tdf['type_id'] == 16) & (tdf['own goal'] != 'Si')])
            else:
                n_goals = len(tdf[tdf['type_id'] == 16])
            n_tack  = len(tdf[tdf['type_id'] == 7])
            n_int   = len(tdf[tdf['type_id'] == 8])
            n_rec   = len(tdf[tdf['type_id'] == 49])
            n_press = len(tdf[(tdf['type_id'] == 49) & (tdf['x'] > 50)])
            n_lost  = len(tdf[
                ((tdf['type_id'] == 1) & (tdf['outcome'] == 0)) |
                (tdf['type_id'].isin([50, 51]))
            ])
            xg  = tdf['xG'].sum() if 'xG' in tdf.columns else 0
            xga = odf['xG'].sum() if 'xG' in odf.columns else 0
            penalty_attempts = [
                row for _, row in _collect_penalty_attempts(df.to_dict('records'), team)
            ]
            n_pen = len(penalty_attempts)
            n_pen_scored = sum(1 for row in penalty_attempts if row.get('event') == 'Goal')
            
            # Buildup outcomes for fallback benchmark values. Displayed build-up
            # rankings are aligned below with the opponent-season helper.
            b_total = b_f3 = b_shot = b_goal = b_turnover = b_danger = 0
            try:
                buildup_res = analyze_buildup_for_match(df, team)
                if buildup_res:
                    b_total = buildup_res.get('total_buildups', 0)
                    outcomes = buildup_res.get('outcomes_10s', {})
                    b_f3 = outcomes.get('f3_entry', 0)
                    b_shot = outcomes.get('shot', 0)
                    b_goal = outcomes.get('goal', 0)
                    b_turnover = outcomes.get('turnover', 0)
                    b_danger = outcomes.get('opp_danger', 0)
            except Exception:
                pass

            if team not in acc:
                acc[team] = dict(m=0, p=0, s=0, sh=0, g=0, t=0, i=0, r=0, pr=0, lost=0, xg=0, xga=0,
                                 pen=0, pen_scored=0,
                                 b_total=0, b_f3=0, b_shot=0, b_goal=0, b_turnover=0, b_danger=0)
            a = acc[team]
            a['m'] += 1; a['p'] += n_pass; a['s'] += n_succ; a['sh'] += n_shots
            a['g'] += n_goals; a['t'] += n_tack; a['i'] += n_int; a['r'] += n_rec; a['pr'] += n_press; a['lost'] += n_lost
            a['xg'] += xg; a['xga'] += xga
            a['pen'] += n_pen; a['pen_scored'] += n_pen_scored
            a['b_total'] += b_total
            a['b_f3'] += b_f3
            a['b_shot'] += b_shot
            a['b_goal'] += b_goal
            a['b_turnover'] += b_turnover
            a['b_danger'] += b_danger

    rows = []
    for team, a in acc.items():
        m = max(a['m'], 1)
        _, buildup_season = get_opponent_buildup_analysis(team)
        if buildup_season:
            b_total = buildup_season.get('total_buildups', 0)
            buildup_outcomes = buildup_season.get('outcomes_10s', {})
            b_f3 = buildup_outcomes.get('f3_entry', 0)
            b_shot = buildup_outcomes.get('shot', 0)
            b_goal = buildup_outcomes.get('goal', 0)
            b_turnover = buildup_outcomes.get('turnover', 0)
            b_danger = buildup_outcomes.get('opp_danger', 0)
        else:
            b_total = a['b_total']
            b_f3 = a['b_f3']
            b_shot = a['b_shot']
            b_goal = a['b_goal']
            b_turnover = a['b_turnover']
            b_danger = a['b_danger']
        b_tot = max(b_total, 1)
        rows.append(dict(team=team,
            passes_pg=a['p']/m, pass_acc=a['s']/max(a['p'],1)*100,
            shots_pg=a['sh']/m, goals_pg=a['g']/m,
            xg_pg=a['xg']/m, xga_pg=a['xga']/m,
            tackles_pg=a['t']/m, interceptions_pg=a['i']/m,
            ball_rec_pg=a['r']/m, press_rec_pg=a['pr']/m,
            ball_lost_pg=a['lost']/m,
            penalty_total=a['pen'],
            penalty_conv_pct=a['pen_scored']/max(a['pen'], 1)*100,
            buildup_f3_pct=b_f3/b_tot*100,
            buildup_shot_pct=b_shot/b_tot*100,
            buildup_goal_pct=b_goal/b_tot*100,
            buildup_turnover_pct=b_turnover/b_tot*100,
            buildup_danger_pct=b_danger/b_tot*100))

    result = pd.DataFrame(rows).set_index('team')
    return result


def _build_benchmark_radar(league_df, rival):
    metric_key = "-".join(col for col, _, _ in RADAR_METRICS)
    cache_key = f"radar_{rival}_{metric_key}"
    if cache_key in _RADAR_CACHE:
        return _RADAR_CACHE[cache_key]

    def pct_series(col, hib):
        s = league_df[col]
        return (s.rank(pct=True) if hib else s.rank(pct=True, ascending=False)) * 100

    pct_df = pd.DataFrame(
        {col: pct_series(col, hib) for col, _, hib in RADAR_METRICS},
        index=league_df.index
    )

    N = len(RADAR_METRICS)
    labels = [lbl for _, lbl, _ in RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    def row_vals(team):
        if team not in pct_df.index:
            return [50] * N
        return pct_df.loc[team, [c for c, *_ in RADAR_METRICS]].tolist()

    goz_v   = row_vals(GOZTEPE) + row_vals(GOZTEPE)[:1]
    rival_v = row_vals(rival) + row_vals(rival)[:1]
    avg_v   = [50] * (N + 1)
    ang     = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'}, facecolor=_PITCH_BG)
    ax.set_facecolor(_PITCH_BG)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    for ring in [25, 50, 75, 100]:
        ax.plot(ang, [ring] * (N + 1), color='white', alpha=0.07, linewidth=0.5)

    ax.plot(ang, avg_v,   color='white',  alpha=0.30, linewidth=1.2, linestyle='--', label='League Avg')
    ax.fill(ang, avg_v,   color='white',  alpha=0.03)
    ax.plot(ang, rival_v, color='#3b82f6', linewidth=1.8, label=_clean(rival))
    ax.fill(ang, rival_v, color='#3b82f6', alpha=0.12)
    ax.plot(ang, goz_v,   color='#ef4444', linewidth=2.3, label='Göztepe')
    ax.fill(ang, goz_v,   color='#ef4444', alpha=0.18)
    ax.scatter(angles, goz_v[:-1], color='#ef4444', s=45, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=9, color='white', fontweight='bold')
    ax.tick_params(axis='x', pad=10)
    ax.set_ylim(0, 108)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_rlabel_position(30)
    ax.set_yticklabels(['25th', '50th', '75th', '100th'], size=6.5, color=(1, 1, 1, 0.35))
    ax.spines['polar'].set_visible(False)
    ax.grid(color='white', alpha=0.07, linewidth=0.5)

    result = _fig_to_b64(fig)
    _RADAR_CACHE[cache_key] = result
    if len(_RADAR_CACHE) > 20:
        _RADAR_CACHE.pop(next(iter(_RADAR_CACHE)))
    return result


def _build_benchmarking_section(rival, opp_name):
    try:
        league_df = _load_league_benchmarks()
    except Exception as e:
        return html.Div(f"Benchmark data unavailable: {e}", className="goz-card-desc")

    if league_df.empty:
        error = _LEAGUE_CACHE.get('error')
        message = f"Benchmark data unavailable: {error}" if error else \
            "Preparing league benchmarks in the background. Tactical tabs are ready to use now."
        return html.Div(className="goz-form-section pm-benchmark-loading", children=[
            html.Div("LEAGUE BENCHMARKING", className="goz-card-title",
                     style={"fontSize": "1.05rem", "marginBottom": "6px"}),
            html.P(message, className="goz-card-desc", style={"margin": 0}),
        ])

    radar_b64 = _build_benchmark_radar(league_df, rival)

    def rank_of(team, col, hib):
        if team not in league_df.index:
            return '-'
        s = league_df[col]
        return int(s.rank(ascending=not hib, method='min')[team])

    def val_of(team, col):
        if team not in league_df.index:
            return 0
        return league_df.loc[team, col]

    n_teams = len(league_df)

    metric_rows = []
    for col, label, hib, fmt in BENCH_METRICS:
        gv = val_of(GOZTEPE, col)
        rv = val_of(rival, col)
        gr = rank_of(GOZTEPE, col, hib)
        rr = rank_of(rival, col, hib)
        rival_better = (hib and rv > gv) or (not hib and rv < gv)
        delta = rv - gv if hib else gv - rv
        delta_str = f"+{abs(delta):.1f}" if delta > 0 else f"−{abs(delta):.1f}"

        metric_rows.append(html.Div(style={
            "display": "flex", "alignItems": "center", "gap": "6px",
            "padding": "7px 10px", "borderRadius": "8px", "marginBottom": "5px",
            "background": "rgba(239,68,68,0.07)" if rival_better else "rgba(34,197,94,0.05)",
            "border": "1px solid rgba(239,68,68,0.18)" if rival_better else "1px solid rgba(34,197,94,0.14)",
        }, children=[
            html.Span(label, style={"flex": "1.8", "fontSize": "0.76rem",
                "color": "var(--text-secondary)", "fontWeight": "500"}),
            html.Div(style={"flex": "1", "textAlign": "center"}, children=[
                html.Span(fmt.format(gv), style={"fontWeight": "700", "color": _RED, "fontSize": "0.88rem"}),
                html.Span(f" #{gr}", style={"fontSize": "0.62rem", "color": "var(--text-secondary)"}),
            ]),
            html.Span("▲" if rival_better else "▼", style={
                "fontSize": "0.75rem", "width": "14px", "textAlign": "center",
                "color": _RED if rival_better else "#22c55e",
            }),
            html.Div(style={"flex": "1", "textAlign": "center"}, children=[
                html.Span(fmt.format(rv), style={"fontWeight": "700", "color": _BLUE, "fontSize": "0.88rem"}),
                html.Span(f" #{rr}", style={"fontSize": "0.62rem", "color": "var(--text-secondary)"}),
            ]),
            html.Span(delta_str, style={
                "fontSize": "0.68rem", "width": "36px", "textAlign": "right",
                "color": _RED if rival_better else "#22c55e", "fontWeight": "600",
            }),
        ]))

    goz_rank_goals = rank_of(GOZTEPE, 'goals_pg', True)
    goz_rank_xg    = rank_of(GOZTEPE, 'xg_pg',    True)
    goz_rank_xga   = rank_of(GOZTEPE, 'xga_pg',   False)

    return html.Div(className="goz-form-section", style={"marginTop": "24px"}, children=[
        html.Div(className="goz-section-header", style={
            "display": "flex", "flexDirection": "column", "alignItems": "flex-start",
            "gap": "6px", "marginBottom": "20px"
        }, children=[
            html.Span("LEAGUE BENCHMARKING", className="goz-card-title", style={"fontSize": "1.3rem", "fontWeight": "bold", "color": "white"}),
            html.P(
                f"Season-wide per-game averages · Göztepe vs {opp_name} vs all {n_teams} teams",
                className="goz-card-desc",
                style={"margin": "0", "fontSize": "0.8rem", "color": "var(--text-secondary)"}
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div("MATCH METRICS — PERCENTILE RANK", style={
                    "fontSize": "0.72rem", "fontWeight": "700", "color": _GOLD,
                    "letterSpacing": "1px", "marginBottom": "8px", "textAlign": "center",
                }),
                html.Img(src=radar_b64, style={
                    "width": "100%", "maxWidth": "500px", "margin": "0 auto",
                    "display": "block", "borderRadius": "8px"
                }),
                html.Div(style={
                    "display": "flex", "justifyContent": "center", "gap": "16px",
                    "marginTop": "12px", "marginBottom": "15px", "flexWrap": "wrap"
                }, children=[
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                        html.Div(style={"width": "10px", "height": "10px", "borderRadius": "50%", "background": "#ef4444"}),
                        html.Span("Göztepe", style={"fontSize": "0.72rem", "color": "white", "fontWeight": "600"}),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                        html.Div(style={"width": "10px", "height": "10px", "borderRadius": "50%", "background": "#3b82f6"}),
                        html.Span(opp_name, style={"fontSize": "0.72rem", "color": "white", "fontWeight": "600"}),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                        html.Div(style={"width": "12px", "height": "1.5px", "background": "white", "opacity": "0.5", "borderStyle": "dashed"}),
                        html.Span("League Average", style={"fontSize": "0.72rem", "color": "var(--text-secondary)"}),
                    ]),
                ]),
            ], md=6),
            dbc.Col([
                html.Div(style={
                    "display": "flex", "gap": "6px", "padding": "0 4px",
                    "marginBottom": "10px", "alignItems": "center",
                }, children=[
                    html.Span("Metric", style={"flex": "1.8", "fontSize": "0.65rem",
                        "color": "var(--text-secondary)", "textTransform": "uppercase"}),
                    html.Span("Göztepe", style={"flex": "1", "textAlign": "center",
                        "fontSize": "0.65rem", "color": _RED, "fontWeight": "700", "textTransform": "uppercase"}),
                    html.Span("", style={"width": "14px"}),
                    html.Span(opp_name[:10], style={"flex": "1", "textAlign": "center",
                        "fontSize": "0.65rem", "color": _BLUE, "fontWeight": "700", "textTransform": "uppercase",
                        "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                    html.Span("Δ", style={"width": "36px", "textAlign": "right",
                        "fontSize": "0.65rem", "color": "var(--text-secondary)"}),
                ]),
                html.Div(children=metric_rows),
                html.Div(style={"marginTop": "16px", "padding": "10px 12px", "borderRadius": "8px",
                    "background": "rgba(255,255,255,0.03)", "border": "1px solid var(--border-color)",
                    "fontSize": "0.72rem", "color": "var(--text-secondary)", "lineHeight": "1.8"}, children=[
                    html.Span("Context: ", style={"fontWeight": "700", "color": _GOLD}),
                    f"Göztepe rank #{goz_rank_goals}/18 in goals scored, #{goz_rank_xg}/18 in xG created, "
                    f"and #{goz_rank_xga}/18 in defensive solidity (xGA). "
                    "▲ = rival outperforms Göztepe on this metric.",
                ]),
            ], md=6),
        ]),
    ])

GOZTEPE = 'Göztepe Spor Kulübü'

TAB_TO_PHASE = {
    'game-plan-tab':  'Game Plan',
    'offensive-tab':  'Offensive',
    'defensive-tab':  'Defensive',
    'off-trans-tab':  'Off. Transitions',
    'def-trans-tab':  'Def. Transitions',
    'set-pieces-tab': 'Set Pieces',
}
TAB_LABELS = {
    'game-plan-tab':  'Game Plan',
    'offensive-tab':  'Offensive',
    'defensive-tab':  'Defensive',
    'off-trans-tab':  'Off. Transitions',
    'def-trans-tab':  'Def. Transitions',
    'set-pieces-tab': 'Set Pieces',
}

_SUFFIXES = ['Spor Kulübü', 'Futbol Kulübü', 'Kulübü', 'Spor A.Ş.', 'A.Ş.', 'S.K.', 'F.K.', 'SK']

def _clean(name):
    result = name
    for s in _SUFFIXES:
        result = result.replace(s, '')
    return result.strip()


def _fig_to_b64(fig):
    """Convert matplotlib figure to a base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=_PITCH_BG, edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def _build_buildup_pitch(coords):
    """Horizontal own-half build-up map with a compact 3x3 density grid."""
    counts = np.zeros((3, 3))
    for c in coords:
        x, y = c.get('x'), c.get('y')
        if x is None or y is None:
            continue
        # Cap/bound values
        x = max(0, min(49.9, float(x)))
        y = max(0, min(99.9, float(y)))
        
        # Grid index
        if x < 16.67: r = 0
        elif x < 33.33: r = 1
        else: r = 2
        
        if y < 33.33: cl = 0
        elif y <= 66.67: cl = 1
        else: cl = 2
        
        counts[cl, r] += 1
        
    total = max(1, sum(1 for c in coords if c.get('x') is not None))
    pcts = (counts / total) * 100
    
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor=_PITCH_BG)
    ax.set_facecolor('#112213')

    for i in range(10):
        x0 = i * 10
        ax.fill_between([x0, x0 + 10], 0, 100,
                        color='#18331b' if i % 2 == 0 else '#122615',
                        alpha=0.9, zorder=0)

    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color='white', alpha=0.52, linewidth=1.5)
    ax.plot([50, 50], [0, 100], color='white', alpha=0.45, linewidth=1.3)
    ax.add_patch(plt.Circle((50, 50), 9.15, color='white', fill=False, alpha=0.30, linewidth=1.0))
    ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color='white', alpha=0.35, linewidth=1.1)
    ax.plot([0, 5.8, 5.8, 0], [40.9, 40.9, 59.1, 59.1], color='white', alpha=0.25, linewidth=0.9)
    ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color='white', alpha=0.20, linewidth=1.0)

    x_centers = [8.33, 25.0, 41.67]
    y_centers = [16.67, 50.0, 83.33]
    x_bounds = [0, 16.67, 33.33, 50.0]
    y_bounds = [0, 33.33, 66.67, 100.0]
    max_pct = max(1.0, pcts.max())

    for cl in range(3):
        for r in range(3):
            val = pcts[cl, r]
            alpha = max(0.06, min(0.62, (val / max_pct) * 0.58))
            ax.fill_between([x_bounds[r], x_bounds[r + 1]], y_bounds[cl], y_bounds[cl + 1],
                            color='#fbbf24', alpha=alpha, zorder=1)
            ax.text(x_centers[r], y_centers[cl], f"{val:.1f}%", color='white',
                    fontsize=11, fontweight='bold', ha='center', va='center', zorder=10)

    for xv in x_bounds[1:-1]:
        ax.plot([xv, xv], [0, 100], color='white', alpha=0.20, linestyle='--', linewidth=1.0)
    for yv in y_bounds[1:-1]:
        ax.plot([0, 50], [yv, yv], color='white', alpha=0.20, linestyle='--', linewidth=1.0)

    ax.text(25, 104, 'BUILD-UP START ZONES — OWN HALF', color='#fbbf24',
            fontsize=9, fontweight='bold', ha='center')
    ax.text(75, 50, 'ATTACKING HALF', color=(1, 1, 1, 0.18),
            fontsize=12, fontweight='bold', ha='center', va='center')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-3, 108)
    ax.set_aspect('equal')
    ax.axis('off')
    return _fig_to_b64(fig)


def _build_buildup_f3_entry_pitch(entry_coords):
    """Show where own-half build-ups first enter the final third in a 3x3 grid."""
    counts = np.zeros((3, 3), dtype=int)
    for coord in entry_coords:
        try:
            x = max(66.6, min(99.9, float(coord.get('x'))))
            y = max(0.0, min(99.9, float(coord.get('y'))))
        except (TypeError, ValueError):
            continue
        x_idx = min(2, int((x - 66.6) / ((100.0 - 66.6) / 3)))
        y_idx = min(2, int(y / (100.0 / 3)))
        counts[y_idx, x_idx] += 1

    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(7.3, 5.2), facecolor=_PITCH_BG)
    ax.set_facecolor('#112213')
    for i in range(6):
        x0 = 66.6 + i * ((100 - 66.6) / 6)
        ax.fill_between([x0, x0 + ((100 - 66.6) / 6)], 0, 100,
                        color='#18331b' if i % 2 == 0 else '#122615', alpha=0.9, zorder=0)

    ax.plot([66.6, 100, 100, 66.6, 66.6], [0, 0, 100, 100, 0],
            color='white', alpha=0.52, linewidth=1.5)
    ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9],
            color='white', alpha=0.42, linewidth=1.1)
    ax.plot([100, 94.2, 94.2, 100], [40.9, 40.9, 59.1, 59.1],
            color='white', alpha=0.28, linewidth=0.9)

    x_bounds = np.linspace(66.6, 100, 4)
    y_bounds = np.linspace(0, 100, 4)
    max_count = max(1, int(counts.max()))
    for y_idx in range(3):
        for x_idx in range(3):
            count = int(counts[y_idx, x_idx])
            alpha = max(0.05, min(0.62, count / max_count * 0.58))
            ax.fill_between([x_bounds[x_idx], x_bounds[x_idx + 1]],
                            y_bounds[y_idx], y_bounds[y_idx + 1],
                            color='#fbbf24', alpha=alpha, zorder=1)
            ax.text((x_bounds[x_idx] + x_bounds[x_idx + 1]) / 2,
                    (y_bounds[y_idx] + y_bounds[y_idx + 1]) / 2,
                    str(count), color='white', fontsize=14, fontweight='bold',
                    ha='center', va='center', zorder=5)

    for x_value in x_bounds[1:-1]:
        ax.plot([x_value, x_value], [0, 100], color='white', alpha=0.20,
                linestyle='--', linewidth=1.0)
    for y_value in y_bounds[1:-1]:
        ax.plot([66.6, 100], [y_value, y_value], color='white', alpha=0.20,
                linestyle='--', linewidth=1.0)

    for y_idx, label in enumerate(['LEFT', 'CENTER', 'RIGHT']):
        ax.text(65.2, (y_bounds[y_idx] + y_bounds[y_idx + 1]) / 2, label,
                color=(1, 1, 1, 0.52), fontsize=7, fontweight='bold',
                ha='right', va='center')
    ax.text(100.5, 50, 'GOAL', color=(1, 1, 1, 0.48), fontsize=7,
            fontweight='bold', rotation=90, ha='left', va='center')

    ax.set_title(f'BUILD-UP-DERIVED FINAL-THIRD ENTRIES — {total} TOTAL',
                 color='#fbbf24', fontsize=10, fontweight='bold', pad=10)
    ax.set_xlim(64.5, 102)
    ax.set_ylim(-3, 103)
    ax.set_aspect('equal')
    ax.axis('off')
    return _fig_to_b64(fig)


def _build_goals_pitch(goals):
    fig, ax = plt.subplots(figsize=(3.8, 7), facecolor=_PITCH_BG)
    ax.set_facecolor(_PITCH_BG)
    
    # Draw attacking half-pitch outline (x from 50 to 100, y from 0 to 100)
    ax.plot([50, 100, 100, 50, 50], [0, 0, 100, 100, 0], color=(1.0, 1.0, 1.0, 0.4), linewidth=1.8)
    # Penalty box
    ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=(1.0, 1.0, 1.0, 0.4), linewidth=1.5)
    # Six yard box
    ax.plot([100, 94.2, 94.2, 100], [40.9, 40.9, 59.1, 59.1], color=(1.0, 1.0, 1.0, 0.25), linewidth=1.0)
    
    # Halfway line arc
    from matplotlib.patches import Arc
    halfway_arc = Arc((50, 50), 18.3, 18.3, theta1=270, theta2=90, color=(1.0, 1.0, 1.0, 0.4), linewidth=1.5)
    ax.add_patch(halfway_arc)
    
    # Scatter coords of goals colored by origin
    origin_colors = {
        'open_play': '#22c55e',    # Green
        'from_cross': '#3b82f6',   # Blue
        'set_piece': '#fbbf24',    # Gold
        'through_ball': '#a855f7', # Purple
        'fast_break': '#ef4444'    # Red
    }
    origin_labels = {
        'open_play': 'Open Play',
        'from_cross': 'Cross',
        'set_piece': 'Set Piece',
        'through_ball': 'Through Ball',
        'fast_break': 'Fast Break'
    }
    
    scattered = set()
    for o_type, color in origin_colors.items():
        sub_x = [g.get('x', 88.0) for g in goals if g.get('origin') == o_type and g.get('x') is not None]
        sub_y = [g.get('y', 50.0) for g in goals if g.get('origin') == o_type and g.get('y') is not None]
        if sub_x:
            ax.scatter(sub_x, sub_y, color=color, s=40, alpha=0.9, label=origin_labels[o_type], edgecolors='white', linewidths=0.6, zorder=5)
            scattered.add(o_type)
            
    ax.set_xlim(49, 101)
    ax.set_ylim(-2, 102)
    ax.axis('off')
    
    if scattered:
        legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=min(len(scattered), 5), framealpha=0.9, facecolor=_PITCH_BG, edgecolor='none', fontsize=7.5)
        for t in legend.get_texts():
            t.set_color('white')
            
    return _fig_to_b64(fig)


def _build_defensive_heatmap(coords):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=_PITCH_BG)
    ax.set_facecolor(_PITCH_BG)
    
    # Full pitch outline
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=(1.0, 1.0, 1.0, 0.4), linewidth=1.5)
    # Midfield line
    ax.plot([50, 50], [0, 100], color=(1.0, 1.0, 1.0, 0.4), linewidth=1.5)
    # Center circle
    center_circle = plt.Circle((50, 50), 9.15, color=(1.0, 1.0, 1.0, 0.4), fill=False, linewidth=1.5)
    ax.add_patch(center_circle)
    
    # Left Penalty box
    ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=(1.0, 1.0, 1.0, 0.3), linewidth=1.2)
    # Right Penalty box
    ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=(1.0, 1.0, 1.0, 0.3), linewidth=1.2)
    
    types_colors = {
        'Tackle': '#ef4444',       # Red
        'Interception': '#3b82f6', # Blue
        'Clearance': '#a855f7',    # Purple
        'Challenge': '#22c55e'     # Green
    }
    
    for t_name, color in types_colors.items():
        sub_x = [c['x'] for c in coords if c.get('type') == t_name]
        sub_y = [c['y'] for c in coords if c.get('type') == t_name]
        if sub_x:
            ax.scatter(sub_x, sub_y, color=color, s=25, alpha=0.75, label=t_name, edgecolors='none', zorder=5)
            
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.axis('off')
    
    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=4, framealpha=0.9, facecolor=_PITCH_BG, edgecolor='none', fontsize=8)
    for t in legend.get_texts():
        t.set_color('white')
        
    return _fig_to_b64(fig)


def _build_phase_bench_section(active_tab, rival, opp_name):
    """Compact phase-specific benchmark comparison strip shown at the top of each tab."""
    try:
        league_df = _load_league_benchmarks()
    except Exception:
        return html.Div()

    if league_df.empty:
        return html.Div()

    metrics = PHASE_BENCH_METRICS.get(active_tab, [])
    if not metrics:
        return html.Div()

    phase_names = {
        'offensive-tab':  'OFFENSIVE PHASE',
        'defensive-tab':  'DEFENSIVE PHASE',
        'off-trans-tab':  'ATTACKING TRANSITIONS',
        'def-trans-tab':  'DEFENSIVE TRANSITIONS',
        'set-pieces-tab': 'SET PIECES',
    }
    phase_name = phase_names.get(active_tab, 'PHASE')
    n_teams = len(league_df)

    def _val(team, col):
        return float(league_df.loc[team, col]) if team in league_df.index else 0.0

    def _rank(team, col, hib):
        if team not in league_df.index:
            return '-'
        return int(league_df[col].rank(ascending=not hib, method='min')[team])

    cards = []
    for col, label, hib, fmt in metrics:
        gv = _val(GOZTEPE, col)
        rv = _val(rival, col)
        gr = _rank(GOZTEPE, col, hib)
        rr = _rank(rival, col, hib)
        rival_better = (hib and rv > gv) or (not hib and rv < gv)
        arrow     = '▲' if rival_better else '▼'
        arrow_col = _RED if rival_better else _GREEN

        cards.append(dbc.Col(html.Div(style={
            'padding': '10px 12px', 'borderRadius': '10px', 'marginBottom': '6px',
            'border': f"1px solid {'rgba(239,68,68,0.28)' if rival_better else 'rgba(34,197,94,0.2)'}",
            'background': f"{'rgba(239,68,68,0.05)' if rival_better else 'rgba(34,197,94,0.04)'}",
        }, children=[
            html.Div(label, style={'fontSize': '0.6rem', 'color': 'var(--text-secondary)',
                                   'fontWeight': '700', 'textTransform': 'uppercase',
                                   'letterSpacing': '0.4px', 'marginBottom': '8px'}),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                html.Div([
                    html.Div(fmt.format(gv), style={'fontSize': '1.05rem', 'fontWeight': '800', 'color': _RED}),
                    html.Div(f'#{gr}/{n_teams}', style={'fontSize': '0.58rem', 'color': 'var(--text-secondary)'}),
                    html.Div('GZP', style={'fontSize': '0.55rem', 'color': _RED, 'opacity': '0.7'}),
                ], style={'textAlign': 'center'}),
                html.Span(arrow, style={'fontSize': '0.85rem', 'color': arrow_col, 'fontWeight': '800'}),
                html.Div([
                    html.Div(fmt.format(rv), style={'fontSize': '1.05rem', 'fontWeight': '800', 'color': _BLUE}),
                    html.Div(f'#{rr}/{n_teams}', style={'fontSize': '0.58rem', 'color': 'var(--text-secondary)'}),
                    html.Div(opp_name[:6], style={'fontSize': '0.55rem', 'color': _BLUE, 'opacity': '0.7'}),
                ], style={'textAlign': 'center'}),
            ]),
        ]), md=2, sm=4, xs=6))

    return html.Div(style={
        'marginBottom': '18px', 'padding': '14px 18px',
        'borderRadius': '12px', 'background': 'rgba(255,255,255,0.02)',
        'border': '1px solid rgba(255,255,255,0.07)',
    }, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                        'alignItems': 'center', 'marginBottom': '12px'}, children=[
            html.Div([
                html.Span('PHASE BENCHMARK — ', style={'fontSize': '0.65rem', 'color': 'var(--text-secondary)'}),
                html.Span(phase_name, style={'fontSize': '0.65rem', 'fontWeight': '800', 'color': _GOLD,
                                             'letterSpacing': '0.5px'}),
            ]),
            html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}, children=[
                    html.Div(style={'width': '7px', 'height': '7px', 'borderRadius': '50%', 'background': _RED}),
                    html.Span('Göztepe', style={'fontSize': '0.6rem', 'color': 'white'}),
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}, children=[
                    html.Div(style={'width': '7px', 'height': '7px', 'borderRadius': '50%', 'background': _BLUE}),
                    html.Span(opp_name, style={'fontSize': '0.6rem', 'color': 'white'}),
                ]),
            ]),
        ]),
        dbc.Row(cards, className='g-2'),
    ])


def _league_rank_for_metric(team, column, higher_is_better=True):
    try:
        league_df = _load_league_benchmarks()
    except Exception:
        return None, None
    if league_df.empty or team not in league_df.index or column not in league_df.columns:
        return None, None
    rank = int(league_df[column].rank(ascending=not higher_is_better, method='min')[team])
    return rank, len(league_df)


def _penalty_total_rank(team):
    if time.time() - _PENALTY_RANK_CACHE.get('timestamp', 0) < 3600:
        data = _PENALTY_RANK_CACHE.get('data') or {}
    else:
        try:
            data = {}
            needed_cols = [
                'team_name', 'Penalty', 'event', 'type_id',
                'period_id', 'time_min', 'time_sec', 'player_name'
            ]
            for filename in os.listdir(get_data_dir()):
                if not filename.endswith('.parquet'):
                    continue
                try:
                    path = os.path.join(get_data_dir(), filename)
                    available_cols = pd.read_parquet(path, columns=['team_name']).columns.tolist()
                    if 'team_name' not in available_cols:
                        continue
                    try:
                        df = pd.read_parquet(path, columns=needed_cols)
                    except Exception:
                        df = pd.read_parquet(path)
                    records = df.to_dict('records')
                    for team_name in df['team_name'].dropna().unique().tolist():
                        if team_name == GOZTEPE:
                            continue
                        data.setdefault(team_name, 0)
                        data[team_name] += len(_collect_penalty_attempts(records, team_name))
                except Exception:
                    continue
            _PENALTY_RANK_CACHE['data'] = data
            _PENALTY_RANK_CACHE['timestamp'] = time.time()
        except Exception:
            return None, None

    if team not in data:
        return None, None
    ranked = sorted(data.items(), key=lambda item: (-item[1], item[0]))
    rank_lookup = {}
    last_value = None
    last_rank = 0
    for idx, (team_name, value) in enumerate(ranked, start=1):
        if value != last_value:
            last_rank = idx
            last_value = value
        rank_lookup[team_name] = last_rank
    return rank_lookup.get(team), len(ranked)


def _rank_from_count_map(data, team):
    if team not in data:
        return None, None
    ranked = sorted(data.items(), key=lambda item: (-item[1], item[0]))
    rank_lookup = {}
    last_value = None
    last_rank = 0
    for idx, (team_name, value) in enumerate(ranked, start=1):
        if value != last_value:
            last_rank = idx
            last_value = value
        rank_lookup[team_name] = last_rank
    return rank_lookup.get(team), len(ranked)


def _set_piece_goal_rank(team):
    if time.time() - _SET_PIECE_GOAL_RANK_CACHE.get('timestamp', 0) < 3600:
        data = _SET_PIECE_GOAL_RANK_CACHE.get('data') or {}
    else:
        try:
            data = {}
            for filename in os.listdir(get_data_dir()):
                if not filename.endswith('.parquet'):
                    continue
                try:
                    df = pd.read_parquet(os.path.join(get_data_dir(), filename))
                    if 'team_name' not in df.columns:
                        continue
                    records = df.to_dict('records')
                    for team_name in df['team_name'].dropna().unique().tolist():
                        if team_name == GOZTEPE:
                            continue
                        data.setdefault(team_name, 0)
                        penalty_goals = sum(
                            1 for _, row in _collect_penalty_attempts(records, team_name)
                            if row.get('event') == 'Goal'
                        )
                        corner_goals = 0
                        direct_fk_goals = 0
                        for i, row in enumerate(records):
                            if row.get('team_name') != team_name:
                                continue

                            if row.get('event') == 'Corner' or _parse_opta_bool(row.get('Corner taken')):
                                start_time = (row.get('time_min') or 0) * 60 + (row.get('time_sec') or 0)
                                for offset in range(1, 7):
                                    if i + offset >= len(records):
                                        break
                                    next_row = records[i + offset]
                                    next_time = (
                                        (next_row.get('time_min') or 0) * 60
                                        + (next_row.get('time_sec') or 0)
                                    )
                                    if next_time - start_time > 10:
                                        break
                                    if next_row.get('team_name') == team_name and next_row.get('event') == 'Goal':
                                        corner_goals += 1
                                        break

                            if (
                                row.get('event') == 'Goal'
                                and row.get('type_id') in [13, 14, 15, 16]
                                and _parse_opta_bool(row.get('Free kick'))
                            ):
                                direct_fk_goals += 1

                        data[team_name] += corner_goals + direct_fk_goals + penalty_goals
                except Exception:
                    continue
            _SET_PIECE_GOAL_RANK_CACHE['data'] = data
            _SET_PIECE_GOAL_RANK_CACHE['timestamp'] = time.time()
        except Exception:
            return None, None

    return _rank_from_count_map(data, team)


def _build_transitions_risk_pitch(transitions_map, is_att=True):
    """Matplotlib full-pitch 9-zone risk map for transitions."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=_PITCH_BG)
    ax.set_facecolor('#1a3a1e')

    sw = 100 / 8
    for i in range(8):
        ax.fill_between([i * sw, (i + 1) * sw], 0, 100,
                        color='#1e4a25' if i % 2 == 0 else '#173519', alpha=0.85, zorder=0)

    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], 'w-', alpha=0.55, lw=1.5, zorder=3)
    ax.plot([50, 50], [0, 100], 'w-', alpha=0.35, lw=1.2, zorder=3)
    from matplotlib.patches import Circle as _MplCircle
    ax.add_patch(_MplCircle((50, 50), 9.15, color='w', fill=False, alpha=0.3, lw=1, zorder=3))
    ax.plot([0, 17, 17, 0],   [21.1, 21.1, 78.9, 78.9], 'w-', alpha=0.35, lw=1.2, zorder=3)
    ax.plot([0, 5.5, 5.5, 0], [40.5, 40.5, 59.5, 59.5], 'w-', alpha=0.22, lw=0.8, zorder=3)
    ax.plot([100, 83, 83, 100],       [21.1, 21.1, 78.9, 78.9], 'w-', alpha=0.35, lw=1.2, zorder=3)
    ax.plot([100, 94.5, 94.5, 100],   [40.5, 40.5, 59.5, 59.5], 'w-', alpha=0.22, lw=0.8, zorder=3)
    ax.plot([-2, 0, 0, -2],       [45.2, 45.2, 54.8, 54.8], 'w-', alpha=0.4, lw=1.5, zorder=3)
    ax.plot([100, 102, 102, 100], [45.2, 45.2, 54.8, 54.8], 'w-', alpha=0.4, lw=1.5, zorder=3)

    x_bounds = [0, 33.3, 66.6, 100]
    y_bounds = [0, 33.3, 66.6, 100]
    zone_color = '#22c55e' if is_att else '#ef4444'
    rate_key   = 'chance_creation_rate' if is_att else 'danger_conceded_rate'
    rate_label = 'CC%' if is_att else 'Danger%'

    all_counts = [transitions_map.get((r, c), {}).get('count', 0) for r in range(3) for c in range(3)]
    max_count  = max(all_counts) if any(v > 0 for v in all_counts) else 1

    for r in range(3):
        for c in range(3):
            d     = transitions_map.get((r, c), {})
            count = d.get('count', 0)
            rate  = d.get(rate_key, 0)
            alpha = max(0.04, min(0.55, (count / max_count) * 0.52))
            is_hot = rate >= 25.0 and count >= 3

            x0, x1 = x_bounds[r], x_bounds[r + 1]
            y0, y1 = y_bounds[c], y_bounds[c + 1]

            ax.fill_between([x0, x1], y0, y1, color=zone_color, alpha=alpha, zorder=1)
            if is_hot:
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        color='#fbbf24', alpha=0.9, lw=2.0, zorder=2)

            xc = (x0 + x1) / 2
            yc = (y0 + y1) / 2
            ax.text(xc, yc + 7, str(count), color='white', fontsize=13, fontweight='bold',
                    ha='center', va='center', zorder=5)
            rc_col = '#fbbf24' if rate >= 25 else zone_color
            ax.text(xc, yc - 4, f'{rate_label}: {rate}%', color=rc_col, fontsize=7.5,
                    fontweight='700', ha='center', va='center', zorder=5)
            ev_lbl = 'regains' if is_att else 'losses'
            ax.text(xc, yc - 11, ev_lbl, color=(1, 1, 1, 0.40), fontsize=6.5,
                    ha='center', va='center', zorder=5)

    for xv in x_bounds[1:-1]:
        ax.plot([xv, xv], [0, 100], 'w--', alpha=0.15, lw=1, zorder=4)
    for yv in y_bounds[1:-1]:
        ax.plot([0, 100], [yv, yv], 'w--', alpha=0.15, lw=1, zorder=4)

    for r, lbl in enumerate(['DEF. THIRD', 'MIDFIELD', 'ATT. THIRD']):
        ax.text((x_bounds[r] + x_bounds[r + 1]) / 2, 103.5, lbl,
                color=(1, 1, 1, 0.55), fontsize=7.5, fontweight='600',
                ha='center', va='bottom')
    for c, lbl in enumerate(['LEFT', 'CENTER', 'RIGHT']):
        ax.text(-4, (y_bounds[c] + y_bounds[c + 1]) / 2, lbl,
                color=(1, 1, 1, 0.55), fontsize=6.5, fontweight='600',
                ha='right', va='center')

    ax.set_xlim(-7, 104)
    ax.set_ylim(-4, 107)
    ax.set_aspect('equal')
    ax.axis('off')
    return _fig_to_b64(fig)


def _build_transition_risk_pitch(coords_list, is_att=True, filter_mode='shots'):
    """
    Matplotlib risk map of transition starting locations (regains/losses)
    along with coordinates of shots taken within 10s.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.5), facecolor=_PITCH_BG)
    ax.set_facecolor('#112213') # Sleek dark-green pitch

    # Pitch stripes
    sw = 100 / 8
    for i in range(8):
        ax.fill_between([i * sw, (i + 1) * sw], 0, 100,
                        color='#18331b' if i % 2 == 0 else '#122615', alpha=0.9, zorder=0)

    # Pitch markings
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], 'w-', alpha=0.5, lw=1.5, zorder=3)
    ax.plot([50, 50], [0, 100], 'w-', alpha=0.3, lw=1.2, zorder=3)
    from matplotlib.patches import Circle as _MplCircle
    ax.add_patch(_MplCircle((50, 50), 9.15, color='w', fill=False, alpha=0.25, lw=1, zorder=3))
    ax.plot([0, 17, 17, 0],   [21.1, 21.1, 78.9, 78.9], 'w-', alpha=0.3, lw=1.2, zorder=3)
    ax.plot([0, 5.5, 5.5, 0], [40.5, 40.5, 59.5, 59.5], 'w-', alpha=0.2, lw=0.8, zorder=3)
    ax.plot([100, 83, 83, 100],       [21.1, 21.1, 78.9, 78.9], 'w-', alpha=0.3, lw=1.2, zorder=3)
    ax.plot([100, 94.5, 94.5, 100],   [40.5, 40.5, 59.5, 59.5], 'w-', alpha=0.2, lw=0.8, zorder=3)

    if filter_mode == 'goals':
        visible_transitions = [
            c for c in coords_list
            if any(sc.get('event') == 'Goal' for sc in c.get('shot_coords', []))
        ]
    else:
        visible_transitions = [c for c in coords_list if c.get('shot_coords')]

    xs = [c['x'] for c in visible_transitions]
    ys = [c['y'] for c in visible_transitions]
    
    regain_color = '#22c55e' if is_att else '#ef4444'
    regain_label = 'Ball Regain Start' if is_att else 'Ball Loss Start'

    if xs:
        # Draw translucent scatter dots of starting actions
        ax.scatter(xs, ys, color=regain_color, alpha=0.55, s=42, label=regain_label, zorder=4, edgecolors='none')
        
    # Now let's extract all shot coordinates and goals
    shots_x = []
    shots_y = []
    goals_x = []
    goals_y = []
    
    for c in visible_transitions:
        for sc in c.get('shot_coords', []):
            if filter_mode == 'goals' and sc.get('event') != 'Goal':
                continue
            if sc.get('event') == 'Goal':
                goals_x.append(sc['x'])
                goals_y.append(sc['y'])
            else:
                shots_x.append(sc['x'])
                shots_y.append(sc['y'])
                
    # Plot non-goal shots as bright orange/gold markers
    if shots_x:
        ax.scatter(shots_x, shots_y, color='#fbbf24', alpha=0.9, s=90, marker='o',
                   label='Shot Attempt (10s)', zorder=5, edgecolors='black', linewidths=1.2)
    # Plot goals as larger glowing stars
    if goals_x:
        ax.scatter(goals_x, goals_y, color='#22c55e', alpha=1.0, s=160, marker='*',
                   label='Goal Converted (10s)', zorder=6, edgecolors='black', linewidths=1.2)
        
    # Draw connections (lines) from transition start to shot coordinate if it exists
    for c in visible_transitions:
        for sc in c.get('shot_coords', []):
            if filter_mode == 'goals' and sc.get('event') != 'Goal':
                continue
            ax.plot([c['x'], sc['x']], [c['y'], sc['y']], color='white', alpha=0.2, linestyle=':', lw=1, zorder=4)

    # Add zone-based shot density indicator
    total_shots = len(shots_x) + len(goals_x)
    box_shots = sum(1 for x, y in zip(shots_x + goals_x, shots_y + goals_y) if x >= 83.0 and 21.1 <= y <= 78.9)
    box_pct = round(box_shots / max(total_shots, 1) * 100, 1) if total_shots > 0 else 0
    
    title_prefix = "OFFENSIVE TRANSITIONS" if is_att else "DEFENSIVE TRANSITIONS"
    mode_label = {'shots': 'SHOT-PRODUCING STARTS', 'goals': 'GOAL-PRODUCING STARTS'}.get(filter_mode, 'SHOT-PRODUCING STARTS')
    ax.set_title(f"{title_prefix} — {mode_label} (10s WINDOW)\n{len(visible_transitions)} transitions shown | {total_shots} shots | {box_pct}% inside penalty box",
                 color='white', fontsize=10, fontweight='bold', pad=10)

    # Legend
    legend = ax.legend(loc='lower left', facecolor='#112213', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=8)
    handles = getattr(legend, 'legend_handles', None) or getattr(legend, 'legendHandles', None)
    if handles:
        for lh in handles:
            lh.set_alpha(1.0)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.axis('off')
    return _fig_to_b64(fig)


def _build_setpiece_corner_pitch(corners):
    """Attacking half pitch showing corner delivery sides and trajectory types."""
    left_c  = corners.get('left', 0)
    right_c = corners.get('right', 0)
    ins     = corners.get('inswinger', 0)
    out     = corners.get('outswinger', 0)
    strg    = corners.get('straight', 0)
    total_c = max(left_c + right_c, 1)

    fig, ax = plt.subplots(figsize=(5.0, 7.5), facecolor=_PITCH_BG)
    ax.set_facecolor('#1a3a1e')

    for i in range(6):
        y0_s = i * (100 / 6)
        ax.fill_between([50, 100], y0_s, y0_s + 100 / 6,
                        color='#1e4a25' if i % 2 == 0 else '#173519', alpha=0.85, zorder=0)

    ax.plot([50, 100, 100, 50, 50], [0, 0, 100, 100, 0], 'w-', alpha=0.55, lw=1.5, zorder=3)
    ax.plot([100, 83, 83, 100],       [21.1, 21.1, 78.9, 78.9], 'w-', alpha=0.5,  lw=1.2, zorder=3)
    ax.plot([100, 94.5, 94.5, 100],   [40.5, 40.5, 59.5, 59.5], 'w-', alpha=0.3,  lw=0.9, zorder=3)
    ax.plot([100, 102, 102, 100],     [45.2, 45.2, 54.8, 54.8], 'w-', alpha=0.5,  lw=2,   zorder=3)
    ax.scatter([89], [50], color='white', s=12, alpha=0.55, zorder=4)
    from matplotlib.patches import Arc as _MplArc
    ax.add_patch(_MplArc((50, 50),   18.3, 18.3, theta1=270, theta2=90, color='w', alpha=0.3, lw=1.2, zorder=3))
    ax.add_patch(_MplArc((100, 0),   4,    4,    theta1=90,  theta2=180, color='w', alpha=0.3, lw=1,   zorder=3))
    ax.add_patch(_MplArc((100, 100), 4,    4,    theta1=180, theta2=270, color='w', alpha=0.3, lw=1,   zorder=3))

    if left_c > 0:
        lp = round(left_c / total_c * 100, 0)
        ax.annotate('', xy=(87, 42), xytext=(100, 3),
                    arrowprops=dict(arrowstyle='->', color='#00e5ff', lw=2.2,
                                   connectionstyle='arc3,rad=-0.3'), zorder=6)
        ax.scatter([100], [2], color='#00e5ff', s=160, zorder=7, edgecolors='white', linewidths=0.8)
        ax.text(97.5, 12, f'LEFT CORNER\n{left_c} kicks  ({lp:.0f}%)',
                color='#00e5ff', fontsize=7, fontweight='bold', ha='right', va='bottom', zorder=7)

    if right_c > 0:
        rp = round(right_c / total_c * 100, 0)
        ax.annotate('', xy=(87, 58), xytext=(100, 97),
                    arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=2.2,
                                   connectionstyle='arc3,rad=0.3'), zorder=6)
        ax.scatter([100], [98], color='#fbbf24', s=160, zorder=7, edgecolors='white', linewidths=0.8)
        ax.text(97.5, 88, f'RIGHT CORNER\n{right_c} kicks  ({rp:.0f}%)',
                color='#fbbf24', fontsize=7, fontweight='bold', ha='right', va='top', zorder=7)

    type_cfgs = [
        ('Inswinger',  ins,  '#22c55e', -6),
        ('Outswinger', out,  '#a855f7',  0),
        ('Straight',   strg, '#f97316',  6),
    ]
    for t_label, cnt, col, dy in type_cfgs:
        if cnt > 0:
            ax.scatter([91], [50 + dy], color=col, s=min(cnt * 14, 240),
                       alpha=0.65, zorder=5, edgecolors='white', linewidths=0.6)
            ax.text(88.5, 50 + dy, f'{t_label}\n{cnt}', color=col, fontsize=6.5,
                    fontweight='700', ha='right', va='center', zorder=5)

    ax.set_xlim(48, 104)
    ax.set_ylim(-4, 104)
    ax.axis('off')
    return _fig_to_b64(fig)


def _safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _goal_mouth_xy(placement):
    """Convert Opta goal-mouth coordinates into a front-facing goal diagram."""
    mouth_y = _safe_float(placement.get('goal_mouth_y'))
    mouth_z = _safe_float(placement.get('goal_mouth_z'))
    if mouth_y is None:
        return None
    x = (mouth_y - GOAL_MOUTH_LEFT_OPT) / (GOAL_MOUTH_RIGHT_OPT - GOAL_MOUTH_LEFT_OPT)
    # Opta goal-mouth height is expressed on an approximately 0-38 scale.
    y = (mouth_z or 0.0) / 38.0
    return x, y


def _draw_goal_mouth(ax, title, placements):
    """Draw a front-facing goal and plot goals, saves, and misses distinctly."""
    ax.set_facecolor('#151f19')
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='white', alpha=0.72, linewidth=2)
    for x in np.linspace(0, 1, 7):
        ax.plot([x, x], [0, 1], color='white', alpha=0.11, linewidth=0.6)
    for y in np.linspace(0, 1, 5):
        ax.plot([0, 1], [y, y], color='white', alpha=0.11, linewidth=0.6)
    for placement in placements:
        point = _goal_mouth_xy(placement)
        if point is None:
            continue
        event = placement.get('event')
        if event == 'Goal':
            color, marker, label = '#22c55e', 'o', 'Goal'
        elif event == 'Saved Shot':
            color, marker, label = '#3b82f6', 's', 'Saved'
        else:
            color, marker, label = '#ef4444', 'X', 'Miss / Post'
        ax.scatter([point[0]], [point[1]], s=82, color=color, marker=marker,
                   edgecolors='black', linewidths=0.8, alpha=0.94, zorder=5,
                   label=label)
        if placement.get('num') is not None:
            ax.text(point[0], point[1], str(placement.get('num')),
                    color='white', fontsize=6, fontweight='bold',
                    ha='center', va='center', zorder=6)
    ax.set_title(title, color='#fbbf24', fontsize=10, fontweight='bold', loc='left', pad=8)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        legend = ax.legend(unique.values(), unique.keys(), loc='upper center',
                           bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False,
                           fontsize=7)
        for text in legend.get_texts():
            text.set_color('white')
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.20, 1.42)
    ax.axis('off')


def _build_setpiece_overview(penalties, freekicks, goalkicks):
    """Football-specific restart panel with goals, deliveries, and destinations."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor=_PITCH_BG)
    fig.subplots_adjust(wspace=0.28)

    penalty_ax, fk_ax, goalkick_ax = axes.flat

    _draw_goal_mouth(penalty_ax, 'PENALTIES — GOAL PLACEMENT', penalties.get('placements', []))
    penalty_ax.text(0.5, 1.10,
                    f"{penalties.get('scored', 0)} scored  |  {penalties.get('saved', 0)} saved  |  {penalties.get('missed', 0)} missed",
                    color='white', fontsize=8, fontweight='bold', ha='center')

    _draw_goal_mouth(fk_ax, 'DIRECT FREE KICKS — GOAL PLACEMENT', freekicks.get('placements', []))
    fk_ax.text(0.5, 1.10,
               f"{freekicks.get('direct_shots', 0)} shots  |  {freekicks.get('direct_goals', 0)} goals",
               color='white', fontsize=8, fontweight='bold', ha='center')

    goalkick_ax.set_facecolor('#151f19')
    zone_counts = np.zeros((2, 3), dtype=int)
    for destination in goalkicks.get('destinations', []):
        x = _safe_float(destination.get('x'))
        y = _safe_float(destination.get('y'))
        if x is not None and y is not None:
            x_idx = 0 if x < 50 else 1
            y_idx = min(2, max(0, int(y / (100 / 3))))
            zone_counts[x_idx, y_idx] += 1

    x_bounds = [0, 50, 100]
    y_bounds = [0, 100 / 3, 200 / 3, 100]
    colors = ['#22c55e', '#3b82f6']
    max_zone = max(1, int(zone_counts.max()))
    for x_idx in range(2):
        for y_idx in range(3):
            count = int(zone_counts[x_idx, y_idx])
            goalkick_ax.fill_between(
                [x_bounds[x_idx], x_bounds[x_idx + 1]],
                y_bounds[y_idx], y_bounds[y_idx + 1],
                color=colors[x_idx], alpha=max(0.08, count / max_zone * 0.52),
            )
            goalkick_ax.text(
                (x_bounds[x_idx] + x_bounds[x_idx + 1]) / 2,
                (y_bounds[y_idx] + y_bounds[y_idx + 1]) / 2,
                str(count), color='white', fontsize=12, fontweight='bold',
                ha='center', va='center',
            )

    goalkick_ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0],
                     color='white', alpha=0.55, linewidth=1.3)
    goalkick_ax.plot([50, 50], [0, 100], color='white', alpha=0.40, linewidth=1.1)
    for y_value in y_bounds[1:-1]:
        goalkick_ax.plot([0, 100], [y_value, y_value], color='white',
                         alpha=0.22, linewidth=0.9, linestyle='--')
    goalkick_ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9],
                     color='white', alpha=0.45, linewidth=1.0)
    goalkick_ax.text(25, 103, 'SHORT (<50)', color='#22c55e',
                     fontsize=8, fontweight='bold', ha='center')
    goalkick_ax.text(75, 103, 'LONG (>=50)', color='#3b82f6',
                     fontsize=8, fontweight='bold', ha='center')
    for y_idx, label in enumerate(['LEFT', 'CENTER', 'RIGHT']):
        goalkick_ax.text(-4, (y_bounds[y_idx] + y_bounds[y_idx + 1]) / 2,
                         label, color=(1, 1, 1, 0.52), fontsize=7,
                         fontweight='bold', ha='right', va='center')

    total = max(1, goalkicks.get('total', 0))
    short_pct = round(goalkicks.get('short', 0) / total * 100, 1)
    long_pct = round(goalkicks.get('long', 0) / total * 100, 1)
    goalkick_ax.set_title('GOAL KICKS — DELIVERY CHOICE', color='#fbbf24',
                          fontsize=10, fontweight='bold', loc='left', pad=8)
    goalkick_ax.text(50, -9,
                     f"{goalkicks.get('short', 0)} short ({short_pct}%)  |  "
                     f"{goalkicks.get('long', 0)} long ({long_pct}%)",
                     color='white', fontsize=8, fontweight='bold', ha='center')
    goalkick_ax.set_xlim(-7, 103)
    goalkick_ax.set_ylim(-13, 103)
    goalkick_ax.axis('off')

    fig.suptitle('SET-PIECE & RESTART VISUALS OVERVIEW', color='white',
                 fontsize=13, fontweight='bold', y=0.99)
    return _fig_to_b64(fig)


def _build_penalty_goal_mouth(penalties):
    placements = penalties.get('placements', [])
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=_PITCH_BG)
    ax.set_facecolor('#101913')

    # Goal frame
    ax.plot([0, 1], [0, 0], color='white', alpha=0.92, linewidth=3)
    ax.plot([0, 0], [0, 1], color='white', alpha=0.92, linewidth=3)
    ax.plot([1, 1], [0, 1], color='white', alpha=0.92, linewidth=3)
    ax.plot([0, 1], [1, 1], color='white', alpha=0.92, linewidth=3)

    # Net grid and zones
    for x in np.linspace(0.2, 0.8, 4):
        ax.plot([x, x], [0, 1], color='white', alpha=0.12, linewidth=1)
    for y in np.linspace(0.25, 0.75, 3):
        ax.plot([0, 1], [y, y], color='white', alpha=0.12, linewidth=1)
    ax.axvline(0.5, color=_GOLD, alpha=0.25, linestyle='--', linewidth=1)

    zone_label_color = (1, 1, 1, 0.55)
    ax.text(0.17, -0.08, 'LEFT', color=zone_label_color, fontsize=8,
            fontweight='bold', ha='center')
    ax.text(0.50, -0.08, 'CENTRE', color=zone_label_color, fontsize=8,
            fontweight='bold', ha='center')
    ax.text(0.83, -0.08, 'RIGHT', color=zone_label_color, fontsize=8,
            fontweight='bold', ha='center')

    for placement in placements:
        point = _goal_mouth_xy(placement)
        if point is None:
            continue
        x, y = point
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        event = placement.get('event')
        if event == 'Goal':
            color, marker, label = _GREEN, 'o', 'Goal'
        elif event == 'Saved Shot':
            color, marker, label = _BLUE, 's', 'Saved'
        else:
            color, marker, label = _RED, 'X', 'Miss / Post'

        ax.scatter([x], [y], s=240, color=color, marker=marker,
                   edgecolors='black', linewidths=1.5, alpha=0.96,
                   label=label, zorder=5)
        num = placement.get('num')
        if num is not None:
            ax.text(x, y, str(num), color='white', fontsize=9,
                    fontweight='900', ha='center', va='center', zorder=6)

        player = placement.get('player')
        minute = placement.get('minute')
        if player:
            label_text = f"{player.split()[-1]} {minute}'" if minute is not None else player.split()[-1]
            ax.annotate(label_text, xy=(x, y), xytext=(0, 16), textcoords='offset points',
                        color='white', fontsize=7, ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='-', color='white', alpha=0.22, lw=0.7),
                        zorder=7)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        legend = ax.legend(unique.values(), unique.keys(), loc='upper center',
                           bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False,
                           fontsize=8)
        for text in legend.get_texts():
            text.set_color('white')

    ax.text(0.5, 1.12, 'PENALTY GOAL MOUTH', color=_GOLD,
            fontsize=13, fontweight='bold', ha='center')
    ax.text(0.5, 1.055,
            f"{penalties.get('scored', 0)} scored | {penalties.get('saved', 0)} saved | {penalties.get('missed', 0)} missed",
            color='white', fontsize=9, fontweight='bold', ha='center')

    if not placements:
        ax.text(0.5, 0.5, 'No penalty attempts recorded', color='white',
                fontsize=12, fontweight='bold', ha='center', va='center', alpha=0.75)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.16, 1.18)
    ax.axis('off')
    fig.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def get_ordinal(n):
    if n == '-':
        return '-'
    try:
        n = int(n)
    except ValueError:
        return str(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def make_progress_bar(label, pct, color, rank_str=None):
    label_span = html.Span(label, style={"color": "var(--text-primary)"})
    if rank_str:
        label_span = html.Span([
            label,
            html.Span(f" ({rank_str})", style={"color": _GOLD, "fontSize": "0.68rem", "marginLeft": "6px", "fontWeight": "700"})
        ])
    return html.Div(style={"marginBottom": "14px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.76rem", "marginBottom": "4px", "fontWeight": "600"}, children=[
            label_span,
            html.Span(f"{pct}%", style={"color": color, "fontWeight": "700"})
        ]),
        html.Div(style={"height": "8px", "background": "rgba(255,255,255,0.06)", "borderRadius": "4px", "overflow": "hidden"}, children=[
            html.Div(style={"width": f"{pct}%", "height": "100%", "background": color, "borderRadius": "4px"})
        ])
    ])


def _build_buildup_rank_section(opponent):
    league_df = _load_league_benchmarks()
    if league_df.empty or opponent not in league_df.index:
        return html.Div(style={
            "padding": "12px", "borderRadius": "8px",
            "background": "rgba(251,191,36,0.06)",
            "border": "1px solid rgba(251,191,36,0.18)",
            "fontSize": "0.72rem", "color": "var(--text-secondary)",
        }, children="League ranking is loading in the background.")

    rank_metrics = [
        ("buildup_f3_pct", "Final Third Entry", True, _GOLD),
        ("buildup_shot_pct", "Shot Creation", True, _BLUE),
        ("buildup_goal_pct", "Goal Conversion", True, _GREEN),
        ("buildup_turnover_pct", "Turnover Risk", False, _RED),
        ("buildup_danger_pct", "Danger after Turnover", False, _PURPLE),
    ]
    n_teams = len(league_df)
    cards = []
    for column, label, higher_is_better, color in rank_metrics:
        value = float(league_df.loc[opponent, column])
        rank = int(league_df[column].rank(
            ascending=not higher_is_better,
            method='min',
        )[opponent])
        cards.append(html.Div(style={
            "padding": "10px", "borderRadius": "8px",
            "background": "rgba(255,255,255,0.025)",
            "border": "1px solid var(--border-color)",
            "textAlign": "center",
        }, children=[
            html.Div(label, style={
                "fontSize": "0.6rem", "fontWeight": "800",
                "color": "var(--text-secondary)", "textTransform": "uppercase",
                "letterSpacing": "0.4px", "minHeight": "30px",
            }),
            html.Div(f"{value:.1f}%", style={
                "fontSize": "1.05rem", "fontWeight": "800", "color": color,
            }),
            html.Div(f"{get_ordinal(rank)} of {n_teams}", style={
                "fontSize": "0.64rem", "fontWeight": "700", "color": "white",
            }),
        ]))

    return html.Div([
        html.Div("LEAGUE RANKING — 10s BUILD-UP OUTCOMES", style={
            "fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD,
            "letterSpacing": "0.5px", "marginBottom": "10px",
        }),
        html.Div(cards, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(110px, 1fr))",
            "gap": "8px",
        }),
    ])


def _build_kpi_section(active_tab, stats, opp_name):
    # Retrieve KPIs based on active_tab
    if active_tab in ('game-plan-tab', 'offensive-tab'):
        return html.Div()
        
    kpi_list = []
    if active_tab == 'defensive-tab':
        line = stats['def_profile'].get('avg_def_line', 0)
        aerial = stats['def_profile'].get('box_aerial_win_pct', 0)
        z14 = stats['def_profile'].get('z14_success_allowed_pct', 0)
        kpi_list = [
            ("PRESSING HEIGHT", f"{line:.1f}m", "Average height of defensive actions"),
            ("BOX AERIAL DOMINANCE", f"{aerial}%", "Aerial duel success rate in defensive box"),
            ("ZONE 14 VULNERABILITY", f"{z14}%", "Opponent pass completion outside penalty area")
        ]
    elif active_tab == 'off-trans-tab':
        all_rec = stats.get('all_recoveries', [])
        total_r = len(all_rec)
        shots_count = sum(1 for x in all_rec if x['shot'])
        goals_count = sum(1 for x in all_rec if x['goal'])
        shot_rate = round(shots_count / max(total_r, 1) * 100, 1) if total_r > 0 else 0
        goal_rate = round(goals_count / max(total_r, 1) * 100, 1) if total_r > 0 else 0
        kpi_list = [
            ("TOTAL BALL REGAINS", f"{total_r}", "Transitions won back over season"),
            ("10s SHOT ATTEMPT RATE", f"{shot_rate}%", "Percentage of regains leading to a shot within 10s"),
            ("GOALS FROM 10s TRANSITIONS", f"{goals_count}", f"{goal_rate}% of regains resulted in a goal")
        ]
    elif active_tab == 'def-trans-tab':
        all_los = stats.get('all_losses', [])
        total_l = len(all_los)
        shots_conceded = sum(1 for x in all_los if x['shot'])
        goals_conceded = sum(1 for x in all_los if x['goal'])
        danger_rate = round(shots_conceded / max(total_l, 1) * 100, 1) if total_l > 0 else 0
        conceded_goal_rate = round(goals_conceded / max(total_l, 1) * 100, 1) if total_l > 0 else 0
        kpi_list = [
            ("TOTAL BALL LOSSES", f"{total_l}", "Transitions lost over season"),
            ("10s DANGER CONCEDED RATE", f"{danger_rate}%", "Percentage of losses leading to opp. shot within 10s"),
            ("GOALS CONCEDED FROM 10s TRANSITIONS", f"{goals_conceded}", f"{conceded_goal_rate}% of losses resulted in a goal")
        ]
    elif active_tab == 'set-pieces-tab':
        ins = stats['corners'].get('inswinger', 0)
        out = stats['corners'].get('outswinger', 0)
        strg = stats['corners'].get('straight', 0)
        pref = "Inswing" if ins >= out and ins >= strg else "Outswing" if out >= ins and out >= strg else "Straight"
        gks = stats['goalkicks'].get('long_pct', 0)
        
        fk_shots = stats['freekicks'].get('direct_shots', 0)
        fk_goals = stats['freekicks'].get('direct_goals', 0)
        fk_conv = round(fk_goals / max(fk_shots, 1) * 100, 1) if fk_shots > 0 else 0
        
        kpi_list = [
            ("CORNER DELIVERY PREF", pref, "Primary corner kick trajectory style"),
            ("LONG GOALKICK RATIO", f"{gks}%", "Goalkicks landing past the halfway line"),
            ("DIRECT FK SHOT CONV.", f"{fk_conv}%", f"{fk_goals} direct free-kick goals from {fk_shots} shots")
        ]
        
    # Render cards
    cards = []
    for label, val, desc in kpi_list:
        cards.append(dbc.Col(html.Div(className="coach-brief-item", style={
            "minHeight": "auto", "padding": "16px 20px", "textAlign": "center"
        }, children=[
            html.Div(label, className="coach-brief-label", style={"fontSize": "0.64rem", "marginBottom": "4px"}),
            html.Div(val, className="coach-brief-text", style={"fontSize": "1.7rem", "fontWeight": "700", "color": _GOLD, "marginBottom": "4px"}),
            html.Div(desc, style={"fontSize": "0.68rem", "color": "var(--text-secondary)"})
        ]), width=4))
        
    return dbc.Row(cards, style={"maxWidth": "1200px", "margin": "0 auto 20px"})


def _largest_buildup_lane(stats):
    zones = stats.get('buildup', {}).get('zone', {})
    lanes = {
        'left': zones.get('left_pct', 0) or 0,
        'center': zones.get('center_pct', 0) or 0,
        'right': zones.get('right_pct', 0) or 0,
    }

    if not any(lanes.values()):
        coords = stats.get('buildup', {}).get('coords', [])
        lanes = {
            'left': sum(1 for c in coords if float(c.get('y', 50)) < 33.3),
            'center': sum(1 for c in coords if 33.3 <= float(c.get('y', 50)) <= 66.6),
            'right': sum(1 for c in coords if float(c.get('y', 50)) > 66.6),
        }
        total = max(sum(lanes.values()), 1)
        lanes = {k: round(v / total * 100, 1) for k, v in lanes.items()}

    lane = max(lanes, key=lanes.get)
    return lane, lanes.get(lane, 0)


def _plan_card(title, body, color=_GOLD, tag=None):
    return html.Div(className="coach-brief-item", style={
        "minHeight": "auto",
        "padding": "16px 18px",
        "height": "100%",
        "borderLeft": f"3px solid {color}",
    }, children=[
        html.Div(tag, className="coach-brief-label", style={
            "fontSize": "0.58rem", "color": color, "marginBottom": "6px",
        }) if tag else None,
        html.Div(title, className="coach-brief-text", style={
            "fontSize": "0.95rem", "fontWeight": "800", "color": "white",
            "marginBottom": "8px",
        }),
        html.Div(body, style={
            "fontSize": "0.74rem", "color": "var(--text-secondary)",
            "lineHeight": "1.55",
        }),
    ])


def _build_game_plan_content(stats, opp_name, opponent):
    buildup = stats.get('buildup', {})
    outcomes = buildup.get('outcomes_10s', {})
    def_profile = stats.get('def_profile', {})
    recoveries = stats.get('all_recoveries', [])
    losses = stats.get('all_losses', [])
    corners = stats.get('corners', {})
    goalkicks = stats.get('goalkicks', {})

    lane, lane_pct = _largest_buildup_lane(stats)
    lane_label = {'left': 'left side', 'center': 'central lane', 'right': 'right side'}[lane]
    force_lane = {'left': 'inside toward our midfield trap', 'center': 'wide, away from Zone 14', 'right': 'inside toward our midfield trap'}[lane]

    total_rec = len(recoveries)
    rec_shot_rate = round(sum(1 for r in recoveries if r.get('shot')) / max(total_rec, 1) * 100, 1) if total_rec else 0
    total_losses = len(losses)
    loss_shot_rate = round(sum(1 for l in losses if l.get('shot')) / max(total_losses, 1) * 100, 1) if total_losses else 0
    loss_f3_rate = round(sum(1 for l in losses if l.get('reached_f3')) / max(total_losses, 1) * 100, 1) if total_losses else 0
    turnover_pct = outcomes.get('turnover_pct', 0)
    danger_after_turnover = outcomes.get('opp_danger_pct', 0)

    def_line = def_profile.get('avg_def_line', 0) or 0
    z14_allowed = def_profile.get('z14_success_allowed_pct', 0) or 0
    aerial = def_profile.get('box_aerial_win_pct', 0) or 0

    if def_line >= 45:
        attacking_route = "Attack the space behind their back line early. Prepare diagonal runs from the weak-side winger and first-time passes after regains."
        route_label = "Exploit High Line"
    elif def_line <= 35 and def_line > 0:
        attacking_route = "Be patient against their lower block. Move them side to side, arrive in Zone 14, then use cutbacks instead of hopeful crosses."
        route_label = "Break Low Block"
    else:
        attacking_route = "Use mixed attacks: secure the first pass, then play quickly into the half-spaces before their midfield can reset."
        route_label = "Control Half-Spaces"

    if aerial >= 58:
        crossing_plan = "Avoid repeated floated crosses into their centre-backs. Prefer low crosses, cutbacks, and second-post runs after switches."
    elif aerial and aerial <= 45:
        crossing_plan = "Attack the box with early crosses and back-post overloads. Their defensive aerial numbers suggest we can compete there."
    else:
        crossing_plan = "Cross selectively after moving their block. The better route is still low delivery into runners, not static box service."

    if rec_shot_rate >= 18:
        rest_defense = "Keep two plus one behind the ball when attacking. On loss, first defender delays, nearest midfielder blocks the forward pass, far fullback tucks in."
    else:
        rest_defense = "Counter-press aggressively after loss, but keep the far-side balance. Their transition shot rate is manageable if the first pass is blocked."

    if loss_shot_rate >= 14 or loss_f3_rate >= 35:
        transition_attack = "When we win it, play vertical immediately. First look: forward run into the channel, second look: Zone 14 support, third look: switch to the far winger."
    else:
        transition_attack = "Use the regain to pin them back, but do not force every counter. If the forward pass is closed, secure possession and attack the next wave."

    corner_pref = "inswingers" if corners.get('inswinger', 0) >= corners.get('outswinger', 0) and corners.get('inswinger', 0) >= corners.get('straight', 0) else "outswingers" if corners.get('outswinger', 0) >= corners.get('straight', 0) else "straight deliveries"
    gk_plan = "Press their short goal kicks" if goalkicks.get('long_pct', 0) < 45 else "Prepare for second balls from long goal kicks"

    priority_cards = dbc.Row([
        dbc.Col(_plan_card(
            "Press Their Build-Up Lane",
            f"{opp_name} start most often through the {lane_label} ({lane_pct}%). Show them {force_lane}, then jump on the backwards pass.",
            _RED,
            "WITHOUT THE BALL",
        ), md=4),
        dbc.Col(_plan_card(
            route_label,
            attacking_route,
            _GOLD,
            "WITH THE BALL",
        ), md=4),
        dbc.Col(_plan_card(
            "Transition Rule",
            transition_attack,
            _GREEN,
            "WHEN WE WIN IT",
        ), md=4),
    ], className="g-3", style={"marginBottom": "18px"})

    detail_cards = dbc.Row([
        dbc.Col(_plan_card(
            "Defensive Transitions",
            f"{opp_name}'s regains become shots {rec_shot_rate}% of the time. {rest_defense}",
            _RED,
        ), md=6),
        dbc.Col(_plan_card(
            "Final Third Choices",
            f"Zone 14 allowed: {z14_allowed}%. Box aerial win rate: {aerial}%. {crossing_plan}",
            _BLUE,
        ), md=6),
        dbc.Col(_plan_card(
            "Set-Piece Preparation",
            f"Main corner profile: {corner_pref}. {gk_plan}. Track their first two corner targets and protect the second-ball zone.",
            _PURPLE,
        ), md=6),
        dbc.Col(_plan_card(
            "Pressing Trigger",
            f"Their build-up turnover rate is {turnover_pct}%, with {danger_after_turnover}% danger after those losses. Trigger pressure on poor body shape, square passes, and first touch toward their own goal.",
            _GOLD,
        ), md=6),
    ], className="g-3")

    principles = html.Div(className="goz-form-section", style={
        "maxWidth": "980px", "margin": "0 auto 20px", "padding": "18px 20px",
    }, children=[
        html.Div(className="goz-section-header", style={
            "marginBottom": "12px",
            "flexDirection": "column",
            "alignItems": "flex-start",
            "gap": "8px",
        }, children=[
            html.Span(f"MATCH PLAN VS {opp_name.upper()}", className="goz-card-title", style={
                "fontSize": "1.15rem",
                "margin": 0,
            }),
            html.P("Coach-facing actions for how Göztepe should play this opponent.", className="goz-card-desc", style={
                "margin": 0,
            }),
        ]),
        dbc.Row([
            dbc.Col(make_progress_bar("Build-up Turnover Target", turnover_pct, _RED), md=3),
            dbc.Col(make_progress_bar("Counter-Attack Opportunity", loss_shot_rate, _GREEN), md=3),
            dbc.Col(make_progress_bar("Opponent Transition Threat", rec_shot_rate, _RED), md=3),
            dbc.Col(make_progress_bar("Zone 14 Access Allowed", z14_allowed, _BLUE), md=3),
        ]),
    ])

    return html.Div([
        principles,
        html.Div(style={"maxWidth": "1180px", "margin": "0 auto"}, children=[
            priority_cards,
            detail_cards,
        ]),
    ])


def _rank_text(team, column, higher_is_better=True):
    try:
        rank, total = _league_rank_for_metric(team, column, higher_is_better)
        return f"#{rank}/{total}"
    except Exception:
        return "n/a"


def _build_pre_match_report(opponent):
    if not opponent:
        return html.Div()

    stats = _get_season_stats(opponent)
    opp_name = _clean(opponent)
    buildup = stats.get('buildup', {})
    corners = stats.get('corners', {})
    goalkicks = stats.get('goalkicks', {})
    recoveries = stats.get('all_recoveries', [])
    losses = stats.get('all_losses', [])
    def_profile = stats.get('def_profile', {})
    lane, lane_pct = _largest_buildup_lane(stats)
    lane_label = {'left': 'Left side', 'center': 'Central lane', 'right': 'Right side'}[lane]
    rec_shot_rate = round(sum(1 for r in recoveries if r.get('shot')) / max(len(recoveries), 1) * 100, 1) if recoveries else 0
    loss_shot_rate = round(sum(1 for l in losses if l.get('shot')) / max(len(losses), 1) * 100, 1) if losses else 0
    turnover_pct = buildup.get('outcomes_10s', {}).get('turnover_pct', 0)
    danger_after_turnover = buildup.get('outcomes_10s', {}).get('opp_danger_pct', 0)
    corner_pref = "inswinger" if corners.get('inswinger', 0) >= corners.get('outswinger', 0) else "outswinger"
    opponent_logo = "/" + TEAM_LOGOS.get(opponent, "assets/logo.png").lstrip("/")

    return html.Div(className="report-page", children=[
        html.Div(className="report-header", children=[
            html.Img(src="/assets/logo.png", className="report-logo"),
            html.Div([
                html.Div("tactIQ", className="report-brand"),
                html.Div("Pre-Match Report", className="report-kicker"),
            ]),
        ]),
        html.Div(className="report-opponent-title", children=[
            html.Img(src=opponent_logo, className="report-opponent-logo"),
            html.H1(opp_name, className="report-title"),
        ]),
        html.P(f"Opponent-focused match plan for {opp_name}, generated from season event data.", className="report-subtitle"),
        html.Div(className="report-section", children=[
            html.H3("Game Plan"),
            html.Ul([
                html.Li(f"Press their main build-up lane: {lane_label} ({lane_pct}%). Force play into our trap before jumping."),
                html.Li(f"On regain, attack quickly: opponent losses become shots {loss_shot_rate}% of the time."),
                html.Li(f"Rest defense priority: their regains become shots {rec_shot_rate}% of the time."),
                html.Li(f"Set-piece prep: main corner tendency is {corner_pref}; long goal-kick rate {goalkicks.get('long_pct', 0)}%."),
            ]),
        ]),
        html.Div(className="report-section", children=[
            html.H3("Key Info"),
            html.Div(className="report-grid", children=[
                html.Div(className="report-pill", children=[html.Strong(f"{turnover_pct}%"), html.Span("Build-up turnover rate")]),
                html.Div(className="report-pill", children=[html.Strong(f"{danger_after_turnover}%"), html.Span("Danger after build-up losses")]),
                html.Div(className="report-pill", children=[html.Strong(f"{def_profile.get('z14_success_allowed_pct', 0)}%"), html.Span("Zone 14 access allowed")]),
                html.Div(className="report-pill", children=[html.Strong(f"{corners.get('total', 0)}"), html.Span("Corners sampled")]),
                html.Div(className="report-pill", children=[html.Strong(f"{len(recoveries)}"), html.Span("Opponent regains")]),
                html.Div(className="report-pill", children=[html.Strong(f"{len(losses)}"), html.Span("Opponent losses")]),
            ]),
        ]),
        html.Div(className="report-section", children=[
            html.H3("League Rankings"),
            html.Div(className="report-grid", children=[
                html.Div(className="report-pill", children=[html.Strong(_rank_text(opponent, 'passes_pg', True)), html.Span("Pass volume")]),
                html.Div(className="report-pill", children=[html.Strong(_rank_text(opponent, 'goals_pg', True)), html.Span("Goals per game")]),
                html.Div(className="report-pill", children=[html.Strong(_rank_text(opponent, 'xga_pg', False)), html.Span("Defensive solidity")]),
            ]),
        ]),
    ])


def _build_tab_content(active_tab, stats, opp_name, opponent):
    if active_tab == 'game-plan-tab':
        return _build_game_plan_content(stats, opp_name, opponent)

    if active_tab == 'offensive-tab':
        buildup_coords = stats['buildup'].get('coords', [])
        b_plot_b64 = _build_buildup_pitch(buildup_coords)
        
        # Outcomes 10s
        outcomes = stats['buildup'].get('outcomes_10s', {})
        f3_pct = outcomes.get('f3_entry_pct', 0)
        shot_pct = outcomes.get('shot_pct', 0)
        goal_pct = outcomes.get('goal_pct', 0)
        turnover_pct = outcomes.get('turnover_pct', 0)
        opp_danger_pct = outcomes.get('opp_danger_pct', 0)
        
        # Goal sequences
        stats.get('goal_sequences', [])
        
        # Calculate preferred attacking/buildup flank
        buildup_zones = stats['buildup'].get('zone', {})
        left_pct = buildup_zones.get('left_pct', 0)
        center_pct = buildup_zones.get('center_pct', 0)
        right_pct = buildup_zones.get('right_pct', 0)
        
        # Fallback using coords density if zone percentages are missing
        if not left_pct and not center_pct and not right_pct:
            left_c = sum(1 for c in buildup_coords if float(c.get('y', 50)) < 33.3)
            center_c = sum(1 for c in buildup_coords if 33.3 <= float(c.get('y', 50)) <= 66.6)
            right_c = sum(1 for c in buildup_coords if float(c.get('y', 50)) > 66.6)
            tot_c = max(1, left_c + center_c + right_c)
            left_pct = round(left_c / tot_c * 100, 1)
            center_pct = round(center_c / tot_c * 100, 1)
            right_pct = round(right_c / tot_c * 100, 1)

        max_val = max(left_pct, center_pct, right_pct)
        if max_val == left_pct:
            pref_side = "Left Flank"
            pref_pct = left_pct
            pref_desc = "They heavily favor build-ups down the left channel, utilizing their left fullback and winger to progress the ball into the final third."
            pref_color = _BLUE
        elif max_val == right_pct:
            pref_side = "Right Flank"
            pref_pct = right_pct
            pref_desc = "They strongly prefer initiating build-ups on the right flank, seeking overloads via their right fullback and wide midfielders."
            pref_color = _RED
        else:
            pref_side = "Center Lane"
            pref_pct = center_pct
            pref_desc = "They prioritize central build-up pathways through their deep playmakers and central defenders, seeking to penetrate lines directly through the middle."
            pref_color = _GOLD

        compact_card_style = {"padding": "16px", "marginBottom": "14px"}

        left_col = dbc.Col(html.Div(className="goz-form-section", style=compact_card_style, children=[
            html.Div(className="goz-section-header", children=[
                html.Span("1. BUILD-UP START ZONES", className="goz-card-title", style={"fontSize": "0.95rem"}),
            ]),
            html.P(
                f"Season view of {stats['buildup'].get('total_buildups', 0)} build-ups "
                f"({stats['buildup'].get('avg_buildups_per_match', 0)} per match).",
                style={"fontSize": "0.68rem", "color": "var(--text-secondary)", "marginBottom": "8px"}
            ),
            html.Img(src=b_plot_b64, style={
                "width": "100%", "maxHeight": "360px", "objectFit": "contain",
                "borderRadius": "10px", "marginBottom": "10px"
            }),
            html.Div(style={"fontSize": "0.7rem", "color": "var(--text-secondary)", "lineHeight": "1.45"}, children=[
                html.Span("Primary lane: ", style={"color": _GOLD, "fontWeight": "700"}),
                html.Span(f"{pref_side} ({pref_pct}%)", style={"color": pref_color, "fontWeight": "800", "fontSize": "0.82rem"}),
                html.Span(f". {pref_desc}", style={"fontStyle": "italic"}),
            ])
        ]), md=6)
        
        # Dynamic F3 entry and outcome breakdown with dropdown selector
        f3_breakdown = html.Div(style={
            "marginTop": "14px", "padding": "16px", "borderRadius": "10px",
            "background": "rgba(255,255,255,0.02)", "border": "1px solid var(--border-color)"
        }, children=[
            html.Div(style={"marginBottom": "10px"}, children=[
                html.Span("FINAL THIRD ENTRY & OUTCOMES", style={"fontSize": "0.78rem", "fontWeight": "800", "color": _GOLD, "letterSpacing": "0.5px"}),
            ]),
            html.Label("SELECT ENTRY FILTER TYPE", className="goz-label", style={"fontSize": "0.68rem", "fontWeight": "700", "color": _GOLD}),
            dcc.Dropdown(
                id='pre-match-f3-entry-type-selector',
                options=[
                    {'label': '🛡️ Build-up Final Third Entries', 'value': 'buildup'},
                    {'label': '🌍 All Final Third Entries (Normal)', 'value': 'all'}
                ],
                value='buildup',
                className="goz-dropdown",
                clearable=False,
                style={"marginBottom": "16px"}
            ),
            
            html.Div(id='pre-match-f3-breakdown-dynamic-content')
        ])

        # Get league standings context for Final Third Entry Rate
        try:
            league_df = _load_league_benchmarks()
        except Exception:
            league_df = pd.DataFrame()

        len(league_df) if not league_df.empty else 18
        if not league_df.empty and opponent in league_df.index:
            s_f3 = league_df['buildup_f3_pct']
            f3_rank = int(s_f3.rank(ascending=False, method='min')[opponent])
            f3_rank_ord = get_ordinal(f3_rank)
            rank_str = f"{f3_rank_ord} in League"
        else:
            rank_str = None

        mid_col_enhanced = dbc.Col(html.Div(className="goz-form-section", style=compact_card_style, children=[
            html.Div(className="goz-section-header", children=[
                html.Span("2. 10s BUILD-UP OUTCOMES", className="goz-card-title", style={"fontSize": "0.95rem"}),
            ]),
            make_progress_bar("Build-up-Derived Final Third Entry Rate", f3_pct, _GOLD, rank_str=rank_str),
            make_progress_bar("Shot Attempt Rate", shot_pct, _BLUE),
            make_progress_bar("Goal Conversion Rate", goal_pct, _GREEN),
            make_progress_bar("Possession Turnover Rate", turnover_pct, _RED),
            make_progress_bar("Conceded Danger after Turnover (10s)", opp_danger_pct, "#a855f7"),
            html.Div(style={"marginTop": "10px", "padding": "9px", "borderRadius": "8px", "background": "rgba(255,255,255,0.02)", "border": "1px solid var(--border-color)", "fontSize": "0.68rem", "color": "var(--text-secondary)"}, children=[
                html.Span("Vulnerability Warning: ", style={"color": _RED, "fontWeight": "700"}),
                "When losing the ball in buildup, they concede a Final Third entry or shot within 10 seconds in ",
                html.Span(f"{opp_danger_pct}%", style={"color": "white", "fontWeight": "700"}), " of the sequences. High-press triggers should be initiated."
            ]),
            html.Div(id='pre-match-buildup-rank-container', style={"marginTop": "12px"},
                     children=_build_buildup_rank_section(opponent)),
            f3_breakdown,
        ]), md=6)

        goal_sequence_card = dbc.Col(html.Div(className="goz-form-section", style=compact_card_style, children=[
            html.Div(className="goz-section-header", style={"marginBottom": "14px"}, children=[
                html.Span("3. GOAL ATTACKS", className="goz-card-title", style={"fontSize": "0.98rem"}),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.Label("FILTER BY GOAL ORIGIN", className="goz-label", style={"fontSize": "0.68rem"}),
                            dcc.Dropdown(
                                id='pre-match-goal-origin-filter',
                                options=[
                                    {'label': '⚽ All Goals', 'value': 'all'},
                                    {'label': '⚡ Open Play', 'value': 'open_play'},
                                    {'label': '📐 From Crosses', 'value': 'from_cross'},
                                    {'label': '🎯 Set Pieces', 'value': 'set_piece'},
                                    {'label': '📥 Through Balls', 'value': 'through_ball'},
                                    {'label': '🛡️ Fast Breaks', 'value': 'fast_break'},
                                ],
                                value='all',
                                className="goz-dropdown",
                                clearable=False,
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("SELECT GOAL TO INSPECT", className="goz-label", style={"fontSize": "0.68rem"}),
                            dcc.Dropdown(
                                id='pre-match-goal-selector',
                                className="goz-dropdown pre-match-goal-dropdown",
                                clearable=False,
                                optionHeight=38,
                            )
                        ], width=6)
                    ], style={"marginBottom": "15px"}),
                    html.Div(id='pre-match-goal-list-container', style={"maxHeight": "360px", "overflowY": "auto"})
                ], md=4),
                dbc.Col([
                    html.Div("COMPLETE GOAL ATTACK — FIRST ACTION TO FINISH", style={"fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD, "marginBottom": "8px", "letterSpacing": "0.5px", "textAlign": "center"}),
                    html.Div(id='pre-match-goal-sequence-graph-container', style={"minHeight": "340px"})
                ], md=8)
            ])
        ]), md=12)

        phase_bench = _build_phase_bench_section('offensive-tab', opponent, opp_name)
        return html.Div([
            phase_bench,
            dbc.Row([left_col, mid_col_enhanced], className="g-3", style={"marginBottom": "14px"}),
            dbc.Row([goal_sequence_card])
        ])

    elif active_tab == 'defensive-tab':
        import utils.visuals as main_visuals
        from göztepehub.utils.transitions_analysis import _load_opponent_matches
        match_dfs = _load_opponent_matches(opponent)
        if match_dfs:
            combined_df = pd.concat([df for fn, df in match_dfs], ignore_index=True)
            try:
                def_profile_b64_raw = main_visuals.plot_defensive_profile(combined_df, opponent)
                def_profile_b64 = f"data:image/png;base64,{def_profile_b64_raw}"
            except Exception as e:
                import traceback
                print(f"Error generating defensive profile: {e}")
                traceback.print_exc()
                def_profile_b64 = ""
        else:
            def_profile_b64 = ""

        left_col = dbc.Col(html.Div(className="goz-form-section", children=[
            html.Div(className="goz-section-header", style={"marginBottom": "20px"}, children=[
                html.Span("Defensive Profile", className="goz-card-title", style={"fontSize": "1.3rem", "fontWeight": "bold", "color": "white"}),
                html.P("Block type, compactness, defensive line height shift between halves, and action breakdown.", className="goz-card-desc"),
            ]),
            html.Div(opp_name.upper(), style={"color": _GOLD, "fontWeight": "700", "fontSize": "0.95rem", "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "15px", "marginTop": "10px"}),
            html.Img(src=def_profile_b64, style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"}) if def_profile_b64 else html.Div("No defensive data available.", style={"padding": "40px", "textAlign": "center", "color": "var(--text-secondary)"})
        ]), md=12)

        phase_bench = _build_phase_bench_section('defensive-tab', opponent, opp_name)
        return html.Div([phase_bench, dbc.Row([left_col])])

    elif active_tab == 'off-trans-tab':
        all_rec = stats.get('all_recoveries', [])
        total_r = len(all_rec)
        avg_passes = round(sum(x['passes'] for x in all_rec) / max(total_r, 1), 1) if total_r > 0 else 0
        avg_carries = round(sum(x['carries'] for x in all_rec) / max(total_r, 1), 1) if total_r > 0 else 0
        f3_entry_rate = round(sum(1 for x in all_rec if x['reached_f3']) / max(total_r, 1) * 100, 1) if total_r > 0 else 0
        shots_count = sum(1 for x in all_rec if x['shot'])
        goals_count = sum(1 for x in all_rec if x['goal'])
        shot_rate = round(shots_count / max(total_r, 1) * 100, 1) if total_r > 0 else 0
        
        # Calculate box shots percentage
        all_shots = []
        for r in all_rec:
            all_shots.extend(r.get('shot_coords', []))
        box_shots = sum(1 for s in all_shots if s['x'] >= 83.0 and 21.1 <= s['y'] <= 78.9)
        box_pct = round(box_shots / max(len(all_shots), 1) * 100, 1) if len(all_shots) > 0 else 0

        # Generate base64 pitch
        pitch_b64 = _build_transition_risk_pitch(all_rec, is_att=True, filter_mode='shots')
        
        # Additional glassmorphic KPI cards for Off. Transitions
        extra_kpis = dbc.Row([
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(34, 197, 94, 0.05)"}, children=[
                html.Div("AVG PASSES IN 10s", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#22c55e"}),
                html.Div(f"{avg_passes}", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": "white"}),
                html.Div("Ball movement speed", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(34, 197, 94, 0.05)"}, children=[
                html.Div("AVG CARRIES IN 10s", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#22c55e"}),
                html.Div(f"{avg_carries}", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": "white"}),
                html.Div("Dribbles/runs after regain", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(34, 197, 94, 0.05)"}, children=[
                html.Div("10s F3 ENTRY RATE", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#22c55e"}),
                html.Div(f"{f3_entry_rate}%", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": _GOLD}),
                html.Div("Final third penetration", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(34, 197, 94, 0.05)"}, children=[
                html.Div("GOALS FROM REGAINS", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#22c55e"}),
                html.Div(f"{goals_count}", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": _GOLD}),
                html.Div("Goals scored within 10s", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
        ], style={"marginBottom": "20px", "maxWidth": "900px", "margin": "0 auto 20px"})

        takeaway_card = html.Div(className="goz-takeaway-card", style={"maxWidth": "900px", "margin": "0 auto", "borderLeftColor": "#22c55e", "padding": "20px"}, children=[
            html.Div("TACTICAL TAKEAWAY — OPPONENT ATTACKING TRANSITIONS", className="goz-card-title", style={"fontSize": "0.95rem", "color": _GOLD, "marginBottom": "10px"}),
            html.P(
                f"Across the season, {opp_name} regained possession {total_r} times. Within exactly 10 seconds of winning the ball: "
                f"they completed an average of {avg_passes} passes and registered {avg_carries} carries to break lines. "
                f"Importantly, they progressed to the final third {f3_entry_rate}% of the time, converting {shot_rate}% of their regains into a shot attempt. "
                f"They scored {goals_count} goals within these 10-second attacking transition windows. "
                f"Of all transition shots taken, {box_pct}% occurred inside the penalty box (visualized as gold circles/stars). "
                f"Göztepe's counter-press must immediately block passing lanes and apply pressure on the ball-winner within the crucial first 3-5 seconds to neutralize their {shot_rate}% transition shot threat.",
                style={"fontSize": "0.78rem", "color": "var(--text-secondary)", "lineHeight": "1.6", "margin": 0}
            )
        ])

        pitch_section = html.Div(className="goz-form-section", style={"marginBottom": "20px", "maxWidth": "950px", "margin": "0 auto 20px"}, children=[
            html.Div(className="goz-section-header", children=[
                html.Span("OFFENSIVE TRANSITIONS — 10s RISK MAP & SHOT LOCATIONS", className="goz-card-title",
                          style={"fontSize": "1.1rem"}),
            ]),
            html.P(
                "Shows shot-producing regains only. Switch to goals to isolate the highest-value attacks.",
                style={"fontSize": "0.74rem", "color": "var(--text-secondary)", "marginBottom": "14px"}
            ),
            dbc.RadioItems(
                id='off-transition-plot-filter',
                options=[
                    {'label': 'Shots Only', 'value': 'shots'},
                    {'label': 'Goals Only', 'value': 'goals'},
                ],
                value='shots',
                inline=True,
                className='pm-tab-radio-group',
                inputClassName='pm-tab-radio-input',
                labelClassName='pm-tab-radio-label',
                style={"marginBottom": "14px"},
            ),
            html.Img(id='off-transition-risk-image', src=pitch_b64, style={"width": "100%", "maxWidth": "900px", "margin": "0 auto",
                                           "display": "block", "borderRadius": "12px", "border": "1px solid var(--border-color)"}),
        ])

        phase_bench = _build_phase_bench_section('off-trans-tab', opponent, opp_name)
        return html.Div([
            phase_bench,
            extra_kpis,
            pitch_section,
            takeaway_card,
        ])

    elif active_tab == 'def-trans-tab':
        all_los = stats.get('all_losses', [])
        total_l = len(all_los)
        avg_passes = round(sum(x['passes'] for x in all_los) / max(total_l, 1), 1) if total_l > 0 else 0
        avg_carries = round(sum(x['carries'] for x in all_los) / max(total_l, 1), 1) if total_l > 0 else 0
        f3_entry_rate = round(sum(1 for x in all_los if x['reached_f3']) / max(total_l, 1) * 100, 1) if total_l > 0 else 0
        shots_conceded = sum(1 for x in all_los if x['shot'])
        goals_conceded = sum(1 for x in all_los if x['goal'])
        danger_rate = round(shots_conceded / max(total_l, 1) * 100, 1) if total_l > 0 else 0
        
        # Calculate conceded box shots percentage
        all_conceded_shots = []
        for l in all_los:
            all_conceded_shots.extend(l.get('shot_coords', []))
        box_shots = sum(1 for s in all_conceded_shots if s['x'] >= 83.0 and 21.1 <= s['y'] <= 78.9)
        box_pct = round(box_shots / max(len(all_conceded_shots), 1) * 100, 1) if len(all_conceded_shots) > 0 else 0

        # Generate base64 pitch
        pitch_b64_def = _build_transition_risk_pitch(all_los, is_att=False, filter_mode='shots')
        
        # Additional glassmorphic KPI cards for Def. Transitions
        extra_kpis_def = dbc.Row([
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(239, 68, 68, 0.05)"}, children=[
                html.Div("AVG OPP PASSES IN 10s", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#ef4444"}),
                html.Div(f"{avg_passes}", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": "white"}),
                html.Div("Conceded passes", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(239, 68, 68, 0.05)"}, children=[
                html.Div("AVG OPP CARRIES IN 10s", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#ef4444"}),
                html.Div(f"{avg_carries}", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": "white"}),
                html.Div("Conceded dribbles/carries", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(239, 68, 68, 0.05)"}, children=[
                html.Div("10s OPP F3 ENTRY RATE", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#ef4444"}),
                html.Div(f"{f3_entry_rate}%", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": _GOLD}),
                html.Div("Conceded F3 penetrations", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center", "background": "rgba(239, 68, 68, 0.05)"}, children=[
                html.Div("GOALS CONCEDED AFTER LOSSES", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px", "color": "#ef4444"}),
                html.Div(f"{goals_conceded}", className="coach-brief-text", style={"fontSize": "1.5rem", "fontWeight": "700", "color": _GOLD}),
                html.Div("Goals conceded within 10s", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3),
        ], style={"marginBottom": "20px", "maxWidth": "900px", "margin": "0 auto 20px"})

        takeaway_card_def = html.Div(className="goz-takeaway-card", style={"maxWidth": "900px", "margin": "0 auto", "borderLeftColor": "#ef4444", "padding": "20px"}, children=[
            html.Div("TACTICAL TAKEAWAY — OPPONENT DEFENSIVE TRANSITIONS", className="goz-card-title", style={"fontSize": "0.95rem", "color": _RED, "marginBottom": "10px"}),
            html.P(
                f"During buildup or attack, {opp_name} lost possession {total_l} times over the season. Within 10 seconds of losing the ball: "
                f"their opponents completed an average of {avg_passes} passes and {avg_carries} carries to hit them on the break. "
                f"Opponents entered their final third in {f3_entry_rate}% of these transitions, creating a shot attempt in {danger_rate}% of cases. "
                f"They conceded {goals_conceded} goals within these 10-second defensive transition windows. "
                f"Crucially, {box_pct}% of these shots were taken inside their penalty box (visualized as gold/yellow circles). "
                f"This indicates a severe vulnerability to direct, vertical counter-attacks. Göztepe must immediately launch direct runs into the final third (Zone 14 and penalty box) "
                f"the instant we win the ball to exploit their slow defensive recovery.",
                style={"fontSize": "0.78rem", "color": "var(--text-secondary)", "lineHeight": "1.6", "margin": 0}
            )
        ])

        pitch_section_def = html.Div(className="goz-form-section", style={"marginBottom": "20px", "maxWidth": "950px", "margin": "0 auto 20px"}, children=[
            html.Div(className="goz-section-header", children=[
                html.Span("DEFENSIVE TRANSITIONS — 10s RISK MAP & CONCEDED SHOT LOCATIONS", className="goz-card-title",
                          style={"fontSize": "1.1rem"}),
            ]),
            html.P(
                "Shows losses that conceded a shot only. Switch to goals to isolate the most dangerous breakdowns.",
                style={"fontSize": "0.74rem", "color": "var(--text-secondary)", "marginBottom": "14px"}
            ),
            dbc.RadioItems(
                id='def-transition-plot-filter',
                options=[
                    {'label': 'Shots Only', 'value': 'shots'},
                    {'label': 'Goals Only', 'value': 'goals'},
                ],
                value='shots',
                inline=True,
                className='pm-tab-radio-group',
                inputClassName='pm-tab-radio-input',
                labelClassName='pm-tab-radio-label',
                style={"marginBottom": "14px"},
            ),
            html.Img(id='def-transition-risk-image', src=pitch_b64_def, style={"width": "100%", "maxWidth": "900px", "margin": "0 auto",
                                               "display": "block", "borderRadius": "12px", "border": "1px solid var(--border-color)"}),
        ])

        phase_bench_def = _build_phase_bench_section('def-trans-tab', opponent, opp_name)
        return html.Div([
            phase_bench_def,
            extra_kpis_def,
            pitch_section_def,
            takeaway_card_def,
        ])

    elif active_tab == 'set-pieces-tab':
        # Column 1: Corners Analysis
        corners = stats['corners']
        c_left = corners.get('left', 0)
        c_right = corners.get('right', 0)
        
        ins = corners.get('inswinger', 0)
        out = corners.get('outswinger', 0)
        strg = corners.get('straight', 0)
        
        takers = corners.get('takers', [])
        targets = corners.get('targets', [])
        
        ins_pct = round(ins / max(ins + out + strg, 1) * 100, 1)
        out_pct = round(out / max(ins + out + strg, 1) * 100, 1)
        strg_pct = round(strg / max(ins + out + strg, 1) * 100, 1)
        
        corner_pitch_b64 = _build_setpiece_corner_pitch(corners)

        left_col = dbc.Col(html.Div(className="goz-form-section", children=[
            html.Div(className="goz-section-header", children=[
                html.Span("CORNERS & RECEIVER TRACING", className="goz-card-title", style={"fontSize": "1.05rem"}),
            ]),

            html.Div("CORNER DELIVERY PITCH", style={"fontSize": "0.68rem", "fontWeight": "800",
                                                     "color": _GOLD, "marginBottom": "6px",
                                                     "letterSpacing": "0.5px"}),
            html.P("Arrow shows delivery side. Bubbles in box = trajectory type (size = frequency).",
                   style={"fontSize": "0.64rem", "color": "var(--text-secondary)", "marginBottom": "8px"}),
            html.Img(src=corner_pitch_b64, style={"width": "100%", "borderRadius": "10px",
                                                   "marginBottom": "14px",
                                                   "border": "1px solid var(--border-color)"}),

            html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px", "fontSize": "0.78rem"}, children=[
                html.Span("Left Corner Volume", style={"color": "var(--text-secondary)"}),
                html.Span(f"{c_left} kicks", style={"fontWeight": "bold", "color": _BLUE})
            ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px", "fontSize": "0.78rem"}, children=[
                html.Span("Right Corner Volume", style={"color": "var(--text-secondary)"}),
                html.Span(f"{c_right} kicks", style={"fontWeight": "bold", "color": _GOLD})
            ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "14px", "fontSize": "0.78rem"}, children=[
                html.Span("Corner Goals Scored", style={"color": "var(--text-secondary)"}),
                html.Span(f"{corners.get('total_goals', 0)} goals", style={"fontWeight": "bold", "color": _GREEN if corners.get('total_goals', 0) > 0 else "white"})
            ]),

            html.Div(style={"borderTop": "1px solid var(--border-color)", "paddingTop": "12px", "marginBottom": "14px"}, children=[
                html.Div("DELIVERY TRAJECTORY PREFERENCES", style={"fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD, "marginBottom": "8px", "letterSpacing": "0.5px"}),
                make_progress_bar("Inswinger corners", ins_pct, _GREEN),
                make_progress_bar("Outswinger corners", out_pct, _BLUE),
                make_progress_bar("Straight corners", strg_pct, _GOLD)
            ]),

            html.Div(style={"borderTop": "1px solid var(--border-color)", "paddingTop": "12px"}, children=[
                html.Div("TOP CORNER TAKERS", style={"fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD, "marginBottom": "8px", "letterSpacing": "0.5px"}),
                html.Div([
                    html.Div(style={"marginBottom": "8px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.76rem", "fontWeight": "600"}, children=[
                            html.Span(name, style={"color": "white"}),
                            html.Span(f"{data_dict['total']} taken", style={"color": _GOLD})
                        ]),
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.68rem", "color": "var(--text-secondary)"}, children=[
                            html.Span(f"Sides: {data_dict['left']} Left / {data_dict['right']} Right"),
                            html.Span(f"{data_dict['goals']} goal seq.", style={"color": _GREEN if data_dict['goals'] > 0 else "var(--text-secondary)", "fontWeight": "700" if data_dict['goals'] > 0 else "normal"})
                        ])
                    ]) for name, data_dict in takers[:3]
                ] if takers else html.Div("No corner stats recorded", style={"fontSize": "0.7rem", "color": "var(--text-secondary)"}))
            ]),

            html.Div(style={"borderTop": "1px solid var(--border-color)", "paddingTop": "12px", "marginTop": "14px"}, children=[
                html.Div("TARGET RECEIVER TRACING (6s)", style={"fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD, "marginBottom": "8px", "letterSpacing": "0.5px"}),
                html.P("Most common targeted receiver on corner delivery sequence:", style={"fontSize": "0.64rem", "color": "var(--text-secondary)", "marginBottom": "8px"}),
                html.Div([
                    html.Div(style={"marginBottom": "8px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.76rem", "fontWeight": "600"}, children=[
                            html.Span(name, style={"color": "white"}),
                            html.Span(f"{data_dict['targeted']} targeted", style={"color": _GOLD})
                        ]),
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.68rem", "color": "var(--text-secondary)"}, children=[
                            html.Span(f"Shots from corners: {data_dict['shots']}"),
                            html.Span(f"{data_dict['goals']} goals scored", style={"color": _GREEN if data_dict['goals'] > 0 else "var(--text-secondary)", "fontWeight": "700" if data_dict['goals'] > 0 else "normal"})
                        ])
                    ]) for name, data_dict in targets[:3]
                ] if targets else html.Div("No corner targets traced", style={"fontSize": "0.7rem", "color": "var(--text-secondary)"}))
            ])
        ]), md=4)
        
        # Column 2: Free Kicks & Penalties Analysis
        fk = stats['freekicks']
        fk_total = fk.get('total', 0)
        fk_shots = fk.get('direct_shots', 0)
        fk_goals = fk.get('direct_goals', 0)
        
        pen = stats['penalties']
        p_total = pen.get('total', 0)
        p_scored = pen.get('scored', 0)
        p_saved = pen.get('saved', 0)
        p_missed = pen.get('missed', 0)
        p_takers = pen.get('takers', [])
        p_placements = pen.get('placements', [])
        p_conv = round(p_scored / max(p_total, 1) * 100, 1)
        p_rank, p_rank_total = _penalty_total_rank(opponent)
        p_rank_text = f"#{p_rank}/{p_rank_total}" if p_rank is not None else "loading"
        set_piece_goals = stats.get(
            'set_piece_goals',
            corners.get('total_goals', 0) + fk_goals + p_scored,
        )
        sp_goal_rank, sp_goal_rank_total = _set_piece_goal_rank(opponent)
        sp_goal_rank_text = (
            f"#{sp_goal_rank}/{sp_goal_rank_total}"
            if sp_goal_rank is not None else "loading"
        )
        penalty_rows = sorted(
            p_placements,
            key=lambda item: (item.get('week') or 0, item.get('minute') or 0, item.get('num') or 0)
        )
        
        mid_col = dbc.Col(html.Div(className="goz-form-section", children=[
            html.Div(className="goz-section-header", children=[
                html.Span("FREEKICKS & PENALTIES", className="goz-card-title", style={"fontSize": "1.05rem"}),
            ]),
            html.Div(style={"marginBottom": "16px"}, children=[
                html.Div("SET-PIECE GOAL OUTPUT", style={
                    "fontSize": "0.68rem", "fontWeight": "800",
                    "color": _GOLD, "marginBottom": "8px", "letterSpacing": "0.5px"
                }),
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between",
                    "fontSize": "0.78rem", "marginBottom": "8px"
                }, children=[
                    html.Span("Total Set-Piece Goals", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{set_piece_goals} · rank {sp_goal_rank_text}", style={
                        "fontWeight": "bold", "color": _GREEN if set_piece_goals > 0 else "white"
                    })
                ]),
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between",
                    "fontSize": "0.72rem", "color": "var(--text-secondary)"
                }, children=[
                    html.Span("Corner / Direct FK / Penalty"),
                    html.Span(f"{corners.get('total_goals', 0)} / {fk_goals} / {p_scored}", style={
                        "fontWeight": "700", "color": "white"
                    })
                ]),
            ]),
            html.Div(style={"marginBottom": "16px"}, children=[
                html.Div("FREE KICK CONVERSION", style={"fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD, "marginBottom": "8px", "letterSpacing": "0.5px"}),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Freekicks Won", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{fk_total} kicks", style={"fontWeight": "bold", "color": "white"})
                ]),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Direct Shots Taken", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{fk_shots} shots", style={"fontWeight": "bold", "color": _BLUE})
                ]),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Direct Goals Scored", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{fk_goals} goals", style={"fontWeight": "bold", "color": _GREEN})
                ])
            ]),
            
            html.Div(style={"borderTop": "1px solid var(--border-color)", "paddingTop": "12px"}, children=[
                html.Div("PENALTIES HISTORY", style={"fontSize": "0.68rem", "fontWeight": "800", "color": _GOLD, "marginBottom": "10px", "letterSpacing": "0.5px"}),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Total Penalty Kicks", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{p_total} · rank {p_rank_text}", style={"fontWeight": "bold", "color": "white"})
                ]),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Penalties Scored", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{p_scored}", style={"fontWeight": "bold", "color": _GREEN})
                ]),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Conversion Rate", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{p_conv}%", style={"fontWeight": "bold", "color": _GOLD})
                ]),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Penalties Saved", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{p_saved}", style={"fontWeight": "bold", "color": _BLUE})
                ]),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                    html.Span("Penalties Missed / Off-Target", style={"color": "var(--text-secondary)"}),
                    html.Span(f"{p_missed}", style={"fontWeight": "bold", "color": _RED})
                ]),
                html.Div("PENALTY TAKERS", style={
                    "fontSize": "0.64rem", "fontWeight": "800", "color": _GOLD,
                    "marginTop": "12px", "marginBottom": "8px", "letterSpacing": "0.5px"
                }),
                html.Div([
                    html.Div(style={
                        "display": "flex", "justifyContent": "space-between",
                        "fontSize": "0.72rem", "marginBottom": "5px",
                    }, children=[
                        html.Span(name, style={"color": "white", "fontWeight": "700"}),
                        html.Span(
                            f"{data['scored']}/{data['total']}",
                            style={"color": _GREEN if data['scored'] == data['total'] else _GOLD, "fontWeight": "700"}
                        )
                    ]) for name, data in p_takers[:4]
                ] if p_takers else html.Div("No penalty takers recorded", style={
                    "fontSize": "0.7rem", "color": "var(--text-secondary)"
                })),
                html.Div("SHOT PLACEMENT LOG", style={
                    "fontSize": "0.64rem", "fontWeight": "800", "color": _GOLD,
                    "marginTop": "12px", "marginBottom": "8px", "letterSpacing": "0.5px"
                }),
                html.Div([
                    html.Div(style={
                        "display": "grid",
                        "gridTemplateColumns": "20px 1fr auto",
                        "gap": "7px",
                        "alignItems": "center",
                        "fontSize": "0.68rem",
                        "padding": "5px 0",
                        "borderTop": "1px solid rgba(255,255,255,0.06)" if idx else "none",
                    }, children=[
                        html.Span(f"#{row.get('num')}", style={"color": _GOLD, "fontWeight": "800"}),
                        html.Span(
                            f"{row.get('player', 'Unknown')} vs {row.get('opponent', '?')} ({row.get('minute', 0)}')",
                            style={"color": "white", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}
                        ),
                        html.Span(
                            f"{row.get('side', 'Unknown')} · {row.get('event', '')}",
                            style={"color": _GREEN if row.get('event') == 'Goal' else _BLUE if row.get('event') == 'Saved Shot' else _RED,
                                   "fontWeight": "700"}
                        ),
                    ]) for idx, row in enumerate(penalty_rows)
                ] if penalty_rows else html.Div("No shot placement data", style={
                    "fontSize": "0.7rem", "color": "var(--text-secondary)"
                }))
            ])
        ]), md=4)
        
        # Column 3: Goalkicks Analysis
        gk = stats['goalkicks']
        gk_total = gk.get('total', 0)
        gk_long = gk.get('long', 0)
        gk_short = gk.get('short', 0)
        gk_long_pct = gk.get('long_pct', 0)
        
        right_col = dbc.Col(html.Div(className="goz-form-section", children=[
            html.Div(className="goz-section-header", children=[
                html.Span("GOALKICKS DISTRIBUTION", className="goz-card-title", style={"fontSize": "1.05rem"}),
            ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                html.Span("Total Goal Kicks", style={"color": "var(--text-secondary)"}),
                html.Span(f"{gk_total}", style={"fontWeight": "bold", "color": "white"})
            ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "8px"}, children=[
                html.Span("Short Goalkicks (<50m)", style={"color": "var(--text-secondary)"}),
                html.Span(f"{gk_short}", style={"fontWeight": "bold", "color": _GREEN})
            ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.78rem", "marginBottom": "16px"}, children=[
                html.Span("Long Goalkicks (>=50m)", style={"color": "var(--text-secondary)"}),
                html.Span(f"{gk_long}", style={"fontWeight": "bold", "color": _BLUE})
            ]),
            
            make_progress_bar("Long Goalkick Preference Ratio", gk_long_pct, _BLUE),
            
            html.Div(style={"fontSize": "0.72rem", "color": "var(--text-secondary)", "lineHeight": "1.5", "marginTop": "14px"}, children=[
                html.Span("Tactical Implication: ", style={"color": _GOLD, "fontWeight": "700"}),
                f"A long goalkick ratio of {gk_long_pct}% indicates they "
                f"{'prefer to play long and fight for second balls.' if gk_long_pct >= 50 else 'tend to build up systematically from the back.'} "
                f"Pressing strategies should be tailored accordingly."
            ])
        ]), md=4)
        
        phase_bench_sp = _build_phase_bench_section('set-pieces-tab', opponent, opp_name)
        penalty_goal_b64 = _build_penalty_goal_mouth(pen)

        # Season avg summary cards row (penalties + goalkicks quick-glance)
        fk_conv = round(fk_goals / max(fk_shots, 1) * 100, 1) if fk_shots > 0 else 0

        summary_cards = dbc.Row([
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center"}, children=[
                html.Div("SET-PIECE GOALS", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px"}),
                html.Div(f"{set_piece_goals}", className="coach-brief-text",
                         style={"fontSize": "1.6rem", "fontWeight": "700", "color": _GREEN}),
                html.Div(f"League rank: {sp_goal_rank_text}", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3, sm=6),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center"}, children=[
                html.Div("PENALTIES", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px"}),
                html.Div(f"{p_scored}/{p_total}", className="coach-brief-text",
                         style={"fontSize": "1.6rem", "fontWeight": "700", "color": _GOLD}),
                html.Div(f"Total rank: {p_rank_text} · Conv. rate: {p_conv}%", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3, sm=6),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center"}, children=[
                html.Div("DIRECT FK GOALS", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px"}),
                html.Div(f"{fk_goals}/{fk_shots}", className="coach-brief-text",
                         style={"fontSize": "1.6rem", "fontWeight": "700", "color": _BLUE}),
                html.Div(f"Conv. rate: {fk_conv}%", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3, sm=6),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center"}, children=[
                html.Div("TOTAL CORNERS", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px"}),
                html.Div(f"{corners.get('total', 0)}", className="coach-brief-text",
                         style={"fontSize": "1.6rem", "fontWeight": "700", "color": _GREEN}),
                html.Div(f"L: {c_left}  /  R: {c_right}", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3, sm=6),
            dbc.Col(html.Div(className="coach-brief-item", style={"minHeight": "auto", "padding": "14px 18px", "textAlign": "center"}, children=[
                html.Div("LONG GOALKICK %", className="coach-brief-label", style={"fontSize": "0.62rem", "marginBottom": "4px"}),
                html.Div(f"{gk_long_pct}%", className="coach-brief-text",
                         style={"fontSize": "1.6rem", "fontWeight": "700", "color": _PURPLE}),
                html.Div(f"Total GKs: {gk_total}", style={"fontSize": "0.66rem", "color": "var(--text-secondary)"}),
            ]), md=3, sm=6),
        ], style={"marginBottom": "20px"})

        penalty_goal_section = html.Div(className="goz-form-section", style={"marginBottom": "20px"}, children=[
            html.Div(className="goz-section-header", children=[
                html.Span("PENALTY GOAL MOUTH", className="goz-card-title", style={"fontSize": "1.05rem"}),
            ]),
            html.P(
                "Penalty dots are plotted inside the goal using Opta goal-mouth coordinates. Numbers match the shot placement log.",
                className="goz-card-desc",
                style={"fontSize": "0.72rem", "marginBottom": "12px"},
            ),
            html.Img(src=penalty_goal_b64, style={
                "width": "100%",
                "maxWidth": "760px",
                "display": "block",
                "margin": "0 auto",
                "borderRadius": "10px",
            }),
        ])

        return html.Div([
            phase_bench_sp,
            summary_cards,
            penalty_goal_section,
            dbc.Row([left_col, mid_col, right_col]),
        ])

    return html.Div()


# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────

def layout():
    matches   = extract_fixture_data(lite=True)
    standings = calculate_standings(matches)
    rivals    = sorted([t for t in standings['Team'].unique() if t != GOZTEPE])

    return html.Div(className="page-wrap", children=[
        html.Div(className="goz-hero", children=[
            html.Div(className="goz-hero-content", children=[
                dcc.Link("← GÖZTEPE HUB", href="/", className="goz-back-link"),
                html.H1("PRE-MATCH ANALYSIS", className="goz-hub-title"),
                html.P("Season-level tactical intelligence & phase analysis", className="goz-hub-subtitle"),
                html.Div(style={"marginTop": "25px", "width": "100%", "maxWidth": "350px"}, children=[
                    html.Label("SELECT UPCOMING OPPONENT", className="goz-label"),
                    dcc.Dropdown(
                        id='pre-match-rival-selector',
                        options=[{'label': r, 'value': r} for r in rivals],
                        value=rivals[0] if rivals else None,
                        className="goz-dropdown",
                        clearable=False,
                    ),
                ]),
                html.Div(className="report-actions", children=[
                    html.Button("Report", type="button", className="btn-print btn-report-print"),
                ]),
            ]),
        ]),

        html.Div(className="content-container", style={"padding": "0 20px 60px"}, children=[
            html.Div(id="pre-match-report-container", className="report-only"),
            html.Div(className="report-screen", children=[
            dcc.Interval(
                id='pre-match-benchmark-refresh',
                interval=5000,
                n_intervals=0,
                max_intervals=120,
            ),
            html.Div(id='pre-match-benchmark-container', style={"marginTop": "30px"}),
            html.Div(className="pm-tab-toolbar", children=[
                dbc.RadioItems(
                    id="pre-match-tabs",
                    options=[
                        {"label": "📋 Game Plan",         "value": "game-plan-tab"},
                        {"label": "⚔️ Offensive",        "value": "offensive-tab"},
                        {"label": "🛡️ Defensive",        "value": "defensive-tab"},
                        {"label": "⚡ Off. Transitions",  "value": "off-trans-tab"},
                        {"label": "🔄 Def. Transitions",  "value": "def-trans-tab"},
                        {"label": "🎯 Set Pieces",        "value": "set-pieces-tab"},
                    ],
                    value="game-plan-tab",
                    inline=True,
                    className="pm-tab-radio-group",
                    inputClassName="pm-tab-radio-input",
                    labelClassName="pm-tab-radio-label",
                ),
            ]),
            html.Div(id='pre-match-kpi-container'),
            html.Div(id='pre-match-tab-content'),
            ]),
        ]),

        html.Footer(className="footer", children=[
            html.Div(className="footer-inner", children=[
                html.Div("© tactIQ Göztepe Hub — Precision Analytics", className="footer-text"),
                html.Img(src="/assets/superlig_logo.jpg", className="superlogo"),
            ])
        ])
    ])


# ──────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────

@callback(
    [Output('pre-match-benchmark-container', 'children'),
     Output('pre-match-report-container', 'children')],
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-benchmark-refresh', 'n_intervals')]
)
def update_pre_match(opponent, _refresh_tick):
    if not opponent:
        return html.Div(), html.Div()

    opp_name  = _clean(opponent)
    benchmark = _build_benchmarking_section(opponent, opp_name)
    report = _build_pre_match_report(opponent)
    return benchmark, report


@callback(
    Output('pre-match-buildup-rank-container', 'children'),
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-benchmark-refresh', 'n_intervals')]
)
def update_buildup_rankings(opponent, _refresh_tick):
    if not opponent:
        return html.Div()
    return _build_buildup_rank_section(opponent)


@callback(
    [Output('pre-match-kpi-container', 'children'),
     Output('pre-match-tab-content', 'children')],
    [Input('pre-match-tabs', 'value'),
     Input('pre-match-rival-selector', 'value')]
)
def update_pre_match_tabs(active_tab, opponent):
    if not opponent or not active_tab:
        return html.Div(), html.Div()
    try:
        stats = _get_season_stats(opponent)
        opp_name = _clean(opponent)
        kpi = _build_kpi_section(active_tab, stats, opp_name)
        content = _build_tab_content(active_tab, stats, opp_name, opponent)
        return kpi, content
    except Exception as e:
        import traceback
        err_msg = f"Error rendering tabs: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return html.Div(f"Error loading KPIs: {e}", style={"color": _RED, "padding": "10px"}), \
               html.Div(f"Error loading content: {e}", style={"color": _RED, "padding": "20px"})


@callback(
    Output('off-transition-risk-image', 'src'),
    [Input('off-transition-plot-filter', 'value'),
     Input('pre-match-rival-selector', 'value')]
)
def update_off_transition_plot(filter_mode, opponent):
    if not opponent:
        return ''
    stats = _get_season_stats(opponent)
    return _build_transition_risk_pitch(
        stats.get('all_recoveries', []),
        is_att=True,
        filter_mode=filter_mode or 'shots',
    )


@callback(
    Output('def-transition-risk-image', 'src'),
    [Input('def-transition-plot-filter', 'value'),
     Input('pre-match-rival-selector', 'value')]
)
def update_def_transition_plot(filter_mode, opponent):
    if not opponent:
        return ''
    stats = _get_season_stats(opponent)
    return _build_transition_risk_pitch(
        stats.get('all_losses', []),
        is_att=False,
        filter_mode=filter_mode or 'shots',
    )


# Shot & Goal Event lists
SHOT_EVENTS = [
    "Chance missed",
    "Goal",
    "Miss",
    "Post",
    "Saved Shot",
    "Temp_Attempt",
    "Temp_Goal",
]
GOAL_EVENTS = {"Goal", "Temp_Goal"}


def _ev_lower(ev) -> str:
    if not isinstance(ev, str):
        return ""
    return ev.lower()


def has_valid_player_name(row_or_value) -> bool:
    """True if player_name is valid."""
    if isinstance(row_or_value, pd.Series):
        pname = row_or_value.get("player_name", "")
    else:
        pname = row_or_value

    if pd.isna(pname) or pname is None:
        return False

    pname_str = str(pname).strip()
    if not pname_str:
        return False

    if pname_str.lower() in {"nan", "none"}:
        return False

    return True


def is_shot_saved_generic(row: pd.Series) -> bool:
    """True if shot was saved or blocked."""
    ev_low = _ev_lower(row.get("event", ""))
    if ("shot" not in ev_low) and (row.get("event") not in SHOT_EVENTS):
        return False

    if any(k in ev_low for k in ["saved shot", "saved", "save", "block", "blocked"]) and ("goal" not in ev_low):
        return True

    outcome_candidates = []
    for col in ["outcome", "shot_outcome", "outcome_name", "shot_outcome_name", "result", "result_name"]:
        if col in row.index:
            outcome_candidates.append(str(row.get(col, "")).lower())

    outcome_text = " ".join(outcome_candidates)
    keywords = ["saved", "save", "saved shot", "savedshot", "goalkeeper save", "save made", "blocked", "block", "parado", "detenido", "bloqueado"]
    return any(kw in outcome_text for kw in keywords)


def is_goal_row(row: pd.Series) -> bool:
    return str(row.get("event", "")) in GOAL_EVENTS


def is_shot_off_target(row: pd.Series) -> bool:
    if is_shot_saved_generic(row):
        return False

    ev = str(row.get("event", ""))
    if ev in {"Miss", "Chance missed"}:
        return True

    outcome_candidates = []
    for col in ["outcome", "shot_outcome", "outcome_name", "shot_outcome_name", "result", "result_name"]:
        if col in row.index:
            outcome_candidates.append(str(row.get(col, "")).lower())

    outcome_text = " ".join(outcome_candidates)
    keywords = ["off target", "wide", "miss", "missed", "fuera"]
    return any(kw in outcome_text for kw in keywords)


def is_error_event(ev: str) -> bool:
    ev_low = _ev_lower(ev)
    return ("error" in ev_low) or ("clearance" in ev_low)


def is_foul_event(ev: str) -> bool:
    ev_low = _ev_lower(ev)
    return "foul" in ev_low


def is_start_delay_event(ev: str) -> bool:
    ev_low = _ev_lower(ev)
    return "start delay" in ev_low


def is_end_delay_event(ev: str) -> bool:
    ev_low = _ev_lower(ev)
    return "end delay" in ev_low


def is_contentious_ref_event(ev: str) -> bool:
    ev_low = _ev_lower(ev)
    return "contentious referee decision" in ev_low


def is_admin_event(ev: str) -> bool:
    return (
        is_start_delay_event(ev) or
        is_end_delay_event(ev) or
        is_contentious_ref_event(ev)
    )


def is_card_event(ev: str) -> bool:
    ev_low = _ev_lower(ev)
    return ev_low == "card"


def is_deleted_row(row: pd.Series) -> bool:
    ev_str = str(row.get("event", "")).lower()
    if "deleted event" in ev_str:
        return True
    if ev_str in IGNORED_EVENT_NAMES:
        return True
    if "penalty faced" in ev_str:
        return True

    candidate_cols = ["Deleted Event", "deleted_event", "is_deleted", "Deleted"]
    for col in candidate_cols:
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str):
                if val.strip().lower() in ("1", "true", "yes", "y"):
                    return True
            else:
                try:
                    if int(val) == 1:
                        return True
                except Exception:
                    pass
    return False


def is_field_event_row(row: pd.Series) -> bool:
    ev = row.get("event", "")
    if is_deleted_row(row):
        return False
    if is_admin_event(ev):
        return False
    if is_card_event(ev):
        return False
    if not has_valid_player_name(row):
        return False
    return True


def is_pass_intercepted(row: pd.Series) -> bool:
    ev_low = _ev_lower(row.get("event", ""))
    if "pass" not in ev_low:
        return False

    outcome_candidates = []
    for col in ["outcome", "pass_outcome", "outcome_name", "pass_outcome_name", "result", "result_name"]:
        if col in row.index:
            outcome_candidates.append(str(row.get(col, "")).lower())

    outcome_text = " ".join(outcome_candidates)
    keywords = ["intercepted", "intercept", "blocked", "cut out", "unsuccessful", "incomplete", "out of play", "ball lost", "perdido", "interceptado", "bloqueado"]
    return any(kw in outcome_text for kw in keywords) or row.get("outcome") == 0 or row.get("outcome") is False or row.get("outcome") == '0'


def is_save_like_event_row(row: pd.Series) -> bool:
    ev_low = _ev_lower(row.get("event", ""))
    return ("save" in ev_low) or ("safe" in ev_low) or ("block" in ev_low)


def find_next_save_event_for_shot(shot_row: pd.Series, df: pd.DataFrame) -> pd.Series | None:
    if "row_index" in shot_row.index:
        start_idx = int(shot_row["row_index"])
    else:
        start_idx = int(shot_row.name)

    shot_team_pos = shot_row.get("team_position")
    tail = df[df["row_index"] > start_idx].copy()
    if tail.empty:
        return None

    ev_low_series = tail["event"].astype(str).str.lower()
    mask_save_word = (
        ev_low_series.str.contains("save", na=False) |
        ev_low_series.str.contains("safe", na=False) |
        ev_low_series.str.contains("block", na=False)
    )

    if "team_position" in tail.columns and isinstance(shot_team_pos, str):
        mask_rival = tail["team_position"] != shot_team_pos
    else:
        mask_rival = True

    if "macro_category" in tail.columns:
        macro_low = tail["macro_category"].astype(str).str.lower()
        mask_macro_ok = ~macro_low.isin(IGNORED_MACRO_CATEGORIES)
    else:
        mask_macro_ok = True

    mask_event_ok = ~ev_low_series.isin(IGNORED_EVENT_NAMES)
    mask_player_ok = tail.apply(has_valid_player_name, axis=1)
    mask_not_deleted = ~tail.apply(is_deleted_row, axis=1)

    mask1 = mask_save_word & mask_rival & mask_macro_ok & mask_event_ok & mask_player_ok & mask_not_deleted
    candidates = tail[mask1].sort_values("row_index")
    if not candidates.empty:
        return candidates.iloc[0]

    if not is_shot_saved_generic(shot_row):
        return None

    for _, r in tail.sort_values("row_index").iterrows():
        if is_deleted_row(r):
            continue

        cat = str(r.get("macro_category", "")).strip().lower()
        if cat in IGNORED_MACRO_CATEGORIES:
            continue

        ev_low = str(r.get("event", "")).strip().lower()
        if ev_low in IGNORED_EVENT_NAMES:
            continue

        if not has_valid_player_name(r):
            continue

        if isinstance(shot_team_pos, str) and "team_position" in r.index:
            if r.get("team_position") == shot_team_pos:
                continue

        return r

    return None


def compute_shot_end_coordinates_default(row):
    float(row.get("x_plot_m", np.nan))
    y0 = float(row.get("y_plot_m", np.nan))

    x_end = row.get("pass_end_x_plot_m", np.nan)
    y_end = row.get("pass_end_y_plot_m", np.nan)

    if not pd.isna(x_end) and not pd.isna(y_end):
        return float(x_end), float(y_end)

    team_pos = row.get("team_position", "home")
    if team_pos == "home":
        goal_x_front = PITCH_LENGTH - GOAL_MARGIN_X
    else:
        goal_x_front = 0.0 + GOAL_MARGIN_X

    if GOAL_MOUTH_Y_COL in row.index and not pd.isna(row.get(GOAL_MOUTH_Y_COL)):
        gm_y = float(row[GOAL_MOUTH_Y_COL])
        gm_y_clamped = min(max(gm_y, GOAL_MOUTH_LEFT_OPT), GOAL_MOUTH_RIGHT_OPT)
        rel_opt = (gm_y_clamped - GOAL_MOUTH_CENTER_OPT) / GOAL_MOUTH_HALF_SPAN_OPT

        goal_center_y = PITCH_WIDTH / 2.0
        goal_half_w   = GOAL_WIDTH_M / 2.0

        goal_y_raw = goal_center_y + rel_opt * goal_half_w
        if team_pos == "home":
            goal_y = goal_y_raw
        else:
            goal_y = PITCH_WIDTH - goal_y_raw
    else:
        goal_y = y0

    return goal_x_front, goal_y


def get_next_event_for_shot_row(shot_row: pd.Series, df: pd.DataFrame) -> pd.Series | None:
    idx = int(shot_row["row_index"])
    tail = df[df["row_index"] > idx].sort_values("row_index")
    for _, r in tail.iterrows():
        cat = str(r.get("macro_category", "")).strip().lower()
        if cat in IGNORED_MACRO_CATEGORIES:
            continue

        ev_low = str(r.get("event", "")).strip().lower()
        if ev_low in IGNORED_EVENT_NAMES:
            continue

        if not has_valid_player_name(r):
            continue

        return r

    return None


def classify_shot_category(row: pd.Series, df: pd.DataFrame) -> str:
    if not is_shot_event(str(row.get("event", ""))):
        return "other_event"

    if is_goal_row(row):
        return "goal"

    if is_shot_saved_generic(row):
        save_row = find_next_save_event_for_shot(row, df)
        if save_row is not None:
            pos = str(save_row.get("position", "")).upper()
            if pos == "GK":
                return "saved_gk"
            else:
                return "blocked"
        else:
            return "saved_other"

    return "other_shot"


def classify_shot_text(row: pd.Series, df: pd.DataFrame) -> str:
    ev = str(row.get("event", ""))

    if ev in GOAL_EVENTS:
        return "Gol"

    if is_shot_saved_generic(row):
        save_row = find_next_save_event_for_shot(row, df)
        if save_row is not None:
            pos = str(save_row.get("position", "")).upper()
            if pos == "GK":
                return "Tiro parado (portero)"
            else:
                return "Tiro bloqueado (jugador)"
        return "Tiro parado/bloqueado"

    if is_shot_off_target(row):
        return "Tiro fuera"

    return "Tiro sin gol"


def is_shot_event(ev: str) -> bool:
    if ev in SHOT_EVENTS:
        return True
    return "shot" in _ev_lower(ev)


def _load_goal_timeline(filename, event_id, team_name):
    from utils.data import get_data_dir
    filepath = os.path.join(get_data_dir(), filename)
    if not os.path.exists(filepath):
        return pd.DataFrame()

    df_match = pd.read_parquet(filepath)
    df_match = df_match.reset_index(drop=False).rename(columns={"index": "row_index"})

    # 1. Normalize coordinates to meters (105x68 size)
    df_match["x_m"] = df_match["x"] * PITCH_LENGTH / 100.0
    df_match["y_m"] = df_match["y"] * PITCH_WIDTH / 100.0

    df_match["Pass End X"] = df_match.get("Pass End X", np.nan)
    df_match["Pass End Y"] = df_match.get("Pass End Y", np.nan)

    try:
        df_match["pass_end_x_m"] = pd.to_numeric(df_match["Pass End X"], errors='coerce') * PITCH_LENGTH / 100.0
        df_match["pass_end_y_m"] = pd.to_numeric(df_match["Pass End Y"], errors='coerce') * PITCH_WIDTH / 100.0
    except Exception:
        df_match["pass_end_x_m"] = np.nan
        df_match["pass_end_y_m"] = np.nan

    flip_away = False
    if 'team_position' in df_match.columns:
        team_pos_series = df_match[df_match['team_name'] == team_name]['team_position']
        if not team_pos_series.empty and team_pos_series.iloc[0] == 'away':
            flip_away = True

    df_match["x_plot_m"] = df_match["x_m"]
    df_match["y_plot_m"] = df_match["y_m"]
    df_match["pass_end_x_plot_m"] = df_match["pass_end_x_m"]
    df_match["pass_end_y_plot_m"] = df_match["pass_end_y_m"]

    if flip_away:
        mask_team = (df_match["team_name"] == team_name)
        df_match.loc[mask_team, "x_plot_m"] = PITCH_LENGTH - df_match.loc[mask_team, "x_m"]
        df_match.loc[mask_team, "y_plot_m"] = PITCH_WIDTH - df_match.loc[mask_team, "y_m"]
        df_match.loc[mask_team, "pass_end_x_plot_m"] = PITCH_LENGTH - df_match.loc[mask_team, "pass_end_x_m"]
        df_match.loc[mask_team, "pass_end_y_plot_m"] = PITCH_WIDTH - df_match.loc[mask_team, "pass_end_y_m"]

        mask_opp = (df_match["team_name"] != team_name)
        df_match.loc[mask_opp, "x_plot_m"] = df_match.loc[mask_opp, "x_m"]
        df_match.loc[mask_opp, "y_plot_m"] = df_match.loc[mask_opp, "y_m"]
        df_match.loc[mask_opp, "pass_end_x_plot_m"] = df_match.loc[mask_opp, "pass_end_x_m"]
        df_match.loc[mask_opp, "pass_end_y_plot_m"] = df_match.loc[mask_opp, "pass_end_y_m"]

    # Find row index of goal event
    def _find_event_rows(column):
        if column not in df_match.columns:
            return pd.DataFrame()
        numeric_target = pd.to_numeric(pd.Series([event_id]), errors='coerce').iloc[0]
        numeric_values = pd.to_numeric(df_match[column], errors='coerce')
        if not pd.isna(numeric_target):
            matches = df_match[numeric_values == numeric_target]
            if not matches.empty:
                return matches
        return df_match[df_match[column].astype(str) == str(event_id)]

    matching_rows = _find_event_rows('event_id')
    if matching_rows.empty and 'id' in df_match.columns:
        matching_rows = _find_event_rows('id')
    if not matching_rows.empty:
        goal_matches = matching_rows[
            (matching_rows['team_name'] == team_name) &
            (matching_rows['event'].astype(str).isin(GOAL_EVENTS))
        ]
        if not goal_matches.empty:
            matching_rows = goal_matches
        else:
            team_matches = matching_rows[matching_rows['team_name'] == team_name]
            if not team_matches.empty:
                matching_rows = team_matches
    if matching_rows.empty:
        try:
            matching_rows = df_match.loc[[int(event_id)]]
        except Exception:
            return pd.DataFrame()

    goal_row_idx = matching_rows.index[0]

    # Trace the complete uninterrupted scoring possession. Stop when the
    # previous team has a real field action or when the period changes.
    lead_up_indices = []
    goal_period = df_match.iloc[goal_row_idx].get('period_id')
    i = goal_row_idx - 1
    while i >= 0:
        row_i = df_match.iloc[i]
        if pd.notna(goal_period) and row_i.get('period_id') != goal_period:
            break
        if row_i.get('team_name') != team_name:
            if is_field_event_row(row_i):
                break
            i -= 1
            continue
        if str(row_i.get('event', '')).lower() in {'out', 'offside pass', 'foul throw-in'}:
            break
        if is_field_event_row(row_i):
            lead_up_indices.append(i)
        i -= 1
        
    lead_up_indices.reverse()
    all_indices = lead_up_indices + [goal_row_idx]
    
    window_raw = df_match.loc[all_indices].copy()
    window_raw["is_goal"] = False
    window_raw.loc[window_raw.index[-1], "is_goal"] = True

    not_deleted_mask = ~window_raw.apply(is_deleted_row, axis=1)
    valid_name_mask = window_raw.apply(has_valid_player_name, axis=1)

    window = window_raw[not_deleted_mask & valid_name_mask].copy().reset_index(drop=True)
    window = window[~window["event"].astype(str).str.lower().eq("card")].reset_index(drop=True)

    if len(window) == 0:
        return pd.DataFrame()

    timeline_rows = []
    action_counter = 1
    prev_ball_end_x = None
    prev_ball_end_y = None
    n = len(window)

    for k in range(n):
        this = window.iloc[k].copy()
        ev_str = str(this["event"])

        # ---- Admin / VAR ----
        if is_admin_event(ev_str):
            if is_contentious_ref_event(ev_str):
                var_row = this.copy()
                var_row["kind"] = "var"
                var_row["event"] = "VAR Review"
                var_row["is_goal"] = False
                var_row["order"] = action_counter
                action_counter += 1

                var_row["x_plot_m"] = PITCH_LENGTH / 2.0
                var_row["y_plot_m"] = PITCH_WIDTH  / 2.0

                if prev_ball_end_x is None:
                    ball_x = var_row["x_plot_m"]
                    ball_y = var_row["y_plot_m"]
                else:
                    ball_x = prev_ball_end_x
                    ball_y = prev_ball_end_y

                var_row["ball_start_x"] = ball_x
                var_row["ball_start_y"] = ball_y
                var_row["ball_end_x"]   = ball_x
                var_row["ball_end_y"]   = ball_y
                var_row["has_line"] = False
                var_row["line_is_dashed"] = False

                prev_ball_end_x, prev_ball_end_y = ball_x, ball_y
                timeline_rows.append(var_row)
            continue

        # ---- Error / Clearance → "puente" ----
        if is_error_event(ev_str):
            if 0 < k < n - 1:
                nxt = window.iloc[k + 1]

                bridge = this.copy()
                bridge["kind"] = "error_bridge"
                bridge["is_goal"] = False
                bridge["order"] = action_counter
                action_counter += 1

                if prev_ball_end_x is None:
                    start_x = this["x_plot_m"]
                    start_y = this["y_plot_m"]
                else:
                    start_x = prev_ball_end_x
                    start_y = prev_ball_end_y

                bridge["ball_start_x"] = start_x
                bridge["ball_start_y"] = start_y
                bridge["ball_end_x"]   = nxt["x_plot_m"]
                bridge["ball_end_y"]   = nxt["y_plot_m"]

                bridge["has_line"] = False
                bridge["line_is_dashed"] = False

                prev_ball_end_x, prev_ball_end_y = bridge["ball_end_x"], bridge["ball_end_y"]
                timeline_rows.append(bridge)
            continue

        # ---- Evento normal ----
        this_kind = "event"
        this["order"] = action_counter
        action_counter += 1

        if is_save_like_event_row(this):
            pos = str(this.get("position", "")).upper()
            if pos == "GK":
                this_kind = "save_gk"
            else:
                this_kind = "save_block"

        is_pass = (
            ev_str == "Pass"
            and not pd.isna(this["pass_end_x_plot_m"])
            and not pd.isna(this["pass_end_y_plot_m"])
        )
        is_shot = is_shot_event(ev_str)

        shot_saved = is_shot and is_shot_saved_generic(this)
        pass_intercepted = is_pass and is_pass_intercepted(this)

        # ====== ball_start / ball_end ======
        if k == 0:
            if is_pass:
                ball_start_x = this["x_plot_m"]
                ball_start_y = this["y_plot_m"]
                ball_end_x   = this["pass_end_x_plot_m"]
                ball_end_y   = this["pass_end_y_plot_m"]
            elif is_shot:
                ball_start_x = this["x_plot_m"]
                ball_start_y = this["y_plot_m"]

                end_x, end_y = compute_shot_end_coordinates_default(this)

                if not is_goal_row(this):
                    next_ev = get_next_event_for_shot_row(this, df_match)
                    if next_ev is not None:
                        ne_x = next_ev.get("x_plot_m")
                        ne_y = next_ev.get("y_plot_m")
                        if not pd.isna(ne_x) and not pd.isna(ne_y):
                            end_x, end_y = float(ne_x), float(ne_y)

                ball_end_x = end_x
                ball_end_y = end_y
            else:
                ball_start_x = this["x_plot_m"]
                ball_start_y = this["y_plot_m"]
                ball_end_x   = this["x_plot_m"]
                ball_end_y   = this["y_plot_m"]
        else:
            if is_pass:
                ball_start_x = this["x_plot_m"]
                ball_start_y = this["y_plot_m"]
                ball_end_x   = this["pass_end_x_plot_m"]
                ball_end_y   = this["pass_end_y_plot_m"]
            elif is_shot:
                ball_start_x = this["x_plot_m"]
                ball_start_y = this["y_plot_m"]

                end_x, end_y = compute_shot_end_coordinates_default(this)

                if not is_goal_row(this):
                    next_ev = get_next_event_for_shot_row(this, df_match)
                    if next_ev is not None:
                        ne_x = next_ev.get("x_plot_m")
                        ne_y = next_ev.get("y_plot_m")
                        if not pd.isna(ne_x) and not pd.isna(ne_y):
                            end_x, end_y = float(ne_x), float(ne_y)

                ball_end_x = end_x
                ball_end_y = end_y
            else:
                ball_start_x = prev_ball_end_x
                ball_start_y = prev_ball_end_y
                ball_end_x   = this["x_plot_m"]
                ball_end_y   = this["y_plot_m"]

        this["ball_start_x"] = ball_start_x
        this["ball_start_y"] = ball_start_y
        this["ball_end_x"]   = ball_end_x
        this["ball_end_y"]   = ball_end_y

        this["has_line"] = bool(is_pass or is_shot)
        this["line_is_dashed"] = bool(shot_saved or pass_intercepted)

        this["kind"] = this_kind

        prev_ball_end_x, prev_ball_end_y = ball_end_x, ball_end_y
        timeline_rows.append(this)

        # ---- CONDUCCIONES (carry)
        if k < n - 1:
            nxt = window.iloc[k + 1]
            same_team   = this.get("team_name") == nxt.get("team_name")
            same_player = this.get("player_name") == nxt.get("player_name")

            base_ok = (
                same_team
                and not is_error_event(str(nxt["event"]))
                and not is_admin_event(str(nxt["event"]))
                and not is_deleted_row(nxt)
            )

            # Conducción tras pase
            if (
                base_ok
                and is_pass
                and not pd.isna(this["pass_end_x_plot_m"])
                and not pd.isna(this["pass_end_y_plot_m"])
            ):
                carry = nxt.copy()
                carry["kind"] = "carry"
                carry["event"] = "Carry"
                carry["is_goal"] = False
                carry["order"] = action_counter
                action_counter += 1

                carry["x_plot_m"] = prev_ball_end_x
                carry["y_plot_m"] = prev_ball_end_y

                carry["ball_start_x"] = prev_ball_end_x
                carry["ball_start_y"] = prev_ball_end_y
                carry["ball_end_x"]   = nxt["x_plot_m"]
                carry["ball_end_y"]   = nxt["y_plot_m"]

                carry["has_line"] = False
                carry["line_is_dashed"] = False

                prev_ball_end_x, prev_ball_end_y = carry["ball_end_x"], carry["ball_end_y"]
                timeline_rows.append(carry)

            # Conducción por mismo jugador (aunque no sea pase)
            elif base_ok and same_player:
                carry = nxt.copy()
                carry["kind"] = "carry"
                carry["event"] = "Carry"
                carry["is_goal"] = False
                carry["order"] = action_counter
                action_counter += 1

                carry["x_plot_m"] = prev_ball_end_x
                carry["y_plot_m"] = prev_ball_end_y

                carry["ball_start_x"] = prev_ball_end_x
                carry["ball_start_y"] = prev_ball_end_y
                carry["ball_end_x"]   = nxt["x_plot_m"]
                carry["ball_end_y"]   = nxt["y_plot_m"]

                carry["has_line"] = False
                carry["line_is_dashed"] = False

                prev_ball_end_x, prev_ball_end_y = carry["ball_end_x"], carry["ball_end_y"]
                timeline_rows.append(carry)

    timeline = pd.DataFrame(timeline_rows).reset_index(drop=True)
    if not timeline.empty:
        timeline["order"] = range(1, len(timeline) + 1)
    return timeline


def create_pitch_figure():
    fig = go.Figure()

    # Green pitch background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=PITCH_LENGTH, y1=PITCH_WIDTH,
        fillcolor="#1a3a1e",
        layer="below",
        line=dict(width=0),
    )

    # Grass stripes
    stripe_width = PITCH_LENGTH / 10
    for i in range(10):
        fig.add_shape(
            type="rect",
            x0=i * stripe_width,
            y0=0,
            x1=(i + 1) * stripe_width,
            y1=PITCH_WIDTH,
            fillcolor="rgba(0, 100, 0, 0.25)" if i % 2 == 0 else "rgba(0, 80, 0, 0.20)",
            opacity=1,
            layer="below",
            line=dict(width=0),
        )

    line_color = "white"
    box_color  = "white"

    field_lines = [
        [[0, 0], [0, PITCH_WIDTH]],
        [[0, PITCH_WIDTH], [PITCH_LENGTH, PITCH_WIDTH]],
        [[PITCH_LENGTH, PITCH_WIDTH], [PITCH_LENGTH, 0]],
        [[PITCH_LENGTH, 0], [0, 0]],
        [[PITCH_LENGTH / 2, 0], [PITCH_LENGTH / 2, PITCH_WIDTH]],
        [[16.5, (PITCH_WIDTH / 2) - 16.5], [16.5, (PITCH_WIDTH / 2) + 16.5]],
        [[PITCH_LENGTH - 16.5, (PITCH_WIDTH / 2) - 16.5],
         [PITCH_LENGTH - 16.5, (PITCH_WIDTH / 2) + 16.5]],
        [[0, (PITCH_WIDTH / 2) - 16.5], [16.5, (PITCH_WIDTH / 2) - 16.5]],
        [[0, (PITCH_WIDTH / 2) + 16.5], [16.5, (PITCH_WIDTH / 2) + 16.5]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) - 16.5],
         [PITCH_LENGTH - 16.5, (PITCH_WIDTH / 2) - 16.5]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) + 16.5],
         [PITCH_LENGTH - 16.5, (PITCH_WIDTH / 2) + 16.5]],
        [[5.5, (PITCH_WIDTH / 2) - 5.5], [5.5, (PITCH_WIDTH / 2) + 5.5]],
        [[PITCH_LENGTH - 5.5, (PITCH_WIDTH / 2) - 5.5],
         [PITCH_LENGTH - 5.5, (PITCH_WIDTH / 2) + 5.5]],
        [[0, (PITCH_WIDTH / 2) - 5.5], [5.5, (PITCH_WIDTH / 2) - 5.5]],
        [[0, (PITCH_WIDTH / 2) + 5.5], [5.5, (PITCH_WIDTH / 2) + 5.5]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) - 5.5],
         [PITCH_LENGTH - 5.5, (PITCH_WIDTH / 2) - 5.5]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) + 5.5],
         [PITCH_LENGTH - 5.5, (PITCH_WIDTH / 2) + 5.5]],
    ]

    for line in field_lines:
        fig.add_shape(
            type="line",
            x0=line[0][0], y0=line[0][1],
            x1=line[1][0], y1=line[1][1],
            line=dict(color=line_color, width=1.8),
            layer="above"
        )

    fig.add_shape(
        type="circle",
        x0=PITCH_LENGTH / 2 - 9.15,
        y0=PITCH_WIDTH / 2 - 9.15,
        x1=PITCH_LENGTH / 2 + 9.15,
        y1=PITCH_WIDTH / 2 + 9.15,
        line=dict(color=line_color, width=1.8),
        layer="above"
    )

    for cx, cy in [
        (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
        (11, PITCH_WIDTH / 2),
        (PITCH_LENGTH - 11, PITCH_WIDTH / 2),
    ]:
        fig.add_shape(
            type="circle",
            x0=cx - 0.3, y0=cy - 0.3,
            x1=cx + 0.3, y1=cy + 0.3,
            fillcolor=line_color,
            line=dict(color=line_color, width=1),
            layer="above",
        )

    goal_center_y  = PITCH_WIDTH / 2.0
    goal_half_w    = GOAL_WIDTH_M / 2.0
    goal_y_bottom  = goal_center_y - goal_half_w
    goal_y_top     = goal_center_y + goal_half_w

    goal_right_x_front = PITCH_LENGTH - GOAL_MARGIN_X
    goal_right_x_back  = goal_right_x_front + GOAL_DEPTH_M

    fig.add_shape(
        type="line",
        x0=goal_right_x_front, y0=goal_y_bottom,
        x1=goal_right_x_back,  y1=goal_y_bottom,
        line=dict(color=box_color, width=3),
        layer="above"
    )
    fig.add_shape(
        type="line",
        x0=goal_right_x_front, y0=goal_y_top,
        x1=goal_right_x_back,  y1=goal_y_top,
        line=dict(color=box_color, width=3),
        layer="above"
    )
    fig.add_shape(
        type="line",
        x0=goal_right_x_back, y0=goal_y_bottom,
        x1=goal_right_x_back, y1=goal_y_top,
        line=dict(color=box_color, width=3),
        layer="above"
    )

    goal_left_x_front = 0.0 + GOAL_MARGIN_X
    goal_left_x_back  = goal_left_x_front - GOAL_DEPTH_M

    fig.add_shape(
        type="line",
        x0=goal_left_x_front, y0=goal_y_bottom,
        x1=goal_left_x_back,  y1=goal_y_bottom,
        line=dict(color=box_color, width=3),
        layer="above"
    )
    fig.add_shape(
        type="line",
        x0=goal_left_x_front, y0=goal_y_top,
        x1=goal_left_x_back,  y1=goal_y_top,
        line=dict(color=box_color, width=3),
        layer="above"
    )
    fig.add_shape(
        type="line",
        x0=goal_left_x_back, y0=goal_y_bottom,
        x1=goal_left_x_back, y1=goal_y_top,
        line=dict(color=box_color, width=3),
        layer="above"
    )

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-4, PITCH_LENGTH + 4],
            fixedrange=True,
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-2, PITCH_WIDTH + 2],
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="#1a3a1e",
        paper_bgcolor="#1a3a1e",
        height=450,
        margin=dict(l=5, r=5, t=5, b=5),
    )
    return fig


def build_static_sequence_figure(timeline: pd.DataFrame, title: str = "") -> go.Figure:
    fig = create_pitch_figure()

    if timeline.empty:
        return fig

    # ---------- PUNTOS DE ACCIÓN ----------
    xs, ys, colors, symbols, texts, hover = [], [], [], [], [], []

    for _, row in timeline.iterrows():
        xs.append(row["x_plot_m"])
        ys.append(row["y_plot_m"])

        ev_str = str(row.get("event", ""))
        is_foul = is_foul_event(ev_str)
        kind = row.get("kind", "event")

        # Colores base por equipo
        if row["team_name"] == GOZTEPE:
            base_color = "#ef4444"
        else:
            base_color = "#3b82f6"

        is_save_gk    = (kind == "save_gk")
        is_save_block = (kind == "save_block")

        # Símbolos / colores por tipo
        if kind == "var":
            color = "#ffd54f"
            symbol = "diamond"
            text_label = "V"
        elif bool(row.get("is_goal", False)):
            color = "#ff4fd8"
            symbol = "star"
            text_label = "⚽"
        else:
            if is_save_gk:
                color = "#4ade80"   # verde
                symbol = "diamond"
                text_label = "🧤"
            elif is_save_block:
                color = "#fb923c"   # naranja
                symbol = "hexagon"
                text_label = "🛡"
            elif is_foul:
                color = "#ffa600"
                symbol = "triangle-up"
                text_label = str(row["order"])
            elif kind == "carry":
                color = base_color
                symbol = "circle-open"
                text_label = str(row["order"])
            elif kind == "error_bridge":
                color = "#9ca3af"
                symbol = "x"
                text_label = str(row["order"])
            else:
                color = base_color
                symbol = "circle"
                text_label = str(row["order"])

        colors.append(color)
        symbols.append(symbol)
        texts.append(text_label)

        minute = int(row.get("time_min", 0))
        second = int(row.get("time_sec", 0))
        t_str  = f"{minute:02d}:{second:02d}"

        if kind == "var":
            hover.append(
                f"{row['order']}. VAR Review<br>"
                f"Time: {t_str}, Period: {row.get('period_id', 1)}"
            )
        else:
            hover.append(
                f"{row['order']}. {row['event']} - {row.get('player_name','')} ({row.get('team_name','')})<br>"
                f"Time: {t_str}, Period: {row.get('period_id', 1)}<br>"
                f"Type: {kind}"
            )

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            color=colors,
            symbol=symbols,
            size=11,
            line=dict(color="rgba(0,0,0,0.6)", width=1),
        ),
        text=texts,
        textposition="top center",
        textfont=dict(size=8, color="#e5e7eb"),
        hovertext=hover,
        hoverinfo="text",
        name="Actions",
        showlegend=False,
    ))

    # ---------- LÍNEAS PASES + TIROS ----------
    pass_solid_x, pass_solid_y = [], []
    pass_dash_x, pass_dash_y   = [], []
    carry_x, carry_y = [], []
    shot_goal_x, shot_goal_y = [], []
    shot_other_x, shot_other_y = [], []

    # --- Pases y tiros (usa has_line) ---
    for _, row in timeline.iterrows():
        if not bool(row.get("has_line", False)):
            continue

        ev_str = str(row["event"])
        x0 = row["ball_start_x"]
        y0 = row["ball_start_y"]
        x1 = row["ball_end_x"]
        y1 = row["ball_end_y"]

        if ev_str == "Pass":
            if bool(row.get("line_is_dashed", False)):
                pass_dash_x.extend([x0, x1, None])
                pass_dash_y.extend([y0, y1, None])
            else:
                pass_solid_x.extend([x0, x1, None])
                pass_solid_y.extend([y0, y1, None])

        elif is_shot_event(ev_str):
            if bool(row.get("is_goal", False)):
                shot_goal_x.extend([x0, x1, None])
                shot_goal_y.extend([y0, y1, None])
            else:
                shot_other_x.extend([x0, x1, None])
                shot_other_y.extend([y0, y1, None])

    # --- Conducciones (carry) como líneas propias ---
    for _, row in timeline.iterrows():
        if row.get("kind") != "carry":
            continue
        x0 = row["ball_start_x"]
        y0 = row["ball_start_y"]
        x1 = row["ball_end_x"]
        y1 = row["ball_end_y"]
        carry_x.extend([x0, x1, None])
        carry_y.extend([y0, y1, None])

    # Successful passes (solid lines)
    if pass_solid_x:
        fig.add_trace(go.Scatter(
            x=pass_solid_x,
            y=pass_solid_y,
            mode="lines",
            line=dict(color="#00eaff", width=2),
            hoverinfo="skip",
            name="Passes",
            showlegend=True,
        ))

    # Intercepted passes (dashed lines)
    if pass_dash_x:
        fig.add_trace(go.Scatter(
            x=pass_dash_x,
            y=pass_dash_y,
            mode="lines",
            line=dict(color="#00eaff", width=3, dash="dash"),
            hoverinfo="skip",
            name="Passes (intercepted)",
            showlegend=True,
        ))

    # Ball carries (dotted white lines)
    if carry_x:
        fig.add_trace(go.Scatter(
            x=carry_x,
            y=carry_y,
            mode="lines",
            line=dict(color="#ffffff", width=2, dash="dot"),
            hoverinfo="skip",
            name="Carries",
            showlegend=True,
        ))

    # Goals
    if shot_goal_x:
        fig.add_trace(go.Scatter(
            x=shot_goal_x,
            y=shot_goal_y,
            mode="lines",
            line=dict(color="#ff6bcb", width=3),
            hoverinfo="skip",
            name="Goals",
            showlegend=True,
        ))

    # Shots off target / other
    if shot_other_x:
        fig.add_trace(go.Scatter(
            x=shot_other_x,
            y=shot_other_y,
            mode="lines",
            line=dict(color="#fb923c", width=1.5),
            hoverinfo="skip",
            name="Shots off target",
            showlegend=True,
        ))

    # ---------- FINAL BALL ----------
    last = timeline.iloc[-1]
    ball_x = last["ball_end_x"]
    ball_y = last["ball_end_y"]

    fig.add_trace(go.Scatter(
        x=[ball_x],
        y=[ball_y],
        mode="markers",
        marker=dict(
            color="#ffffff",
            symbol="circle",
            size=8,
            line=dict(color="#00f5ff", width=2),
        ),
        hoverinfo="skip",
        name="Final Ball",
        showlegend=True,
    ))

    # ---------- EXTRA LEGEND (EMOJIS) ----------
    # ⭐ Goal
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            color="#ff4fd8",
            symbol="star",
            size=10,
            line=dict(color="rgba(0,0,0,0.6)", width=1),
        ),
        name="Goal ⭐",
        showlegend=True,
        hoverinfo="skip",
    ))

    # 🧤 Goalkeeper Save
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers+text",
        marker=dict(
            color="#4ade80",
            symbol="diamond",
            size=10,
            line=dict(color="rgba(0,0,0,0.6)", width=1),
        ),
        text=["🧤"],
        textposition="top center",
        name="Goalkeeper Save 🧤",
        showlegend=True,
        hoverinfo="skip",
    ))

    # 🛡 Shot Block
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers+text",
        marker=dict(
            color="#fb923c",
            symbol="hexagon",
            size=10,
            line=dict(color="rgba(0,0,0,0.6)", width=1),
        ),
        text=["🛡"],
        textposition="top center",
        name="Shot Block 🛡",
        showlegend=True,
        hoverinfo="skip",
    ))

    # ⚠️ Foul
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            color="#ffa600",
            symbol="triangle-up",
            size=10,
            line=dict(color="rgba(0,0,0,0.6)", width=1),
        ),
        name="Foul ⚠️",
        showlegend=True,
        hoverinfo="skip",
    ))

    # ❌ Error / Clearance
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            color="#9ca3af",
            symbol="x",
            size=10,
            line=dict(color="rgba(0,0,0,0.6)", width=1),
        ),
        name="Error ❌",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#e5e7eb", size=8),
        ),
    )
    return fig


@callback(
    [Output('pre-match-goal-selector', 'options'),
     Output('pre-match-goal-selector', 'value')],
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-goal-origin-filter', 'value')]
)
def update_goal_selector_options(opponent, origin_filter):
    if not opponent:
        return [], None
        
    stats = _get_season_stats(opponent)
    goals = stats.get('goal_sequences', [])
    
    if origin_filter and origin_filter != 'all':
        goals = [g for g in goals if g.get('origin') == origin_filter]
        
    if not goals:
        return [], None
        
    options = []
    for g in goals:
        label = f"{g['minute']}' - {g['player']}"
        value = f"{g['filename']}|{g['event_id']}|{g['player']}|{g['minute']}"
        options.append({'label': label, 'value': value})
        
    default_val = options[0]['value'] if options else None
    return options, default_val


@callback(
    Output('pre-match-goal-sequence-graph-container', 'children'),
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-goal-selector', 'value')]
)
def update_goal_sequence_graph(opponent, selected_goal_value):
    if not opponent or not selected_goal_value:
        return html.Div("Select a goal to visualize the complete attacking possession.", style={
            "textAlign": "center", "padding": "20px", "color": "var(--text-secondary)", "fontSize": "0.85rem"
        })
        
    try:
        parts = selected_goal_value.split('|')
        filename = parts[0]
        event_id = parts[1]
        player = parts[2]
        minute = parts[3]
        
        timeline = _load_goal_timeline(filename, event_id, opponent)
        if timeline.empty:
            return html.Div("Complete attacking possession data is not available for this goal.", style={
                "textAlign": "center", "padding": "20px", "color": "var(--text-secondary)", "fontSize": "0.85rem"
            })
            
        fig = build_static_sequence_figure(timeline, f"Goal sequence by {player} ({minute}')")
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    except Exception as e:
        return html.Div(f"Error loading goal sequence: {e}", style={"color": _RED, "fontSize": "0.8rem", "padding": "10px"})


@callback(
    Output('pre-match-goal-list-container', 'children'),
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-goal-origin-filter', 'value')]
)
def update_goal_list(opponent, origin_filter):
    if not opponent:
        return html.Div()
        
    stats = _get_season_stats(opponent)
    goals = stats.get('goal_sequences', [])
    
    # Filter by origin
    if origin_filter and origin_filter != 'all':
        goals = [g for g in goals if g.get('origin') == origin_filter]
        
    if not goals:
        return html.Div("No goals matching this filter found for the season.", style={
            "textAlign": "center", "padding": "20px", "color": "var(--text-secondary)", "fontSize": "0.85rem"
        })
        
    goal_rows = []
    origin_labels = {
        'open_play': 'Open Play',
        'from_cross': 'Cross',
        'set_piece': 'Set Piece',
        'through_ball': 'Through Ball',
        'fast_break': 'Fast Break'
    }
    origin_badges_colors = {
        'open_play': "rgba(34,197,94,0.12)",
        'from_cross': "rgba(59,130,246,0.12)",
        'set_piece': "rgba(251,191,36,0.12)",
        'through_ball': "rgba(168,85,247,0.12)",
        'fast_break': "rgba(239,68,68,0.12)"
    }
    origin_colors = {
        'open_play': '#22c55e',
        'from_cross': '#3b82f6',
        'set_piece': '#fbbf24',
        'through_ball': '#a855f7',
        'fast_break': '#ef4444'
    }
    
    for g in goals:
        origin_type = g.get('origin', 'open_play')
        goal_value = f"{g['filename']}|{g['event_id']}|{g['player']}|{g['minute']}"
        
        goal_rows.append(html.Div(id={'type': 'pre-match-goal-row', 'value': goal_value}, n_clicks=0, className='pre-match-goal-row', style={
            "display": "flex", "alignItems": "center", "gap": "10px",
            "padding": "10px 14px", "borderRadius": "10px", "marginBottom": "8px",
            "background": "rgba(255,255,255,0.03)", "border": "1px solid var(--border-color)",
            "cursor": "pointer",
            "transition": "border-color 0.2s, background 0.2s",
        }, children=[
            html.Div(style={"width": "36px", "height": "36px", "borderRadius": "50%",
                            "background": "rgba(251,191,36,0.1)", "display": "flex",
                            "alignItems": "center", "justifyContent": "center",
                            "border": "1.5px solid #fbbf24", "fontWeight": "bold", "color": "#fbbf24", "fontSize": "0.85rem"},
                     children=f"{g['minute']}'"),
            html.Div(style={"flex": "2"}, children=[
                html.Div(g['player'], style={"fontWeight": "700", "fontSize": "0.9rem", "color": "white"}),
                html.Div(f"Week {g['week']} vs {g['opponent']}", style={"fontSize": "0.74rem", "color": "var(--text-secondary)"}),
            ]),
            html.Div(style={
                "padding": "4px 10px", "borderRadius": "6px", "fontSize": "0.72rem", "fontWeight": "700",
                "background": origin_badges_colors.get(origin_type, "rgba(255,255,255,0.12)"),
                "color": origin_colors.get(origin_type, "white"),
                "border": f"1px solid {origin_colors.get(origin_type, 'rgba(255,255,255,0.15)')}"
            }, children=origin_labels.get(origin_type, 'Open Play'))
        ]))
        
    return html.Div([
        html.Div("Click a goal below, or select one from the dropdown, to inspect its full sequence.", style={
            "fontSize": "0.68rem",
            "color": "var(--text-secondary)",
            "marginBottom": "8px",
        }),
        *goal_rows,
    ])


@callback(
    Output('pre-match-goal-selector', 'value', allow_duplicate=True),
    Input({'type': 'pre-match-goal-row', 'value': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def select_goal_from_list(_clicks):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict):
        return triggered.get('value')
    return dash.no_update


# ──────────────────────────────────────────────────────────────
# Dynamic Final Third Entry Analytics
# ──────────────────────────────────────────────────────────────

_F3_ENTRIES_CACHE = {}

def _get_f3_entries_stats(team_name, only_buildup=False):
    cache_key = (team_name, only_buildup)
    if cache_key in _F3_ENTRIES_CACHE:
        return _F3_ENTRIES_CACHE[cache_key]

    from göztepehub.utils.transitions_analysis import _load_opponent_matches
    match_dfs = _load_opponent_matches(team_name)
    
    total_entries = 0
    shots_count = 0
    goals_count = 0
    shot_coords = []
    
    SHOT_EVENTS_SET = {'Goal', 'Miss', 'Saved Shot', 'Post'}
    
    for fn, df in match_dfs:
        df_clean = df.copy()
        flip = False
        if 'team_position' in df_clean.columns:
            pos_series = df_clean[df_clean['team_name'] == team_name]['team_position']
            if not pos_series.empty and pos_series.iloc[0] == 'away':
                flip = True

        if flip:
            df_clean['x'] = 100.0 - df_clean['x']
            df_clean['y'] = 100.0 - df_clean['y']
            if 'Pass End X' in df_clean.columns:
                try:
                    df_clean['Pass End X'] = 100.0 - pd.to_numeric(df_clean['Pass End X'], errors='coerce')
                except:
                    pass
            if 'Pass End Y' in df_clean.columns:
                try:
                    df_clean['Pass End Y'] = 100.0 - pd.to_numeric(df_clean['Pass End Y'], errors='coerce')
                except:
                    pass

        records = df_clean.to_dict('records')
        
        from göztepehub.utils.buildup_analysis import _extract_possession_sequences, _detect_f3_entry_event
        
        seq_with_idx = _extract_possession_sequences(records, team_name)
        
        for seq, _ in seq_with_idx:
            if not seq:
                continue
            
            start_x = seq[0]['x']
            if only_buildup:
                if start_x >= 50:
                    continue
            else:
                if start_x >= 66.6:
                    continue
            
            entry = _detect_f3_entry_event(seq)
            if entry is None:
                continue
            
            total_entries += 1
            entry_time = entry['entry_time']
            entry_idx = entry['entry_event_idx']
            
            has_shot = False
            has_goal = False
            seq_shot_coords = None
            
            for idx in range(entry_idx, len(seq)):
                ev = seq[idx]
                ev_time = ev['time_min'] * 60 + ev['time_sec']
                if ev_time - entry_time > 10:
                    break
                
                if ev['event'] in SHOT_EVENTS_SET:
                    has_shot = True
                    is_g = (ev['event'] == 'Goal')
                    if is_g:
                        has_goal = True
                    sx = ev.get('x')
                    sy = ev.get('y')
                    if sx is not None and sy is not None:
                        try:
                            seq_shot_coords = {
                                'x': float(sx),
                                'y': float(sy),
                                'is_goal': is_g
                            }
                        except:
                            pass
                    break
            
            if has_shot:
                shots_count += 1
                if seq_shot_coords:
                    shot_coords.append(seq_shot_coords)
            if has_goal:
                goals_count += 1
                
    if total_entries > 0:
        shot_pct = round(shots_count / total_entries * 100, 1)
        goal_pct = round(goals_count / total_entries * 100, 1)
        nothing_pct = round(max(0.0, 100.0 - shot_pct), 1)
    else:
        shot_pct = 0.0
        goal_pct = 0.0
        nothing_pct = 0.0
        
    result = {
        'total_entries': total_entries,
        'shot_pct': shot_pct,
        'goal_pct': goal_pct,
        'nothing_pct': nothing_pct,
        'shot_coords': shot_coords
    }
    _F3_ENTRIES_CACHE[cache_key] = result
    return result


def _build_f3_shots_pitch(shot_coords, title_prefix='FINAL-THIRD ENTRIES'):
    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=_PITCH_BG)
    ax.set_facecolor(_PITCH_BG)
    
    # Draw attacking half-pitch outline (x from 50 to 100, y from 0 to 100)
    ax.plot([50, 100, 100, 50, 50], [0, 0, 100, 100, 0], color='white', alpha=0.55, linewidth=1.5)
    # Penalty box
    ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color='white', alpha=0.42, linewidth=1.2)
    # Six yard box
    ax.plot([100, 94.2, 94.2, 100], [40.9, 40.9, 59.1, 59.1], color='white', alpha=0.28, linewidth=0.9)
    
    # Halfway line arc
    from matplotlib.patches import Arc
    halfway_arc = Arc((50, 50), 18.3, 18.3, theta1=270, theta2=90, color='white', alpha=0.42, linewidth=1.2)
    ax.add_patch(halfway_arc)
    
    # Plot shots
    goals_x = [s['x'] for s in shot_coords if s['is_goal']]
    goals_y = [s['y'] for s in shot_coords if s['is_goal']]
    
    shots_x = [s['x'] for s in shot_coords if not s['is_goal']]
    shots_y = [s['y'] for s in shot_coords if not s['is_goal']]
    
    if shots_x:
        ax.scatter(shots_x, shots_y, color='#3b82f6', s=55, alpha=0.75, label='Shot (No Goal)', edgecolors='white', linewidths=0.6, zorder=5)
    if goals_x:
        ax.scatter(goals_x, goals_y, color='#22c55e', s=110, alpha=0.95, label='Goal', edgecolors='white', linewidths=0.9, zorder=6)
        
    ax.set_xlim(48, 102)
    ax.set_ylim(-3, 103)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend inside pitch
    ax.legend(loc='lower left', framealpha=0.9, facecolor='#111f12', edgecolor='none', fontsize=8, labelcolor='white')
    
    ax.set_title(f'{title_prefix} — SHOTS AFTER ENTRY ({len(shot_coords)} TOTAL)',
                 color='#fbbf24', fontsize=9.5, fontweight='bold', pad=8)
    return _fig_to_b64(fig)


@callback(
    Output('pre-match-f3-breakdown-dynamic-content', 'children'),
    [Input('pre-match-f3-entry-type-selector', 'value'),
     Input('pre-match-rival-selector', 'value')]
)
def update_f3_breakdown(entry_type, opponent):
    if not opponent:
        return html.Div("No opponent selected.")
        
    opp_name = _clean(opponent)
    only_buildup = (entry_type == 'buildup')
    
    try:
        data = _get_f3_entries_stats(opponent, only_buildup=only_buildup)
        total_entries = data['total_entries']
        shot_pct = data['shot_pct']
        goal_pct = data['goal_pct']
        nothing_pct = data['nothing_pct']
        shot_coords = data['shot_coords']
        
        children_list = []
        
        if only_buildup:
            mode_label = "BUILD-UP FINAL THIRD ENTRIES"
            mode_color = _GOLD
            mode_hint = "Only possessions that started in own half and then entered the final third."
            desc_text = f"Own-half build-ups that reached the final third within 10 seconds: {total_entries} entries."
        else:
            mode_label = "ALL FINAL THIRD ENTRIES"
            mode_color = _BLUE
            mode_hint = "Every possession sequence entering the final third, including transitions and higher starts."
            desc_text = f"All final third entries from open possession sequences: {total_entries} entries."

        # Generate the pitch visual containing only the filtered shots
        pitch_b64 = _build_f3_shots_pitch(shot_coords, title_prefix=mode_label)
            
        children_list.extend([
            html.Div(style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "gap": "10px", "marginBottom": "10px", "flexWrap": "wrap",
            }, children=[
                html.Span(mode_label, style={
                    "fontSize": "0.72rem", "fontWeight": "900", "color": mode_color,
                    "letterSpacing": "0.5px", "padding": "5px 8px",
                    "borderRadius": "999px", "background": f"{mode_color}22",
                    "border": f"1px solid {mode_color}55",
                }),
                html.Span(desc_text, style={
                    "fontSize": "0.72rem", "fontWeight": "700", "color": "white",
                }),
            ]),
            html.P(mode_hint, style={"fontSize": "0.68rem", "color": "var(--text-secondary)", "marginBottom": "12px"}),
            html.Img(src=pitch_b64, style={
                "width": "100%", "maxWidth": "860px", "margin": "0 auto 16px",
                "display": "block", "borderRadius": "10px", "border": "1px solid var(--border-color)"
            })
        ])
            
        # Outcomes progress bars
        children_list.extend([
            html.Div("AFTER FINAL-THIRD ENTRY — WHAT HAPPENED?", style={
                "fontSize": "0.72rem", "fontWeight": "800", "color": _GOLD,
                "letterSpacing": "0.4px", "marginBottom": "10px"}),
            html.P("Outcomes measured within 10 seconds of crossing into the final third:", style={
                "fontSize": "0.68rem", "color": "var(--text-secondary)", "marginBottom": "12px"}),
            make_progress_bar("Shot Attempt Rate (of F3 entries)", shot_pct, _BLUE),
            make_progress_bar("Goal Conversion Rate (of F3 entries)", goal_pct, _GREEN),
            make_progress_bar("No Shot — Possession Lost / Recycled", nothing_pct, _RED),
            html.Div(style={
                "marginTop": "12px", "padding": "8px 10px", "borderRadius": "8px",
                "background": "rgba(251,191,36,0.07)", "border": "1px solid rgba(251,191,36,0.2)",
                "fontSize": "0.68rem", "color": "var(--text-secondary)"
            }, children=[
                html.Span("Scouting Note: ", style={"fontWeight": "700", "color": _GOLD}),
                f"When {opp_name} enters the final third via "
                f"{'an own-half build-up' if only_buildup else 'any attacking transition/possession'}, they convert to a shot "
                f"{shot_pct}% of the time. "
                f"{'Compact shape to limit shot creation after F3 entry.' if shot_pct > 35 else 'Apply pressure to deny direct pathways after F3 entry.'}"
            ])
        ])
        
        return html.Div(children_list)
        
    except Exception as e:
        import traceback
        print(f"Error in update_f3_breakdown callback: {e}")
        traceback.print_exc()
        return html.Div(f"Error loading final third breakdown: {e}", style={"color": _RED, "padding": "10px"})
