
import pandas as pd
import numpy as np
import os

from utils.data import get_data_dir, extract_fixture_data, calculate_standings

# Import predict_xg from the main app
try:
    import importlib.util
    _main_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _xg_model_path = os.path.join(_main_app_dir, 'utils', 'xg_model.py')
    if os.path.exists(_xg_model_path):
        _spec = importlib.util.spec_from_file_location('main_xg_model', _xg_model_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _predict_xg = _mod.predict_xg
    else:
        _predict_xg = lambda df: df
except Exception:
    _predict_xg = lambda df: df

# Module-level cache for team events
_TEAM_EVENTS_CACHE = {}


def _load_all_team_events(team_name):
    """Load and concatenate all match events for a given team across the season."""
    if team_name in _TEAM_EVENTS_CACHE:
        return _TEAM_EVENTS_CACHE[team_name]
    
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    all_dfs = []
    match_count = 0
    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' in df.columns and team_name in df['team_name'].values:
                df['source_file'] = filename
                # Apply xG model so the 'xG' column is populated
                try:
                    df = _predict_xg(df)
                except Exception:
                    pass
                all_dfs.append(df)
                match_count += 1
        except Exception:
            continue
    
    if not all_dfs:
        result = (pd.DataFrame(), 0)
    else:
        result = (pd.concat(all_dfs, ignore_index=True), match_count)
    
    _TEAM_EVENTS_CACHE[team_name] = result
    return result


def _safe_div(a, b, decimals=2):
    if b == 0:
        return 0
    return round(a / b, decimals)


# ============================================================
# DEFENSIVE PHASE METRICS
# ============================================================

def calc_defensive_metrics(team_name):
    """
    Returns defensive phase metrics as a dict of {metric_label: value}.
    All per-game metrics are averaged across the season.
    """
    matches = extract_fixture_data(lite=True)
    standings_df = calculate_standings(matches)
    
    if standings_df.empty:
        return {}
    
    team_row = standings_df[standings_df['Team'] == team_name]
    if team_row.empty:
        return {}
    
    team_row = team_row.iloc[0]
    played = team_row['Played']
    
    # Load all events for advanced metrics
    all_events, match_count = _load_all_team_events(team_name)
    
    if all_events.empty or match_count == 0:
        return {}
    
    team_events = all_events[all_events['team_name'] == team_name]
    opp_events = all_events[all_events['team_name'] != team_name]
    
    # Tackles
    tackles = len(team_events[team_events['event'] == 'Tackle'])
    
    # Interceptions
    interceptions = len(team_events[team_events['event'] == 'Interception'])
    
    # Clearances
    clearances = len(team_events[team_events['event'] == 'Clearance'])
    
    # Defensive line height
    def_actions = ['Tackle', 'Interception', 'Challenge', 'Foul', 'Ball recovery', 'Ball Recovery', 'Clearance']
    def_events = team_events[team_events['event'].isin(def_actions)]
    line_height = round(def_events['x'].mean(), 1) if not def_events.empty else 50.0
    
    # High press %
    total_def = len(def_events)
    high_def = len(def_events[def_events['x'] > 50]) if total_def > 0 else 0
    high_press_pct = round((high_def / total_def) * 100, 1) if total_def > 0 else 0.0
    
    # PPDA
    opp_passes_own_half = len(opp_events[(opp_events['event'] == 'Pass') & (opp_events['x'] < 60)])
    team_def_in_opp = len(team_events[
        (team_events['event'].isin(['Tackle', 'Interception', 'Challenge', 'Foul', 'Ball recovery', 'Ball Recovery'])) &
        (team_events['x'] > 40)
    ])
    ppda = round(opp_passes_own_half / team_def_in_opp, 1) if team_def_in_opp > 0 else 0
    
    # GA and xGA from standings (need full standings data)
    matches_full = extract_fixture_data(lite=True)
    ga = 0
    xga = 0.0
    for m in matches_full:
        t1, t2 = m['team_names']
        if t1 == team_name:
            ga += m['stats']['team2']['goals']
            xga += m['stats']['team2'].get('xg', 0)
        elif t2 == team_name:
            ga += m['stats']['team1']['goals']
            xga += m['stats']['team1'].get('xg', 0)
            
    # Fallback: if xga is still 0.0, compute directly from loaded opponent events
    if xga == 0.0 and 'xG' in opp_events.columns:
        shot_events = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
        shot_mask = opp_events['event'].isin(shot_events)
        xga = round(opp_events.loc[shot_mask, 'xG'].sum(), 2)
        
    return {
        'Goals Conceded / Game': _safe_div(ga, played),
        'xGA / Game': _safe_div(xga, played),
        'PPDA': ppda,
        'Tackles / Game': _safe_div(tackles, match_count),
        'Interceptions / Game': _safe_div(interceptions, match_count),
        'Clearances / Game': _safe_div(clearances, match_count),
        'Defensive Line Height': line_height,
        'High Press %': f"{high_press_pct}%",
    }


# ============================================================
# DEFENSIVE TRANSITIONS METRICS
# ============================================================

def calc_defensive_transition_metrics(team_name):
    """
    Returns defensive transition metrics.
    """
    all_events, match_count = _load_all_team_events(team_name)
    
    if all_events.empty or match_count == 0:
        return {}
    
    team_events = all_events[all_events['team_name'] == team_name]
    opp_events = all_events[all_events['team_name'] != team_name]
    
    # Ball recoveries
    ball_recoveries = len(team_events[team_events['event'].isin(['Ball recovery', 'Ball Recovery'])])
    
    # Ball recoveries in opponent half (x > 50)
    recoveries_opp_half = len(team_events[
        (team_events['event'].isin(['Ball recovery', 'Ball Recovery'])) &
        (team_events['x'] > 50)
    ])
    
    # Counter-press success % (recoveries in opp half / total recoveries)
    counter_press_pct = round((recoveries_opp_half / ball_recoveries) * 100, 1) if ball_recoveries > 0 else 0.0
    
    # PPDA
    opp_passes_own_half = len(opp_events[(opp_events['event'] == 'Pass') & (opp_events['x'] < 60)])
    team_def_in_opp = len(team_events[
        (team_events['event'].isin(['Tackle', 'Interception', 'Challenge', 'Foul', 'Ball recovery', 'Ball Recovery'])) &
        (team_events['x'] > 40)
    ])
    ppda = round(opp_passes_own_half / team_def_in_opp, 1) if team_def_in_opp > 0 else 0
    
    # Build-up disruption %
    opp_passes_own = opp_events[(opp_events['event'] == 'Pass') & (opp_events['x'] < 50)]
    if not opp_passes_own.empty:
        opp_succ = opp_passes_own[opp_passes_own['outcome'] == 1]
        bdp = round(100 - (len(opp_succ) / len(opp_passes_own)) * 100, 1)
    else:
        bdp = 0.0
    
    return {
        'Ball Recoveries / Game': _safe_div(ball_recoveries, match_count),
        'Recoveries in Opp. Half / Game': _safe_div(recoveries_opp_half, match_count),
        'Counter-Press Success %': f"{counter_press_pct}%",
        'PPDA': ppda,
        'Build-up Disruption %': f"{bdp}%",
    }


# ============================================================
# OFFENSIVE PHASE METRICS
# ============================================================

def calc_offensive_metrics(team_name):
    """
    Returns offensive phase metrics.
    """
    matches = extract_fixture_data(lite=True)
    standings_df = calculate_standings(matches)
    
    if standings_df.empty:
        return {}
    
    team_row = standings_df[standings_df['Team'] == team_name]
    if team_row.empty:
        return {}
    
    team_row = team_row.iloc[0]
    played = team_row['Played']
    
    all_events, match_count = _load_all_team_events(team_name)
    
    if all_events.empty or match_count == 0:
        return {}
    
    team_events = all_events[all_events['team_name'] == team_name]
    
    # GF and xG
    gf = team_row['GF']
    xg_total = 0.0
    for m in matches:
        t1, t2 = m['team_names']
        if t1 == team_name:
            xg_total += m['stats']['team1'].get('xg', 0)
        elif t2 == team_name:
            xg_total += m['stats']['team2'].get('xg', 0)
    
    # Fallback: if xg_total is still 0, compute directly from loaded events
    # (extract_fixture_data cache may not have had xG applied)
    if xg_total == 0.0 and 'xG' in team_events.columns:
        shot_events = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
        shot_mask = team_events['event'].isin(shot_events)
        xg_total = round(team_events.loc[shot_mask, 'xG'].sum(), 2)
    
    # Shots (type_id 13=Miss, 14=Post, 15=Saved Shot, 16=Goal)
    shot_types = [13, 14, 15, 16]
    has_own_goal = 'own goal' in team_events.columns
    if has_own_goal:
        shots = len(team_events[(team_events['type_id'].isin(shot_types)) & (team_events['own goal'] != 'Si')])
        shots_on_target = len(team_events[(team_events['type_id'].isin([15, 16])) & (team_events['own goal'] != 'Si')])
    else:
        shots = len(team_events[team_events['type_id'].isin(shot_types)])
        shots_on_target = len(team_events[team_events['type_id'].isin([15, 16])])
    
    # Pass accuracy
    total_passes = len(team_events[team_events['type_id'] == 1])
    success_passes = len(team_events[(team_events['type_id'] == 1) & (team_events['outcome'] == 1)])
    pass_accuracy = round((success_passes / total_passes) * 100, 1) if total_passes > 0 else 0
    
    # Progressive passes
    passes = team_events[(team_events['event'] == 'Pass') & (team_events['outcome'] == 1)].copy()
    prog_passes = 0
    if not passes.empty and 'Pass End X' in passes.columns:
        passes['start_dist'] = np.sqrt((100 - passes['x'])**2 + (50 - passes['y'])**2)
        passes['end_dist'] = np.sqrt((100 - passes['Pass End X'])**2 + (50 - passes['Pass End Y'])**2)
        prog_passes = len(passes[passes['end_dist'] <= passes['start_dist'] * 0.75])
    
    # Field tilt
    all_f3_passes = all_events[(all_events['event'] == 'Pass') & (all_events['x'] >= 66.6)]
    if not all_f3_passes.empty:
        team_f3 = all_f3_passes[all_f3_passes['team_name'] == team_name]
        field_tilt = round((len(team_f3) / len(all_f3_passes)) * 100, 1)
    else:
        field_tilt = 50.0
    
    # xT
    xt = 0.0
    if not passes.empty and 'Pass End X' in passes.columns:
        if 'start_dist' not in passes.columns:
            passes['start_dist'] = np.sqrt((100 - passes['x'])**2 + (50 - passes['y'])**2)
            passes['end_dist'] = np.sqrt((100 - passes['Pass End X'])**2 + (50 - passes['Pass End Y'])**2)
        passes['xt_added'] = passes['start_dist'] - passes['end_dist']
        xt = round(passes[passes['xt_added'] > 0]['xt_added'].sum() / 100.0, 2)

    return {
        'Goals Scored / Game': _safe_div(gf, played),
        'xG / Game': _safe_div(xg_total, played),
        'Shots / Game': _safe_div(shots, match_count),
        'Shots on Target / Game': _safe_div(shots_on_target, match_count),
        'Pass Accuracy': f"{pass_accuracy}%",
        'Progressive Passes / Game': _safe_div(prog_passes, match_count),
        'Field Tilt %': f"{field_tilt}%",
        'xT / Game': _safe_div(xt, match_count),
    }


# ============================================================
# OFFENSIVE TRANSITIONS METRICS
# ============================================================

def calc_offensive_transition_metrics(team_name):
    """
    Returns offensive transition metrics.
    """
    all_events, match_count = _load_all_team_events(team_name)
    
    if all_events.empty or match_count == 0:
        return {}
    
    team_events = all_events[all_events['team_name'] == team_name]
    
    # Fast break goals
    has_fast_break = 'Fast break' in team_events.columns
    if has_fast_break:
        fb_goals = len(team_events[(team_events['type_id'] == 16) & (team_events['Fast break'] == 'Si')])
        fb_shots_types = [13, 14, 15, 16]
        fb_shots = len(team_events[(team_events['type_id'].isin(fb_shots_types)) & (team_events['Fast break'] == 'Si')])
    else:
        fb_goals = 0
        fb_shots = 0
    
    # Directness (forward distance / total distance of passes)
    passes = team_events[team_events['event'] == 'Pass'].copy()
    directness = 0.0
    if not passes.empty and 'Pass End X' in passes.columns:
        dx = passes['Pass End X'] - passes['x']
        dy = passes['Pass End Y'] - passes['y']
        fwd = np.maximum(dx, 0).sum()
        total = np.sqrt(dx**2 + dy**2).sum()
        directness = round(fwd / total, 2) if total > 0 else 0.0
    
    # Progressive passes
    succ_passes = team_events[(team_events['event'] == 'Pass') & (team_events['outcome'] == 1)].copy()
    prog_passes = 0
    if not succ_passes.empty and 'Pass End X' in succ_passes.columns:
        succ_passes['start_dist'] = np.sqrt((100 - succ_passes['x'])**2 + (50 - succ_passes['y'])**2)
        succ_passes['end_dist'] = np.sqrt((100 - succ_passes['Pass End X'])**2 + (50 - succ_passes['Pass End Y'])**2)
        prog_passes = len(succ_passes[succ_passes['end_dist'] <= succ_passes['start_dist'] * 0.75])
    
    # Carries into final third (Take On events ending in final third is a proxy)
    take_ons_f3 = len(team_events[
        (team_events['event'] == 'Take On') &
        (team_events['x'] > 66) &
        (team_events['outcome'] == 1)
    ])
    
    return {
        'Fast Break Goals': fb_goals,
        'Counter-Attack Shots': fb_shots,
        'Directness': directness,
        'Progressive Passes / Game': _safe_div(prog_passes, match_count),
        'Carries into Final 3rd / Game': _safe_div(take_ons_f3, match_count),
    }


# ============================================================
# SET PIECES METRICS
# ============================================================

def calc_set_piece_metrics(team_name):
    all_events, match_count = _load_all_team_events(team_name)
    if all_events.empty or match_count == 0:
        return {}
    
    team_events = all_events[all_events['team_name'] == team_name]
    opp_events = all_events[all_events['team_name'] != team_name]
    
    # 1. Corners (type_id 6)
    corners = len(team_events[team_events['type_id'] == 6])
    opp_corners = len(opp_events[opp_events['type_id'] == 6])
    
    # 2. Set piece shots
    has_sp_col = 'Set piece' in team_events.columns
    if has_sp_col:
        sp_shots = len(team_events[(team_events['type_id'].isin([13,14,15,16])) & (team_events['Set piece'] == 'Si')])
        opp_sp_shots = len(opp_events[(opp_events['type_id'].isin([13,14,15,16])) & (opp_events['Set piece'] == 'Si')])
        sp_goals = len(team_events[(team_events['type_id'] == 16) & (team_events['Set piece'] == 'Si')])
        opp_sp_goals = len(opp_events[(opp_events['type_id'] == 16) & (opp_events['Set piece'] == 'Si')])
    else:
        sp_shots = 0
        opp_sp_shots = 0
        sp_goals = 0
        opp_sp_goals = 0

    # 3. Penalties
    has_penalty_col = 'Penalty' in team_events.columns
    if has_penalty_col:
        pen_goals = len(team_events[(team_events['type_id'] == 16) & (team_events['Penalty'] == 'Si')])
    else:
        pen_goals = 0
        
    return {
        'Corners Won / Game': _safe_div(corners, match_count),
        'Set Piece Shots / Game': _safe_div(sp_shots, match_count),
        'Set Piece Goals': sp_goals,
        'Penalty Goals': pen_goals,
        'Corners Conceded / Game': _safe_div(opp_corners, match_count),
        'Set Piece Shots Conceded / Game': _safe_div(opp_sp_shots, match_count),
        'Set Piece Goals Conceded': opp_sp_goals,
    }


# ============================================================
# MASTER FUNCTION
# ============================================================

PHASE_CALCULATORS = {
    'Defensive': calc_defensive_metrics,
    'Def. Transitions': calc_defensive_transition_metrics,
    'Offensive': calc_offensive_metrics,
    'Off. Transitions': calc_offensive_transition_metrics,
    'Set Pieces': calc_set_piece_metrics,
}

# Metrics where LOWER is better for that team
LOWER_IS_BETTER = {
    'Goals Conceded / Game', 'xGA / Game', 'PPDA',
    'Corners Conceded / Game', 'Set Piece Shots Conceded / Game', 'Set Piece Goals Conceded',
}

def get_phase_metrics(phase_name, team_name):
    """
    Returns a dict of {metric_label: value} for the given phase and team.
    """
    calc_fn = PHASE_CALCULATORS.get(phase_name)
    if calc_fn is None:
        return {}
    return calc_fn(team_name)
