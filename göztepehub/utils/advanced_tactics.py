import pandas as pd
import numpy as np
from utils.data import extract_fixture_data, calculate_standings

def identify_playmaker(df, team_name):
    """
    Identifies the playmaker based on pass volume, progressive passes, and xT (if available).
    Returns dict of stats for the top player.
    """
    team_df = df[df['team_name'] == team_name]
    passes = team_df[(team_df['event'] == 'Pass') & (team_df['outcome'].astype(str).isin(['1', 'True', 'Success']))]
    
    if passes.empty:
        return {"name": "Bilinmiyor", "passes": 0, "prog_passes": 0}
        
    pass_counts = passes['player_name'].value_counts()
    
    # Try to find progressive passes manually if not already tagged
    if 'Pass End X' in passes.columns and 'x' in passes.columns:
        prog_passes = passes[passes['Pass End X'] > passes['x'] + 10] # simple proxy 10 units forward
    else:
        prog_passes = pd.DataFrame()
        
    prog_counts = prog_passes['player_name'].value_counts() if not prog_passes.empty else pd.Series(dtype=int)
    
    # Combine scores
    scores = {}
    for p in pass_counts.index:
        scores[p] = pass_counts.get(p, 0) * 0.5 + prog_counts.get(p, 0) * 1.5
        
    if not scores:
        return {"name": "Bilinmiyor", "passes": 0, "prog_passes": 0}
        
    best_player = max(scores, key=scores.get)
    return {
        "name": best_player,
        "passes": pass_counts.get(best_player, 0),
        "prog_passes": prog_counts.get(best_player, 0)
    }

def analyze_15s_rule(df, team_name):
    """
    Analyzes what happens 15 seconds after entering the final third.
    Categorizes outcomes into Shot, Loss, Cross.
    """
    if 'time_min' not in df.columns or 'time_sec' not in df.columns:
         return {"Shot": 0, "Loss": 0, "Cross": 0, "Other": 0, "Total": 0}
         
    df_sorted = df.sort_values(['period_id', 'time_min', 'time_sec']) if 'period_id' in df.columns else df.sort_values(['time_min', 'time_sec'])
    
    entries = []
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i-1]
        curr = df_sorted.iloc[i]
        if curr['team_name'] == team_name and curr.get('x', 0) >= 66.6 and prev.get('x', 0) < 66.6:
            entries.append(i)
            
    outcomes = {"Shot": 0, "Loss": 0, "Cross": 0, "Other": 0}
    
    for idx in entries:
        start_row = df_sorted.iloc[idx]
        start_time = start_row['time_min'] * 60 + start_row['time_sec']
        
        outcome_found = False
        for j in range(idx+1, len(df_sorted)):
            curr_row = df_sorted.iloc[j]
            curr_time = curr_row['time_min'] * 60 + curr_row['time_sec']
            
            if curr_time - start_time > 15:
                break
                
            ev_str = str(curr_row.get('event', '')).lower()
            
            if curr_row['team_name'] != team_name:
                outcomes["Loss"] += 1
                outcome_found = True
                break
                
            if any(s in ev_str for s in ['shot', 'goal', 'miss', 'post']):
                outcomes["Shot"] += 1
                outcome_found = True
                break
                
            if 'cross' in ev_str or str(curr_row.get('cross', '')).lower() == 'true':
                outcomes["Cross"] += 1
                outcome_found = True
                break
                
        if not outcome_found:
            outcomes["Other"] += 1
            
    outcomes["Total"] = len(entries)
    return outcomes

def analyze_xg_chain_origins(df, team_name):
    """Finds origin of shots (Cross, Set Piece, Frontal)."""
    team_df = df[df['team_name'] == team_name]
    shots = team_df[team_df['event'].astype(str).str.lower().str.contains('shot|goal|miss|post')]
    
    origins = {"Cross": 0, "Set Piece": 0, "Frontal/Open": 0}
    
    for idx, shot in shots.iterrows():
        try:
             full_idx = df.index.get_loc(idx)
             prev_evs = df.iloc[max(0, full_idx-3):full_idx]
             
             is_sp = any(True for _, r in prev_evs.iterrows() if r.get('team_name') == team_name and any(sp in str(r.get('event', '')).lower() for sp in ['corner', 'free kick']))
             is_cross = any(True for _, r in prev_evs.iterrows() if r.get('team_name') == team_name and ('cross' in str(r.get('event', '')).lower() or str(r.get('cross', '')).lower() == 'true'))
             
             if is_sp:
                 origins["Set Piece"] += 1
             elif is_cross:
                 origins["Cross"] += 1
             else:
                 origins["Frontal/Open"] += 1
        except:
             origins["Frontal/Open"] += 1
             
    return origins

def get_defensive_profile_match(df, team_name):
    """Returns penalty box aerials, High press ratio, vulnerable flanks."""
    team_df = df[df['team_name'] == team_name]
    
    aerials = team_df[(team_df['event'].astype(str).str.contains('Aerial', case=False))]
    box_aerials = aerials[(aerials['x'] < 17) & (aerials['y'] > 21.1) & (aerials['y'] < 78.9)]
    box_aerial_wins = box_aerials[box_aerials['outcome'].astype(str).isin(['1', 'True', 'Success'])]
    
    opp_crosses = df[(df['team_name'] != team_name) & (df['event'].astype(str).str.contains('Cross', case=False) | (df.get('cross', '').astype(str).str.lower() == 'true'))]
    left_crosses = len(opp_crosses[opp_crosses['y'] > 50])
    right_crosses = len(opp_crosses[opp_crosses['y'] <= 50])
    
    return {
        "box_aerials_total": len(box_aerials),
        "box_aerials_won": len(box_aerial_wins),
        "vulnerable_flank": "Sol (Left)" if left_crosses > right_crosses else "Sağ (Right)" if right_crosses > left_crosses else "Dengeli",
        "opp_crosses": left_crosses + right_crosses
    }

def get_recent_form(team_name):
    """Returns W/D/L form for the team."""
    matches = extract_fixture_data(lite=True)
    df = calculate_standings(matches)
    
    if df is None or df.empty or team_name not in df['Team'].values:
        return {"form": "N/A", "avg_points": 0}
        
    team_stats = df[df['Team'] == team_name].iloc[0]
    points = team_stats['Points']
    played = team_stats['Played']
    avg = round(points / played, 2) if played > 0 else 0
    form_str = f"{team_stats['Won']}W {team_stats['Drawn']}D {team_stats['Lost']}L"
    return {"form": form_str, "avg_points": avg}

def get_goal_typologies(df, team_name):
    """Analyzes goals scored by the team."""
    team_df = df[(df['team_name'] == team_name) & (df['event'] == 'Goal')]
    if team_df.empty:
        return {"Header": 0, "Inside Box": 0, "Outside Box": 0}
        
    typology = {"Header": 0, "Inside Box": 0, "Outside Box": 0}
    for _, row in team_df.iterrows():
        ev_str = str(row.get('description', '')).lower() + " " + str(row.get('event', '')).lower()
        if 'head' in ev_str or 'kafa' in ev_str:
            typology["Header"] += 1
            
        x, y = row.get('x', 0), row.get('y', 0)
        if x > 83 and 21.1 < y < 78.9:
            typology["Inside Box"] += 1
        else:
            typology["Outside Box"] += 1
            
    return typology

def get_15min_intervals(df, team_name):
    """Groups goals into 15-min intervals."""
    team_df = df[(df['team_name'] == team_name) & (df['event'] == 'Goal')]
    intervals = {"0-15": 0, "16-30": 0, "31-45": 0, "46-60": 0, "61-75": 0, "76-90": 0}
    
    if 'time_min' in team_df.columns:
        for _, row in team_df.iterrows():
            m = row['time_min']
            if m <= 15: intervals["0-15"] += 1
            elif m <= 30: intervals["16-30"] += 1
            elif m <= 45: intervals["31-45"] += 1
            elif m <= 60: intervals["46-60"] += 1
            elif m <= 75: intervals["61-75"] += 1
            else: intervals["76-90"] += 1
            
    return intervals

def get_recovery_types(df, team_name):
    """Differentiates counter-pressing vs interceptions."""
    team_df = df[df['team_name'] == team_name]
    recoveries = team_df[team_df['event'].astype(str).str.contains('Recovery|Interception|Challenge|Tackle', case=False)]
    
    types = {"Counter-Press": 0, "Interception": 0, "Defensive 1v1": 0, "Other": 0}
    
    for _, row in recoveries.iterrows():
        ev = str(row.get('event', '')).lower()
        if 'interception' in ev:
            types["Interception"] += 1
        elif 'tackle' in ev or 'challenge' in ev:
            types["Defensive 1v1"] += 1
        else:
            if row.get('x', 0) > 50:
                 types["Counter-Press"] += 1
            else:
                 types["Other"] += 1
                 
    return types

def get_squad_minutes(df, team_name):
    """Approximates most played players by position."""
    team_df = df[df['team_name'] == team_name]
    if 'position' not in team_df.columns:
        top = team_df['player_name'].value_counts().head(3).index.tolist()
        return {"GK": "Unknown", "CB": "Unknown", "Attacker": top[0] if len(top) > 0 else "Unknown"}
        
    gk = team_df[team_df['position'] == 'GK']['player_name'].value_counts()
    cb = team_df[team_df['position'].astype(str).str.contains('CB|Center Back')]['player_name'].value_counts()
    atk = team_df[team_df['position'].astype(str).str.contains('FW|Striker|Wing')]['player_name'].value_counts()
    
    return {
        "GK": gk.index[0] if not gk.empty else "Bilinmiyor",
        "CB": cb.index[0] if not cb.empty else "Bilinmiyor",
        "Attacker": atk.index[0] if not atk.empty else "Bilinmiyor"
    }

def get_radar_data(df, team_name):
    """Aggregates data for Offensive and Defensive Radars."""
    team_df = df[df['team_name'] == team_name]
    
    try:
        npxg = round(team_df[(team_df['event'] == 'Shot') & (team_df['type_id'] != 9)]['xG'].sum(), 2)
    except:
        npxg = 0
        
    if 'key_pass' in team_df.columns:
        key_passes = len(team_df[(team_df['event'] == 'Pass') & (team_df['key_pass'].astype(str).str.lower() == 'true')])
    else:
        key_passes = 0
        
    if key_passes == 0 and 'outcome' in team_df.columns: 
        pass_end_x = team_df.get('Pass End X', pd.Series([0]*len(team_df), index=team_df.index))
        key_passes = len(team_df[(team_df['event'] == 'Pass') & (pass_end_x > 83) & (team_df['outcome'] == 1)])
        
    touches_in_box = len(team_df[(team_df['x'] > 83) & (team_df['y'] > 21.1) & (team_df['y'] < 78.9)])
    dribbles = len(team_df[(team_df['event'] == 'Take On') & (team_df['outcome'] == 1)])
    
    rec_data = get_recovery_types(df, team_name)
    recoveries = rec_data["Other"] + rec_data["Counter-Press"]
    interceptions = rec_data["Interception"]
    def_1v1 = rec_data["Defensive 1v1"]
    aerials_won = len(team_df[(team_df['event'] == 'Aerial') & (team_df['outcome'] == 1)])
    
    return {
        "Offense": {
            "categories": ['NPxG (x10)', 'Kilit Pas', 'C.Sahası Aksiyonu', 'Başarılı Dripling'],
            "values": [npxg * 10, key_passes, touches_in_box, dribbles]
        },
        "Defense": {
            "categories": ['Top Kazanım', 'Araya Girme', 'Defansif 1v1', 'Hava Topu (Kaz.)'],
            "values": [recoveries, interceptions, def_1v1, aerials_won]
        }
    }

def get_set_piece_stats(df, team_name):
    """Calculates corners, free kicks, penalties and associated xG."""
    team_df = df[df['team_name'] == team_name]
    
    corners = len(team_df[team_df['event'].astype(str).str.lower() == 'corner'])
    fks = len(team_df[team_df['event'].astype(str).str.lower().str.contains('free kick', case=False)])
    pens = len(team_df[(team_df['event'] == 'Shot') & (team_df.get('type_id', 0) == 9)])
    
    # Try to find xG from these
    sp_shots = team_df[(team_df['event'].astype(str).str.lower().str.contains('shot|goal|miss|post'))]
    sp_xg = 0
    # Simplified approach: If shot is penalty, it's set piece. 
    # For others, we rely on the origins function or if previous event was a set piece.
    # Since we have xG chain origins, we can just return that.
    origins = analyze_xg_chain_origins(df, team_name)
    
    return {
        "Corners": corners,
        "Free Kicks": fks,
        "Penalties": pens,
        "Set Piece Shots Ratio": f"{origins['Set Piece']}/{origins['Set Piece'] + origins['Cross'] + origins['Frontal/Open']}"
    }
