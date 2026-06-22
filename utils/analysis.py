
import pandas as pd
import numpy as np

# ============================================================
# SWOT ANALYSIS
# ============================================================


def get_turnovers_and_outcomes(df, team_name, window_seconds=15):
    """
    Identifies turnovers for a team and checks for subsequent opponent shots/goals.
    
    Args:
        df (pd.DataFrame): Match dataframe.
        team_name (str): The team to analyze for turnovers.
        window_seconds (int): Time window to look for outcomes after turnover.
        
    Returns:
        pd.DataFrame: DataFrame containing turnover events with outcome details.
    """
    df = df.copy()
    
    # Ensure time sorted
    if 'period_id' in df.columns and 'time_min' in df.columns:
        df = df.sort_values(by=['period_id', 'time_min', 'time_sec', 'event_id'])
    
    # Identify Turnovers
    # 1. Failed Passes by Team
    failed_passes = (df['team_name'] == team_name) & (df['event'] == 'Pass') & (df['outcome'] == 0)
    
    # 2. Failed Take Ons
    failed_takeons = (df['team_name'] == team_name) & (df['event'] == 'Take On') & (df['outcome'] == 0)
    
    # 3. Dispossessed
    dispossessed = (df['team_name'] == team_name) & (df['event'] == 'Dispossessed')
    
    # 4. Errors
    errors = (df['team_name'] == team_name) & (df['event'] == 'Error')
    
    turnover_mask = failed_passes | failed_takeons | dispossessed | errors
    turnovers = df[turnover_mask].copy()
    
    if turnovers.empty:
        return pd.DataFrame()
    
    # Initialize outcome columns
    turnovers['consequence'] = 'None'
    turnovers['consequence_x'] = np.nan
    turnovers['consequence_y'] = np.nan
    turnovers['consequence_time'] = np.nan
    
    # Filter opponent events for checking outcomes
    # Opponent is NOT team_name
    opponent_events = df[df['team_name'] != team_name]
    
    # Shot types
    shot_types = ['Goal', 'Miss', 'Post', 'Attempt Saved']
    
    for idx, turnover in turnovers.iterrows():
        # Define window
        t_period = turnover['period_id']
        t_min = turnover['time_min']
        t_sec = turnover['time_sec']
        t_time_total = t_min * 60 + t_sec
        
        # Determine opponent
        # Logic: Events in same period, within window, by opponent
        window_events = opponent_events[
            (opponent_events['period_id'] == t_period) & 
            ((opponent_events['time_min'] * 60 + opponent_events['time_sec']) >= t_time_total) &
            ((opponent_events['time_min'] * 60 + opponent_events['time_sec']) <= t_time_total + window_seconds)
        ]
        
        if window_events.empty:
            continue
            
        # Check for Goals first (highest priority)
        goals = window_events[window_events['event'] == 'Goal']
        if not goals.empty:
            goal = goals.iloc[0]
            turnovers.at[idx, 'consequence'] = 'Goal'
            turnovers.at[idx, 'consequence_x'] = goal['x']
            turnovers.at[idx, 'consequence_y'] = goal['y']
            continue
            
        # Check for Shots
        shots = window_events[window_events['event'].isin(shot_types)]
        if not shots.empty:
            shot = shots.iloc[0]
            turnovers.at[idx, 'consequence'] = 'Shot'
            turnovers.at[idx, 'consequence_x'] = shot['x']
            turnovers.at[idx, 'consequence_y'] = shot['y']
            
    return turnovers

# ============================================================
# BUILDING UP ANALYSIS (SALIDAS DE BALÓN)
# ============================================================

def _ev_lower(ev):
    if not isinstance(ev, str): return ""
    return ev.lower()

def is_field_event_row(row):
    """
    Checks if event is a valid field event (not deleted, not admin).
    """
    ev = str(row.get("event", ""))
    
    # Check deleted
    if "deleted event" in ev.lower(): return False
    
    # Check admin
    if any(k in ev.lower() for k in ["start delay", "end delay", "contentious referee decision"]):
        return False
        
    return True


def is_salida_start_row(row):
    """
    Identify triggers for a "Building Up" sequence:
    - Event in defensive third (Third 0).
    - Recovery, Interception, Tackle, Keeper action, or Set Piece pass.
    """
    if not is_field_event_row(row):
        return False
        
    # Assuming x is already normalized to plot coordinates (or we check both if unsure)
    # But usually this is called on data that might be raw Opta (0-100)
    # Let's assume input df has 'x_plot_m' or we calculate third based on 'x' (0-100)
    
    # Use 'x' (0-100) for safety if available, else x_plot_m
    x_val = row.get('x', row.get('x_plot_m', 50))
    limit = 100 / 3.0 # Opta coordinates
    
    if x_val > limit: # Not in defensive third
        return False

    ev_low = _ev_lower(row.get("event", ""))

    # Start keywords
    start_keywords = [
        "ball recovery", "interception", "tackle",
        "keeper pick-up", "keeper pickup", "smother",
        "keeper sweeper", "save", "claim"
    ]
    if any(k in ev_low for k in start_keywords):
        return True

    # Set pieces (Pass with qualifier)
    if row.get("event", "") == "Pass":
        # Check standard qualifier columns if they exist
        # Robust check against columns in dataframe
        # We look for columns that might indicate a restart
        for col in row.index:
            if not isinstance(col, str): continue
            col_lower = col.lower()
            if any(k in col_lower for k in ["goal kick", "free kick", "throw in", "corner"]):
                # Check if value is truthy
                val = str(row.get(col, "")).strip().lower()
                if val in ("si", "yes", "y", "1", "true"):
                    return True
    return False


def detect_counter_attacks(df):
    """
    Detects counter-attack sequences.
    Definition:
    - Starts in own half (x < 50) with a recovery/interception.
    - Reaches opponent box (x > 83, 21.1 < y < 78.9) or results in a shot within 20 seconds.
    - High speed or directness metric could be added later.
    """
    counters = []
    
    if df.empty: return []

    # Ensure indices
    if not isinstance(df.index, pd.RangeIndex):
         df = df.reset_index(drop=True)

    n = len(df)
    i = 0
    
    # Pre-calculate shot indices for speed? Or just scan. N is small per match usually.
    
    while i < n:
        row = df.iloc[i]
        
        # 1. Start Condition: Recovery in Own Half
        # Check standard recovery events
        ev_low = _ev_lower(row.get("event", ""))
        is_recovery = any(k in ev_low for k in ["ball recovery", "interception", "tackle", "keeper pick-up", "save"])
        
        # x check: Own Half (< 50) (Assuming 0-100 coord system which seems standard for this app)
        x_val = row.get('x', row.get('x_plot_m', 100)) # Default high to fail check if missing
        
        if is_recovery and x_val < 50:
            team = row['team_name']
            start_time = row['time_min'] * 60 + row['time_sec']
            
            # Scan forward for outcome
            j = i + 1
            has_outcome = False
            seq_indices = [i]
            
            while j < n:
                curr = df.iloc[j]
                
                # Time limit (e.g. 20s for a fast transition)
                curr_time = curr['time_min'] * 60 + curr['time_sec']
                if (curr_time - start_time) > 20: 
                    break
                
                # Possession change
                if curr.get('team_name') != team:
                    break
                
                seq_indices.append(j)
                
                # Check End Condition: Shot or Box Entry
                curr_ev = _ev_lower(curr.get("event", ""))
                curr_x = curr.get('x', 0)
                curr_y = curr.get('y', 0)
                
                # Shot
                if any(s in curr_ev for s in ['goal', 'miss', 'post', 'attempt saved', 'shot']):
                    has_outcome = True
                    break
                    
                # Box Entry (x > 83, y in 21.1-78.9)
                if curr_x > 83 and 21.1 <= curr_y <= 78.9:
                    has_outcome = True
                    # Don't break immediately, might lead to shot in next few seconds?
                    # But for "Counter detection", reaching box is usually enough success.
                    # Let's see if next event is shot.
                    if j+1 < n:
                        next_ev = df.iloc[j+1]
                        if next_ev['team_name'] == team:
                            next_ev_name = _ev_lower(next_ev.get("event", ""))
                            if any(s in next_ev_name for s in ['goal', 'miss', 'post', 'attempt saved', 'shot']):
                                seq_indices.append(j+1)
                    break
                
                j += 1
            
            if has_outcome and len(seq_indices) > 1:
                counters.append(seq_indices)
                i = j # Skip
            else:
                i += 1
        else:
            i += 1
            
    return counters

# ============================================================
# PHASE 5: SET PIECES
# ============================================================

def get_set_pieces(df, team_name):
    """
    Extracts set piece events for a team.
    Returns dictionary with 'corners', 'free_kicks', and 'penalties'.
    """
    df_team = df[df['team_name'] == team_name].copy()
    
    # 1. Corners
    # Opta: qualifier 'CornerTaken'? Or Event 'Corner'? -> Usually 'Pass' with qualifier per earlier check
    # But usually there is a 'Corner Awarded' event, followed by execution.
    # Let's look for "Pass" with "Corner" context or qualifiers.
    # Often cleaner to look for specific event names if standardized.
    # Let's assume standard names or fallback to qualifiers.
    
    corners = pd.DataFrame()
    free_kicks = pd.DataFrame()
    penalties = pd.DataFrame()
    

    if not df_team.empty:
        # Check qualifiers based method (more robust for granular data)
        # 1. Corners
        # Find column that likely represents 'Corner Taken'
        corner_col = None
        for col in df_team.columns:
            if not isinstance(col, str): continue
            if 'corner' in col.lower() and 'taken' in col.lower():
                corner_col = col
                break
                
        if corner_col:
             # Robust check: lowercase and strip
             corners = df_team[df_team[corner_col].astype(str).str.lower().str.strip().isin(['1', 'true', 'yes', 'si', 'y'])].copy()
        
        # 2. Free Kicks (Dangerous)
        fk_col = None
        for col in df_team.columns:
            if not isinstance(col, str): continue
            if 'free' in col.lower() and 'kick' in col.lower() and 'taken' in col.lower():
                fk_col = col
                break
                
        if fk_col:
             free_kicks = df_team[df_team[fk_col].astype(str).str.lower().str.strip().isin(['1', 'true', 'yes', 'si', 'y'])].copy()
             # Filter dangerous? (Attacking Third)
             free_kicks = free_kicks[free_kicks['x'] > 60]

        penalty_col = None
        for col in df_team.columns:
            if not isinstance(col, str): continue
            if col.lower().strip() == 'penalty':
                penalty_col = col
                break

        if penalty_col:
            penalty_events = df_team[df_team[penalty_col].astype(str).str.lower().str.strip().isin(['1', 'true', 'yes', 'si', 'y'])].copy()
            shot_events = ['Goal', 'Miss', 'Saved Shot', 'Post']
            penalties = penalty_events[penalty_events['event'].isin(shot_events)].copy()
        
    return {
        "corners": corners,
        "free_kicks": free_kicks,
        "penalties": penalties,
    }
