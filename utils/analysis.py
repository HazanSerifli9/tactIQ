
import pandas as pd
import numpy as np

# ============================================================
# SWOT ANALYSIS
# ============================================================

def generate_swot(df_concat, team_name, target_teams_list=None):
    """
    Generates SWOT analysis based on team's statistical percentile rank vs rivals.
    df_concat: DataFrame containing ALL teams' data (for league comparison).
    team_name: Team to analyze.
    """
    if df_concat.empty:
        return {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
        
    # 1. Calc Per-Match Stats for ALL teams
    # Group by Team, MatchID (source_file)
    # We need a way to group matches. Assuming 'source_file' or 'match_id' column exists? 
    # Or just aggregate totals and divide by match count if match id missing.
    # `goztepe.py` loads multiple files. `df_concat` has them all.
    
    # Metric Dictionary
    # We define what constitutes a strength/weakness
    
    # Aggregation
    # Just sum stat per team and normalize by games played?
    # Simple approach:
    
    stats = {}
    teams = df_concat['team_name'].unique()
    
    for t in teams:
        t_df = df_concat[df_concat['team_name'] == t]
        if t_df.empty: continue
        
        # Estimate matches by unique time blocks or files? 
        # Using 'period_id' count isn't enough.
        # Use simple heuristic: count of 'Kick Off'?
        # Or just use raw totals if matches count is similar? 
        # Risky.
        # Better: Average per 90 (Total events / 60 / 90?)
        # Let's rely on event counts relative to others.
        
        goals = len(t_df[t_df['event'] == 'Goal'])
        shots = len(t_df[t_df['event'].isin(['Shot', 'Goal', 'Miss', 'Post', 'Attempt Saved'])])
        passes = len(t_df[(t_df['event'] == 'Pass') & (t_df['outcome'].isin([1, 'Successful']))])
        interceptions = len(t_df[t_df['event'] == 'Interception'])
        tackles_won = len(t_df[(t_df['event'] == 'Tackle') & (t_df['outcome'] == 1)])
        
        # Derived
        conversion = (goals / shots * 100) if shots > 0 else 0
        
        stats[t] = {
            "Goals": goals,
            "Shots": shots,
            "Passes": passes,
            "Interceptions": interceptions,
            "Tackles Won": tackles_won,
            "Conversion %": conversion
        }
        
    if team_name not in stats:
        return {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []} # Should not happen
        
    # 2. Ranking
    # Compare team_name vs others
    
    my_stats = stats[team_name]
    ranks = {}
    
    for metric in my_stats:
        # Get list of values for this metric
        all_vals = [stats[t][metric] for t in teams]
        # Rank: Higher is better?
        # Percentile
        val = my_stats[metric]
        pct = (sum(v <= val for v in all_vals) / len(all_vals)) * 100
        ranks[metric] = pct
        
    # 3. Classify SWOT
    swot = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    
    # Internal (Strengths/Weaknesses)
    for m, p in ranks.items():
        if p >= 75:
            swot["Strengths"].append(f"High {m} (Top {int(100-p)}%)")
        elif p <= 25:
            swot["Weaknesses"].append(f"Low {m} (Bottom {int(p)}%)")
            
    # External (Opportunities/Threats) - Derived from potential matchups?
    # Or generically: "Weakness in Possession implies Threat from Pressing teams"?
    # Simple logic for MVP:
    
    if "Passes" in str(swot["Weaknesses"]):
        swot["Threats"].append("Vulnerable to High Press due to low passing retention")
    if "Interceptions" in str(swot["Strengths"]):
        swot["Opportunities"].append("Counter-attack situations from high turnovers won")
        
    # Fill defaults if empty
    if not swot["Strengths"]: swot["Strengths"].append("Balanced Profile")
    if not swot["Weaknesses"]: swot["Weaknesses"].append("No specific deficiencies")
    
    return swot

def generate_tactical_profile(df_concat, team_name):
    """
    Generates data for the Tactical Profile (Radar + Insights).
    """
    if df_concat.empty:
        return {}
        
    stats = {}
    teams = df_concat['team_name'].unique()
    
    # helper for metrics
    def get_team_metrics(t, df):
        t_df = df[df['team_name'] == t]
        if t_df.empty: return None
        
        matches = t_df['source_file'].nunique()
        if matches == 0: matches = 1 # avoid div by zero
        
        # 1. Attack: xG per 90
        xg = 0
        if 'expectedGoals' in t_df.columns:
            xg = t_df['expectedGoals'].fillna(0).sum()
        elif 'xG' in t_df.columns:
            xg = t_df['xG'].fillna(0).sum()
        xg_p90 = xg / matches
        
        # 2. Control: Possession (Passes)
        passes = len(t_df[t_df['type_id'] == 1])
        passes_p90 = passes / matches
        
        # 3. Intensity: PPDA (Lower is better, but for radar we might invert or normalize)
        # PPDA = Opponent Passes / Defensive Actions
        # We need opponent data for this match. df_concat has it. 
        # But for simpler league comparison, let's use "Defensive Actions P90" as proxy for intensity?
        # Or just "Ball Recoveries P90"
        def_actions = len(t_df[t_df['type_id'].isin([4, 49, 12, 74, 44])]) # foul, recovery, interception, blocked pass, clearance roughly
        # Let's try to be more precise if possible, but basic count is okay for MVP profile.
        def_p90 = def_actions / matches
        
        # 4. Defense: Goals Conceded per 90 (Reverse metric)
        # We need to find goals where this team is opponent.
        # This is hard with just t_df.
        # Alternative: "Shots Conceded P90"
        # We need opponent shots.
        # Let's use "Interceptions P90" as a positive layout metric for now to be safe.
        interceptions = len(t_df[t_df['type_id'] == 12])
        int_p90 = interceptions / matches
        
        # 5. Efficiency: Goals / xG
        goals = len(t_df[t_df['event'] == 'Goal'])
        efficiency = (goals / xg) if xg > 0 else 0
        
        return {
            "xG p90": xg_p90,
            "Passes p90": passes_p90,
            "Intensity": def_p90,
            "Interceptions": int_p90,
            "Efficiency": efficiency
        }

    # Calculate for all teams
    league_metrics = {'xG p90': [], 'Passes p90': [], 'Intensity': [], 'Interceptions': [], 'Efficiency': []}
    team_metrics = None
    
    for t in teams:
        m = get_team_metrics(t, df_concat)
        if m:
            for k, v in m.items():
                league_metrics[k].append(v)
            if t == team_name:
                team_metrics = m
                
    if not team_metrics: return {}
    
    # Calculate Percentiles (0-100)
    profile = {}
    for k, v in team_metrics.items():
        all_vals = league_metrics[k]
        pct = (sum(x <= v for x in all_vals) / len(all_vals)) * 100
        profile[k] = pct
        
    # Generate Insights
    insights = []
    
    # 1. Threat
    if profile["xG p90"] > 75:
        insights.append({"type": "threat", "title": "High Attack Threat", "desc": f"Top {int(100-profile['xG p90'])}% in xG creation."})
    elif profile["Efficiency"] > 80:
         insights.append({"type": "threat", "title": "Clinical Finishers", "desc": "Overperforms xG significantly."})
         
    # 2. Weakness
    if profile["Interceptions"] < 25:
        insights.append({"type": "weakness", "title": "Passive Defense", "desc": "Low interception rate implies space between lines."})
    if profile["Intensity"] < 25:
        insights.append({"type": "weakness", "title": "Low Intensity", "desc": "Allows opponents time on the ball."})
        
    # 3. Style
    if profile["Passes p90"] > 70:
        insights.append({"type": "info", "title": "Possession Based", "desc": "Prefers to control the game with passing."})
    else:
        insights.append({"type": "info", "title": "Direct Play", "desc": "Likely to play long or counter."})

    return {"radar_data": profile, "insights": insights}


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

def compute_third(x, pitch_length=100):
    if x < pitch_length / 3.0:
        return 0
    elif x < 2 * pitch_length / 3.0:
        return 1
    return 2

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

def detect_building_up_sequences(df):
    """
    Detects sequences starting in defensive third and progressing to middle third.
    Returns a list of lists (indices).
    """
    sequences = []
    
    # Ensure indices are accessible
    if not isinstance(df.index, pd.RangeIndex):
         df = df.reset_index(drop=True)
         
    n = len(df)
    i = 0
    
    while i < n:
        row = df.iloc[i]
        
        # Check start condition
        if is_salida_start_row(row):
            team = row['team_name']
            seq_indices = [i]
            reached_middle = False
            
            j = i + 1
            while j < n:
                next_row = df.iloc[j]
                
                # Filter non-field events but keep scanning
                if not is_field_event_row(next_row):
                    j += 1
                    continue
                    
                # Break if possession changes
                if next_row.get('team_name') != team:
                    break
                    
                seq_indices.append(j)
                
                # Check progression
                x_val = next_row.get('x', next_row.get('x_plot_m', 0))
                # Third 1 start (33.3) to Third 2 start (66.6)
                if x_val >= (100/3.0):
                    reached_middle = True
                    
                # Break if reached attacking third
                if x_val >= (200/3.0):
                    break
                    
                # Limit length
                if len(seq_indices) > 20: 
                    break
                    
                j += 1
            
            if reached_middle and len(seq_indices) > 1:
                sequences.append(seq_indices)
                
            i = j # Skip processed events
        else:
            i += 1
            
    return sequences

# ============================================================
# PHASE 2: ATTACKING TRANSITION (COUNTER ATTACKS)
# ============================================================

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
    Returns dictionary with 'corners' and 'free_kicks'.
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
        
    return {
        "corners": corners,
        "free_kicks": free_kicks
    }

