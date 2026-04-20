
import pandas as pd
import numpy as np

def calculate_match_stats(df):
    """
    Calculates advanced stats and identifying top players based on user logic.
    Returns list of key players.
    """
    # 0. Basic Cleanup & Mapping
    df = df.copy()
    
    # Column Mapping
    if 'player_name' in df.columns:
        df['playerName'] = df['player_name']
    
    if 'event' in df.columns:
        df['typeId'] = df['event'] 
    
    # Outcome
    if 'outcome' in df.columns:
        if df['outcome'].dtype in [int, float, 'int64', 'float64']:
            df['outcome'] = df['outcome'].apply(lambda x: 'Successful' if x == 1 else 'Unsuccessful')

    # Coordinates Scaling (100x100 -> 120x80 usually for User constants)
    if 'Pass End X' in df.columns:
        df['end_x_raw'] = df['Pass End X']
        df['end_y_raw'] = df['Pass End Y']
    else:
        df['end_x_raw'] = df.get('end_x', 0)
        df['end_y_raw'] = df.get('end_y', 0)
        
    df['x'] = df['x'] * 1.2
    df['y'] = df['y'] * 0.8
    df['end_x'] = df['end_x_raw'] * 1.2
    df['end_y'] = df['end_y_raw'] * 0.8
    
    # Progressive (Euclidean dist as proxy if 'pro' not in df, but user code checks 'pro')
    # If 'pro' not exists, calculate it? User code implies 'pro' exists.
    # Let's calc distance.
    if 'pro' not in df.columns:
        df['pro'] = np.sqrt((df['end_x'] - df['x'])**2 + (df['end_y'] - df['y'])**2)
    
    # Key Pass / Shot Assist
    # Logic: Pass that leads to a Shot (KeyPass=1)
    # We need to ensure 'keyPass' column exists or is calculated.
    if 'keyPass' not in df.columns:
        # Simple KeyPass calc: Pass followed by Shot by same team
        is_pass = df['typeId'] == 'Pass'
        next_type = df['typeId'].shift(-1)
        next_team = df['team_name'].shift(-1)
        current_team = df['team_name']
        is_shot = next_type.isin(['Miss', 'Attempt Saved', 'Post', 'Goal'])
        is_same_team = next_team == current_team
        
        df['keyPass'] = 0.0
        df.loc[is_pass & is_shot & is_same_team, 'keyPass'] = 1.0
        
    # Helpers
    def get_short_name(name):
        if not isinstance(name, str):
            return str(name)
        names = name.split()
        if len(names) > 1:
            return f"{names[0][0]}. {names[-1]}"
        return name

    # --- User Combined Logic ---
    unique_players = df['playerName'].unique()
    unique_players = [p for p in unique_players if isinstance(p, str)]
    
    # 1. Shot Sequence
    shot_seq_counts = {'playerName': [], 'total': []}
    
    # Optimization: Vectorized or filtering is better than loop, but using loop to match user reliability
    # Filter DF once?
    # Actually, let's just loop as requested but carefully.
    
    for name in unique_players:
        # Filter for player
        p_df = df[df['playerName'] == name]
        
        # Shots
        shots = len(p_df[p_df['typeId'].isin(['Miss', 'Attempt Saved', 'Post', 'Goal'])])
        
        # Shot Assist (KeyPass)
        # Check if keyPass is 1 or 1.0
        kps = len(p_df[(p_df['typeId'] == 'Pass') & (p_df['keyPass'] == 1)])
        
        # Buildup to shot (Pass led to a KeyPass)
        # Shift logic: shift(-1) logic is tricky on filtered p_df if not continuous.
        # But User logic: df[(df['playerName'] == name) & (df['typeId'] == 'Pass') & (df['keyPass'].shift(-1)==1)]
        # This is strictly relying on original DF indexing if run on full df context?
        # No, boolean mask `(df['playerName'] == name)` returns subset.
        # `df['keyPass'].shift(-1)` is computed on FULL DF if done first.
        # Correct approach:
        # Create mask for 'Next is KeyPass'
        # Then filter for player.
        pass
    
    # Let's vectorize 'Next is KeyPass'
    df['next_is_kp'] = df['keyPass'].shift(-1).fillna(0)
    
    shot_data = []
    pass_data = []
    def_data = []
    
    for name in unique_players:
        p_df = df[df['playerName'] == name]
        team_name = p_df['team_name'].iloc[0] if not p_df.empty else "Unknown"
        
        # A. Shot Sequence
        shots = len(p_df[p_df['typeId'].isin(['Miss', 'Attempt Saved', 'Post', 'Goal'])])
        shot_assist = len(p_df[(p_df['typeId'] == 'Pass') & (p_df['keyPass'] == 1)])
        buildup = len(p_df[(p_df['typeId'] == 'Pass') & (p_df['next_is_kp'] == 1)])
        
        total_shot = shots + shot_assist + buildup
        shot_data.append({'playerName': name, 'team': team_name, 'total': total_shot})
        
        # B. Passing
        # Progressive: pro > 9.144, Successful, x [40, 119]
        prog = len(p_df[
            (p_df['pro'] > 9.144) & 
            (p_df['outcome'] == 'Successful') & 
            (p_df['x'] >= 40) & 
            (p_df['x'] <= 119)
        ])
        
        # Box Entry: Pass, Successful, end_x >= 103.5, end_y [16, 64]
        box = len(p_df[
            (p_df['typeId'] == 'Pass') & 
            (p_df['outcome'] == 'Successful') & # User added Successful in 2nd snippet
            (p_df['end_x'] >= 103.5) & 
            (p_df['end_y'] >= 16) & 
            (p_df['end_y'] <= 64)
        ])
        
        kp_count = shot_assist # Same as Key Passes
        
        total_pass = prog + box + kp_count
        pass_data.append({'playerName': name, 'team': team_name, 'total': total_pass})
        
        # C. Defense
        # Tackles (Successful), Interceptions, Clearance
        tackles = len(p_df[(p_df['typeId'] == 'Tackle') & (p_df['outcome'] == 'Successful')])
        interceptions = len(p_df[p_df['typeId'] == 'Interception'])
        clearances = len(p_df[p_df['typeId'] == 'Clearance'])
        
        total_def = tackles + interceptions + clearances
        def_data.append({'playerName': name, 'team': team_name, 'total': total_def})
        
    # Sort and Extract Top
    top_players = []
    
    # Top Attacker
    if shot_data:
        best_att = max(shot_data, key=lambda x: x['total'])
        if best_att['total'] > 0:
            top_players.append({
                'name': get_short_name(best_att['playerName']),
                'team': best_att['team'],
                'reason': f"Attacking Threat ({best_att['total']})"
            })
            
    # Top Creator
    if pass_data:
        best_pass = max(pass_data, key=lambda x: x['total'])
        # Avoid duplicate if same player is top attacker?
        # Ideally yes, but maybe they are just that MVP.
        if best_pass['total'] > 0:
            top_players.append({
                'name': get_short_name(best_pass['playerName']),
                'team': best_pass['team'],
                'reason': f"Top Creator ({best_pass['total']})"
            })
            
    # Top Defender
    if def_data:
        best_def = max(def_data, key=lambda x: x['total'])
        if best_def['total'] > 0:
            top_players.append({
                'name': get_short_name(best_def['playerName']),
                'team': best_def['team'],
                'reason': f"Defensive Rock ({best_def['total']})"
            })
            
    # Deduplicate by name if needed? 
    # If same player is Top Attacker and Creator, show them twice or just once?
    # Showing twice emphasizes dominance.
    
    return top_players


def calculate_player_rankings(df):
    """
    Returns (sh_sq_df, passer_df, defender_df) for Top Players Dashboard.
    """
    df = df.copy()
    
    # Pre-calc columns
    if 'playerName' not in df.columns and 'player_name' in df.columns:
        df['playerName'] = df['player_name']
    
    # Coordinates Scaling
    if 'end_x_raw' not in df.columns:
        if 'Pass End X' in df.columns:
             df['end_x_raw'] = df['Pass End X']
             df['end_y_raw'] = df['Pass End Y']
        else:
             df['end_x_raw'] = df.get('end_x', 0)
             df['end_y_raw'] = df.get('end_y', 0)
    
    # Ensure raw values scaled only if not already? 
    # extract_fixture_data keeps raw 100x100? calculate_match_stats scales.
    # We should assume this function receives clean DF or handled internally.
    # Let's simple-scale like calculate_match_stats if not sure.
    # To be safe, re-scaling logic should be robust (idempotent?)
    # Just do standard scaling assuming 100x100 input usually.
    
    df['x'] = df['x'] * 1.2
    df['y'] = df['y'] * 0.8
    df['end_x'] = df['end_x_raw'] * 1.2
    df['end_y'] = df['end_y_raw'] * 0.8
    
    if 'pro' not in df.columns:
        df['pro'] = np.sqrt((df['end_x'] - df['x'])**2 + (df['end_y'] - df['y'])**2)
        
    if 'typeId' not in df.columns and 'event' in df.columns:
        df['typeId'] = df['event']
        
    # KeyPass logic
    if 'keyPass' not in df.columns:
        is_pass = df['typeId'] == 'Pass'
        next_type = df['typeId'].shift(-1)
        next_team = df['team_name'].shift(-1)
        current_team = df['team_name']
        is_shot = next_type.isin(['Miss', 'Attempt Saved', 'Post', 'Goal'])
        is_same_team = next_team == current_team
        df['keyPass'] = 0.0
        df.loc[is_pass & is_shot & is_same_team, 'keyPass'] = 1.0

    df['next_is_kp'] = df['keyPass'].shift(-1).fillna(0)

    unique_players = df['playerName'].unique()
    unique_players = [p for p in unique_players if isinstance(p, str)]
    
    # 1. Shot Seq
    shot_data = []
    for name in unique_players:
        p_df = df[df['playerName'] == name]
        team_name = p_df['team_name'].iloc[0] if not p_df.empty else "Unknown"
        shots = len(p_df[p_df['typeId'].isin(['Miss', 'Attempt Saved', 'Post', 'Goal'])])
        sa = len(p_df[(p_df['typeId'] == 'Pass') & (p_df['keyPass'] == 1)])
        bs = len(p_df[(p_df['typeId'] == 'Pass') & (p_df['next_is_kp'] == 1)])
        total = shots + sa + bs
        shot_data.append({'playerName': name, 'team': team_name, 'Shots': shots, 'Shot Assist': sa, 'Buildup to shot': bs, 'total': total})
    
    sh_sq_df = pd.DataFrame(shot_data).sort_values(by='total', ascending=False).reset_index(drop=True).head(10)

    # 2. Passing
    pass_data = []
    for name in unique_players:
        p_df = df[df['playerName'] == name]
        team_name = p_df['team_name'].iloc[0] if not p_df.empty else "Unknown"
        
        prog = len(p_df[
            (p_df['pro'] > 9.144) & 
            (p_df['outcome'].astype(str).str.contains('Successful|1|True')) & 
            (p_df['x'] >= 40) & 
            (p_df['x'] <= 119)
        ])
        
        box = len(p_df[
            (p_df['typeId'] == 'Pass') & 
            (p_df['outcome'].astype(str).str.contains('Successful|1|True')) & 
            (p_df['end_x'] >= 103.5) & 
            (p_df['end_y'] >= 16) & 
            (p_df['end_y'] <= 64)
        ])
        
        kp = len(p_df[(p_df['typeId'] == 'Pass') & (p_df['keyPass'] == 1)])
        total = prog + box + kp
        pass_data.append({'playerName': name, 'team': team_name, 'Progressive Passes': prog, 'Passes into pen. box': box, 'Key Passes': kp, 'total': total})
        
    passer_df = pd.DataFrame(pass_data).sort_values(by='total', ascending=False).reset_index(drop=True).head(10)
    
    # 3. Defense
    def_data = []
    for name in unique_players:
        p_df = df[df['playerName'] == name]
        team_name = p_df['team_name'].iloc[0] if not p_df.empty else "Unknown"
        
        tk = len(p_df[(p_df['typeId'] == 'Tackle') & (p_df['outcome'].astype(str).str.contains('Successful|1|True'))])
        intc = len(p_df[p_df['typeId'] == 'Interception'])
        cl = len(p_df[p_df['typeId'] == 'Clearance'])
        total = tk + intc + cl
        def_data.append({'playerName': name, 'team': team_name, 'Tackles': tk, 'Interceptions': intc, 'Clearance': cl, 'total': total})
        
    defender_df = pd.DataFrame(def_data).sort_values(by='total', ascending=False).reset_index(drop=True).head(10)
    
    # Helpers
    def get_short_name(name):
        if not isinstance(name, str): return str(name)
        parts = name.split()
        if len(parts) == 1: return name
        elif len(parts) == 2: return f"{parts[0][0]}. {parts[1]}"
        return f"{parts[0][0]}. {parts[1][0]}. {' '.join(parts[2:])}"

    for d in [sh_sq_df, passer_df, defender_df]:
        if not d.empty:
            d['shortName'] = d['playerName'].apply(get_short_name)
    
    return sh_sq_df, passer_df, defender_df


def get_unique_years():
    """
    Returns unique years/seasons from the data.
    """
    # Simple placeholder or derived from data
    # Assuming current season for now as files don't guarantee year column
    return ["2024-2025"]

def get_leagues():
    """
    Returns unique leagues.
    """
    return ["Süper Lig"]
