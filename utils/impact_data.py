
import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from utils.data import get_data_dir

def process_impact_data(league: str = "Süper Lig", year: str = "2024", min_mins: int = 400) -> pd.DataFrame:
    """
    Calculate Player Impact Metrics quantifying Team Performance ON vs OFF the pitch.
    
    Metrics:
    - Impact on Creation: (Team Threat For ON/90) - (Team Threat For OFF/90)
    - Impact on Concession: (Team Threat Against ON/90) - (Team Threat Against OFF/90)
    
    'Threat' is defined as Successful Progressive Passes + Box Entries.

    Args:
        min_mins (int): Minimum minutes played to reduce variance.

    Returns:
        pd.DataFrame: Summary dataframe sorted by Creative Impact.
    """
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    player_aggregates: Dict[str, Dict[str, Any]] = {}
    
    for filename in files:
        try:
            file_path = os.path.join(data_dir, filename)
            df = pd.read_parquet(file_path)
            
            if df.empty or 'event' not in df.columns:
                continue
                
            # --- 1. Identify Match Metadata ---
            teams = df['team_name'].unique()
            if len(teams) < 2: continue
            team_a, team_b = teams[0], teams[1]
            match_duration = df['time_min'].max()
            
            # --- 2. Track Player Minutes (On-Pitch Intervals) ---
            # Default: Everyone starts (0 to End)
            # Corrections: Apply Sub On/Off events
            
            match_players = {} # player -> {team, intervals: [[start, end]]}
            
            # Initialize potential players
            all_actors = df['player_name'].unique()
            for p in all_actors:
                if pd.isna(p): continue
                p_team = df[df['player_name'] == p]['team_name'].iloc[0]
                match_players[p] = {'team': p_team, 'intervals': [[0, match_duration]]}
            
            # Fix Subtract Sub Offs
            sub_off_events = df[df['event'] == 'Player Off']
            for _, row in sub_off_events.iterrows():
                p, t = row['player_name'], row['time_min']
                if p in match_players:
                    match_players[p]['intervals'][-1][1] = t
            
            # Fix Add Sub Ons
            sub_on_events = df[df['event'] == 'Player on']
            for _, row in sub_on_events.iterrows():
                p, t = row['player_name'], row['time_min']
                # If player subbed on, they didn't start. Reset interval to [t, end]
                if p in match_players:
                    match_players[p]['intervals'] = [[t, match_duration]]
                else:
                    match_players[p] = {'team': row['team_name'], 'intervals': [[t, match_duration]]}

            # --- 3. Identify 'Threat' Events for Stream Analysis ---
            # Ensure numeric coords
            for col in ['Pass End X', 'Pass End Y', 'x', 'y']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filter Successful Passes Only
            succ_pass = (df['type_id'] == 1) & (df['outcome'] == 1) & (df['Pass End X'].notna())
            
            if succ_pass.any():
                pass_df = df[succ_pass]
                x1, y1 = pass_df['x'], pass_df['y']
                x2, y2 = pass_df['Pass End X'], pass_df['Pass End Y']
                
                # Logic: Progressive or Box Entry
                dist_start = np.sqrt((100-x1)**2 + (50-y1)**2)
                dist_end = np.sqrt((100-x2)**2 + (50-y2)**2)
                progression = dist_start - dist_end
                
                is_prog = (progression > dist_start * 0.25) | ((progression > 10) & (x1 > 40))
                
                start_in_box = (x1 > 83) & (y1 > 21) & (y1 < 79)
                end_in_box = (x2 > 83) & (y2 > 21) & (y2 < 79)
                is_box_entry = end_in_box & (~start_in_box)
                
                threat_mask = is_prog | is_box_entry
                threat_events = pass_df[threat_mask]
            else:
                threat_events = pd.DataFrame()

            # Separate Threat Events by Team
            if not threat_events.empty:
                home_threats = threat_events[threat_events['team_name'] == team_a]
                away_threats = threat_events[threat_events['team_name'] == team_b]
            else:
                home_threats = pd.DataFrame()
                away_threats = pd.DataFrame()
            
            # --- 4. Attribution: Accumulate ON/OFF Stats per Player ---
            for p, info in match_players.items():
                p_team = info['team']
                
                # Check intervals
                for start, end in info['intervals']:
                    duration = end - start
                    if duration <= 0: continue
                    
                    if p not in player_aggregates:
                        player_aggregates[p] = {
                            'team': p_team,
                            'mins_played': 0,
                            'team_mins': 0,
                            'on_for': 0, 'on_against': 0,
                            'total_for': 0, 'total_against': 0
                        }
                    
                    stats = player_aggregates[p]
                    stats['mins_played'] += duration
                    stats['team_mins'] += match_duration
                    
                    # Calculate Stats "While ON"
                    if p_team == team_a:
                        # FOR: Home events in interval
                        # AGAINST: Away events in interval
                        if not home_threats.empty:
                            on_for = len(home_threats[(home_threats['time_min'] >= start) & (home_threats['time_min'] <= end)])
                        else:
                            on_for = 0
                            
                        if not away_threats.empty:
                            on_against = len(away_threats[(away_threats['time_min'] >= start) & (away_threats['time_min'] <= end)])
                        else:
                            on_against = 0
                            
                        stats['total_for'] += len(home_threats)
                        stats['total_against'] += len(away_threats)

                    else: # Away Team
                        if not away_threats.empty:
                            on_for = len(away_threats[(away_threats['time_min'] >= start) & (away_threats['time_min'] <= end)])
                        else:
                            on_for = 0
                            
                        if not home_threats.empty:
                            on_against = len(home_threats[(home_threats['time_min'] >= start) & (home_threats['time_min'] <= end)])
                        else:
                            on_against = 0
                            
                        stats['total_for'] += len(away_threats)
                        stats['total_against'] += len(home_threats)
                        
                    stats['on_for'] += on_for
                    stats['on_against'] += on_against
                    
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            continue

    # --- 5. Final Calculation (Per 90 Differentials) ---
    results = []
    
    for p, stats in player_aggregates.items():
        if stats['mins_played'] < min_mins:
            continue
            
        off_mins = stats['team_mins'] - stats['mins_played']
        # Filter: needs at least ~one full game OFF (90 mins) to have a valid comparison?
        # Let's be lenient but fair.
        
        off_for = stats['total_for'] - stats['on_for']
        off_against = stats['total_against'] - stats['on_against']
        
        # Calculate Per 90s
        on_90 = 90.0 / stats['mins_played']
        off_90 = 90.0 / off_mins if off_mins > 0 else 0
        
        rate_on_create = stats['on_for'] * on_90
        rate_off_create = off_for * off_90
        
        rate_on_concede = stats['on_against'] * on_90
        rate_off_concede = off_against * off_90
        
        # Impact = Rate ON - Rate OFF
        impact_creation = rate_on_create - rate_off_create
        impact_concession = rate_on_concede - rate_off_concede # Positive = Conceded MORE while ON
        
        results.append({
            'name': p,
            'team': stats['team'],
            'impact_creation': round(impact_creation, 2),
            'impact_concession': round(impact_concession, 2),
            'mins': stats['mins_played'],
            'on_create': round(rate_on_create, 2),
            'off_create': round(rate_off_create, 2)
        })
        
    df_result = pd.DataFrame(results)
    if not df_result.empty:
        df_result = df_result.sort_values('impact_creation', ascending=False)
        
    return df_result
