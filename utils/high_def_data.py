
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List
from utils.data import get_data_dir

def process_high_def_data(league: str = "Süper Lig", year: str = "2024", min_mins: int = 90) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Calculate High Defensive Actions (pressing intensity).
    
    Metrics:
    - High Defensive Action: Tackle, Interception, Recovery, Block, Clearance, Foul 
      committed in the attacking third (x > 67).
    - Normalized per 100 Opponent Passes in their own Defensive Third (x < 33).

    Args:
        min_mins (int): Minimum volume filter (calculated via proxy).

    Returns:
        tuple: (Summary DataFrame, Plot details)
    """
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    player_stats: Dict[str, Dict[str, Any]] = {}
    team_def_passes_cache: Dict[Tuple[Any, str], int] = {} 
    
    # Event Codes
    # 7: Tackle, 8: Int, 49: Recovery, 74: Block, 12: Clearance, 4: Foul
    HIGH_DEF_TYPES = [7, 8, 49, 74, 12, 4]
    
    for filename in files:
        try:
            file_path = os.path.join(data_dir, filename)
            df = pd.read_parquet(file_path)
            
            if df.empty or 'event' not in df.columns:
                continue
            
            match_id = df['match_id'].iloc[0] if 'match_id' in df.columns else filename
            home_team = df[df['team_position'] == 'home']['team_name'].iloc[0]
            away_team = df[df['team_position'] == 'away']['team_name'].iloc[0]
            
            # --- 1. Calculate Normalization Base (Opponent Defensive Passes) ---
            # Passes (Type 1) in their own defensive third (x < 33)
            home_def_passes = len(df[(df['team_name'] == home_team) & (df['type_id'] == 1) & (df['x'] < 33)])
            away_def_passes = len(df[(df['team_name'] == away_team) & (df['type_id'] == 1) & (df['x'] < 33)])
            
            # Store 'Passes Faced' for the PRESSING team
            team_def_passes_cache[(match_id, home_team)] = away_def_passes # Home faces Away's passes
            team_def_passes_cache[(match_id, away_team)] = home_def_passes
            
            # --- 2. Filter High Defensive Actions ---
            # Action event, x > 67 (Attacking Third)
            high_def_actions = df[
                (df['type_id'].isin(HIGH_DEF_TYPES)) &
                (df['x'] > 67) &
                (df['outcome'] == 1) # Successful actions only (simplified)
            ].copy()
            
            for _, event in high_def_actions.iterrows():
                p = event['player_name']
                if pd.isna(p): continue
                
                if p not in player_stats:
                    player_stats[p] = {
                        'team': event['team_name'],
                        'count': 0,
                        'matches': set(),
                        'events': []
                    }
                
                player_stats[p]['count'] += 1
                player_stats[p]['matches'].add(match_id)
                player_stats[p]['events'].append({'x': event['x'], 'y': event['y']})
                
        except Exception as e:
            continue

    # --- 3. Normalization and Aggregation ---
    results = []
    plot_data = {}
    
    for p, stats in player_stats.items():
        # Calculate total opportunity (Opponent Passes Faced)
        opp_passes_faced = 0
        for mid in stats['matches']:
            key = (mid, stats['team'])
            if key in team_def_passes_cache:
                opp_passes_faced += team_def_passes_cache[key]
        
        if opp_passes_faced < 50: # Minimum sample size
            continue
            
        norm_factor = opp_passes_faced / 100.0
        metric = stats['count'] / norm_factor
        
        results.append({
            'name': p,
            'team': stats['team'],
            'high_def_total': stats['count'],
            'high_def_per_100_opp_passes': round(metric, 2),
            'opp_def_passes': opp_passes_faced
        })
        
        plot_data[p] = stats['events']
        
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('high_def_per_100_opp_passes', ascending=False)
        
    return df, plot_data
