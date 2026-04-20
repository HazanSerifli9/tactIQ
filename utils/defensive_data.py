
import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from utils.data import get_data_dir

def process_defensive_data(league: str = "Süper Lig", year: str = "2024", min_opp_passes: int = 100) -> pd.DataFrame:
    """
    Process defensive metrics normalized by opponent volume.
    
    Metrics:
    - Balls Won (Interceptions + Blocks + Tackles + Aerials)
    - Recoveries
    
    Normalized per 100 Opposition Passes.

    Returns:
        pd.DataFrame: Summary table
    """
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    player_stats: Dict[str, Dict[str, Any]] = {}
    team_opp_passes: Dict[Tuple[Any, str], int] = {}
    
    DEF_TYPES = [7, 8, 44, 49, 74] # Tackle, Int, Aerial, Recovery, Block
    
    for filename in files:
        try:
            file_path = os.path.join(data_dir, filename)
            df = pd.read_parquet(file_path)
            
            if df.empty or 'event' not in df.columns:
                continue
                
            match_id = df['match_id'].iloc[0] if 'match_id' in df.columns else filename
            home = df[df['team_position'] == 'home']['team_name'].iloc[0]
            away = df[df['team_position'] == 'away']['team_name'].iloc[0]
            
            # Count Team Passes (for Opponent Normalization)
            home_passes = len(df[(df['team_name'] == home) & (df['type_id'] == 1)])
            away_passes = len(df[(df['team_name'] == away) & (df['type_id'] == 1)])
            
            # Store what each team *FACED*
            team_opp_passes[(match_id, home)] = away_passes
            team_opp_passes[(match_id, away)] = home_passes
            
            # Filter Defensive Events
            def_events = df[df['type_id'].isin(DEF_TYPES) & (df['outcome'] == 1)]
            
            for _, row in def_events.iterrows():
                p = row['player_name']
                if pd.isna(p): continue
                
                if p not in player_stats:
                    player_stats[p] = {
                        'team': row['team_name'],
                        'tackle': 0, 'interception': 0, 'aerial': 0,
                        'recovery': 0, 'block': 0,
                        'matches': set()
                    }
                
                stats = player_stats[p]
                stats['matches'].add(match_id)
                
                tid = row['type_id']
                if tid == 7: stats['tackle'] += 1
                elif tid == 8: stats['interception'] += 1
                elif tid == 44: stats['aerial'] += 1
                elif tid == 49: stats['recovery'] += 1
                elif tid == 74: stats['block'] += 1

        except Exception as e:
            continue
            
    # Aggregation
    results = []
    
    for p, stats in player_stats.items():
        # Calculate opportunity
        total_opp_passes = 0
        for mid in stats['matches']:
            key = (mid, stats['team'])
            if key in team_opp_passes:
                total_opp_passes += team_opp_passes[key]
                
        if total_opp_passes < min_opp_passes:
            continue
            
        norm = total_opp_passes / 100.0
        
        balls_won = (stats['interception'] + stats['block'] + stats['tackle'] + stats['aerial']) / norm
        recovery_rate = stats['recovery'] / norm
        
        results.append({
            'name': p,
            'team': stats['team'],
            'balls_won_norm': round(balls_won, 2),
            'recovery_norm': round(recovery_rate, 2),
            'tackle': stats['tackle'],
            'interception': stats['interception'],
            'recovery': stats['recovery'],
            'opp_passes': total_opp_passes
        })
        
    return pd.DataFrame(results)
