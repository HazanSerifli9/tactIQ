
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Any
from utils.data import get_data_dir
from utils.xt_data import calculate_xt_for_events
from utils.cache import disk_cache

@disk_cache
def process_zonal_data(league: str = "Süper Lig", year: str = "2024", min_mins: int = 300) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Identify the top threat creator in each zone of a 6x5 pitch grid.
    
    A 'Threat Event' is defined as:
    1. A Progressive Pass (moves ball >25% closer to goal).
    2. A Pass into the Penalty Box.

    Args:
        league (str): Competition context.
        year (str): Season year.
        min_mins (int): Minimum minutes played filter.

    Returns:
        tuple: (grid_results list, num_rows, num_cols)
    """
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    # Grid Configuration
    ROWS = 5
    COLS = 6
    PITCH_WIDTH_X = 100
    PITCH_HEIGHT_Y = 100
    
    # Storage
    zone_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    player_mins: Dict[str, int] = {}
    player_teams: Dict[str, str] = {}
    
    for filename in files:
        try:
            file_path = os.path.join(data_dir, filename)
            df = pd.read_parquet(file_path)
            
            if df.empty or 'event' not in df.columns:
                continue
            
            # --- 1. Calculate Minutes Played (Approximation) ---
            match_duration = df['time_min'].max()
            players = df['player_name'].unique()
            
            # Initialize all players with full match duration
            current_match_mins = {}
            for p in players:
                if pd.isna(p): continue
                current_match_mins[p] = match_duration
                # Cache team name
                if p not in player_teams:
                     team = df[df['player_name'] == p]['team_name'].iloc[0]
                     if p == 'U. Çakır':
                         player_teams[p] = 'Galatasaray Spor Kulübü'
                     else:
                         player_teams[p] = team
            
            # Adjust for Substitutions
            sub_on = df[df['event'] == 'Player on']
            for _, row in sub_on.iterrows():
                p, t = row['player_name'], row['time_min']
                current_match_mins[p] = match_duration - t
                
            sub_off = df[df['event'] == 'Player Off']
            for _, row in sub_off.iterrows():
                p, t = row['player_name'], row['time_min']
                current_match_mins[p] = t
            
            # Aggregate minutes
            for p, m in current_match_mins.items():
                player_mins[p] = player_mins.get(p, 0) + m

            # --- 2. Identify Threat Events ---
            df_passes_xt = calculate_xt_for_events(df)
            if df_passes_xt.empty:
                continue
                
            # Allow Goalkeepers to remain
            # if 'position' in df_passes_xt.columns:
            #     df_passes_xt = df_passes_xt[df_passes_xt['position'].fillna('') != 'GK']
                
            threat_events = df_passes_xt[df_passes_xt['xT'] > 0].copy()
            
            if threat_events.empty:
                continue
                
            # --- 3. Bin Events into Zones ---
            for _, row in threat_events.iterrows():
                p = row['player_name']
                if pd.isna(p): continue
                
                x, y = row['x'], row['y']
                xt_val = row['xT']
                
                # Calculate Bin Index
                col_idx = min(int(x / (PITCH_WIDTH_X / COLS)), COLS - 1)
                row_idx = min(int(y / (PITCH_HEIGHT_Y / ROWS)), ROWS - 1)
                
                key = (row_idx, col_idx)
                if key not in zone_stats:
                    zone_stats[key] = {}
                
                zone_stats[key][p] = zone_stats[key].get(p, 0.0) + xt_val
                    
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            continue
            
    # --- 4. Determine "King" of Each Zone ---
    grid_results = []
    
    for r in range(ROWS):
        for c in range(COLS):
            key = (r, c)
            if key not in zone_stats:
                continue
                
            stats = zone_stats[key]
            
            best_player = None
            best_val = -1.0
            best_count = 0
            
            for p, xt_sum in stats.items():
                mins = player_mins.get(p, 0)
                if mins < min_mins:
                    continue
                
                if xt_sum > best_val:
                    best_val = xt_sum
                    best_player = p
            
            if best_player:
                grid_results.append({
                    'row': r,
                    'col': c,
                    'player': best_player,
                    'team': player_teams.get(best_player, 'Unknown'),
                    'value': round(best_val, 2)
                })
                
    return grid_results, ROWS, COLS
