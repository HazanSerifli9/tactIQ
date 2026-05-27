
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from utils.data import get_data_dir
from utils.cache import disk_cache

@disk_cache
def process_box_entry_data(league: str = "Süper Lig", year: str = "2024") -> List[Dict[str, Any]]:
    """
    Identify Top 20 players by Box Entries and extract their event coordinates.
    A 'Box Entry' is defined as a successful pass or carry that originates outside 
    the penalty area and ends inside it.

    Args:
        league (str): League name for filtering (future support).
        year (str): Season year for data retrieval.

    Returns:
        List[Dict]: A list of dictionaries containing player name, team, entry count, 
                    and arrays of starting coordinates (x, y).
    """
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    player_entries: Dict[str, Dict[str, Any]] = {} 
    
    # Opta Box Dimensions (Approximate)
    BOX_X_MIN = 83
    BOX_Y_MIN = 21
    BOX_Y_MAX = 79

    for filename in files:
        try:
            file_path = os.path.join(data_dir, filename)
            df = pd.read_parquet(file_path)
            
            if df.empty or 'event' not in df.columns:
                continue
            
            # Ensure numeric coordinates
            cols_to_numeric = ['x', 'y', 'Pass End X', 'Pass End Y']
            for col in cols_to_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # --- Logic 1: Successful Passes into Box ---
            # Type 1 = Pass, Outcome 1 = Successful
            successful_passes = (df['type_id'] == 1) & (df['outcome'] == 1) & (df['Pass End X'].notna())
            
            if successful_passes.any():
                pass_events = df[successful_passes].copy()
                
                # Vectorized check for Box Entry
                x_start, y_start = pass_events['x'], pass_events['y']
                x_end, y_end = pass_events['Pass End X'], pass_events['Pass End Y']
                
                start_in_box = (x_start > BOX_X_MIN) & (y_start > BOX_Y_MIN) & (y_start < BOX_Y_MAX)
                end_in_box = (x_end > BOX_X_MIN) & (y_end > BOX_Y_MIN) & (y_end < BOX_Y_MAX)
                
                # Valid entry: Ends IN box, Started OUT of box
                is_valid_entry = end_in_box & (~start_in_box)
                
                entry_events = pass_events[is_valid_entry]
                
                for _, row in entry_events.iterrows():
                    player_name = row['player_name']
                    if pd.isna(player_name):
                        continue
                    
                    if player_name not in player_entries:
                        player_entries[player_name] = {
                            'team': row['team_name'], 
                            'x': [], 
                            'y': [], 
                            'count': 0
                        }
                    
                    player_entries[player_name]['x'].append(row['x'])
                    player_entries[player_name]['y'].append(row['y'])
                    player_entries[player_name]['count'] += 1

            # --- Logic 2: Carries into Box (Future Expansion) ---
            # Currently relying on explicit pass events. Carries inferred from tracking data
            # or sequential event logic would be added here.

        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            continue

    # Sort players by count and take Top 20
    sorted_players = sorted(player_entries.items(), key=lambda item: item[1]['count'], reverse=True)
    top_20 = sorted_players[:20]
    
    results = []
    for player, data in top_20:
        results.append({
            'name': player,
            'team': data['team'],
            'count': data['count'],
            'x': data['x'],
            'y': data['y']
        })
        
    return results
