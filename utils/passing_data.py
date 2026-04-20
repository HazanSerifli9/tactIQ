
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List
from utils.data import get_data_dir

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def process_passing_data(league: str = "Süper Lig", year: str = "2024", min_passes: int = 30) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Process passing data to calculate Progressive and Box Entry metrics.
    
    Metrics:
    - Progressive Pass: Moves ball >25% closer to goal OR >10m closer (if starting outside defensive 40%).
    - Box Pass: Enters the opposition penalty box.
    
    Returns:
        tuple: (Summary DF, Plotting Data Dict)
    """
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    player_stats: Dict[str, Dict[str, Any]] = {}
    
    for filename in files:
        try:
            file_path = os.path.join(data_dir, filename)
            df = pd.read_parquet(file_path)
            
            if df.empty or 'event' not in df.columns:
                continue
                
            # Filter for Successful Passes
            # Type 1 = Pass, Outcome 1 = Success
            succ_pass_mask = (df['type_id'] == 1) & (df['outcome'] == 1)
            passes = df[succ_pass_mask].copy()
            
            if passes.empty:
                continue
                
            # Ensure coords
            cols_num = ['x', 'y', 'Pass End X', 'Pass End Y']
            for c in cols_num:
                passes[c] = pd.to_numeric(passes[c], errors='coerce')
            
            passes = passes.dropna(subset=['Pass End X', 'Pass End Y'])
            
            for _, row in passes.iterrows():
                p = row['player_name']
                if pd.isna(p): continue
                
                if p not in player_stats:
                    player_stats[p] = {
                        'team': row['team_name'],
                        'total_passes': 0, 'prog_passes': 0, 'box_passes': 0,
                        'details': []
                    }
                
                stats = player_stats[p]
                stats['total_passes'] += 1
                
                x1, y1 = row['x'], row['y']
                x2, y2 = row['Pass End X'], row['Pass End Y']
                
                # Progressive Logic
                dist_start = calculate_distance(x1, y1, 100, 50)
                dist_end = calculate_distance(x2, y2, 100, 50)
                progression = dist_start - dist_end
                
                is_prog = (progression > dist_start * 0.25) | ((progression > 10.0) & (x1 > 40))
                
                # Box Logic
                start_in_box = (x1 > 83) and (21 < y1 < 79)
                end_in_box = (x2 > 83) and (21 < y2 < 79)
                is_box = end_in_box and not start_in_box
                
                if is_prog: stats['prog_passes'] += 1
                if is_box: stats['box_passes'] += 1
                
                if is_prog:
                    stats['details'].append({
                        'x': x1, 'y': y1,
                        'endX': x2, 'endY': y2,
                        'box_entry': is_box
                    })

        except Exception as e:
            continue
            
    # Summary
    summary = []
    plot_data = {}
    
    for p, stats in player_stats.items():
        if stats['total_passes'] < min_passes:
            continue
            
        pass_norm = stats['total_passes'] / 100.0
        
        summary.append({
            'name': p,
            'team': stats['team'],
            'prog_passes_per_100': round(stats['prog_passes'] / pass_norm, 2),
            'box_passes_per_100': round(stats['box_passes'] / pass_norm, 2),
            'total_prog_passes': stats['prog_passes'],
            'total_box_passes': stats['box_passes'],
            'total_passes': stats['total_passes']
        })
        
        plot_data[p] = stats['details']
        
    df = pd.DataFrame(summary)
    if not df.empty:
        df = df.sort_values('total_prog_passes', ascending=False)
        
    return df, plot_data
