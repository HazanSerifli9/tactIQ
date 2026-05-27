
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List
from utils.data import get_data_dir
from utils.cache import disk_cache
from shared.logger import get_logger

logger = get_logger(__name__)

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

@disk_cache
def process_carry_data(league: str = "Süper Lig", year: str = "2024", min_distance: float = 3.0, max_time: float = 10.0) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Infer 'Carry' events from sequential data.
    
    A Carry is defined as:
    - Same player performing two consecutive events.
    - Moved > min_distance.
    - Elapsed time < max_time.

    Returns:
        tuple: (Summary DataFrame, Plot details)
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
                
            # Normalize timestamp for sequential sorting
            if 'time_min' in df.columns and 'time_sec' in df.columns:
                df['total_seconds'] = df['time_min'] * 60 + df['time_sec']
                df = df.sort_values(by=['period_id', 'total_seconds'])
            else:
                # Assuming implicit csv order
                pass
            
            # Process by Period (don't carry over half-time)
            for _, period_df in df.groupby('period_id'):
                
                # Vectorized Arrays for speed
                players = period_df['player_name'].values
                teams = period_df['team_name'].values
                xs = period_df['x'].values
                ys = period_df['y'].values
                times = period_df['total_seconds'].values if 'total_seconds' in period_df.columns else np.arange(len(period_df))
                
                # Iterate Pairs
                # i = Start of potential carry
                # i+1 = End of potential carry
                
                for i in range(len(period_df) - 1):
                    p1, p2 = players[i], players[i+1]
                    
                    # Must be same player
                    if pd.isna(p1) or p1 != p2:
                        continue
                        
                    t1, t2 = times[i], times[i+1]
                    if (t2 - t1) > max_time:
                        continue
                        
                    # Calculate movement
                    x1, y1 = xs[i], ys[i]
                    x2, y2 = xs[i+1], ys[i+1]
                    dist = calculate_distance(x1, y1, x2, y2)
                    
                    if dist >= min_distance:
                        # VALID CARRY
                        if p1 not in player_stats:
                            player_stats[p1] = {
                                'team': teams[i],
                                'prog_dist': 0,
                                'prog_count': 0,
                                'box_count': 0,
                                'total_dist': 0,
                                'carries': []
                            }
                        
                        stats = player_stats[p1]
                        stats['total_dist'] += dist
                        
                        # Progression Analysis
                        dist_goal_start = calculate_distance(x1, y1, 100, 50)
                        dist_goal_end = calculate_distance(x2, y2, 100, 50)
                        progression = dist_goal_start - dist_goal_end
                        
                        is_prog = progression > 5.0 # Threshold: 5m closer
                        
                        # Box Entry Analysis
                        start_in = (x1 > 83) and (21 < y1 < 79)
                        end_in = (x2 > 83) and (21 < y2 < 79)
                        is_box = end_in and not start_in
                        
                        if is_prog: 
                            stats['prog_dist'] += progression
                            stats['prog_count'] += 1
                            
                        if is_box:
                            stats['box_count'] += 1
                            
                        if is_prog or is_box:
                            stats['carries'].append({
                                'x_start': x1, 'y_start': y1,
                                'x_end': x2, 'y_end': y2,
                                'progressive': is_prog,
                                'box_entry': is_box
                            })

        except Exception as e:
            logger.debug("Skipped player row in carry calc: %s", e)
            continue
            
    # Summarize
    summary = []
    plot_data = {}
    
    for p, stats in player_stats.items():
        summary.append({
            'name': p,
            'team': stats['team'],
            'prog_distance': round(stats['prog_dist'], 1),
            'prog_carry_count': stats['prog_count'],
            'box_entries': stats['box_count'],
            'total_distance': round(stats['total_dist'], 1)
        })
        plot_data[p] = stats['carries']
        
    df = pd.DataFrame(summary)
    if not df.empty:
        df = df.sort_values('prog_distance', ascending=False)
        
    return df, plot_data
