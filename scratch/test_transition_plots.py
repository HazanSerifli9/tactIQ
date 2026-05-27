import sys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# Add the project directory to sys.path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_match_dataframe
from utils.visuals import TACTIQ_BG, TACTIQ_FG, TACTIQ_HOME, TACTIQ_AWAY

def test_transition_outcomes():
    filename = "alanya-goztepe.parquet"
    print(f"Loading match data for {filename}...")
    df = get_match_dataframe(filename)
    if df is None:
        return

    # Ensure scaled coordinates
    df['x_scaled'] = df['x'] * 1.2
    df['y_scaled'] = df['y'] * 0.8
    if 'Pass End X' in df.columns:
        df['end_x_scaled'] = pd.to_numeric(df['Pass End X'], errors='coerce').fillna(0) * 1.2
        df['end_y_scaled'] = pd.to_numeric(df['Pass End Y'], errors='coerce').fillna(0) * 0.8
    else:
        df['end_x_scaled'] = df['x_scaled']
        df['end_y_scaled'] = df['y_scaled']

    # Ensure expanded_minute or abs_time
    if 'expanded_minute' not in df.columns:
        if 'minute' in df.columns and 'second' in df.columns:
            df['abs_time'] = df['minute'] * 60 + df['second']
        else:
            df['abs_time'] = df.index
    else:
        df['abs_time'] = df['expanded_minute'] * 60

    team_name = "Göztepe Spor Kulübü"
    print(f"Testing transition calculations for {team_name}...")

    # Filter ball gains (Tackle outcome=1, Interception, BallRecovery)
    is_gain = (
        (df['event'] == 'BallRecovery') | 
        (df['event'] == 'Ball recovery') |
        (df['event'] == 'Interception') |
        ((df['event'] == 'Tackle') & (df['outcome'] == 1)) |
        (df['type_id'] == 49) |
        (df['type_id'] == 8) |
        ((df['type_id'] == 7) & (df['outcome'] == 1))
    )
    df_gains = df[(df['team_name'] == team_name) & is_gain].copy()
    total_gains = len(df_gains)
    print(f"Found {total_gains} ball gains.")

    if total_gains == 0:
        return

    # Analyze "What Happened"
    # For each gain, scan the next 12 seconds in the same match
    shot_count = 0
    retained_count = 0
    loss_count = 0

    for idx, row in df_gains.iterrows():
        match_id = row['match_id']
        start_time = row['abs_time']
        
        # Scan subsequent events in the same match within 12 seconds
        next_events = df[
            (df['match_id'] == match_id) & 
            (df['abs_time'] > start_time) & 
            (df['abs_time'] <= start_time + 12)
        ].sort_values('abs_time')
        
        outcome_found = False
        has_successful_pass = False
        
        for _, ev in next_events.iterrows():
            # If opponent gets the ball, it might be a turnover unless they shot first
            if ev['team_name'] != team_name:
                # If opponent recovery/interception/tackle happens, did we already retain?
                # We'll continue scanning for shots first, but if opponent has a shot, it's definitely a loss.
                if ev['event'] in ['Shot', 'Goal', 'Missed', 'Saved']:
                    break
                continue
                
            # If same team had a shot/goal: transition led to shot!
            if ev['event'] in ['Shot', 'Goal', 'Missed', 'Saved', 'Attempt Saved', 'SavedShot'] or ev['type_id'] in [13, 14, 15, 16]:
                shot_count += 1
                outcome_found = True
                break
                
            # If same team had a successful pass, we retained possession
            if ev['event'] == 'Pass' and ev['outcome'] == 1:
                has_successful_pass = True

        if not outcome_found:
            if has_successful_pass:
                retained_count += 1
            else:
                loss_count += 1

    print(f"Outcomes - Led to Shot: {shot_count} ({shot_count/total_gains*100:.1f}%), Retained: {retained_count} ({retained_count/total_gains*100:.1f}%), Lost: {loss_count} ({loss_count/total_gains*100:.1f}%)")

    # 9-zone counts
    # X thirds: Def < 40, Mid [40, 80], Att >= 80
    # Y thirds: Right < 26.67, Center [26.67, 53.33], Left >= 53.33
    zones = {
        'Def Left': 0, 'Def Center': 0, 'Def Right': 0,
        'Mid Left': 0, 'Mid Center': 0, 'Mid Right': 0,
        'Att Left': 0, 'Att Center': 0, 'Att Right': 0
    }

    for _, row in df_gains.iterrows():
        x = row['x_scaled']
        y = row['y_scaled']
        
        # Determine X third
        if x < 40:
            x_third = 'Def'
        elif x < 80:
            x_third = 'Mid'
        else:
            x_third = 'Att'
            
        # Determine Y third
        if y < 26.67:
            y_third = 'Right'
        elif y < 53.33:
            y_third = 'Center'
        else:
            y_third = 'Left'
            
        zones[f"{x_third} {y_third}"] += 1

    print("9-Zone counts:", zones)

if __name__ == "__main__":
    test_transition_outcomes()
