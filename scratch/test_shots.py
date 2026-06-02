import pandas as pd
import os
import sys

sys.path.append(os.path.abspath('.'))
from utils.data import get_data_dir

data_dir = get_data_dir()
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

SHOT_EVENTS = ["Chance missed", "Goal", "Miss", "Post", "Saved Shot", "Temp_Attempt", "Temp_Goal"]
shot_types = [13, 14, 15, 16]

print("Inspecting match files...")
found_penalties = 0
found_freekicks = 0

for filename in files:
    df = pd.read_parquet(os.path.join(data_dir, filename))
    # Filter shots
    shots = df[df['event'].isin(SHOT_EVENTS) | df['type_id'].isin(shot_types)]
    if shots.empty:
        continue
    
    # Print columns on shots to see qualifiers
    cols = [c for c in shots.columns if shots[c].notna().any()]
    
    # Check if there are penalties or direct free kicks
    pen_cols = [c for c in cols if 'pen' in c.lower()]
    fk_cols = [c for c in cols if 'free' in c.lower() or 'fk' in c.lower() or 'direct' in c.lower()]
    
    if pen_cols or fk_cols:
        # Let's count them
        for c in pen_cols:
            non_empty = shots[shots[c].astype(str).str.lower().isin(['1', 'si', 'true', 'yes']) | (shots[c] == 1)]
            if not non_empty.empty:
                print(f"\nMatch: {filename}")
                print(f"  Penalty column '{c}' has {len(non_empty)} rows")
                print("  Sample row values:")
                print(non_empty[['team_name', 'player_name', 'event', 'type_id', c]].head(1).to_dict('records'))
                found_penalties += len(non_empty)
                
        for c in fk_cols:
            non_empty = shots[shots[c].astype(str).str.lower().isin(['1', 'si', 'true', 'yes']) | (shots[c] == 1)]
            if not non_empty.empty:
                print(f"\nMatch: {filename}")
                print(f"  Free kick/direct column '{c}' has {len(non_empty)} rows")
                print("  Sample row values:")
                print(non_empty[['team_name', 'player_name', 'event', 'type_id', c]].head(1).to_dict('records'))
                found_freekicks += len(non_empty)
                
    if found_penalties > 3 and found_freekicks > 3:
        break
