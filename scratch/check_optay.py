import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import get_match_dataframe

def check_optay():
    filename = "alanya-goztepe.parquet"
    df = get_match_dataframe(filename)
    if df is None:
        return
        
    shot_types = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
    shots = df[df['event'].isin(shot_types)].copy()
    
    on_target = shots[shots['event'].isin(['Goal', 'Attempt Saved', 'Saved Shot'])].copy()
    on_target = on_target.dropna(subset=['Goal Mouth Y Coordinate', 'Goal Mouth Z Coordinate'])
    
    if not on_target.empty:
        ys = on_target['Goal Mouth Y Coordinate'].astype(float)
        zs = on_target['Goal Mouth Z Coordinate'].astype(float)
        print("For on_target shots:")
        print(f"Goal Mouth Y Coordinate range: min={ys.min()}, max={ys.max()}, mean={ys.mean()}")
        print(f"Goal Mouth Z Coordinate range: min={zs.min()}, max={zs.max()}, mean={zs.mean()}")
        print("All values:")
        for idx, row in on_target.iterrows():
            print(f"Player: {row['player_name']}, Event: {row['event']}, Y: {row['Goal Mouth Y Coordinate']}, Z: {row['Goal Mouth Z Coordinate']}")
    else:
        print("No on_target shots found with goal mouth coordinates.")

if __name__ == "__main__":
    check_optay()
