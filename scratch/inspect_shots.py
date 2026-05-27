import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import get_match_dataframe

def inspect_shots():
    filename = "alanya-goztepe.parquet"
    df = get_match_dataframe(filename)
    if df is None:
        print("DF is None")
        return
        
    shot_types = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
    shots = df[df['event'].isin(shot_types)].copy()
    print("Found columns:")
    cols = [c for c in df.columns if 'goal' in c.lower() or 'mouth' in c.lower() or 'coordinate' in c.lower() or 'mouth' in c.lower()]
    print(cols)
    
    print("\nSample shot data columns related to coordinates:")
    shot_cols = [c for c in df.columns if any(k in c.lower() for k in ['x', 'y', 'z', 'coord', 'mouth', 'blocked'])]
    print(shot_cols)
    
    if not shots.empty:
        # Check non-null values in columns
        for c in shot_cols:
            non_null = shots[c].dropna()
            if not non_null.empty:
                print(f"Column '{c}' has {len(non_null)} non-null values. Sample values: {non_null.head(3).tolist()}")
            else:
                print(f"Column '{c}' is completely empty/null for shots.")
                
if __name__ == "__main__":
    inspect_shots()
