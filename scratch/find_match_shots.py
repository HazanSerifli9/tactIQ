import pandas as pd
import os

data_dir = "/Users/hazanserifli/Desktop/tactıq/raw_data"
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

for filename in files:
    try:
        df = pd.read_parquet(os.path.join(data_dir, filename))
        if 'team_name' not in df.columns:
            continue
        teams = df['team_name'].unique().tolist()
        if 'Antalyaspor' in [t.split()[0] for t in teams]:
            # Look for shots
            shot_types = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
            shots = df[df['event'].isin(shot_types)].copy()
            shots['typeId'] = shots['event']
            
            for team in teams:
                if 'Antalyaspor' in team:
                    t_shots = shots[shots['team_name'] == team]
                    on_target = t_shots[t_shots['typeId'].isin(['Attempt Saved', 'Saved Shot', 'Goal'])]
                    on_target = on_target.dropna(subset=['Goal Mouth Y Coordinate', 'Goal Mouth Z Coordinate'])
                    if len(on_target) == 7:
                        print(f"Match found: {filename} - {team} has 7 shots on target:")
                        for idx, row in on_target.iterrows():
                            print(f"  Player: {row.get('player_name')}, Jersey: {row.get('Jersey Number')}, Event: {row.get('event')}, Y: {row.get('Goal Mouth Y Coordinate')}, Z: {row.get('Goal Mouth Z Coordinate')}")
    except Exception as e:
        continue
