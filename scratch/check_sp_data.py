import pandas as pd

df = pd.read_parquet("/Users/hazanserifli/Desktop/tactıq/raw_data/alanya-goztepe.parquet")
team_name = df['team_name'].unique()[0]
print(f"Analyzing team: {team_name}")
df_team = df[df['team_name'] == team_name]

# Check columns related to corners
corner_col = None
for col in df_team.columns:
    if 'corner' in col.lower() and 'taken' in col.lower():
        corner_col = col
        break

if corner_col:
    corners = df_team[df_team[corner_col].astype(str).str.lower().str.strip().isin(['1', 'true', 'yes', 'si', 'y'])].copy()
    print(f"Found {len(corners)} corners. Let's see some columns of first 2 corners:")
    cols_to_print = ['x', 'y', 'Pass End X', 'Pass End Y', 'outcome', 'player_name']
    cols_to_print = [c for c in cols_to_print if c in corners.columns]
    print(corners[cols_to_print].head(2))
    
    print("\nAll qualifiers in first corner row:")
    row = corners.iloc[0]
    for col in corners.columns:
        if pd.notna(row[col]) and row[col] != 0 and row[col] != '0' and row[col] is not False:
            print(f"  {col}: {row[col]}")
else:
    print("No corner column found!")
