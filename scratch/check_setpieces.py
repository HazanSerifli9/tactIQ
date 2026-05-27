import pandas as pd
import os

df = pd.read_parquet("/Users/hazanserifli/Desktop/tactıq/raw_data/alanya-goztepe.parquet")
print("All column names in df containing 'corner' or 'kick' or 'free':")
cols = [c for c in df.columns if any(k in c.lower() for k in ['corner', 'kick', 'free', 'setpiece', 'set piece'])]
print(cols)

print("\nLet's see unique values of event column:")
print(df['event'].unique().tolist())

# Check if there are any events with 'Corner' or 'Free' in them
print("\nUnique events containing 'Corner' or 'Free' or 'Set':")
print([e for e in df['event'].unique() if any(k in str(e).lower() for k in ['corner', 'free', 'set'])])

# Check for qualifiers
print("\nQualifiers or columns with 'si' or '1' or 'yes' that could be set pieces:")
for col in cols:
    val_counts = df[col].value_counts().head(5)
    print(f"Column: {col}, value counts:\n{val_counts}")
