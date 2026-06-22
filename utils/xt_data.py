import numpy as np

# Karun Singh's xT Grid (12x8)
# Source: https://karun.in/blog/expected-threat.html
# Row-major order (8 rows, 12 columns)
xT_grid_12x8 = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126277, 0.01248344, 0.01473596, 0.0174506,  0.02122129, 0.02756312, 0.03485072, 0.0379259 ],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949,  0.01214719, 0.0138454,  0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00887958, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685614, 0.01935132, 0.0241224,  0.02855202, 0.05491138, 0.06442595],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01991071, 0.02385149, 0.03511326, 0.10805005, 0.25745126],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01991071, 0.02385149, 0.03511326, 0.10805005, 0.25745126],
    [0.00887958, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685614, 0.01935132, 0.0241224,  0.02855202, 0.05491138, 0.06442595],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949,  0.01214719, 0.0138454,  0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126277, 0.01248344, 0.01473596, 0.0174506,  0.02122129, 0.02756312, 0.03485072, 0.0379259 ]
])

_ROWS, _COLS = xT_grid_12x8.shape


def get_xt_value(x, y):
    x = max(0, min(100, x))
    y = max(0, min(100, y))

    col = int(x / (100.0 / _COLS))
    row = int(y / (100.0 / _ROWS))

    if col >= _COLS:
        col = _COLS - 1
    if row >= _ROWS:
        row = _ROWS - 1

    return xT_grid_12x8[row, col]


def calculate_xt_for_events(df):
    df = df.copy()
    df['xT'] = 0.0

    if 'Pass End X' not in df.columns or 'Pass End Y' not in df.columns:
        return df

    pass_mask = (df['event'] == 'Pass') & (df['outcome'] == 1)

    if pass_mask.any():
        start_xt = df.loc[pass_mask].apply(
            lambda row: get_xt_value(row['x'], row['y']), axis=1
        )
        end_xt = df.loc[pass_mask].apply(
            lambda row: get_xt_value(row['Pass End X'], row['Pass End Y']), axis=1
        )
        df.loc[pass_mask, 'xT'] = end_xt.values - start_xt.values

    if 'Carry' in df['event'].unique():
        carry_mask = (df['event'] == 'Carry') & (df['outcome'] == 1)
        if carry_mask.any() and 'Pass End X' in df.columns:
            start_xt = df.loc[carry_mask].apply(
                lambda row: get_xt_value(row['x'], row['y']), axis=1
            )
            end_xt = df.loc[carry_mask].apply(
                lambda row: get_xt_value(row['Pass End X'], row['Pass End Y']), axis=1
            )
            df.loc[carry_mask, 'xT'] = end_xt.values - start_xt.values

    return df
