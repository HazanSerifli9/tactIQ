
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import os

from utils.data import get_data_dir
from utils.analysis import get_turnovers_and_outcomes

dash.register_page(__name__, path='/transitions_animation')

# ============================================================
# 1. CONSTANTS & CONFIG
# ============================================================
PITCH_LENGTH = 100
PITCH_WIDTH = 100
GOAL_HEIGHT_OPT = 45.2
GOAL_TOP_OPT = 54.8

METER_PER_X = 1.05
METER_PER_Y = 0.68

def opta_to_meters_x(x):
    return x * METER_PER_X

def opta_to_meters_y(y):
    return y * METER_PER_Y

# ============================================================
# 2. DATA LOADING HELPER
# ============================================================

def load_match_data(team_name="Göztepe Spor Kulübü"):
    data_dir = get_data_dir()
    priority_files = ["goztepe-fb.parquet", "goztepe-gs.parquet", "Gaziantep-Göztepe.parquet"]

    selected_file = None
    for f in priority_files:
        if os.path.exists(os.path.join(data_dir, f)):
            selected_file = f
            break

    if not selected_file:
         all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and ('goztepe' in f.lower() or 'göztepe' in f.lower())]
         if all_files:
             selected_file = all_files[0]

    if not selected_file:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_parquet(os.path.join(data_dir, selected_file))

    if 'event_id' not in df.columns:
        df['event_id'] = range(len(df))

    df['x_plot_m'] = opta_to_meters_x(df['x'])
    df['y_plot_m'] = opta_to_meters_y(df['y'])
    df['pass_end_x_plot_m'] = opta_to_meters_x(df['Pass End X'])
    df['pass_end_y_plot_m'] = opta_to_meters_y(df['Pass End Y'])

    turnovers = get_turnovers_and_outcomes(df, team_name)

    dangerous_turnovers = turnovers[turnovers['consequence'].isin(['Goal', 'Shot'])].copy()

    if dangerous_turnovers.empty:
        dangerous_turnovers['label'] = []
    else:
        dangerous_turnovers['label'] = dangerous_turnovers.apply(
            lambda r: f"{int(r['time_min'])}:{int(r['time_sec']):02d} - {r['consequence']} ({r['player_name']})", axis=1
        )

    return df, dangerous_turnovers


FULL_DF, TURNOVERS_DF = load_match_data()


# ============================================================
# 3. TIMELINE BUILDER
# ============================================================

def build_timeline(turnover_row, df):
    """
    Builds the timeline starting from the turnover event until the consequence (Goal/Shot).
    """
    start_idx = turnover_row.name

    subset = df.loc[start_idx:].copy()


    t_min = turnover_row['time_min']
    t_sec = turnover_row['time_sec']



    timeline_rows = []




    for _, row in subset.iterrows():
        if row['team_name'] != turnover_row['team_name']:
            evt = row['event']
            if evt in ['Goal', 'Miss', 'Post', 'Attempt Saved']:
                timeline_rows.append(row)
                break

        timeline_rows.append(row)


        curr_time = row['time_min']*60 + row['time_sec']
        start_time = t_min*60 + t_sec
        if (curr_time - start_time) > 20:
            break

    df_timeline = pd.DataFrame(timeline_rows).reset_index(drop=True)

    # Animation timing
    df_timeline['n_sub'] = 10
    df_timeline.loc[df_timeline['event'].isin(['Goal', 'Miss', 'Post', 'Attempt Saved']), 'n_sub'] = 20
    df_timeline.loc[df_timeline['event'] == 'Pass', 'n_sub'] = 15


    df_timeline['order'] = range(1, len(df_timeline) + 1)

    df_timeline['kind'] = 'event'
    df_timeline.loc[df_timeline['event'] == 'Pass', 'kind'] = 'pass'
    df_timeline.loc[df_timeline['event'] == 'Goal', 'kind'] = 'goal'

    df_timeline['ball_start_x'] = df_timeline['x_plot_m']
    df_timeline['ball_start_y'] = df_timeline['y_plot_m']

    df_timeline['ball_end_x'] = df_timeline['ball_start_x']
    df_timeline['ball_end_y'] = df_timeline['ball_start_y']

    mask_pass = (df_timeline['event'] == 'Pass') & (df_timeline['pass_end_x_plot_m'].notna())
    df_timeline.loc[mask_pass, 'ball_end_x'] = df_timeline.loc[mask_pass, 'pass_end_x_plot_m']
    df_timeline.loc[mask_pass, 'ball_end_y'] = df_timeline.loc[mask_pass, 'pass_end_y_plot_m']

    df_timeline['event'].isin(['Goal', 'Miss', 'Post', 'Attempt Saved'])



    return df_timeline

# ============================================================
# 4. PLOT BUILDER
# ============================================================

def create_pitch():
    fig = go.Figure()

    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color="white"), fillcolor="#313332")

    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="white"))
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="white"))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="white"))

    fig.update_layout(
        xaxis=dict(range=[-5, 110], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-5, 73], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="#313332",
        paper_bgcolor="#313332",
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)")
    )
    return fig

def build_traces(timeline, step, substep):
    traces = []

    past = timeline[timeline['order'] <= step]

    traces.append(go.Scatter(
        x=past['x_plot_m'], y=past['y_plot_m'],
        mode='markers+text',
        marker=dict(size=10, color=past.apply(lambda x: 'red' if x['team_name'] != 'Göztepe Spor Kulübü' else 'orange', axis=1)),
        text=past['order'],
        textfont=dict(color='white'),
        name='Events'
    ))

    if not past.empty:
        curr = past.iloc[-1]

        pct = substep / curr['n_sub']
        bx = curr['ball_start_x'] + (curr['ball_end_x'] - curr['ball_start_x']) * pct
        by = curr['ball_start_y'] + (curr['ball_end_y'] - curr['ball_start_y']) * pct

        traces.append(go.Scatter(
            x=[bx], y=[by],
            mode='markers',
            marker=dict(size=8, color='white', symbol='circle'),
            name='Ball'
        ))

        if curr['kind'] == 'pass':
             traces.append(go.Scatter(
                x=[curr['ball_start_x'], curr['ball_end_x']],
                y=[curr['ball_start_y'], curr['ball_end_y']],
                mode='lines',
                line=dict(color='yellow', width=2),
                name='Pass Path'
            ))

    return traces

# ============================================================
# 5. LAYOUT
# ============================================================

layout = html.Div([
    html.H1("Defensive Transition Animation", style={"textAlign": "center", "color": "white"}),

    html.Div([
        html.Label("Select Dangerous Turnover:", style={"color": "white"}),
        dcc.Dropdown(
            id='turnover-dropdown',
            options=[{'label': row['label'], 'value': idx} for idx, row in TURNOVERS_DF.iterrows()],
            placeholder="Select a turnover...",
            style={"color": "black"}
        )
    ], style={"maxWidth": "600px", "margin": "0 auto", "padding": "20px"}),

    dcc.Graph(id='animation-graph'),

    html.Div([
        html.Button("Play", id="btn-play", n_clicks=0, className="btn btn-primary"),
        html.Button("Reset", id="btn-reset", n_clicks=0, className="btn btn-secondary", style={"marginLeft": "10px"}),
    ], style={"textAlign": "center", "marginTop": "20px"}),

    dcc.Interval(id='anim-interval', interval=200, disabled=True)
])

# ============================================================
# 6. CALLBACKS
# ============================================================

@dash.callback(
    [Output('animation-graph', 'figure'),
     Output('anim-interval', 'disabled')],
    [Input('turnover-dropdown', 'value'),
     Input('anim-interval', 'n_intervals'),
     Input('btn-play', 'n_clicks'),
     Input('btn-reset', 'n_clicks')]
)
def update_animation(turnover_idx, n_intervals, play_clicks, reset_clicks):
    ctx = dash.callback_context
    ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None



    if turnover_idx is None:
        return create_pitch(), True

    turnover_row = TURNOVERS_DF.loc[turnover_idx]
    timeline = build_timeline(turnover_row, FULL_DF)

    fig = create_pitch()


    fig.add_trace(go.Scatter(
        x=timeline['x_plot_m'], y=timeline['y_plot_m'],
        mode='markers+text',
        marker=dict(size=12, color=['red' if t != 'Göztepe Spor Kulübü' else 'orange' for t in timeline['team_name']]),
        text=timeline['order'],
        textposition="top center",
        textfont=dict(color='white'),
        name='Sequence'
    ))


    for i in range(len(timeline)-1):
        curr = timeline.iloc[i]
        next_ev = timeline.iloc[i+1]
        fig.add_annotation(
            x=next_ev['x_plot_m'], y=next_ev['y_plot_m'],
            ax=curr['x_plot_m'], ay=curr['y_plot_m'],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="yellow", opacity=0.7
        )

    fig.update_layout(title=f"Turnover leading to {turnover_row['consequence']}")

    return fig, True # Keep animation disabled for this simplified version
