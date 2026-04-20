import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# --- Data Modules ---
from utils.stats import get_unique_years, get_leagues
from utils.carry_data import process_carry_data
from utils.impact_data import process_impact_data
from utils.zonal_data import process_zonal_data
from utils.box_entry_data import process_box_entry_data

# --- Visual Modules ---
from utils.stats_visuals import generate_league_top_players_plot
from utils.carry_visuals import generate_carry_plot
from utils.impact_visuals import generate_impact_chart
from utils.zonal_visuals import generate_zonal_map
from utils.box_entry_visuals import generate_box_entry_grid

dash.register_page(__name__, path='/player-analysis', title='TactIQ | Player Analysis')


def layout():
    # 1. League Top Performers
    try:
        top_players_img = generate_league_top_players_plot()
    except Exception as e:
        print(f"Error generating top players plot: {e}")
        top_players_img = None

    # 2. Top Carriers
    try:
        carry_df, carry_details = process_carry_data()
        carry_img = generate_carry_plot(carry_df, carry_details)
    except Exception as e:
        print(f"Error generating carry plot: {e}")
        carry_img = None

    # 3. Player Impact
    try:
        impact_df = process_impact_data()
        impact_create_img = generate_impact_chart(impact_df, 'creation')
        impact_concede_img = generate_impact_chart(impact_df, 'concession')
    except Exception as e:
        print(f"Error generating impact plots: {e}")
        impact_create_img = None
        impact_concede_img = None

    # 4. Zonal Threat Kings
    try:
        zonal_grid, rows, cols = process_zonal_data()
        zonal_map_img = generate_zonal_map(zonal_grid, rows, cols)
    except Exception as e:
        print(f"Error generating zonal map: {e}")
        zonal_map_img = None

    # 5. Box Entry Distributions
    try:
        box_entries = process_box_entry_data()
        box_entry_img = generate_box_entry_grid(box_entries)
    except Exception as e:
        print(f"Error generating box entry plot: {e}")
        box_entry_img = None

    return html.Div([
        # --- Header ---
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("Player Analysis Hub", className="text-white fw-bold mb-4"),
                    html.P("Comprehensive analysis of player performance across multiple dimensions.", className="text-white-50")
                ])
            ])
        ], fluid=True, className="py-4"),

        # --- Section 1: League Overview ---
        dbc.Container([
            html.H4("League Top Performers", className="text-white fw-bold mb-3 border-bottom border-secondary pb-2"),
            dbc.Row([
                dbc.Col([
                    html.Img(src=top_players_img, style={'width': '100%', 'border-radius': '10px'}) if top_players_img else html.Div("No data")
                ], width=12)
            ], className="mb-5"),
        ], fluid=True),

        # --- Section 2: Carrying ---
        dbc.Container([
            html.H4("Ball Carrying Specialists", className="text-white fw-bold mb-3 border-bottom border-secondary pb-2"),
            dbc.Row([
                dbc.Col([
                    html.Img(src=carry_img, style={'width': '100%', 'border-radius': '10px'}) if carry_img else html.Div("No data")
                ], width=12)
            ], className="mb-5"),
        ], fluid=True),

        # --- Section 3: Advanced Threat Distribution ---
        dbc.Container([
            html.H4("Advanced Threat Distribution", className="text-white fw-bold mb-3 border-bottom border-secondary pb-2"),
            dbc.Row([
                dbc.Col([
                    html.Img(src=zonal_map_img, style={'width': '100%', 'border-radius': '10px'}) if zonal_map_img else html.Div("No data")
                ], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Img(src=box_entry_img, style={'width': '100%', 'border-radius': '10px'}) if box_entry_img else html.Div("No data")
                ], width=12)
            ], className="mb-5")
        ], fluid=True),

        # --- Section 4: Player Impact (On/Off) ---
        dbc.Container([
            html.H4("Player Impact (On/Off Pitch)", className="text-white fw-bold mb-3 border-bottom border-secondary pb-2"),
            dbc.Row([
                dbc.Col([
                    html.Img(src=impact_create_img, style={'width': '100%', 'border-radius': '10px'}) if impact_create_img else html.Div("No data")
                ], width=6),
                dbc.Col([
                    html.Img(src=impact_concede_img, style={'width': '100%', 'border-radius': '10px'}) if impact_concede_img else html.Div("No data")
                ], width=6)
            ], className="mb-5"),
        ], fluid=True),

    ], style={'padding-bottom': '100px'})
