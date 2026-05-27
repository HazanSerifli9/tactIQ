import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

from utils.stats_visuals import generate_league_top_players_plot
from utils.zonal_data import process_zonal_data
from utils.zonal_visuals import generate_zonal_map
from utils.box_entry_data import process_box_entry_data
from utils.box_entry_visuals import generate_box_entry_grid

dash.register_page(__name__, path='/player-analysis', title='TactIQ | Player Analysis')

_IMG_STYLE = {'width': '100%', 'border-radius': '10px'}
_LOADING_PLACEHOLDER = html.Div("Could not load chart.", style={"color": "#6b7280", "padding": "20px"})


def layout():
    return html.Div([
        dcc.Interval(id="pa-load-trigger", interval=1, max_intervals=1),

        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("Player Analysis Hub", className="text-white fw-bold mb-4"),
                    html.P("Comprehensive analysis of player performance across multiple dimensions.",
                           className="text-white-50"),
                ])
            ])
        ], fluid=True, className="py-4"),

        dbc.Container([
            html.H4("League Top Performers",
                    className="text-white fw-bold mb-3 border-bottom border-secondary pb-2"),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(html.Div(id="pa-top-players"), type="circle", color="#FDE636"),
                ], width=12),
            ], className="mb-5"),
        ], fluid=True),

        dbc.Container([
            html.H4("Advanced Threat Distribution",
                    className="text-white fw-bold mb-3 border-bottom border-secondary pb-2"),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(html.Div(id="pa-zonal"), type="circle", color="#FDE636"),
                ], width=12),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(html.Div(id="pa-box-entry"), type="circle", color="#FDE636"),
                ], width=12),
            ], className="mb-5"),
        ], fluid=True),

    ], style={'padding-bottom': '100px'})


@callback(Output("pa-top-players", "children"), Input("pa-load-trigger", "n_intervals"))
def _load_top_players(_):
    try:
        img = generate_league_top_players_plot()
        return html.Img(src=img, style=_IMG_STYLE) if img else _LOADING_PLACEHOLDER
    except Exception as e:
        print(f"Error generating top players plot: {e}")
        return _LOADING_PLACEHOLDER


@callback(Output("pa-zonal", "children"), Input("pa-load-trigger", "n_intervals"))
def _load_zonal(_):
    try:
        zonal_grid, rows, cols = process_zonal_data()
        img = generate_zonal_map(zonal_grid, rows, cols)
        return html.Img(src=img, style=_IMG_STYLE) if img else _LOADING_PLACEHOLDER
    except Exception as e:
        print(f"Error generating zonal map: {e}")
        return _LOADING_PLACEHOLDER


@callback(Output("pa-box-entry", "children"), Input("pa-load-trigger", "n_intervals"))
def _load_box_entry(_):
    try:
        box_entries = process_box_entry_data()
        img = generate_box_entry_grid(box_entries)
        return html.Img(src=img, style=_IMG_STYLE) if img else _LOADING_PLACEHOLDER
    except Exception as e:
        print(f"Error generating box entry plot: {e}")
        return _LOADING_PLACEHOLDER
