
import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data

dash.register_page(__name__, path='/analysis', title='TactIQ | Match Analysis')

def layout():
    matches = extract_fixture_data(lite=True)
    matches.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)

    return html.Div([
        html.Header([
            dbc.Row([
                dbc.Col(html.Img(src="/assets/superlig_logo.jpg", style={"height": "100px", "borderRadius": "50%"}), width=2),
                dbc.Col([
                    html.Div("Central Data Hub", style={"color": "var(--accent-color)", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "2px"}),
                    html.H1("Match Analysis", style={"fontSize": "3rem", "marginBottom": "10px"}),
                    html.P("Select a match to view detailed post-match reports.", style={"color": "var(--text-secondary)"}),
                ], width=8, className="text-center"),
                dbc.Col(width=2)
            ], className="mb-4 align-items-center")
        ], style={"marginBottom": "40px"}),

        html.Div([
            dcc.Input(
                id="match-search",
                type="text",
                placeholder="Search for teams or dates...",
                className="search-input",
                style={
                    "width": "100%", "maxWidth": "500px",
                    "background": "rgba(255, 255, 255, 0.05)",
                    "border": "1px solid var(--border-color)",
                    "padding": "15px 25px",
                    "borderRadius": "30px",
                    "color": "white",
                    "fontSize": "1rem",
                    "outline": "none"
                }
            )
        ], style={"marginBottom": "30px", "display": "flex", "justifyContent": "center"}),

        html.Div(id="matches-list", className="matches-list", style={"display": "flex", "flexDirection": "column", "gap": "15px"})
    ], className="container", style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"})

@callback(
    Output("matches-list", "children"),
    Input("match-search", "value")
)
def update_matches(search_value):
    matches = extract_fixture_data(lite=True)
    matches.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)

    if search_value:
        search_value = search_value.lower()
        matches = [m for m in matches if search_value in m['match_name'].lower() or search_value in m['date']]

    children = []
    for match in matches:
        if 'source_file' in match:
            filename = match['source_file']
            link = f"/analysis/{filename}"
            
            children.append(
                dcc.Link([
                    html.Div(match.get('date', 'Unknown Date'), className="match-date", style={"color": "var(--text-secondary)", "fontSize": "0.9rem", "width": "120px"}),
                    html.Div([
                        html.Span(match['team_names'][0], className="team-home", style={"textAlign": "right", "flex": "1"}),
                        html.Span("vs", className="vs-badge", style={"background": "rgba(255, 255, 255, 0.1)", "color": "var(--text-secondary)", "padding": "2px 8px", "borderRadius": "4px", "fontSize": "0.8rem", "fontWeight": "400", "margin": "0 10px"}),
                        html.Span(match['team_names'][1], className="team-away", style={"textAlign": "left", "flex": "1"})
                    ], className="match-teams", style={"flex": "1", "display": "flex", "alignItems": "center", "gap": "5px", "fontWeight": "600", "fontSize": "1.1rem"}),
                    html.Div("→", className="match-arrow", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "fontWeight": "700"})
                ], href=link, className="match-list-item match-card", style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "textDecoration": "none", "color": "var(--text-main)"})
            )
    
    return children
