
import dash
from dash import html, dcc, Input, Output, State, ALL, callback, MATCH
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data

dash.register_page(__name__, path='/fixtures', title='TactIQ | Fixtures')

def layout():
    matches = extract_fixture_data()
    weeks = {}
    for m in matches:
        w = m['week']
        if w not in weeks:
            weeks[w] = []
        weeks[w].append(m)

    sorted_weeks = sorted(weeks.keys())

    content = []
    
    content.append(html.Header([
        html.Div("2025-2026 Season", style={"color": "var(--accent-color)", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "2px"}),
        html.H1("Süper Lig Fixtures", style={"fontSize": "3rem", "marginBottom": "10px"}),
        html.P("Complete schedule and match results.", style={"color": "var(--text-secondary)"})
    ], style={"textAlign": "center", "marginBottom": "50px"}))

    modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Match Statistics", id="modal-title"), close_button=True),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter(
            dbc.Button("View Detailed Analysis", id="modal-analysis-btn", href="#", external_link=True, color="danger")
        )
    ], id="stats-modal", size="lg", is_open=False, centered=True)
    content.append(modal)

    for week in sorted_weeks:
        content.append(html.H2(f"Week {week}", style={"borderLeft": "4px solid var(--accent-color)", "paddingLeft": "15px", "margin": "40px 0 20px"}))
        
        matches_in_week = weeks[week]
        grid = html.Div(className="matches-grid", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))", "gap": "20px"})
        
        cards = []
        for i, match in enumerate(matches_in_week):
            match_id = match.get('source_file', f'w{week}-m{i}')
            
            card = html.Div([
                html.Div([
                    html.Span("Finished", className="match-status", style={"background": "rgba(42, 157, 143, 0.2)", "color": "#2a9d8f", "padding": "4px 8px", "borderRadius": "4px", "fontSize": "0.75rem", "fontWeight": "700"}),
                    html.Span(f"{match['date']} • {match['time']}", style={"color": "var(--text-secondary)", "fontSize": "0.85rem"})
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "15px"}),
                
                html.Div([
                    html.Div([
                        html.Img(src=f"/{match['logos'][0]}", style={"height": "50px", "marginBottom": "5px"}),
                        html.Span(match['team_names'][0], style={"fontWeight": "600", "fontSize": "0.9rem"})
                    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "flex": "1"}),
                    
                    html.Div([
                        html.Span(f"{match['stats']['team1']['goals']} - {match['stats']['team2']['goals']}", style={"fontSize": "1.8rem", "fontWeight": "700", "color": "white"}), 
                    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "padding": "0 10px"}),
                    
                    html.Div([
                        html.Img(src=f"/{match['logos'][1]}", style={"height": "50px", "marginBottom": "5px"}),
                        html.Span(match['team_names'][1], style={"fontWeight": "600", "fontSize": "0.9rem"})
                    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "flex": "1"}),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "20px"}),
                
                html.Div(match['venue'], style={"textAlign": "center", "color": "var(--text-secondary)", "fontSize": "0.8rem", "marginBottom": "15px"}),
                

                html.Div([
                    html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "15px 0"}),
                    
                    html.Div([
                        html.H5("Match Stats", style={"fontSize": "1rem", "color": "var(--accent-color)", "marginBottom": "10px"}),
                        html.Div([
                            html.Div([html.Span(str(match['stats']['team1']['shots'])), html.Small("Shots", className="text-muted"), html.Span(str(match['stats']['team2']['shots']))], className="d-flex justify-content-between", style={"fontSize": "0.85rem", "marginBottom": "5px"}),
                            html.Div([html.Span(str(match['stats']['team1']['shots_on_target'])), html.Small("On Target", className="text-muted"), html.Span(str(match['stats']['team2']['shots_on_target']))], className="d-flex justify-content-between", style={"fontSize": "0.85rem", "marginBottom": "5px"}),
                            html.Div([html.Span(str(match['stats']['team1']['passes'])), html.Small("Passes", className="text-muted"), html.Span(str(match['stats']['team2']['passes']))], className="d-flex justify-content-between", style={"fontSize": "0.85rem", "marginBottom": "5px"}),
                            html.Div([html.Span(f"{match['stats']['team1']['pass_accuracy']}%"), html.Small("Pass Acc.", className="text-muted"), html.Span(f"{match['stats']['team2']['pass_accuracy']}%")], className="d-flex justify-content-between", style={"fontSize": "0.85rem", "marginBottom": "5px"}),
                        ])
                    ], style={"marginBottom": "20px"}),
                    
                    html.Div([
                        html.H5("Top Players", style={"fontSize": "1rem", "color": "var(--accent-color)", "marginBottom": "10px"}),
                        html.Div([
                            html.Div([
                                html.Span(p['name'], style={"fontWeight": "600"}),
                                html.Span(p['reason'], style={"fontSize": "0.75rem", "color": "var(--text-secondary)", "marginLeft": "auto"})
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px", "borderBottom": "1px solid rgba(255,255,255,0.05)", "paddingBottom": "3px"}) 
                            for p in match.get('key_players', [])[:3]
                        ])
                    ]),
                    
                    dbc.Button("Detailed Analysis →", href=f"/analysis/{match.get('source_file')}", color="danger", size="sm", style={"width": "100%", "marginTop": "20px"})
                    
                ], style={"background": "rgba(0,0,0,0.2)", "borderRadius": "8px", "padding": "15px", "marginTop": "10px"}),

                
            ], className="match-card")
            cards.append(card)
        
        grid.children = cards
        content.append(grid)

    return html.Div(content, className="container", style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"})

