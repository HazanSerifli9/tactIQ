import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Oswald:wght@500;700&display=swap"

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY, FONT_URL])
app.title = "tactIQ"
server = app.server

def navbar():
    return html.Nav([
        html.A([
            html.Img(src="/assets/logo.png", style={"height": "40px", "marginRight": "10px"}), 
            html.Span("tactIQ", className="logo-text")
        ], href="/", className="d-flex align-items-center text-decoration-none me-auto"), 

        html.Div([
            dcc.Link("Home", href="/", className="nav-link custom-link"),
            dcc.Link("Matches", href="/analysis", className="nav-link custom-link"),
            dcc.Link("Teams", href="/team-analysis", className="nav-link custom-link"),
            dcc.Link("Players", href="/player-analysis", className="nav-link custom-link"),
            html.A("Göztepe Hub", href="http://127.0.0.1:8051", target="_blank", className="nav-link custom-link highlight-link"),
            dcc.Link("Fixtures", href="/fixtures", className="nav-link custom-link"),
            
        ], className="d-flex gap-4 align-items-center ms-auto"),
        
    ], className="navbar fixed-top px-4 py-3 glass-nav")

app.layout = html.Div([
    navbar(),
    html.Div([
        dash.page_container
    ], className="content-container", style={"paddingTop": "120px"})
])

if __name__ == "__main__":
    app.run(debug=True)
