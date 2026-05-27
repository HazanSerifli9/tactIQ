import sys
import os

# Ensure the project root (one level up) is on sys.path so that
# the shared/ and utils/ packages are importable from göztepehub.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Must be set before any matplotlib/mplsoccer import — Dash runs callbacks
# on background threads which crash the macOS GUI backend.
import matplotlib
matplotlib.use('Agg')

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Oswald:wght@500;700&display=swap"
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.join(_APP_DIR, "pages"),
    assets_folder=os.path.join(_APP_DIR, "assets"),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY, FONT_URL],
)
app.title = "Göztepe Hub"
server = app.server

def navbar():
    return html.Nav([
        html.A([
            html.Img(src="/assets/logo.png", style={"height": "40px", "marginRight": "10px"}), 
            html.Span("tactIQ", className="logo-text")
        ], href="http://127.0.0.1:8050/", className="d-flex align-items-center text-decoration-none me-auto"), 

        html.Div([
            dcc.Link("Overview", href="/", className="nav-link custom-link highlight-link"),
            dcc.Link("Pre-Match", href="/pre-match", className="nav-link custom-link"),
            dcc.Link("Rival Scout", href="/rival-scout", className="nav-link custom-link"),
            dcc.Link("Post-Match", href="/post-match", className="nav-link custom-link"),
            dcc.Link("Trends", href="/trends", className="nav-link custom-link"),
        ], className="d-flex gap-4 align-items-center ms-auto"),
    ], className="navbar fixed-top px-4 py-3 glass-nav")

app.layout = html.Div([
    navbar(),
    html.Div([
        dash.page_container
    ], className="content-container", style={"paddingTop": "120px"})
])

if __name__ == "__main__":
    app.run(debug=True, port=8051, use_reloader=False)
