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
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Oswald:wght@500;700&display=swap"
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
TACTIQ_MAIN_URL = os.environ.get("TACTIQ_MAIN_URL", "http://127.0.0.1:8050/")

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
    brand = html.A([
        html.Img(src="/assets/logo.png", className="tq-nav-logo"),
        html.Span("tactIQ", className="logo-text"),
    ], href=TACTIQ_MAIN_URL, className="d-flex align-items-center text-decoration-none")

    links = html.Div([
        dcc.Link("Overview", href="/", className="nav-link custom-link highlight-link"),
        dcc.Link("Pre-Match", href="/pre-match", className="nav-link custom-link"),
        dcc.Link("Rival Scout", href="/rival-scout", className="nav-link custom-link"),
        dcc.Link("Post-Match", href="/post-match", className="nav-link custom-link"),
        dcc.Link("Trends", href="/trends", className="nav-link custom-link"),
    ], id="goz-nav-links", className="tq-nav-links")

    toggle = dbc.Button(
        html.Span(className="tq-burger"),
        id="goz-nav-toggle",
        className="tq-nav-toggle",
        color="link",
        n_clicks=0,
        title="Toggle navigation",
    )

    return html.Nav([brand, toggle, links],
                    className="navbar fixed-top px-4 py-3 glass-nav tq-nav")


@callback(
    Output("goz-nav-links", "className"),
    Input("goz-nav-toggle", "n_clicks"),
    State("goz-nav-links", "className"),
    prevent_initial_call=True,
)
def _toggle_nav(_n_clicks, current_class):
    classes = set((current_class or "").split())
    if "is-open" in classes:
        classes.discard("is-open")
    else:
        classes.add("is-open")
    return " ".join(classes)


app.layout = html.Div([
    navbar(),
    html.Div(dash.page_container, className="content-container tq-content"),
])

if __name__ == "__main__":
    app.run(debug=True, port=8051, use_reloader=False)
