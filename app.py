import os

from shared.matplotlib_config import configure_matplotlib

configure_matplotlib()

import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Oswald:wght@500;700&display=swap"
GOZTEPE_HUB_URL = os.environ.get("GOZTEPE_HUB_URL", "http://127.0.0.1:8051")

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.DARKLY, FONT_URL])
app.title = "tactIQ"
server = app.server


def navbar():
    brand = html.A([
        html.Img(src="/assets/logo.png", className="tq-nav-logo"),
        html.Span("tactIQ", className="logo-text"),
    ], href="/", className="d-flex align-items-center text-decoration-none")

    links = html.Div([
        dcc.Link("Home", href="/", className="nav-link custom-link"),
        dcc.Link("Matches", href="/analysis", className="nav-link custom-link"),
        dcc.Link("Teams", href="/team-analysis", className="nav-link custom-link"),
        dcc.Link("Players", href="/player-analysis", className="nav-link custom-link"),
        html.A("Göztepe Hub", href=GOZTEPE_HUB_URL, target="_blank",
               className="nav-link custom-link highlight-link"),
        dcc.Link("Fixtures", href="/fixtures", className="nav-link custom-link"),
    ], id="tq-nav-links", className="tq-nav-links")

    toggle = dbc.Button(
        html.Span(className="tq-burger"),
        id="tq-nav-toggle",
        className="tq-nav-toggle",
        color="link",
        n_clicks=0,
        title="Toggle navigation",
    )

    return html.Nav([brand, toggle, links],
                    className="navbar fixed-top px-4 py-3 glass-nav tq-nav")


@callback(
    Output("tq-nav-links", "className"),
    Input("tq-nav-toggle", "n_clicks"),
    State("tq-nav-links", "className"),
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
    app.run(debug=True)
