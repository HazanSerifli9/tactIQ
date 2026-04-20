import dash
from dash import html

dash.register_page(__name__, path='/post-match', title='Göztepe Hub | Post-Match')

def layout():
    return html.Div(
        style={"textAlign": "center", "padding": "100px 20px", "color": "#888"},
        children=[
            html.H2("Coming Soon", style={"color": "#fbbf24", "marginBottom": "10px"}),
            html.P("Post-Match Analysis is currently unavailable."),
        ]
    )
