import dash
from dash import html, dash_table
from utils.data import extract_fixture_data, calculate_standings

dash.register_page(__name__, path="/standings", title='TactIQ | Standings')

def layout():
    # Fetch data
    matches = extract_fixture_data(lite=True)
    df = calculate_standings(matches)
    
    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[
                    html.H1("League Standings", className="page-title"),
                    
                    html.Div(
                        className="standings-container",
                        children=[
                            dash_table.DataTable(
                                data=df.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in df.columns],
                                style_as_list_view=True,
                                style_header={
                                    'backgroundColor': '#1e1e1e',
                                    'fontWeight': 'bold',
                                    'color': 'white',
                                    'border': '1px solid #333'
                                },
                                style_cell={
                                    'backgroundColor': '#121212',
                                    'color': '#e0e0e0',
                                    'border': '1px solid #333',
                                    'textAlign': 'center',
                                    'padding': '10px'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': '#1a1a1a',
                                    },
                                    {
                                        'if': {'column_id': 'Team'},
                                        'textAlign': 'left',
                                        'paddingLeft': '20px',
                                        'fontWeight': 'bold'
                                    },
                                    {
                                        'if': {'column_id': 'Points'},
                                        'fontWeight': 'bold',
                                        'color': '#FFD700' # Gold for points
                                    }
                                ],
                            )
                        ]
                    )
                ]
            )
        ]
    )
