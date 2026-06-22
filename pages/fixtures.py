import dash
from dash import html
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data

dash.register_page(__name__, path='/fixtures', title='TactIQ | Fixtures')


def layout():
    content = [
        html.Header([
            html.Div("2025-2026 Season", className="tq-eyebrow"),
            html.H1("Süper Lig Fixtures", className="tq-page-title tq-page-title--lg"),
            html.P("Complete schedule and match results.", className="tq-page-subtitle"),
        ], className="tq-page-header"),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Match Statistics", id="modal-title"), close_button=True),
            dbc.ModalBody(id="modal-body"),
            dbc.ModalFooter(
                dbc.Button("View Detailed Analysis", id="modal-analysis-btn", href="#",
                           external_link=True, color="danger")
            ),
        ], id="stats-modal", size="lg", is_open=False, centered=True),
    ]

    content.extend(_fixture_sections())

    return html.Div(content, className="tq-page tq-page--narrow")


def _stat_row(label, home_val, away_val):
    return html.Div([
        html.Span(home_val),
        html.Small(label, className="text-muted"),
        html.Span(away_val),
    ], className="tq-stat-row")


def _key_player_row(player):
    return html.Div([
        html.Span(player['name'], className="tq-keyplayer-row__name"),
        html.Span(player['reason'], className="tq-keyplayer-row__reason"),
    ], className="tq-keyplayer-row")


def _match_card(match):
    s1, s2 = match['stats']['team1'], match['stats']['team2']
    return html.Div([
        html.Div([
            html.Span("Finished", className="tq-match-card__status"),
            html.Span(f"{match['date']} • {match['time']}", className="tq-match-card__kickoff"),
        ], className="tq-match-card__meta"),

        html.Div([
            html.Div([
                html.Img(src=f"/{match['logos'][0]}", className="tq-match-card__logo"),
                html.Span(match['team_names'][0], className="tq-match-card__team-name"),
            ], className="tq-match-card__team"),

            html.Div(
                f"{s1['goals']} - {s2['goals']}",
                className="tq-match-card__score",
            ),

            html.Div([
                html.Img(src=f"/{match['logos'][1]}", className="tq-match-card__logo"),
                html.Span(match['team_names'][1], className="tq-match-card__team-name"),
            ], className="tq-match-card__team"),
        ], className="tq-match-card__teams"),

        html.Div(match['venue'], className="tq-match-card__venue"),

        html.Div([
            html.Hr(className="tq-match-card__hr"),

            html.Div([
                html.H5("Match Stats", className="tq-match-card__subheading"),
                html.Div([
                    _stat_row("Shots", s1['shots'], s2['shots']),
                    _stat_row("On Target", s1['shots_on_target'], s2['shots_on_target']),
                    _stat_row("Passes", s1['passes'], s2['passes']),
                    _stat_row("Pass Acc.", f"{s1['pass_accuracy']}%", f"{s2['pass_accuracy']}%"),
                ]),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.H5("Top Players", className="tq-match-card__subheading"),
                html.Div([_key_player_row(p) for p in match.get('key_players', [])[:3]]),
            ]),

            dbc.Button("Detailed Analysis →", href=f"/analysis/{match.get('source_file')}",
                       color="danger", size="sm",
                       style={"width": "100%", "marginTop": "20px"}),

        ], className="tq-match-card__details"),
    ], className="tq-match-card")


def _fixture_sections():
    matches = extract_fixture_data()

    weeks = {}
    for m in matches:
        weeks.setdefault(m['week'], []).append(m)

    content = []
    for week in sorted(weeks):
        content.append(html.H2(f"Week {week}", className="tq-week-heading"))
        content.append(html.Div(
            [_match_card(m) for m in weeks[week]],
            className="tq-matches-grid",
        ))

    if not content:
        return [html.Div("No matches found.", className="tq-plot-placeholder")]
    return content
