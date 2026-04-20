import plotly.graph_objects as go
import numpy as np

def draw_pitch(fig=None, theme='green'):
    if fig is None:
        fig = go.Figure()

    PITCH_LENGTH = 100
    PITCH_WIDTH = 100

    if theme == 'dark':
        bg_color = "rgba(0,0,0,0)"
        line_color = "rgba(255,255,255,0.15)"
        grid_color = "rgba(255,255,255,0.05)"
    else:
        bg_color = "#14532d"
        line_color = "#f9fafb"
        grid_color = "rgba(209,250,229,0.3)"

    # Background
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH,
        fillcolor=bg_color,
        layer="below",
        line=dict(width=0),
    )

    # 6x4 Tactical Grid
    n_vertical = 6
    n_horizontal = 4

    for i in range(1, n_vertical):
        x = i * PITCH_LENGTH / n_vertical
        fig.add_shape(
            type="line",
            x0=x, y0=0, x1=x, y1=PITCH_WIDTH,
            line=dict(color=grid_color, width=1, dash="dot"),
            layer="below",
        )

    for j in range(1, n_horizontal):
        y = j * PITCH_WIDTH / n_horizontal
        fig.add_shape(
            type="line",
            x0=0, y0=y, x1=PITCH_LENGTH, y1=y,
            line=dict(color=grid_color, width=1, dash="dot"),
            layer="below",
        )

    # Main field lines
    field_lines = [
        [[0, 0], [0, PITCH_WIDTH]],
        [[0, PITCH_WIDTH], [PITCH_LENGTH, PITCH_WIDTH]],
        [[PITCH_LENGTH, PITCH_WIDTH], [PITCH_LENGTH, 0]],
        [[PITCH_LENGTH, 0], [0, 0]],
        # Halfway line
        [[PITCH_LENGTH / 2, 0], [PITCH_LENGTH / 2, PITCH_WIDTH]],
        # Penalty areas (16.5m approximation for 100x100 Opta)
        [[17.0, (PITCH_WIDTH / 2) - 21.1], [17.0, (PITCH_WIDTH / 2) + 21.1]],
        [[PITCH_LENGTH - 17.0, (PITCH_WIDTH / 2) - 21.1], [PITCH_LENGTH - 17.0, (PITCH_WIDTH / 2) + 21.1]],
        [[0, (PITCH_WIDTH / 2) - 21.1], [17.0, (PITCH_WIDTH / 2) - 21.1]],
        [[0, (PITCH_WIDTH / 2) + 21.1], [17.0, (PITCH_WIDTH / 2) + 21.1]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) - 21.1], [PITCH_LENGTH - 17.0, (PITCH_WIDTH / 2) - 21.1]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) + 21.1], [PITCH_LENGTH - 17.0, (PITCH_WIDTH / 2) + 21.1]],
        # 6-yard boxes (5.5m approx)
        [[5.8, (PITCH_WIDTH / 2) - 9.1], [5.8, (PITCH_WIDTH / 2) + 9.1]],
        [[PITCH_LENGTH - 5.8, (PITCH_WIDTH / 2) - 9.1], [PITCH_LENGTH - 5.8, (PITCH_WIDTH / 2) + 9.1]],
        [[0, (PITCH_WIDTH / 2) - 9.1], [5.8, (PITCH_WIDTH / 2) - 9.1]],
        [[0, (PITCH_WIDTH / 2) + 9.1], [5.8, (PITCH_WIDTH / 2) + 9.1]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) - 9.1], [PITCH_LENGTH - 5.8, (PITCH_WIDTH / 2) - 9.1]],
        [[PITCH_LENGTH, (PITCH_WIDTH / 2) + 9.1], [PITCH_LENGTH - 5.8, (PITCH_WIDTH / 2) + 9.1]],
    ]

    for line in field_lines:
        fig.add_trace(go.Scatter(
            x=[line[0][0], line[1][0]],
            y=[line[0][1], line[1][1]],
            mode="lines",
            line=dict(color=line_color, width=1.5 if theme=='dark' else 2),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Center circle
    circle_x = [PITCH_LENGTH / 2 + 9.15 * np.cos(t) for t in np.linspace(0, 2 * np.pi, 200)]
    circle_y = [PITCH_WIDTH / 2 + 9.15 * np.sin(t) for t in np.linspace(0, 2 * np.pi, 200)]

    fig.add_trace(go.Scatter(
        x=circle_x,
        y=circle_y,
        mode="lines",
        line=dict(color=line_color, width=1.5 if theme=='dark' else 2),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Center spot and penalty spots
    fig.add_trace(go.Scatter(
        x=[PITCH_LENGTH / 2, 11.5, PITCH_LENGTH - 11.5],
        y=[PITCH_WIDTH / 2, PITCH_WIDTH / 2, PITCH_WIDTH / 2],
        mode="markers",
        marker=dict(color=line_color, size=4),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Goals
    goal_center_y = PITCH_WIDTH / 2.0
    goal_half_w = 7.32 / 2.0 * (100/68) # adjust for opta coords
    goal_y_bottom = goal_center_y - goal_half_w
    goal_y_top = goal_center_y + goal_half_w
    GOAL_DEPTH = 2.0
    GOAL_MARGIN = 0.2

    # Right Goal
    goal_right_x_front = PITCH_LENGTH - GOAL_MARGIN
    goal_right_x_back = goal_right_x_front + GOAL_DEPTH

    for (x0, y0, x1, y1) in [
        (goal_right_x_front, goal_y_bottom, goal_right_x_back, goal_y_bottom),
        (goal_right_x_front, goal_y_top, goal_right_x_back, goal_y_top),
        (goal_right_x_back, goal_y_bottom, goal_right_x_back, goal_y_top),
    ]:
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#f9fafb", width=3),
            layer="above"
        )

    # Left Goal
    goal_left_x_front = GOAL_MARGIN
    goal_left_x_back = goal_left_x_front - GOAL_DEPTH

    for (x0, y0, x1, y1) in [
        (goal_left_x_front, goal_y_bottom, goal_left_x_back, goal_y_bottom),
        (goal_left_x_front, goal_y_top, goal_left_x_back, goal_y_top),
        (goal_left_x_back, goal_y_bottom, goal_left_x_back, goal_y_top),
    ]:
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#f9fafb", width=3),
            layer="above"
        )
        
    # Build-up zone markers (33.3 and 66.6 horizontal lines to show L/C/R)
    # The user provided code uses grid, but we should make sure our L/C/R zones are somewhat visible.
    # We already have a 6x4 tactical grid, which divides the pitch.

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3, PITCH_LENGTH + 3],
            fixedrange=True,
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3, PITCH_WIDTH + 3],
            fixedrange=True,
            scaleanchor="x",
            scaleratio=0.68,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
    )

    return fig
