import io
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import Pitch

warnings.filterwarnings("ignore")

_cached_game = None


def load_open_game():
    global _cached_game
    if _cached_game is None:
        from databallpy import get_open_game
        _cached_game = get_open_game(verbose=False, use_cache=True)
    return _cached_game


def get_game_metadata():
    g = load_open_game()
    td = g.tracking_data
    alive = td[td["ball_status"] == "alive"]
    return {
        "home": g.home_team_name,
        "away": g.away_team_name,
        "home_score": g.home_score,
        "away_score": g.away_score,
        "home_formation": g.home_formation,
        "away_formation": g.away_formation,
        "pitch": g.pitch_dimensions,
        "frame_rate": g.frame_rate,
        "total_frames": len(alive),
        "alive_index": alive.index.tolist(),
    }


def get_player_columns(td_frame):
    home_cols = [c[:-2] for c in td_frame.index if c.endswith("_x") and c.startswith("home_") and pd.notna(td_frame[c])]
    away_cols = [c[:-2] for c in td_frame.index if c.endswith("_x") and c.startswith("away_") and pd.notna(td_frame[c])]
    return home_cols, away_cols


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def render_frame(frame_idx: int, show_pitch_control: bool = True):
    g = load_open_game()
    td = g.tracking_data
    pitch_length, pitch_width = g.pitch_dimensions

    frame = td.iloc[frame_idx]
    home_cols, away_cols = get_player_columns(frame)

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#1a1a2e",
        line_color="#4a5568",
        goal_type="box",
        linewidth=1.5,
    )

    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    if show_pitch_control:
        try:
            pc = g.tracking_data.get_pitch_control(
                start_idx=frame_idx,
                end_idx=frame_idx + 1,
                n_x_bins=106,
                n_y_bins=68,
            )
            if pc.ndim == 3:
                pc = pc[0]

            home_color = np.array([0.18, 0.55, 0.9, 1.0])
            away_color = np.array([0.9, 0.3, 0.3, 1.0])
            rgb = np.zeros((*pc.shape, 4))
            home_mask = pc > 0.5
            away_mask = pc < 0.5
            alpha_home = np.clip((pc - 0.5) * 2, 0, 1)
            alpha_away = np.clip((0.5 - pc) * 2, 0, 1)
            rgb[home_mask] = home_color * alpha_home[home_mask, None]
            rgb[away_mask] = away_color * alpha_away[away_mask, None]
            rgb[home_mask, 3] = alpha_home[home_mask] * 0.55
            rgb[away_mask, 3] = alpha_away[away_mask] * 0.55

            ax.imshow(
                rgb,
                extent=[-pitch_length / 2, pitch_length / 2, -pitch_width / 2, pitch_width / 2],
                origin="lower",
                aspect="auto",
                zorder=1,
            )
        except Exception:
            pass

    # Draw home players
    for col in home_cols:
        x, y = frame.get(f"{col}_x"), frame.get(f"{col}_y")
        if pd.notna(x) and pd.notna(y):
            ax.scatter(x, y, s=200, c="#2d9cdb", edgecolors="white", linewidths=1.5, zorder=4)
            num = col.split("_")[-1]
            ax.text(x, y, num, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=5)

    # Draw away players
    for col in away_cols:
        x, y = frame.get(f"{col}_x"), frame.get(f"{col}_y")
        if pd.notna(x) and pd.notna(y):
            ax.scatter(x, y, s=200, c="#eb5757", edgecolors="white", linewidths=1.5, zorder=4)
            num = col.split("_")[-1]
            ax.text(x, y, num, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=5)

    # Draw ball
    bx, by = frame.get("ball_x"), frame.get("ball_y")
    if pd.notna(bx) and pd.notna(by):
        ax.scatter(bx, by, s=120, c="white", edgecolors="#f9ca24", linewidths=2.5, zorder=6)

    g_meta = get_game_metadata()
    home_name = g_meta["home"].replace(" ", "\n")
    away_name = g_meta["away"].replace(" ", "\n")

    legend_patches = [
        plt.scatter([], [], s=150, c="#2d9cdb", edgecolors="white", linewidths=1.5, label=g.home_team_name),
        plt.scatter([], [], s=150, c="#eb5757", edgecolors="white", linewidths=1.5, label=g.away_team_name),
        plt.scatter([], [], s=80, c="white", edgecolors="#f9ca24", linewidths=2, label="Ball"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        facecolor="#0f0f1a",
        edgecolor="#4a5568",
        labelcolor="white",
        fontsize=9,
    )

    period = frame.get("period_id", "?")
    gametime = frame.get("gametime_td", "")
    ax.set_title(
        f"Period {period}  |  {gametime}  |  {g.home_team_name} {g.home_score} – {g.away_score} {g.away_team_name}",
        color="white",
        fontsize=11,
        pad=10,
    )

    return fig_to_base64(fig)
