import os
import io
import base64
import numpy as np
from typing import List, Dict, Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch
from PIL import Image

from utils.data import TEAM_LOGOS
from utils.cache import disk_cache
from shared.logger import get_logger

logger = get_logger(__name__)

TACTIQ_BG    = "#1a1a2e"
CARD_BG      = "#16213e"
ACCENT       = "#FDE636"
TEXT_WHITE   = "#ffffff"
TEXT_MUTED   = "#9ca3af"

THREAT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "ThreatCmap", [TACTIQ_BG, "#118ab2", "#06d6a0", "#ffd166"]
)


def _add_team_logo(ax, team_name: str, zoom: float = 0.055) -> None:
    logo_path = TEAM_LOGOS.get(team_name)
    if not logo_path:
        return
    full_path = os.path.abspath(logo_path)
    if not os.path.exists(full_path):
        return
    try:
        img = Image.open(full_path).convert("RGBA")
        imagebox = OffsetImage(img, zoom=zoom)
        # inside the dark label strip at bottom-right
        ab = AnnotationBbox(
            imagebox, (0.88, 0.11),
            xycoords="axes fraction",
            frameon=False,
            zorder=6,
        )
        ax.add_artist(ab)
    except Exception as e:
        logger.debug("Logo render skipped for %s: %s", team_name, e)


@disk_cache
def generate_box_entry_grid(
    players_data: List[Dict[str, Any]],
    league_name: str = "Süper Lig",
) -> Optional[str]:
    """4×5 grid of box-entry heatmaps, one cell per player."""
    if not players_data:
        return None

    pitch = VerticalPitch(
        pitch_type="opta",
        pitch_color=CARD_BG,
        line_color="#4b5563",
        line_zorder=2,
        half=True,
        linewidth=0.8,
    )

    fig, axs = pitch.grid(
        nrows=4, ncols=5,
        axis=False,
        title_height=0.07,
        endnote_height=0.03,
        grid_height=0.82,
        figheight=18,
    )
    fig.set_facecolor(TACTIQ_BG)

    axs_flat = axs["pitch"].flatten()

    for idx, ax in enumerate(axs_flat):
        if idx >= len(players_data):
            ax.set_facecolor(TACTIQ_BG)
            ax.axis("off")
            continue

        p = players_data[idx]
        x, y = np.array(p["x"]), np.array(p["y"])

        # ── KDE heatmap ────────────────────────────────────────────────
        if len(x) > 4:
            try:
                pitch.kdeplot(
                    x, y, ax=ax,
                    cmap=THREAT_CMAP,
                    fill=True, levels=12,
                    alpha=0.85, zorder=0,
                )
            except Exception as e:
                logger.debug("KDE skipped for %s: %s", p["name"], e)

        # ── Scatter dots ───────────────────────────────────────────────
        pitch.scatter(x, y, ax=ax, s=12, color="white", alpha=0.55, zorder=3)

        # ── Rank badge (top-left, axes fraction) ──────────────────────
        ax.text(
            0.04, 0.95, f"#{idx + 1}",
            transform=ax.transAxes,
            color=ACCENT, fontsize=9, fontweight="bold",
            va="top", ha="left", zorder=7,
            bbox=dict(facecolor="#111827", alpha=0.8,
                      edgecolor=ACCENT, boxstyle="round,pad=0.25", lw=0.8),
        )

        # ── Entry count badge (top-right, axes fraction) ───────────────
        ax.text(
            0.96, 0.95, f"{p['count']}",
            transform=ax.transAxes,
            color=TEXT_WHITE, fontsize=9, fontweight="bold",
            va="top", ha="right", zorder=7,
            bbox=dict(facecolor="#374151", alpha=0.85,
                      edgecolor="none", boxstyle="round,pad=0.25"),
        )

        # ── Player name + team — inside the pitch at the bottom ──────────
        name_parts = p["name"].split()
        if len(name_parts) > 1:
            short = f"{name_parts[0][0]}. {name_parts[-1]}"
        else:
            short = p["name"]

        team_short = (
            p["team"]
            .replace(" Kulübü", "").replace(" Spor", "")
            .replace(" Jimnastik", "").replace(" Futbol", "")
            .strip()
        )

        # Dark background strip across the bottom of the cell
        ax.add_patch(plt.Rectangle(
            (0, 0), 1, 0.22,
            transform=ax.transAxes,
            color="#0d1117", alpha=0.82, zorder=6,
        ))

        ax.text(
            0.5, 0.14, short,
            transform=ax.transAxes,
            color=TEXT_WHITE, fontsize=9, fontweight="bold",
            va="center", ha="center", zorder=8,
        )
        ax.text(
            0.5, 0.05, team_short,
            transform=ax.transAxes,
            color=TEXT_MUTED, fontsize=7.5,
            va="center", ha="center", zorder=8,
        )

        # ── Team logo (bottom-right corner, inside strip) ─────────────
        _add_team_logo(ax, p["team"], zoom=0.055)

    # ── Title area ─────────────────────────────────────────────────────
    title_ax = axs["title"]
    title_ax.set_facecolor(TACTIQ_BG)
    title_ax.axis("off")

    title_ax.text(
        0.0, 0.75,
        f"Top 20 Box Entry Sources  |  {league_name}  |  Season 2025/26",
        color=TEXT_WHITE, fontsize=20, fontweight="bold",
        va="center", ha="left", transform=title_ax.transAxes,
    )
    title_ax.text(
        0.0, 0.2,
        "Players ranked by successful passes or carries that enter the penalty area from outside",
        color=TEXT_MUTED, fontsize=11,
        va="center", ha="left", transform=title_ax.transAxes,
    )

    # ── Endnote / legend ───────────────────────────────────────────────
    endnote_ax = axs["endnote"]
    endnote_ax.set_facecolor(TACTIQ_BG)
    endnote_ax.axis("off")

    legend_elements = [
        Patch(facecolor="#ffd166", edgecolor="none", label="High Volume"),
        Patch(facecolor="#06d6a0", edgecolor="none", label="Moderate Volume"),
        Patch(facecolor="#118ab2", edgecolor="none", label="Low Volume"),
        Line2D([0], [0], marker="o", color="w", label="Pass Start",
               markerfacecolor="white", markersize=5, linestyle="None"),
    ]
    endnote_ax.legend(
        handles=legend_elements,
        loc="center left", bbox_to_anchor=(0.0, 0.5),
        ncol=4, frameon=False,
        labelcolor="white", fontsize=10,
        handlelength=1.2, columnspacing=1.5,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"
