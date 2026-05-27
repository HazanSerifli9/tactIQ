import sys
import os

# Add the project directory to sys.path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats_visuals import (
    generate_league_top_players_plot,
    generate_team_ball_winning_plot,
    generate_team_common_zonal_actions_plot,
    generate_team_threat_creation_plot,
    generate_team_space_allowed_plot,
    generate_tactical_style_scatter,
    generate_style_metrics_table,
)
from utils.zonal_data import process_zonal_data
from utils.zonal_visuals import generate_zonal_map
from utils.box_entry_data import process_box_entry_data
from utils.box_entry_visuals import generate_box_entry_grid

def test_other_plots():
    funcs = [
        ("generate_league_top_players_plot", generate_league_top_players_plot),
        ("generate_team_ball_winning_plot", generate_team_ball_winning_plot),
        ("generate_team_common_zonal_actions_plot", generate_team_common_zonal_actions_plot),
        ("generate_team_threat_creation_plot", generate_team_threat_creation_plot),
        ("generate_team_space_allowed_plot", generate_team_space_allowed_plot),
        ("generate_tactical_style_scatter", generate_tactical_style_scatter),
        ("generate_style_metrics_table", generate_style_metrics_table),
    ]

    for name, func in funcs:
        try:
            print(f"Testing {name}...")
            res = func()
            print(f"  {name} success: returned length {len(res) if res else 0}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    try:
        print("Testing zonal map...")
        zonal_grid, rows, cols = process_zonal_data()
        res = generate_zonal_map(zonal_grid, rows, cols)
        print(f"  generate_zonal_map success: returned length {len(res) if res else 0}")
    except Exception as e:
        print(f"  generate_zonal_map FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("Testing box entry map...")
        box_entries = process_box_entry_data()
        res = generate_box_entry_grid(box_entries)
        print(f"  generate_box_entry_grid success: returned length {len(res) if res else 0}")
    except Exception as e:
        print(f"  generate_box_entry_grid FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_other_plots()
