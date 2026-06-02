import sys
import os

# Add paths
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./göztepehub'))

# Mock dash before any import
import dash
dash.register_page = lambda *args, **kwargs: None

import pandas as pd
from göztepehub.pages.pre_match import _get_season_stats, _build_tab_content, _build_buildup_pitch, _load_goal_timeline

try:
    print("Testing offensive tab details for 'Alanyaspor Kulübü'...")
    stats = _get_season_stats('Alanyaspor Kulübü')
    
    print("1. Testing _build_buildup_pitch...")
    coords = stats['buildup'].get('coords', [])
    print(f"Number of starting coordinates: {len(coords)}")
    b_plot_b64 = _build_buildup_pitch(coords)
    print(f"Buildup pitch successfully plotted! Length of base64 string: {len(b_plot_b64)}")
    
    print("2. Testing _build_tab_content for 'offensive-tab'...")
    content_off = _build_tab_content('offensive-tab', stats, 'Alanyaspor', 'Alanyaspor Kulübü')
    print("Success building 'offensive-tab' content!")
    
    print("3. Testing _load_goal_timeline and slicing...")
    goals = stats.get('goal_sequences', [])
    if goals:
        g = goals[0]
        print(f"Goal to inspect: Week {g['week']} by {g['player']} (event_id: {g['event_id']}, filename: {g['filename']})")
        timeline = _load_goal_timeline(g['filename'], g['event_id'], 'Alanyaspor Kulübü')
        print(f"Timeline size: {len(timeline)}")
        print("Sequence orders in timeline:", list(timeline['order']))
    else:
        print("No goal sequences found in stats.")
        
    print("\nALL OFFENSIVE TESTS COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
