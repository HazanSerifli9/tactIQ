import sys
import os

# Add paths
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./göztepehub'))

# Mock dash before any import
import dash
dash.register_page = lambda *args, **kwargs: None

import pandas as pd
from göztepehub.pages.pre_match import _get_season_stats

try:
    print("Testing _get_season_stats for 'Alanyaspor Kulübü'...")
    stats = _get_season_stats('Alanyaspor Kulübü')
    print("Keys in stats:", list(stats.keys()))
    print("transitions_att count:", len(stats.get('transitions_att', {})))
    print("transitions_def count:", len(stats.get('transitions_def', {})))
    
    # Test building content for off-trans-tab
    from göztepehub.pages.pre_match import _build_tab_content
    print("Building tab content for off-trans-tab...")
    content_off = _build_tab_content('off-trans-tab', stats, 'Alanyaspor', 'Alanyaspor Kulübü')
    print("Success off-trans-tab!")
    
    print("Building tab content for def-trans-tab...")
    content_def = _build_tab_content('def-trans-tab', stats, 'Alanyaspor', 'Alanyaspor Kulübü')
    print("Success def-trans-tab!")
except Exception as e:
    import traceback
    traceback.print_exc()
