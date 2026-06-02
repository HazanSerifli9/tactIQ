import sys
import os

# Add paths
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./göztepehub'))

# Mock dash before any import
import dash
dash.register_page = lambda *args, **kwargs: None

from göztepehub.pages.pre_match import _get_season_stats

try:
    print("Running set pieces count calculation check for 'Alanyaspor Kulübü'...")
    stats = _get_season_stats('Alanyaspor Kulübü')
    
    print("\nCalculated Set Pieces Stats:")
    print("Penalties:", stats['penalties'])
    print("Free Kicks Direct Shots & Goals:", stats['freekicks'])
    print("Corners Total & Trajectories:", stats['corners'])
    print("Goal Kicks Total:", stats['goalkicks'])
    print("\nSUCCESS!")
except Exception as e:
    import traceback
    traceback.print_exc()
