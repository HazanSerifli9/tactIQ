import sys
import os

# Set paths
sys.path.insert(0, '/Users/hazanserifli/Desktop/tactıq')
sys.path.insert(0, '/Users/hazanserifli/Desktop/tactıq/göztepehub')

import dash
# Mock __file__ for __main__ module to satisfy page registration checks
sys.modules['__main__'].__file__ = __file__

# Create app specifying correct pages folder
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder='/Users/hazanserifli/Desktop/tactıq/göztepehub/pages'
)

from pages.pre_match import _get_season_stats, _build_tab_content

# Kayserispor test
print("Starting tab rendering validation...")
stats = _get_season_stats('Kayserispor')
print("Season stats computed!")

content_off = _build_tab_content('offensive-tab', stats, 'Kayserispor', 'Kayserispor')
print("Offensive tab content loaded successfully!")

content_def = _build_tab_content('defensive-tab', stats, 'Kayserispor', 'Kayserispor')
print("Defensive tab content loaded successfully!")

content_off_tr = _build_tab_content('off-trans-tab', stats, 'Kayserispor', 'Kayserispor')
print("Offensive transitions tab loaded successfully!")

content_def_tr = _build_tab_content('def-trans-tab', stats, 'Kayserispor', 'Kayserispor')
print("Defensive transitions tab loaded successfully!")

content_set = _build_tab_content('set-pieces-tab', stats, 'Kayserispor', 'Kayserispor')
print("Set Pieces tab loaded successfully!")

print("\nSUCCESS: All 5 tabs rendered flawlessly with no errors or color exceptions!")
