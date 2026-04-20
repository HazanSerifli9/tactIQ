import sys
import os

# Ensure tactıq is at front of sys path, exactly like app.py
sys.path.insert(0, os.path.abspath('.'))
# (göztepehub is implicitly in sys.path when running from inside the directory, but let's append it to simulate that)
sys.path.append(os.path.abspath('göztepehub'))

import dash
# Run Dash from göztepehub folder to simulate running app.py
app = dash.Dash(__name__, use_pages=True, pages_folder='göztepehub/pages')

from göztepehub.pages.goztepe import layout
try:
    print("Testing layout function...")
    layout_result = layout()
    print("Layout function executed successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
