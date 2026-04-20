import sys
import os

root = os.path.abspath('.')
hub = os.path.abspath('göztepehub')

sys.path = [root, hub] + sys.path

try:
    from utils.why_we_lose import calc_why_we_lose
    print("SUCCESS: calc_why_we_lose imported.")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
