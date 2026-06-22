# tactIQ

Football analytics platform for the Turkish Süper Lig, built with Dash. Includes a general-purpose match/team/player explorer and a dedicated **Göztepe Hub** with pre-match scouting, post-match reports, and trend tracking.

## Apps

The repository ships two separate Dash applications:

| App           | Entry point         | Default port | Pages                                                                                       |
| ------------- | ------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| tactIQ        | `app.py`            | 8050         | home, analysis, fixtures, team analysis, player analysis, transitions, wyscout, standings, match report |
| Göztepe Hub   | `göztepehub/app.py` | 8051         | landing, pre-match, post-match, rival scout, trends                                         |

Both apps share `utils/` (analysis + visualizations) and `shared/` (logger, constants).

## Project layout

```
.
├── app.py                       # tactIQ Dash app
├── pages/                       # tactIQ page modules (auto-loaded by Dash)
├── utils/                       # shared analysis + visualization modules
├── shared/                      # logger, constants
├── assets/                      # logos and CSS (gitignored)
├── raw_data/                    # match data (gitignored)
└── göztepehub/
    ├── app.py                   # Göztepe Hub Dash app
    ├── pages/                   # Göztepe Hub page modules
    └── utils/                   # Göztepe-specific helpers (some re-export from ../utils)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`raw_data/` and `assets/` are private and not committed. The apps expect:

- Match event data as `.parquet` files under `raw_data/`
- Team logos under `assets/` and `göztepehub/assets/`
- Two trained models under `utils/`: `tactiq_xg_model.json`, `tactiq_obv_for_model.json`, `tactiq_obv_against_model.json`

## Running

Run each app separately:

```bash
# tactIQ (default port 8050)
python app.py

# Göztepe Hub (port 8051 — hardcoded in göztepehub/app.py)
python göztepehub/app.py
```

The tactIQ navbar links to `http://127.0.0.1:8051` for the Göztepe Hub, so start both when you want cross-navigation.

## Dependencies

Core stack:

- **Dash** + **dash-bootstrap-components** + **plotly** — web app and interactive charts
- **pandas**, **numpy**, **scipy**, **pyarrow** — data processing and parquet I/O
- **matplotlib**, **seaborn**, **mplsoccer**, **highlight-text**, **Pillow** — static pitch plots and rendering
- **scikit-learn**, **xgboost** — xG and OBV models
