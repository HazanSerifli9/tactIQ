"""
Pitch coordinate constants — all values use the 0-100 normalised scale
where (0,0) is the bottom-left corner of the pitch and (100,100) is the
top-right corner.  Import from here instead of scattering magic numbers.

Usage:
    from shared.constants import HALFWAY_LINE, ATT_THIRD_MIN, BOX_Y_MIN
"""

# ---------------------------------------------------------------------------
# Pitch halves
# ---------------------------------------------------------------------------
HALFWAY_LINE: float = 50.0          # centre of the pitch (x-axis)
OWN_HALF_MAX: float = 50.0          # maximum x still in own half
ATTACKING_HALF_MIN: float = 50.0    # minimum x in attacking half

# ---------------------------------------------------------------------------
# Thirds (x-axis)
# ---------------------------------------------------------------------------
DEF_THIRD_MAX: float = 33.3         # defensive third upper boundary
MID_THIRD_MIN: float = 33.3         # middle third lower boundary
MID_THIRD_MAX: float = 66.6         # middle third upper boundary
ATT_THIRD_MIN: float = 66.6         # attacking / final third lower boundary

# ---------------------------------------------------------------------------
# Penalty boxes
# ---------------------------------------------------------------------------
DEF_BOX_X_MAX: float = 17.0        # own penalty box right edge
ATT_BOX_X_MIN: float = 83.0        # opponent penalty box left edge
BOX_Y_MIN: float = 21.1            # penalty box bottom edge
BOX_Y_MAX: float = 78.9            # penalty box top edge

# ---------------------------------------------------------------------------
# Goal mouth
# ---------------------------------------------------------------------------
GOAL_Y_MIN: float = 36.8
GOAL_Y_MAX: float = 63.2
GOAL_CENTER_Y: float = 50.0
GOAL_X_HOME: float = 100.0         # right-hand goal (attacking direction)
GOAL_X_AWAY: float = 0.0           # left-hand goal

# ---------------------------------------------------------------------------
# Zone 14 / Half-space (attacking)
# ---------------------------------------------------------------------------
ZONE14_X_MIN: float = 66.6
ZONE14_Y_MIN: float = 33.3
ZONE14_Y_MAX: float = 66.6

# ---------------------------------------------------------------------------
# Flanks (y-axis)
# ---------------------------------------------------------------------------
LEFT_FLANK_MAX: float = 33.3       # left flank upper boundary
RIGHT_FLANK_MIN: float = 66.6      # right flank lower boundary
CENTRAL_Y_MIN: float = 33.3
CENTRAL_Y_MAX: float = 66.6

# ---------------------------------------------------------------------------
# PPDA pressing thresholds
# ---------------------------------------------------------------------------
PPDA_OPP_HALF_MAX: float = 60.0    # opponent passes counted up to this x
PPDA_DEF_ACTION_MIN: float = 40.0  # defensive actions counted from this x

# ---------------------------------------------------------------------------
# Progressive pass threshold
# ---------------------------------------------------------------------------
PROGRESSIVE_PASS_RATIO: float = 0.75  # end_dist must be ≤ start_dist * ratio

# ---------------------------------------------------------------------------
# xA dangerous pass distance to goal
# ---------------------------------------------------------------------------
XA_DANGER_RADIUS: float = 25.0
