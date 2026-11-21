"""Workspace and probe constants for the initial plate search.

Shared numbers also live in ``configs/default.yaml`` under ``snake_search``.
Keep the two copies the same; ``tests/test_search.py`` checks that they match.
"""

import numpy as np

# The corners of the search space to probe in, in meters.
SEARCH_SPACE_CORNERS = np.array([
    [0.32, -0.20],
    [0.70, -0.20],
    [0.70, 0.20],
    [0.32, 0.20],
])

TABLETOP_HEIGHT = 0.0001  # meters | the z coordinate of the tabletop
SEARCH_MODE_OFFSET = 0.03  # meters | the z coordinate above the tabletop to search at
TRAVEL_MODE_OFFSET = 0.10 # meters | the z coordinate above the tabletop to freely traverse over the plate

# Bottom of the probe to the middle of the grasp region:
# shaft length (38.10 mm) plus half the square top (19.05 mm).
VIRTUAL_PROBE_LENGTH = (38.10 + 19.05 / 2) / 1000  # meters

# When searching/probing, the EE height should be SEARCH_MODE_HEIGHT
# When traveling, e.g. going to the other side of the plate, the EE height should be TRAVEL_MODE_HEIGHT
# When switching between modes, simply move vertically to the desired height
SEARCH_MODE_HEIGHT = VIRTUAL_PROBE_LENGTH + SEARCH_MODE_OFFSET
TRAVEL_MODE_HEIGHT = VIRTUAL_PROBE_LENGTH + TRAVEL_MODE_OFFSET

SEARCH_SPEED = 0.015  # m/s during contact sensing
TRAVEL_SPEED = 0.030  # m/s during lift, traverse, and descent

FORCE_THRESHOLD = 2.0  # Newtons | Cartesian force that counts as hitting a barrier

PROBE_LOOKAHEAD = 0.10 # meters | The distance to position the target position ahead of the current position
POS_EPS = 0.01 # meters | Target is reached when the error is less than this (endpoint)
