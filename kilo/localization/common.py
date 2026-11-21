"""Workspace and probe constants for the initial plate search.

Shared numbers also live in ``configs/default.yaml`` under ``snake_search``.
Keep the two copies the same; ``tests/test_search.py`` checks that they match.
"""

import numpy as np

# The corners of the search space to probe in, in meters.
SEARCH_SPACE_CORNERS = np.array([
    [0.20, -0.20],
    [0.70, -0.20],
    [0.70, 0.20],
    [0.20, 0.20],
])

TABLETOP_HEIGHT = 0.40  # meters | the z coordinate of the tabletop
EE_FLOAT_HEIGHT = 0.03  # meters | the z coordinate above the tabletop to search at

# Bottom of the probe to the middle of the grasp region:
# shaft length (38.10 mm) plus half the square top (19.05 mm).
VIRTUAL_PROBE_LENGTH = (38.10 + 19.05 / 2) / 1000  # meters

FORCE_THRESHOLD = 2.0  # Newtons | Cartesian force that counts as hitting a barrier
