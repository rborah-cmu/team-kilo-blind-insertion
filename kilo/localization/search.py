"""Plan the initial snake pattern that finds the plate by touch.

The probe and the pin are the same part. Sweep the workspace in
``common.SEARCH_SPACE_CORNERS`` at ``EE_FLOAT_HEIGHT`` above the tabletop,
and treat a Cartesian force above ``FORCE_THRESHOLD`` as contact.
"""

import numpy as np


def generate_snake_waypoints(spacing: float):
    pass


def search_line_from(line: np.ndarray):
    pass
