"""Plan the initial snake pattern that finds the plate by touch.

The probe and the pin are the same part. Sweep the workspace in
``common.SEARCH_SPACE_CORNERS`` at ``EE_FLOAT_HEIGHT`` above the tabletop,
and treat a Cartesian force above ``FORCE_THRESHOLD`` as contact.
"""

import numpy as np


def generate_snake_waypoints(spacing: float) -> list[np.ndarray]:
    """Corner points of a square-wave sweep over ``common.SEARCH_SPACE_CORNERS``.

    The rectangle is axis-aligned. The first pass starts at the far corner
    (max x, min y) and runs toward the robot to (min x, min y); each next pass
    steps ``spacing`` toward +y and reverses direction. A final pass always
    runs along the max-y edge, so the last gap may be shorter than ``spacing``.

    Args:
        spacing: distance between neighbouring passes, in metres.

    Returns:
        List of (x, y) points; consecutive points are joined by straight moves.
    """
    from .common import SEARCH_SPACE_CORNERS

    if not spacing > 0:
        raise ValueError("spacing must be positive")

    corners = np.asarray(SEARCH_SPACE_CORNERS, dtype=float)
    (x_min, y_min), (x_max, y_max) = corners.min(axis=0), corners.max(axis=0)

    width = y_max - y_min
    num_steps = int(np.floor(width / spacing + 1e-9))
    ys = np.minimum(y_min + spacing * np.arange(num_steps + 1), y_max)
    if y_max - ys[-1] > 1e-9:
        ys = np.append(ys, y_max)

    waypoints = []
    for i, y in enumerate(ys):
        far, near = np.array([x_max, y]), np.array([x_min, y])
        waypoints.extend([far, near] if i % 2 == 0 else [near, far])
    return waypoints


def get_ray_intersection(
    ray: np.ndarray, bounding_box: np.ndarray
) -> np.ndarray | None:
    """Return the nearest forward intersection with the rectangle boundary.

    ``ray`` is ``[origin_xy, direction_xy]``; the direction need not be a unit
    vector. ``bounding_box`` contains the four axis-aligned rectangle corners,
    as in ``common.SEARCH_SPACE_CORNERS``. An interior origin returns the exit
    point; an origin on the boundary returns itself. A missed box returns None.
    """
    ray = np.asarray(ray, dtype=float)
    bounding_box = np.asarray(bounding_box, dtype=float)
    if ray.shape != (2, 2) or not np.isfinite(ray).all():
        raise ValueError("ray must contain a finite 2D origin and direction")
    if bounding_box.shape != (4, 2) or not np.isfinite(bounding_box).all():
        raise ValueError("bounding_box must contain four finite 2D corners")

    origin, direction = ray
    if not np.any(direction):
        raise ValueError("ray direction must be nonzero")
    lower = bounding_box.min(axis=0)
    upper = bounding_box.max(axis=0)
    if np.any(lower >= upper):
        raise ValueError("bounding_box must have positive width and height")

    # Intersect the ray's parameter intervals for the x and y slabs.
    t_enter, t_exit = -np.inf, np.inf
    for axis in range(2):
        if direction[axis] == 0:
            if not lower[axis] <= origin[axis] <= upper[axis]:
                return None
            continue
        t1 = (lower[axis] - origin[axis]) / direction[axis]
        t2 = (upper[axis] - origin[axis]) / direction[axis]
        t_enter = max(t_enter, min(t1, t2))
        t_exit = min(t_exit, max(t1, t2))

    if t_enter > t_exit or t_exit < 0:
        return None
    inside = np.all((lower <= origin) & (origin <= upper))
    if inside and np.any((origin == lower) | (origin == upper)):
        return origin.copy()
    t = t_exit if inside else t_enter
    return origin + t * direction
