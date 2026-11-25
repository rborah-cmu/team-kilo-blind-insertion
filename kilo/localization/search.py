"""Plan the initial snake pattern that finds the plate by touch.

The probe and the pin are the same part. Sweep the workspace in
``common.SEARCH_SPACE_CORNERS`` at ``SEARCH_MODE_OFFSET`` above the tabletop,
and treat a Cartesian force above ``FORCE_THRESHOLD`` as contact.
"""

from __future__ import annotations

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

# Upon collision, call this function with ray direction [fx, fy] and ray origin as EE [x,y]
def generate_ray_probe_waypoints(
    ray: np.ndarray, bounding_box: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return two wall waypoints spanning the ray's line through the box.

    ray: np.array([[origin_x, origin_y], [direction_x, direction_y]])
    The collision origin is inside the box and the direction is nonzero.
    Start at the forward wall intersection, then probe to the opposite wall.
    """
    ray = np.asarray(ray, dtype=float)
    start_waypoint = get_ray_intersection(ray, bounding_box)
    end_waypoint = get_ray_intersection(np.array([ray[0], -ray[1]]), bounding_box)
    return start_waypoint, end_waypoint

def get_ray_intersection(
    ray: np.ndarray, bounding_box: np.ndarray
) -> np.ndarray | None:
    """Return the nearest strictly forward intersection with the box boundary.

    ray: np.array([[origin_x, origin_y], [direction_x, direction_y]])
    bounding_box: as defined in common.SEARCH_SPACE_CORNERS
    
    An interior or boundary origin returns the exit point ahead of it.
    Boundary rays pointing outward return None, as do rays missing the box.
    """
    ray = np.asarray(ray, dtype=float)
    bounding_box = np.asarray(bounding_box, dtype=float)

    origin, direction = ray
    lower = bounding_box.min(axis=0)
    upper = bounding_box.max(axis=0)

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

    if t_enter > t_exit or t_exit <= 0:
        return None
    inside = np.all((lower <= origin) & (origin <= upper))
    t = t_exit if inside else t_enter
    return origin + t * direction
