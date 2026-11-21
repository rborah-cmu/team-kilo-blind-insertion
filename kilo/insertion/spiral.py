"""Scripted spiral search: the Phase 2 baseline and the base the RL policy corrects."""

from __future__ import annotations

import numpy as np


def spiral_waypoints(center: np.ndarray, pitch: float, max_radius: float, step: float) -> np.ndarray:
    """Archimedean spiral of (x, y) waypoints around ``center``, evenly spaced along the path.

    Args:
        center: (x, y) start of the spiral, e.g. the estimated hole centre.
        pitch: distance between successive turns; keep it below the hole
            clearance plus tip rounding so no gap is skipped.
        max_radius: stop once the spiral reaches this radius.
        step: approximate path length between waypoints.

    Returns:
        (N, 2) array of waypoints, starting at ``center``.
    """
    if pitch <= 0 or step <= 0 or max_radius <= 0:
        raise ValueError("pitch, step and max_radius must be positive")
    b = pitch / (2.0 * np.pi)  # r = b * theta
    points = [np.zeros(2)]
    theta = 0.0
    while b * theta < max_radius:
        r = b * theta
        # Arc length per radian is sqrt(r^2 + b^2); advance theta by about one step.
        theta += step / np.hypot(r, b)
        r = min(b * theta, max_radius)
        points.append(np.array([r * np.cos(theta), r * np.sin(theta)]))
    return np.asarray(center, dtype=float) + np.array(points)
