"""Four-contact plate localization shared by simulation and hardware."""

import numpy as np
from scipy.optimize import least_squares

from kilo.config import load_config


def localize_plate(waypoints, probe, travel):
    """Sweep, sample the other faces, and fit the plate from four contacts."""
    for point in waypoints[1:]:
        hit = probe(point)
        if hit is not None:
            break
    else:
        raise RuntimeError("Sweep ended without touching the plate")

    def touch_from(origin, outward):
        _, target = travel(np.array([origin, outward]))
        print("Crossing from", origin, "toward", target, flush=True)
        contact = probe(target)
        if contact is None:
            raise RuntimeError("Crossing ended without touching the plate")
        print("Contact:", contact, flush=True)
        return contact

    first, force = hit
    print("First contact:", hit, flush=True)
    normal = force / np.linalg.norm(force)
    opposite, opposite_force = touch_from(first, -normal)
    midpoint = (first + opposite) / 2
    tangent = np.array([-normal[1], normal[0]])
    left, left_force = touch_from(midpoint, tangent)
    right, right_force = touch_from(midpoint, -tangent)

    # The tip centre sits one shaft radius outside the contacted face or corner.
    cfg = load_config()
    half_sizes = np.array(cfg["box"]["size"][:2]) / 2
    radius = cfg["peg"]["shaft_radius"]
    samples = np.array([first, opposite, left, right])
    forces = np.array([force, opposite_force, left_force, right_force])
    normals = forces / np.linalg.norm(forces, axis=1, keepdims=True)

    def residual(pose):
        center, angle = pose[:2], pose[2]
        c, s = np.cos(angle), np.sin(angle)
        axes = np.array([[c, -s], [s, c]])
        local = (samples - center) @ axes
        closest = np.clip(local, -half_sizes, half_sizes)
        offsets = local - closest
        distances = np.linalg.norm(offsets, axis=1)
        predicted = offsets @ axes.T / np.maximum(distances[:, None], 1e-9)
        # Forces can include friction, so use them primarily to resolve yaw.
        return np.r_[distances - radius, (.004 * (predicted - normals)).ravel()]

    fits = [least_squares(residual, np.r_[samples.mean(axis=0), yaw])
            for yaw in np.linspace(0, np.pi, 12, endpoint=False)]
    pose = min(fits, key=lambda fit: fit.cost).x
    return pose[:2], np.degrees(pose[2]) % 180
