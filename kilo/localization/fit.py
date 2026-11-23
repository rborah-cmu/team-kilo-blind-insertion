"""Fit the box pose (centre x, y and yaw) to probe contacts.

Same model as scripts/localize.py: the probe tip centre sits one shaft radius
outside the box outline, and the contact force roughly points out of the face.
Shared by the simulation and the real-robot experiments so both fit the same way.

All lengths in metres, angles in radians unless named *_deg.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

DEFAULT_FORCE_WEIGHT = 0.004  # localize.py's value; lower values did not generalize (docs/experiments/phase1_refit*)


@dataclass
class BoxFit:
    center: np.ndarray
    yaw: float
    max_gap: float                      # largest |distance - radius| over the contacts
    cost: float
    rivals: list = field(default_factory=list)  # other local minima: (center, yaw, cost, max_gap)

    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(self.yaw) % 180)

    def ambiguous(self, min_shift: float = 0.005, min_deg: float = 5.0, cost_ratio: float = 3.0) -> bool:
        """True if a clearly different pose explains the contacts nearly as well."""
        return any(cost <= cost_ratio * max(self.cost, 1e-12) and distinct(self.center, self.yaw, c, y, min_shift, min_deg)
                   for c, y, cost, _ in self.rivals)


def _axes(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def point_gaps(center, yaw, samples, half_sizes, radius) -> np.ndarray:
    """|distance from each contact to the box outline - probe radius|."""
    axes = _axes(yaw)
    local = (np.asarray(samples) - center) @ axes
    off = local - np.clip(local, -half_sizes, half_sizes)
    return np.abs(np.linalg.norm(off, axis=1) - radius)


def pose_distance(c1, y1, c2, y2) -> tuple[float, float]:
    """(centre distance in metres, yaw difference in degrees, modulo 180)."""
    dyaw = (np.degrees(y1 - y2) + 90) % 180 - 90
    return float(np.linalg.norm(np.asarray(c1) - c2)), float(abs(dyaw))


def distinct(c1, y1, c2, y2, min_shift: float = 0.005, min_deg: float = 5.0) -> bool:
    shift, deg = pose_distance(c1, y1, c2, y2)
    return shift >= min_shift or deg >= min_deg


def fit_box(samples, forces, half_sizes, radius, force_weight: float = DEFAULT_FORCE_WEIGHT,
            n_starts: int = 12) -> BoxFit:
    """Least-squares box pose from contact points and contact forces.

    Args:
        samples: (n, 2) probe-tip centres at contact.
        forces: (n, 2) contact forces on the probe (any scale; only direction is used).
        half_sizes: box half extents (x, y) in the box frame.
        radius: probe shaft radius.
        force_weight: weight on matching force directions to face normals.
        n_starts: yaw initializations spread over 180 degrees.
    """
    samples = np.asarray(samples, dtype=float)
    forces = np.asarray(forces, dtype=float)
    half_sizes = np.asarray(half_sizes, dtype=float)
    if len(samples) < 3:
        raise ValueError("need at least 3 contacts to fit a box")
    normals = forces / np.maximum(np.linalg.norm(forces, axis=1, keepdims=True), 1e-12)

    def residual(pose):
        center, angle = pose[:2], pose[2]
        axes = _axes(angle)
        local = (samples - center) @ axes
        off = local - np.clip(local, -half_sizes, half_sizes)
        d = np.linalg.norm(off, axis=1)
        pred = off @ axes.T / np.maximum(d[:, None], 1e-9)
        return np.r_[d - radius, (force_weight * (pred - normals)).ravel()]

    fits = sorted((least_squares(residual, np.r_[samples.mean(axis=0), y])
                   for y in np.linspace(0, np.pi, n_starts, endpoint=False)), key=lambda f: f.cost)
    best = fits[0]
    rivals = []
    for f in fits[1:]:
        c, y = f.x[:2], f.x[2]
        if not distinct(best.x[:2], best.x[2], c, y) or any(not distinct(c, y, rc, ry) for rc, ry, _, _ in rivals):
            continue
        rivals.append((c.copy(), float(y), float(f.cost), float(point_gaps(c, y, samples, half_sizes, radius).max())))
    return BoxFit(best.x[:2].copy(), float(best.x[2]), float(point_gaps(best.x[:2], best.x[2], samples, half_sizes, radius).max()),
                  float(best.cost), rivals)


def box_faces(center, yaw, half_sizes):
    """The four faces as (outward normal, face centre), in the base frame."""
    axes = _axes(yaw)
    faces = []
    for axis in (0, 1):
        for sign in (1.0, -1.0):
            n = sign * axes[:, axis]
            faces.append((n, np.asarray(center) + n * half_sizes[axis]))
    return faces
