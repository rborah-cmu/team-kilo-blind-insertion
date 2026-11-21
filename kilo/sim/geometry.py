"""Convex collision geometry for the peg and the base with a hole.

MuJoCo collides every mesh as its convex hull, so a block with a hole must be
split into convex pieces. The base is cut into angular wedges around the hole
centre; each wedge's inner face is a plane tangent to the hole circle, so the
simulated hole is never smaller than the CAD hole (at most r / cos(pi / n) - r
larger at the wedge seams, 0.03 mm for r = 12.7 mm and n = 48).

All lengths in metres.
"""

from __future__ import annotations

import numpy as np


def peg_shaft_vertices(radius: float, length: float, chamfer: float, n: int = 64) -> np.ndarray:
    """Vertices of the chamfered cylindrical shaft, hanging down from z = 0.

    The circle is inscribed (vertices on the radius), so the simulated shaft is
    never larger than the CAD shaft.
    """
    a = 2.0 * np.pi * np.arange(n) / n
    ring = np.stack([np.cos(a), np.sin(a)], axis=1)
    top = np.column_stack([radius * ring, np.zeros(n)])
    body = np.column_stack([radius * ring, np.full(n, -(length - chamfer))])
    tip = np.column_stack([(radius - chamfer) * ring, np.full(n, -length)])
    return np.vstack([top, body, tip])


def _ray_rect_exit(origin: np.ndarray, angle: float, half_x: float, half_y: float) -> np.ndarray:
    """Point where a ray from ``origin`` (inside the rectangle) leaves it."""
    d = np.array([np.cos(angle), np.sin(angle)])
    ts = []
    for axis, half in ((0, half_x), (1, half_y)):
        if abs(d[axis]) > 1e-12:
            for bound in (-half, half):
                t = (bound - origin[axis]) / d[axis]
                if t > 0:
                    ts.append(t)
    return origin + min(ts) * d


def base_wedges(
    size: tuple[float, float, float],
    hole_center: tuple[float, float],
    hole_radius: float,
    chamfer: float,
    n: int = 48,
) -> list[np.ndarray]:
    """Convex wedges that together make a block with a vertical through-hole.

    Args:
        size: full block size (x, y, z); the block spans [-x/2, x/2] x [-y/2, y/2] x [0, z].
        hole_center: (x, y) of the hole centre in the block frame.
        hole_radius: hole radius.
        chamfer: 45-degree chamfer on the hole's top edge.
        n: number of wedges.

    Returns:
        A list of (k, 3) vertex arrays, one per wedge.
    """
    hx, hy, height = size[0] / 2.0, size[1] / 2.0, size[2]
    c = np.asarray(hole_center, dtype=float)
    if not (abs(c[0]) + hole_radius + chamfer < hx and abs(c[1]) + hole_radius + chamfer < hy):
        raise ValueError("hole does not fit inside the block")
    corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    corner_angles = np.mod(np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0]), 2.0 * np.pi)
    stretch = 1.0 / np.cos(np.pi / n)  # puts the wedge's inner face tangent to the circle

    wedges = []
    for k in range(n):
        a0, a1 = 2.0 * np.pi * k / n, 2.0 * np.pi * (k + 1) / n
        outer = [_ray_rect_exit(c, a0, hx, hy)]
        outer += [corners[i] for i in np.argsort(corner_angles) if a0 < corner_angles[i] < a1]
        outer.append(_ray_rect_exit(c, a1, hx, hy))
        verts = []
        for p in outer:
            verts += [[p[0], p[1], 0.0], [p[0], p[1], height]]
        for a in (a0, a1):
            u = np.array([np.cos(a), np.sin(a)])
            low = c + hole_radius * stretch * u
            top = c + (hole_radius + chamfer) * stretch * u
            verts += [[low[0], low[1], 0.0], [low[0], low[1], height - chamfer], [top[0], top[1], height]]
        wedges.append(np.array(verts))
    return wedges
