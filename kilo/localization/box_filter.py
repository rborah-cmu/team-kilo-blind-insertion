"""Particle filter over the box's pose (centre x, y and rotation) from side contacts.

TODO (Phase 1 lead):
- State: (x, y, yaw) of the box centre; prior uniform over the search region
  in ``configs/default.yaml``.
- Probing uses the side of the vertical cylindrical shaft. A sideways contact
  puts the wall one shaft radius (peg.shaft_radius, 12.5 mm) from the peg's
  axis along the direction of travel; convert to a wall point before updating.
- Contact update: score each particle by the distance from the wall point to
  the box outline at that pose (Gaussian in that distance).
- Free-path update: remove particles whose box would overlap the region the
  shaft swept without contact (the path inflated by the shaft radius).
- Next probe: the side whose predicted contact point varies most across particles.
- Stop when the centre's standard deviation is below about 2 mm. The hole search
  only needs the centre, since the hole's prior region is a disc around it.
"""

from __future__ import annotations

import numpy as np


class BoxPoseFilter:
    def __init__(self, box_size_xy: np.ndarray, search_region: dict, num_particles: int = 5000):
        raise NotImplementedError

    def update_contact(self, point: np.ndarray) -> None:
        raise NotImplementedError

    def update_free_path(self, start: np.ndarray, end: np.ndarray) -> None:
        raise NotImplementedError

    def next_probe(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (start point, direction) for the next approach-until-contact move."""
        raise NotImplementedError

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return mean (x, y, yaw) and covariance."""
        raise NotImplementedError
