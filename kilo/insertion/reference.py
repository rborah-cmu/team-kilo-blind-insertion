"""Contact-responsive reference using only calibrated tool FK and surface height."""
from __future__ import annotations
import numpy as np
from kilo.insertion.spiral import spiral_waypoints


class InsertionReference:
    SEARCH, CAPTURE, INSERT = 0, 1, 2

    def __init__(self, cfg, estimate_xy, top_z, tip):
        self.cfg = cfg
        self.dt = 1.0 / cfg['policy_rate_hz']
        self.top_z = top_z
        self.path = spiral_waypoints(estimate_xy, cfg['spiral_pitch'], cfg['spiral_max_radius'],
                                     cfg['search_speed'] * self.dt)
        self.index = 0
        self.phase = self.SEARCH
        self.capture_steps = 0
        self.cooldown = 0
        self.lock_xy = np.asarray(tip[:2]).copy()
        self.z = float(tip[2])
        self.surface_z = None
        self.touch_steps = 0

    def step(self, tip, shaft_length, force_z):
        # Measure the surface in the calibrated TCP frame on first touch. Using
        # the fixture's world height directly confuses grasp-z error with entry.
        if self.surface_z is None:
            self.touch_steps = self.touch_steps + 1 if force_z > self.cfg['touch_force'] else 0
            if self.touch_steps >= self.cfg['touch_samples']:
                self.surface_z = float(tip[2])
            elif self.top_z - tip[2] > self.cfg['engagement_depth']:
                # A centered start may enter without ever touching the top.
                self.surface_z = self.top_z
            else:
                self._descend(tip, shaft_length, self.cfg['search_z_lead'])
                return np.r_[self.path[0], self.z]
        depth = self.surface_z - tip[2]
        self.cooldown = max(0, self.cooldown - 1)
        if self.phase == self.SEARCH and self.cooldown == 0 and depth > self.cfg['capture_depth']:
            self.phase = self.CAPTURE
            self.lock_xy = tip[:2].copy()
            self.capture_steps = 0
        if self.phase == self.CAPTURE:
            self.capture_steps += 1
            if depth > self.cfg['engagement_depth']:
                self.phase = self.INSERT
            elif self.capture_steps * self.dt > self.cfg['capture_timeout']:
                self.phase = self.SEARCH
                self.index = min(self.index + 10, len(self.path) - 1)
                self.cooldown = round(0.2 / self.dt)
        if self.phase == self.INSERT and depth < self.cfg['capture_depth']:
            self.phase = self.SEARCH
        xy = self.path[self.index] if self.phase == self.SEARCH else self.lock_xy
        if self.phase == self.SEARCH and np.linalg.norm(xy - tip[:2]) < self.cfg['tracking_tolerance']:
            self.index = min(self.index + 1, len(self.path) - 1)
        lead = self.cfg['search_z_lead'] if self.phase == self.SEARCH else self.cfg['insert_z_lead']
        self._descend(tip, shaft_length, lead)
        return np.r_[xy, self.z]

    def _descend(self, tip, shaft_length, lead):
        # Extra nominal travel accommodates uncertain tool length. Contact and
        # bounded spring displacement stop the physical shoulder at the surface.
        surface = self.surface_z if self.surface_z is not None else self.top_z
        self.z = max(surface - shaft_length - self.cfg['seating_overtravel'], tip[2] - lead,
                     self.z - self.cfg['descent_rate'] * self.dt)
