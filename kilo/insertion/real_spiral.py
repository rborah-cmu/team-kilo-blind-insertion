"""Scripted spiral insertion for the real Franka (no RL).

Adapted from main's simulation baseline, which succeeded 50/50:
  kilo/insertion/reference.py  InsertionReference (SEARCH -> CAPTURE -> INSERT)
  kilo/insertion/spiral.py     spiral_waypoints (used unchanged)
  kilo/insertion/env.py        Cartesian impedance + bounded downward preload
Those files are not modified. Differences for hardware:
  - the surface is found from the measured upward force residual (wrench estimate),
    not MuJoCo constraint forces, with a real-robot threshold;
  - the preload lead is larger (real friction and wrench noise);
  - a seat depth ends the run (the sim ran to full depth);
  - safety limits (force, excursion, tracking, timeout) end the run.

The logic is pure: ``step(tip, force_z)`` returns the target tip position and the xy
stiffness for this tick. The robot runner streams it as a Cartesian impedance target.
"""

from __future__ import annotations

import numpy as np

from kilo.insertion.spiral import spiral_waypoints


class RealSpiralReference:
    APPROACH, SEARCH, CAPTURE, INSERT, SEATED = "approach", "search", "capture", "insert", "seated"

    def __init__(self, cfg: dict, estimate_xy, expected_top_z: float, tip):
        self.cfg = cfg
        self.dt = 1.0 / cfg["policy_rate_hz"]
        self.estimate_xy = np.asarray(estimate_xy, float)
        self.expected_top_z = float(expected_top_z)
        self.path = spiral_waypoints(self.estimate_xy, cfg["spiral_pitch"], cfg["spiral_max_radius"],
                                     cfg["search_speed"] * self.dt)
        self.index = 0
        self.phase = self.APPROACH
        self.capture_steps = 0
        self.cooldown = 0
        self.lock_xy = self.estimate_xy.copy()
        self.z = float(tip[2])
        self.surface_z = None
        self.touch_steps = 0
        self.spiral_laps = 0
        self.touch_reason = None
        self.seat_reason = None
        self._z_hist = []  # recent tip z during the approach (stall detection)
        self._depth_hist = []  # recent depth during INSERT (bottomed-out detection)
        self._last_advance = 0  # tick of the last spiral advance (stuck detection)
        self._tick = 0
        self.skips = 0

    @property
    def depth_reference(self):
        return self.surface_z if self.surface_z is not None else self.expected_top_z

    def step(self, tip, force_z):
        """One tick. ``force_z``: upward contact force on the peg (N, residual, + = pushed up)."""
        tip = np.asarray(tip, float)
        cfg = self.cfg
        self._tick += 1
        if self.phase == self.APPROACH:
            self.touch_steps = self.touch_steps + 1 if force_z > cfg["touch_force"] else 0
            # Kinematic touch: stopped descending while the spring is compressed (the real
            # wrench estimate under-reads the preload).
            window = max(2, int(round(cfg.get("stall_seconds", .3) / self.dt)))
            self._z_hist = (self._z_hist + [float(tip[2])])[-window:]
            stalled = (len(self._z_hist) == window
                       and max(self._z_hist) - min(self._z_hist) < cfg.get("stall_motion", .0003)
                       and tip[2] - self.z >= cfg.get("stall_lag", .006))
            if self.touch_steps >= cfg["touch_samples"] or stalled:
                self.surface_z = float(tip[2])
                self.touch_reason = "force" if self.touch_steps >= cfg["touch_samples"] else "stall"
                self.phase = self.SEARCH
                self._last_advance = self._tick
            elif self.expected_top_z - tip[2] > cfg["engagement_depth"]:
                # Started right over the hole and dropped in without touching the top.
                self.surface_z = self.expected_top_z
                self.phase = self.INSERT
                self.lock_xy = tip[:2].copy()
            else:
                self.z = max(self.z - cfg["descent_rate"] * self.dt, tip[2] - cfg["search_z_lead"])
                return np.r_[self.path[0], self.z], cfg["search_xy_stiffness"]

        depth = self.surface_z - tip[2]
        self.cooldown = max(0, self.cooldown - 1)
        if self.phase == self.SEARCH and self.cooldown == 0 and depth > cfg["capture_depth"]:
            self.phase = self.CAPTURE
            self.lock_xy = tip[:2].copy()
            self.capture_steps = 0
        if self.phase == self.CAPTURE:
            self.capture_steps += 1
            if depth > cfg["engagement_depth"]:
                self.phase = self.INSERT
            elif self.capture_steps * self.dt > cfg["capture_timeout"]:
                self.phase = self.SEARCH
                self.index = min(self.index + 10, len(self.path) - 1)
                self.cooldown = round(0.2 / self.dt)
        if self.phase == self.INSERT:
            window = max(2, int(round(cfg.get("seat_stall_seconds", 1.5) / self.dt)))
            self._depth_hist = (self._depth_hist + [float(depth)])[-window:]
            bottomed = (depth >= cfg.get("min_seat_depth", cfg["seat_depth"]) and len(self._depth_hist) == window
                        and max(self._depth_hist) - min(self._depth_hist) < cfg.get("seat_stall_motion", .0003))
            shallow_stall = (depth < cfg.get("min_seat_depth", cfg["seat_depth"]) and len(self._depth_hist) == window
                             and max(self._depth_hist) - min(self._depth_hist) < cfg.get("seat_stall_motion", .0003))
            if depth >= cfg["seat_depth"] or bottomed:
                self.phase = self.SEATED
                self.seat_reason = "depth" if depth >= cfg["seat_depth"] else "bottomed out"
            elif shallow_stall:
                # Jammed on the chamfer/edge, not in the bore: do not park here, move on.
                self.phase = self.SEARCH
                self.index = min(self.index + 10, len(self.path) - 1)
                self.cooldown = round(0.5 / self.dt)
                self._depth_hist = []
                self._last_advance = self._tick
                self.skips += 1
            elif depth < cfg["capture_depth"]:
                self.phase = self.SEARCH
                self._depth_hist = []
                self._last_advance = self._tick
        else:
            self._depth_hist = []
        if self.phase == self.SEARCH:
            xy = self.path[self.index]
            stuck = (self._tick - self._last_advance) * self.dt > cfg.get("stuck_seconds", 1e9)
            if np.linalg.norm(xy - tip[:2]) < cfg["tracking_tolerance"] or stuck:
                step = 5 if stuck else 1  # caught on an edge: skip ahead instead of pushing forever
                self.skips += int(stuck)
                if self.index + step > len(self.path) - 1:
                    self.index, self.spiral_laps = 0, self.spiral_laps + 1  # start the spiral again
                else:
                    self.index += step
                self._last_advance = self._tick
        else:
            xy = self.lock_xy
        lead = cfg["search_z_lead"] if self.phase in (self.SEARCH, self.CAPTURE) else cfg["insert_z_lead"]
        # Bounded preload, as in sim: never command more than `lead` below the tip.
        self.z = tip[2] - lead
        kxy = cfg["search_xy_stiffness"] if self.phase == self.SEARCH else cfg["insert_xy_stiffness"]
        return np.r_[xy, self.z], kxy

    @property
    def done(self):
        return self.phase == self.SEATED
