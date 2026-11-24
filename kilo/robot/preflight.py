"""Geometric-only backend for planning the existing adaptive touch localizer.

Contact comes from a stated hypothetical box, never from the actual robot.
Full probe rays are checked even though their nominal execution stops early.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .interface import ProbeResult, RobotBackend, RobotState, tool_rotation
from .planning import PlanningFailure
from .workspace import HandWorkspace


class PreflightRobot(RobotBackend):
    def __init__(self, geometry, cfg, scene, start_q, box, *, yaw=np.pi, elbow=0., step=.005, force_angle_error=0.):
        self.geometry, self.cfg = geometry, cfg
        self.q = np.asarray(start_q).copy()
        self.yaw, self.elbow, self.step = float(yaw), float(elbow), float(step)
        c, s = np.cos(force_angle_error), np.sin(force_angle_error)
        self.force_rotation = np.array([[c, -s], [s, c]])
        self.box = np.asarray(box, float)
        self.half = np.array(scene["box"]["size"][:2]) / 2
        self.box_top = cfg["workspace"]["table_z"] + scene["box"]["size"][2]
        self.radius = float(scene["peg"]["shaft_radius"])
        c, s = np.cos(self.box[2]), np.sin(self.box[2])
        self.axes = np.array([[c, -s], [s, c]])
        self.hand = HandWorkspace(cfg["workspace"])
        self.paths, self.motions = {}, []
        self.stiffness, self.preferred_hz, self.t = 1500., 50., 0.

    def state(self):
        p, _ = self.geometry.tip_pose(self.q)
        return RobotState(self.t, p, self.yaw, np.zeros(3), np.zeros(3), q=self.q.copy())

    def _save(self, kind, path, gap, start, target, **info):
        key = f"path{len(self.paths):03d}"
        self.paths[key] = path
        self.motions.append(dict(kind=kind, path=key, samples=len(path), start=np.asarray(start).tolist(),
                                 target=np.asarray(target).tolist(), certified_clearance_m=gap, **info))

    def reorient(self):
        start = self.state().tip
        seed = self.q.copy()
        seed[6] += np.pi
        goal = self.geometry.inverse(start, self.yaw, seed, elbow=self.elbow)
        path, gap = self.geometry.check_joint_edge(self.q, goal)
        # Only a high-clearance startup rotation, not a general joint travel bypass.
        tips = np.array([self.geometry.tip_pose(q)[0] for q in path])
        if np.linalg.norm(tips - start, axis=1).max() > .002:
            raise PlanningFailure("startup wrist rotation shifts tip >2 mm")
        self._save("startup_joint_rotation", path, gap, start, start)
        self.q = goal

    def move(self, tip, yaw, speed):
        st = self.state()
        self.hand.check(st.tip[:2], np.asarray(tip)[:2], st.yaw, yaw)
        path, gap = self.geometry.cartesian_path(self.q, tip, yaw, elbow=self.elbow, step=self.step)
        self._save("travel", path, gap, st.tip, tip)
        self.q, self.yaw = path[-1], yaw

    def _distance(self, xy):
        local = (np.asarray(xy) - self.box[:2]) @ self.axes
        delta = local - np.clip(local, -self.half, self.half)
        return float(np.linalg.norm(delta) - self.radius)

    def _contact(self, start, target):
        delta = target - start
        if np.linalg.norm(delta[:2]) < 1e-6:
            if self._distance(start[:2]) <= 0 and target[2] <= self.box_top <= start[2]:
                t = (self.box_top - start[2]) / delta[2]
                return t, np.array([0., 0., 5.])
            return None
        if min(start[2], target[2]) >= self.box_top:
            return None
        if self._distance(start[:2]) <= 0:
            raise PlanningFailure("side probe starts inside hypothetical peg/box contact envelope")
        n = max(2, int(np.ceil(np.linalg.norm(delta[:2]) / .001)))
        ts = np.linspace(0, 1, n + 1)
        for low, high in zip(ts[:-1], ts[1:]):
            if self._distance((start + high * delta)[:2]) <= 0:
                t = brentq(lambda f: self._distance((start + f * delta)[:2]), low, high)
                xy = (start + t * delta)[:2]
                local = (xy - self.box[:2]) @ self.axes
                normal = (local - np.clip(local, -self.half, self.half)) @ self.axes.T
                normal /= np.linalg.norm(normal)
                normal = self.force_rotation @ normal
                return t, np.r_[normal * 5, 0.]
        return None

    def probe(self, target, yaw, cfg, stop_on_arrival=True):
        start = self.state().tip
        target = np.asarray(target)
        self.hand.check(start[:2], target[:2], self.yaw, yaw)
        # Check the whole commanded ray, including continuation past the nominal
        # contact. The unknown real contact must not be used to hide unsafe poses.
        full, gap = self.geometry.cartesian_path(self.q, target, yaw, elbow=self.elbow, step=self.step)
        hit = self._contact(start, target)
        end = target if hit is None else start + hit[0] * (target - start)
        self._save("full_guarded_ray", full, gap, start, target,
                   nominal_contact=hit is not None, nominal_stop=end.tolist())
        actual, actual_gap = self.geometry.cartesian_path(self.q, end, yaw, elbow=self.elbow, step=self.step)
        self._save("nominal_probe_prefix", actual, actual_gap, start, end)
        self.q, self.yaw = actual[-1], yaw
        force = np.zeros(3) if hit is None else hit[1]
        result = ProbeResult(hit is not None, end.copy(), force, float(np.linalg.norm(force)),
                             float(np.linalg.norm(end - start)), np.zeros(3), samples=[])
        if hit is not None and cfg.retreat > 0:
            direction = (target - start) / np.linalg.norm(target - start)
            self.move(end - cfg.retreat * direction, yaw, .01)
        return result

    def begin_stream(self):
        raise NotImplementedError("offline planner cannot stream robot commands")

    def stream_target(self, tip, yaw):
        raise NotImplementedError("offline planner cannot stream robot commands")

    def end_stream(self):
        pass

    def tick(self, dt):
        self.t += dt
