"""Deterministic minimum-jerk plant and contact spring for executor fault tests.

This is a software test double, not identified Franka dynamics. It never imports ROS.
"""

import numpy as np
from scipy.optimize import brentq

from .interface import RobotState, yaw_of_tool


class VirtualClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class OfflineJointTransport:
    def __init__(self, geometry, q, clock, *, box=None, table_z=.0072, spring=6000.):
        self.geometry, self.q, self.clock = geometry, np.asarray(q).copy(), clock
        self.box = None if box is None else np.asarray(box, float)
        self.box_top = table_z + .0508
        self.spring = spring
        self.segment, self.commands, self.stop_count = None, [], 0
        self.contact_q, self.contact_tip, self.contact_normal = None, None, None
        self.last_force = np.array([.4, -.1, -.4])
        self.start_tip = geometry.tip_pose(self.q)[0]
        self.fault, self.fault_after, self.controller_errors = None, np.inf, []
        self.force_angle_error = 0.
        self.early_done = False
        self.hold = False
        self.buffer_seconds = .2
        self.joint_lag = np.zeros(7)

    def _distance(self, p):
        if self.box is None:
            return 1.
        c, s = np.cos(self.box[2]), np.sin(self.box[2])
        axes = np.array([[c, -s], [s, c]])
        local = (p[:2] - self.box[:2]) @ axes
        off = local - np.clip(local, -np.array([.1267, .169]) / 2, np.array([.1267, .169]) / 2)
        side = np.linalg.norm(off) - .0125
        return max(float(side), float(p[2] - self.box_top))

    def _normal(self, p):
        if abs(p[2] - self.box_top) < 1e-5:
            return np.array([0., 0., 1.])
        c, s = np.cos(self.box[2]), np.sin(self.box[2])
        axes = np.array([[c, -s], [s, c]])
        local = (p[:2] - self.box[:2]) @ axes
        half = np.array([.1267, .169]) / 2
        n = (local - np.clip(local, -half, half)) @ axes.T
        n /= max(np.linalg.norm(n), 1e-12)
        return np.r_[n, 0.]

    # Streamed dynamic joint-impedance skill: first-order lag toward the latest target.
    def begin_stream(self, q, duration, thresholds, k_gains, d_gains):
        self.state()
        self.commands.append(dict(kind="stream", q=np.asarray(q).copy(), start_q=self.q.copy(),
                                  duration=duration, thresholds=thresholds, time=self.clock.now,
                                  k_gains=list(k_gains), d_gains=list(d_gains)))
        self.stream = dict(target=self.q.copy(), t=self.clock.now, end=self.clock.now + duration,
                           thresholds=thresholds, tau=np.divide(d_gains, k_gains))  # per-joint d/k lag
        self.stream_targets = 0
        self.stream_mode_active = True

    def stream_target(self, q):
        if getattr(self, "stream", None) is None:
            return
        self.stream["target"] = np.asarray(q, float).copy()
        self.stream_targets += 1

    def _stream_state(self, force):
        s = self.stream
        dt = self.clock.now - s["t"]
        s["t"] = self.clock.now
        if dt > 0:
            # stream_bias models friction: the impedance settles this far from the target.
            aim = s["target"] + getattr(self, "stream_bias", np.zeros(7))
            new_q = self.q + (aim - self.q) * np.minimum(1., dt / np.maximum(s["tau"], 1e-3))
        else:
            new_q = self.q.copy()
        p_target, _ = self.geometry.tip_pose(s["target"])
        if self.box is not None and self._distance(self.geometry.tip_pose(new_q)[0]) < -1e-10:
            if self.contact_q is None:
                hit = brentq(lambda t: self._distance(self.geometry.tip_pose(self.q + t * (new_q-self.q))[0]),
                             0, 1) if self._distance(self.geometry.tip_pose(self.q)[0]) > 1e-9 else 0.
                self.contact_q = self.q + hit * (new_q - self.q)
                self.contact_tip = self.geometry.tip_pose(self.contact_q)[0]
                self.contact_normal = self._normal(self.contact_tip)
            self.q = self.contact_q.copy()
        elif self.contact_q is not None and self._distance(p_target) >= 0:
            self.contact_q = None
            self.q = new_q
        elif self.contact_q is None:
            self.q = new_q
        if self.contact_q is not None:
            penetration = max(0., float((self.contact_tip - p_target) @ self.contact_normal))
            n = self.contact_normal.copy()
            c, sn = np.cos(self.force_angle_error), np.sin(self.force_angle_error)
            n[:2] = np.array([[c, -sn], [sn, c]]) @ n[:2]
            # Joint impedance is far softer than the block: model ~2500 N/m at the tip.
            force += n * min(self.spring, 2500.) * penetration
        force += np.array([1.5, 0., 0.])
        if self.clock.now >= s["end"] or np.any(np.abs(force) > np.asarray(s["thresholds"][:3])):
            self.stream = None
        return force

    def state(self):
        force = np.array([.4, -.1, -.4])
        if getattr(self, "stream", None) is not None:
            force = self._stream_state(force)
        elif self.segment is not None:
            a, b, t0, duration, thresholds = self.segment
            f = min(1., max(0., (self.clock.now - t0) / duration))
            if self.hold:
                f = 0.
            if self.early_done:
                f = min(f, .1)
            s = 10 * f**3 - 15 * f**4 + 6 * f**5
            commanded = a + s * (b - a)
            p, _ = self.geometry.tip_pose(commanded)
            if self.box is not None and self._distance(p) < -1e-10:
                if self.contact_q is None:
                    start_p = self.geometry.tip_pose(a)[0]
                    if self._distance(start_p) <= 1e-9:
                        hit = 0.
                    else:
                        hit = brentq(lambda t: self._distance(self.geometry.tip_pose(a + t * (commanded-a))[0]), 0, 1)
                    self.contact_q = a + hit * (commanded - a)
                    self.contact_tip = self.geometry.tip_pose(self.contact_q)[0]
                    self.contact_normal = self._normal(self.contact_tip)
                self.q = self.contact_q.copy()
                penetration = max(0., float((self.contact_tip - p) @ self.contact_normal))
                n = self.contact_normal.copy()
                c, sn = np.cos(self.force_angle_error), np.sin(self.force_angle_error)
                n[:2] = np.array([[c, -sn], [sn, c]]) @ n[:2]
                force += n * self.spring * penetration
            else:
                self.q = commanded.copy() + self.joint_lag * s
                self.contact_q = None
            # Repeatable controller-start offset plus modest travel drift.
            force += np.array([1.5, 0., 0.])
            if self.clock.now - t0 >= duration + self.buffer_seconds and not self.early_done:
                self.segment = None
            if np.any(np.abs(force) > np.asarray(thresholds[:3])):
                self.segment = None
        p, R = self.geometry.tip_pose(self.q)
        # Posture-dependent estimator bias persists when the skill ends.
        force += 10. * (p - self.start_tip)
        yaw = yaw_of_tool(R)
        if self.clock.now >= self.fault_after:
            if self.fault == "joint3":
                self.q[2] += .03
                p, R = self.geometry.tip_pose(self.q)
                yaw = yaw_of_tool(R)
            elif self.fault == "yaw":
                yaw += .10
            elif self.fault == "force":
                force += [30., 0., 0.]
            elif self.fault == "tip":
                p = p + [.02, 0, 0]
        self.last_force = force
        return RobotState(self.clock.now, p.copy(), yaw, force.copy(), np.zeros(3), self.q.copy())

    def send(self, q, duration, thresholds, impedances):
        self.state()
        self.commands.append(dict(q=np.asarray(q).copy(), start_q=self.q.copy(), duration=duration, thresholds=thresholds,
                                  impedances=impedances, time=self.clock.now))
        self.segment = (self.q.copy(), np.asarray(q).copy(), self.clock.now, duration, thresholds)
        self.stream_mode_active = False

    def done(self):
        if getattr(self, "stream_mode_active", False):
            return getattr(self, "stream", None) is None
        return self.segment is None or (self.early_done and self.clock.now-self.segment[2] > .1)

    def errors(self):
        return self.controller_errors

    def stop(self):
        self.state()
        self.segment = None
        self.stream = None
        self.contact_q = None
        self.stop_count += 1
