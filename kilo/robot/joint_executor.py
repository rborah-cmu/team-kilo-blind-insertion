"""Checked joint-edge execution with continuous posture/force supervision.

The transport owns hardware calls; this module also runs against a deterministic
offline transport. It never substitutes a Cartesian command for a joint path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .interface import ProbeResult, SafetyStop, tool_rotation, yaw_of_tool
from .planning import PlanningFailure


@dataclass(frozen=True)
class CheckedJointPath:
    joints: np.ndarray
    yaw: float
    startup: bool
    digest: str


class JointPathExecutor:
    def __init__(self, geometry, transport, options=None, clock=time):
        self.geometry, self.transport, self.clock = geometry, transport, clock
        o = options or {}
        self.yaw = float(o.get("yaw", np.pi))
        self.elbow = float(o.get("joint3", 0.))
        self.step = float(o.get("cartesian_step", .002))
        # Free-space travel may merge many checked 2 mm samples into one joint
        # skill. 2026-11-24: chains of ~0.2-0.3 s goto_joints skills hit
        # joint_motion_generator_velocity/acceleration_discontinuity at a skill
        # start after 71 good transitions (run 20261124-132653-localize), so
        # travel uses as few skill transitions as the per-edge checks allow.
        self.travel_step = float(o.get("travel_step", self.step))
        # Collision/IK sampling remains dense; command spacing is independent.
        # Force feedback is polled throughout each command, not just at knots.
        self.probe_step = float(o.get("probe_step", self.step))
        self.poll = float(o.get("poll_seconds", .02))
        self.min_seconds = float(o.get("min_segment_seconds", .5))
        self.max_velocity = float(o.get("max_joint_velocity", .15))
        self.max_acceleration = float(o.get("max_joint_acceleration", .3))
        self.joint_tolerance = float(o.get("joint_tolerance", .01))
        self.end_tolerance = float(o.get("endpoint_joint_tolerance", .003))
        self.probe_end_tolerance = float(o.get("probe_endpoint_joint_tolerance", .006))
        self.tip_tolerance = float(o.get("tip_tolerance", .004))
        self.angle_tolerance = float(o.get("orientation_tolerance", .03))
        self.clearance = float(o.get("extra_clearance", .002))
        self.startup_height = float(o.get("startup_min_height", .20))
        self.impedances = list(o.get("joint_impedances", [600, 600, 600, 600, 250, 150, 50]))
        # "segments": one JointPositionSkill per checked edge (original executor).
        # "stream": one frankapy dynamic joint-impedance skill per path (see
        # FrankaJointTransport.begin_stream); the wrist startup stays a segment.
        self.mode = str(o.get("execution", "segments"))
        self.stream_poll = float(o.get("stream_poll_seconds", .01))
        self.k_gains = [float(k) for k in o.get("stream_k_gains", [600, 600, 600, 600, 250, 150, 50])]
        self.d_gains = [float(d) for d in o.get("stream_d_gains", [50, 50, 50, 50, 30, 25, 15])]
        self.stream_end_tolerance = float(o.get("stream_endpoint_tolerance", .005))
        self.stream_settle = float(o.get("stream_settle_seconds", 2.))
        self.max_lag = float(o.get("stream_max_tip_lag", .006))
        # path_checks=false (user decision 2026-11-24) disables the path-discrepancy stops:
        # tip off path, joint off path, and tip lag behind the streamed target. Force limits,
        # contact detection, controller errors, joint-3/orientation limits, frankapy virtual
        # walls and joint limits all stay active.
        self.path_checks = bool(o.get("path_checks", True))
        self.min_ramp = float(o.get("stream_min_ramp_seconds", .3))
        # Settle integral: friction holds a joint-impedance wrist a few mrad short of the
        # endpoint (2026-11-24 run 161055: joint 7 stopped 3.6 mrad short with k7=150).
        # During the settle phase only, the streamed target creeps past the endpoint by
        # gain * integral(error), bounded per joint, until the arm actually arrives.
        self.settle_gain = float(o.get("stream_settle_gain", 3.))
        self.settle_max_offset = float(o.get("stream_settle_max_offset", .01))
        self.torque_margin = float(o.get("stream_torque_margin", 20.))
        if abs(np.angle(np.exp(1j*(self.yaw-np.pi)))) > 1e-5 or abs(self.elbow) > 1e-5:
            raise ValueError("executor requires the checked wrist-pi / joint-3-zero posture")
        positive = [self.step, self.poll, self.min_seconds, self.max_velocity, self.max_acceleration,
                    self.joint_tolerance, self.end_tolerance, self.probe_end_tolerance, self.tip_tolerance,
                    self.angle_tolerance, self.clearance]
        if (not np.all(np.isfinite(positive)) or min(positive) <= 0 or self.step > .005
                or self.max_velocity > .25 or self.max_acceleration > .3
                or self.probe_end_tolerance > self.joint_tolerance
                or not self.step <= self.travel_step <= .10
                or not self.step <= self.probe_step <= min(self.travel_step, .02)
                or len(self.impedances) != 7 or min(self.impedances) <= 0):
            raise ValueError("invalid planned joint-executor limits")
        # Gains capped near frankapy's defaults; d7 stays <= 15 for 1 kHz stability.
        k_cap, d_cap = [1000]*4 + [500, 600, 150], [60]*4 + [35, 35, 15]
        if (self.mode not in ("segments", "stream") or len(self.k_gains) != 7 or len(self.d_gains) != 7
                or any(not 0 < k <= c for k, c in zip(self.k_gains, k_cap))
                or any(not 0 <= d <= c for d, c in zip(self.d_gains, d_cap))
                or not .002 <= self.stream_poll <= .02
                or not 0 < self.stream_end_tolerance <= self.joint_tolerance
                or not 0 < self.max_lag <= .01 or not 0 < self.stream_settle <= 5
                or not .1 <= self.min_ramp <= 2
                or not 0 <= self.settle_gain <= 10 or not 0 <= self.settle_max_offset <= .02
                or not 5 <= self.torque_margin <= 30):
            raise ValueError("invalid streamed joint-executor settings")
        self.last_samples, self.last_records, self.active = [], [], False
        self.last_stats = {}

    @staticmethod
    def _digest(q, yaw, startup):
        return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes()
                              + np.array([yaw, float(startup)], dtype="<f8").tobytes()).hexdigest()

    def _checked(self, joints, yaw, startup=False):
        q = np.asarray(joints, float).copy()
        if q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or not np.all(np.isfinite(q)):
            raise SafetyStop("invalid joint path")
        origin, _ = self.geometry.tip_pose(q[0])
        for a, b in zip(q[:-1], q[1:]):
            checked, _ = self.geometry.check_joint_edge(a, b, min_clearance=self.clearance)
            for sample in checked:
                if abs(sample[2] - self.elbow) > self.joint_tolerance:
                    raise SafetyStop("planned path violates joint 3 posture")
                p, R = self.geometry.tip_pose(sample)
                if startup:
                    if p[2] < self.startup_height or np.linalg.norm(p - origin) > .002:
                        raise SafetyStop("startup path leaves its high-clearance rotation envelope")
                elif np.linalg.norm(Rotation.from_matrix(tool_rotation(yaw) @ R.T).as_rotvec()) > .005:
                    raise SafetyStop("planned path violates fixed wrist orientation")
        _, end_R = self.geometry.tip_pose(q[-1])
        if np.linalg.norm(Rotation.from_matrix(tool_rotation(yaw) @ end_R.T).as_rotvec()) > .005:
            raise SafetyStop("path does not end at the required wrist orientation")
        q.flags.writeable = False
        return CheckedJointPath(q, float(yaw), bool(startup), self._digest(q, yaw, startup))

    def plan(self, target, yaw, max_step=None):
        self._require_yaw(yaw)
        st = self.transport.state()
        self._posture(st, st.q, startup=False)
        try:
            q, _ = self.geometry.cartesian_path(st.q, target, self.yaw, elbow=self.elbow,
                                                 step=self.step, min_clearance=self.clearance)
            return self._checked(self._command_knots(q, max_step), self.yaw)
        except PlanningFailure as exc:
            raise SafetyStop(f"joint-path preflight failed before motion: {exc}") from exc

    def _command_knots(self, path, max_step=None):
        """Merge collision-proof subdivision points, not whole unchecked rays.

        Each emitted command advances the tip by at most ``max_step`` (the
        configured 2 mm by default; travel may use more), and its actual joint
        interpolation is independently rechecked for clearance, a straight tip
        line within 1 mm, and the fixed wrist orientation.
        """
        step = self.step if max_step is None else float(max_step)
        if not self.step <= step <= self.travel_step + 1e-12:
            raise PlanningFailure("command step outside the configured limits")
        points = np.array([self.geometry.tip_pose(q)[0] for q in path])
        output, i = [path[0]], 0
        while i < len(path)-1:
            j = i+1
            while j+1 < len(path) and np.linalg.norm(points[j+1]-points[i]) <= step + 1e-7:
                j += 1
            while True:
                checked, _ = self.geometry.check_joint_edge(path[i], path[j], min_clearance=self.clearance)
                valid = True
                direction = points[j]-points[i]
                length2 = float(direction@direction)
                for q in checked:
                    p, R = self.geometry.tip_pose(q)
                    f = float(np.clip((p-points[i])@direction/max(length2, 1e-20), 0, 1))
                    if (np.linalg.norm(p-(points[i]+f*direction)) > .001
                            or np.linalg.norm(Rotation.from_matrix(tool_rotation(self.yaw)@R.T).as_rotvec()) > .005):
                        valid = False
                        break
                if valid:
                    break
                j -= 1
                if j <= i:
                    raise PlanningFailure("could not form a checked joint command")
            output.append(path[j])
            i = j
        return np.asarray(output)

    def plan_startup(self):
        st = self.transport.state()
        if st.tip[2] < self.startup_height:
            raise SafetyStop("planned wrist startup requires a high-clearance pose; reposition first")
        if abs(st.q[2] - self.elbow) > self.joint_tolerance:
            raise SafetyStop("startup joint 3 is outside its required posture")
        seed = st.q.copy()
        seed[6] += np.pi
        try:
            goal = self.geometry.inverse(st.tip, self.yaw, seed, elbow=self.elbow)
            q, _ = self.geometry.check_joint_edge(st.q, goal, min_clearance=self.clearance)
            tips = np.array([self.geometry.tip_pose(v)[0] for v in q])
            if np.linalg.norm(tips - st.tip, axis=1).max() > .002:
                raise SafetyStop("startup wrist rotation moves the tip more than 2 mm")
            # One minimum-jerk joint rotation traverses this same checked edge.
            return self._checked(np.array([st.q, goal]), self.yaw, startup=True)
        except PlanningFailure as exc:
            raise SafetyStop(f"startup preflight failed before motion: {exc}") from exc

    def _require_yaw(self, yaw):
        if abs(np.angle(np.exp(1j * (yaw - self.yaw)))) > 1e-5:
            raise SafetyStop("requested wrist yaw differs from the checked 180-degree posture")

    def _posture(self, st, reference, *, startup):
        if st.q is None or not np.all(np.isfinite([*st.q, *st.tip, st.yaw, *st.force])):
            raise SafetyStop("invalid robot feedback")
        if abs(st.q[2] - self.elbow) > self.joint_tolerance:
            raise SafetyStop("joint 3 left the planned posture")
        p, R = self.geometry.tip_pose(reference)
        actual_p, actual_R = self.geometry.tip_pose(st.q)
        desired_yaw = yaw_of_tool(R) if startup else self.yaw
        yaw_error = abs(np.angle(np.exp(1j * (st.yaw - desired_yaw))))
        if yaw_error > self.angle_tolerance:
            raise SafetyStop("wrist yaw left the checked posture")
        if np.linalg.norm(Rotation.from_matrix(R @ actual_R.T).as_rotvec()) > self.angle_tolerance:
            raise SafetyStop("tool orientation left the checked path")
        if self.path_checks and np.linalg.norm(st.tip - p) > self.tip_tolerance:
            raise SafetyStop("peg tip left the checked path")
        if np.linalg.norm(st.tip - actual_p) > .002:
            raise SafetyStop("measured tool pose disagrees with the calibrated planning model")
        clearance = self.geometry.clearance(st.q)
        self.last_stats["max_joint3_error_rad"] = max(self.last_stats.get("max_joint3_error_rad", 0.),
                                                     abs(st.q[2] - self.elbow))
        self.last_stats["max_wrist_tracking_error_rad"] = max(self.last_stats.get("max_wrist_tracking_error_rad", 0.), yaw_error)
        self.last_stats["min_actual_model_clearance_m"] = min(self.last_stats.get("min_actual_model_clearance_m", np.inf), clearance)
        if clearance <= 0:
            raise SafetyStop("actual arm entered a shelf, wall, or table keep-out")

    def duration(self, a, b, speed, padded=False):
        delta = float(np.max(np.abs(b - a)))
        pa, _ = self.geometry.tip_pose(a)
        pb, _ = self.geometry.tip_pose(b)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be positive")
        distance = float(np.linalg.norm(pb - pa))
        if padded:
            delta += 2 * max(self.end_tolerance, self.probe_end_tolerance)
            distance += 2 * self.tip_tolerance
        return max(self.min_seconds, 1.875 * delta / self.max_velocity,
                   np.sqrt(5.774 * delta / self.max_acceleration), distance / speed)

    def execute(self, path, speed, settings, *, detect_contact=False):
        """Execute checked edges. Stop on contact; never resume a hardware backstop."""
        # Stream mode streams EVERY path, including the wrist startup and tiny joint-only
        # corrections (joint-space minimum-jerk profile). 2026-11-24 run 154857-localize: a
        # single 19.6 s JointPositionSkill wrist turn also raised
        # joint_motion_generator_*_discontinuity mid-motion, so no motion generator is used.
        if self.mode == "stream" and hasattr(self.transport, "begin_stream"):
            return self._execute_stream(path, speed, settings, detect_contact=detect_contact)
        self.last_stats = {}
        self.last_samples, self.last_records, self.last_raw_samples = [], [], []
        if not isinstance(path, CheckedJointPath) or path.digest != self._digest(path.joints, path.yaw, path.startup):
            raise SafetyStop("unchecked or modified joint path")
        self._require_yaw(path.yaw)
        # Revalidate every edge before dispatch, including externally loaded plans.
        try:
            self._checked(path.joints, path.yaw, path.startup)
        except PlanningFailure as exc:
            raise SafetyStop(f"joint-path revalidation failed before motion: {exc}") from exc
        st0 = self.transport.state()
        self._posture(st0, path.joints[0], startup=path.startup)
        if np.max(np.abs(st0.q - path.joints[0])) > self.end_tolerance:
            raise SafetyStop("robot moved since path preflight; replan before execution")
        origin = st0.tip.copy()
        final_tip, _ = self.geometry.tip_pose(path.joints[-1])
        length = float(np.linalg.norm(final_tip - origin))
        direction = (final_tip - origin) / max(length, 1e-12)
        self.last_samples, self.last_records, self.last_raw_samples = [], [], []
        self._posture(st0, path.joints[0], startup=path.startup)
        # Bias only once for the complete probe. Never erase an obstruction by
        # re-zeroing at each joint waypoint or after a force backstop.
        readings = []
        for _ in range(max(1, int(settings.bias_seconds / self.poll))):
            readings.append(self.transport.state().force)
            self.clock.sleep(self.poll)
        bias = np.mean(readings, axis=0)
        baseline, previous_tip = bias.copy(), st0.tip.copy()
        thresholds = (np.abs(bias) + settings.safety_force).tolist() + (np.abs(st0.torque) + 10.).tolist()
        peak, over, started = 0., 0, self.clock.monotonic()
        # Check the full path timeout budget before any command, rather than
        # stopping midair because dense but deliberately slow waypoints take time.
        durations = [self.duration(a, b, speed, padded=True) for a, b in zip(path.joints[:-1], path.joints[1:])]
        if sum(durations) + len(durations) * .25 > settings.timeout:
            raise SafetyStop("planned path exceeds the allowed timeout; shorten it or review the time budget")
        try:
            endpoint_limit = self.probe_end_tolerance if detect_contact else self.end_tolerance
            for index, (a, b, duration) in enumerate(zip(path.joints[:-1], path.joints[1:], durations)):
                st = self.transport.state()
                if np.max(np.abs(st.q - a)) > endpoint_limit:
                    raise SafetyStop("joint waypoint was not reached; refusing the next segment")
                self._posture(st, a, startup=path.startup)
                # The controller starts from measured feedback, not a fictional
                # exact endpoint. Recheck that small connector before dispatch.
                self.geometry.check_joint_edge(st.q, b, min_clearance=self.clearance)
                if self.transport.errors():
                    raise SafetyStop("robot reports an active controller error before dispatch")
                actual_start = st.q.copy()
                command_start = (np.asarray(self.transport.command_start(), float)
                                 if hasattr(self.transport, "command_start") else actual_start)
                if np.max(np.abs(command_start-actual_start)) > self.joint_tolerance:
                    raise SafetyStop("controller's desired joints disagree with measured posture")
                self.geometry.check_joint_edge(command_start, b, min_clearance=self.clearance)
                self._posture(st, command_start, startup=path.startup)
                duration = self.duration(command_start, b, speed)
                delta = b - command_start
                t0 = self.clock.monotonic()
                self.last_stats["last_segment"] = dict(
                    index=index, measured_start_q=actual_start.tolist(),
                    desired_start_q=command_start.tolist(), goal_q=b.tolist(),
                    duration_s=duration, endpoint_limit_rad=endpoint_limit)
                self.active = True
                self.transport.send(b, duration, thresholds, self.impedances)
                done_at = None
                while True:
                    st = self.transport.state()
                    t = self.clock.monotonic()
                    # Cache existing feedback before safety checks. Never add a
                    # blocking state query ahead of an urgent stop request.
                    self.last_stats["last_feedback"] = dict(
                        elapsed_s=t-t0, q=st.q.tolist(), tip=st.tip.tolist(),
                        force=st.force.tolist(),
                        endpoint_error_rad=float(np.max(np.abs(st.q-b))))
                    if self.transport.errors():
                        raise SafetyStop("robot reports an active controller error")
                    fraction = float(np.clip(np.dot(st.q - command_start, delta)
                                             / max(float(delta @ delta), 1e-20), 0, 1))
                    reference = command_start + fraction * delta
                    self._posture(st, reference, startup=path.startup)
                    if self.path_checks and np.max(np.abs(st.q - reference)) > self.joint_tolerance:
                        raise SafetyStop("joint tracking left the checked edge")
                    residual = st.force - baseline
                    fixed_residual = st.force - bias
                    mag = float(np.linalg.norm(residual))
                    self.last_stats["last_residual_n"] = mag
                    peak = max(peak, mag)
                    self.last_samples.append((t - started, *st.tip, *residual))
                    self.last_raw_samples.append((t - started, *st.tip, *st.force))
                    if mag > settings.safety_force or np.linalg.norm(fixed_residual) > settings.safety_force:
                        raise SafetyStop("probe exceeded the force safety limit")
                    over = over + 1 if mag > settings.force_threshold else 0
                    if detect_contact and over >= settings.debounce:
                        self.transport.stop()
                        self.active = False
                        # Preserve the detection pose/force, before recoil/retreat.
                        return ProbeResult(True, st.tip.copy(), residual.copy(), peak,
                                           float((st.tip - origin) @ direction), bias, self.last_samples)
                    if mag <= settings.force_threshold:
                        moved = float(np.linalg.norm(st.tip - previous_tip))
                        baseline += min(1., moved / .005 + self.poll / 5.) * residual
                    previous_tip = st.tip.copy()
                    if self.transport.done():
                        self.active = False
                        if done_at is None:
                            done_at = t
                            termination = dict(self.last_stats["last_feedback"])
                            termination["premature"] = done_at-t0 < duration-self.poll
                            termination["joint_error_rad"] = (st.q-b).tolist()
                            # The skill is already finished. Capture q_d and
                            # controller flags before cleanup resets them.
                            if hasattr(self.transport, "diagnostics"):
                                try:
                                    termination["controller"] = self.transport.diagnostics()
                                except Exception as diagnostic_error:
                                    termination["diagnostic_error"] = str(diagnostic_error)
                            self.last_stats["termination"] = termination
                        # Every dispatched skill has ContactTerminationHandler:
                        # installed course code normally ends at duration+buffer,
                        # so an earlier finish is a backstop/preemption, even if
                        # the tiny waypoint happens to lie within joint tolerance.
                        premature = done_at - t0 < duration - self.poll
                        # A retreat can unload the force present during its
                        # initial bias sample. The probe's 4 N contact debounce
                        # must not veto an otherwise completed free-space move;
                        # the force safety limit above still applies to both.
                        if (not premature and (not detect_contact or over == 0)
                                and np.max(np.abs(st.q - b)) <= endpoint_limit):
                            break
                        # Allow a stopped contact a brief debounce window, with
                        # no new command, then report early termination explicitly.
                        if t - done_at >= max(.1, settings.debounce * self.poll):
                            raise SafetyStop("joint skill ended early or short (backstop, preemption, or tracking failure)")
                    if t - t0 > duration + 2. or t - started > settings.timeout:
                        raise SafetyStop("joint segment timed out")
                    self.clock.sleep(self.poll)
                self.last_records.append(dict(index=index, duration=duration,
                                              endpoint_error=float(np.max(np.abs(st.q - b)))))
            st = self.transport.state()
            return ProbeResult(False, st.tip.copy(), st.force - baseline, peak,
                               float((st.tip - origin) @ direction), bias, self.last_samples)
        except BaseException as exc:
            self.last_stats["failure"] = dict(type=type(exc).__name__, message=str(exc))
            if hasattr(self.transport, "diagnostics"):
                try:
                    self.last_stats["error_diagnostics"] = self.transport.diagnostics()
                except Exception as diagnostic_error:
                    self.last_stats["error_diagnostics"] = dict(diagnostic_error=str(diagnostic_error))
            try:
                self.transport.stop()
            except Exception as stop_error:
                self.active = True  # close() must retry, rather than assuming success
                raise SafetyStop(f"execution failed and stop request failed: {stop_error}; use the e-stop") from exc
            self.active = False
            if isinstance(exc, PlanningFailure):
                raise SafetyStop(f"measured joint connector failed clearance checks: {exc}") from exc
            raise

    # ------------------------------------------------------------ streamed path
    def joint_profile(self, joints):
        """Joint-space minimum-jerk profile along the checked knots (no tip-speed term).

        Used for the wrist startup and joint-only corrections, whose tip barely moves.
        Same peak joint velocity and acceleration limits as the segment executor.
        """
        q = np.asarray(joints, float)
        dq = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        s_knots = np.r_[0., np.cumsum(dq)]
        length = float(s_knots[-1])
        total = max(self.min_seconds, 1.875 * length / self.max_velocity,
                    float(np.sqrt(5.774 * length / self.max_acceleration)), 2 * self.min_ramp)

        def s_of(t):
            x = min(max(t / total, 0.), 1.)
            return length * (10 * x**3 - 15 * x**4 + 6 * x**5)

        def q_of(t):
            if length <= 0:
                return q[-1].copy()
            si = s_of(t)
            k = int(np.clip(np.searchsorted(s_knots, si, side="right") - 1, 0, len(q) - 2))
            span = s_knots[k+1] - s_knots[k]
            f = 1. if span <= 0 else (si - s_knots[k]) / span
            return q[k] + min(max(f, 0.), 1.) * (q[k+1] - q[k])

        # The start offset blends out over the first quarter of the motion.
        return dict(total=total, ramp=total / 4, cruise_speed=None, length=length, q_of=q_of, s_of=s_of)

    def stream_profile(self, joints, speed):
        """Time-parameterize the checked knots along tip arc length.

        Joint targets follow the piecewise-linear checked edges; the tip moves at
        a constant speed between quintic (smoothstep) ramps. Peak joint velocity
        and acceleration respect the executor limits on every edge.
        """
        q = np.asarray(joints, float)
        tips = np.array([self.geometry.tip_pose(v)[0] for v in q])
        ds = np.linalg.norm(np.diff(tips, axis=0), axis=1)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be positive")
        if np.any(ds < 1e-6):
            raise SafetyStop("streamed path has a knot without tip motion")
        s_knots = np.r_[0., np.cumsum(ds)]
        length = float(s_knots[-1])
        gain = float(np.max(np.max(np.abs(np.diff(q, axis=0)), axis=1) / ds))  # rad per m of tip
        vc = min(float(speed), self.max_velocity / gain)
        ramp = max(self.min_ramp, 1.875 * vc * gain / self.max_acceleration)
        if length < vc * ramp:
            ramp = max(self.min_ramp, float(np.sqrt(1.875 * length * gain / self.max_acceleration)))
            vc = length / ramp
        total = ramp + length / vc

        def ramp_s(x):
            x = min(max(x, 0.), 1.)
            return vc * ramp * (2.5 * x**4 - 3 * x**5 + x**6)

        def s_of(t):
            if t <= 0:
                return 0.
            if t >= total:
                return length
            if t < ramp:
                return ramp_s(t / ramp)
            if t > total - ramp:
                return length - ramp_s((total - t) / ramp)
            return vc * ramp / 2 + vc * (t - ramp)

        def q_of(t):
            si = s_of(t)
            k = int(np.clip(np.searchsorted(s_knots, si, side="right") - 1, 0, len(q) - 2))
            f = (si - s_knots[k]) / (s_knots[k+1] - s_knots[k])
            return q[k] + min(max(f, 0.), 1.) * (q[k+1] - q[k])

        return dict(total=total, ramp=ramp, cruise_speed=vc, length=length, q_of=q_of, s_of=s_of)

    def _streamable(self, path):
        """Tiny joint-only corrections (no tip motion) stay single segments."""
        tips = np.array([self.geometry.tip_pose(v)[0] for v in path.joints])
        return bool(np.all(np.linalg.norm(np.diff(tips, axis=0), axis=1) >= 1e-4))

    @staticmethod
    def _nearest_on_path(q, joints):
        a, b = joints[:-1], joints[1:]
        d = b - a
        f = np.clip(np.einsum("ij,ij->i", q - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-20), 0, 1)
        points = a + f[:, None] * d
        err = np.max(np.abs(points - q), axis=1)
        k = int(np.argmin(err))
        return points[k], float(err[k])

    def _execute_stream(self, path, speed, settings, *, detect_contact=False):
        self.last_stats = {}
        self.last_samples, self.last_records, self.last_raw_samples = [], [], []
        if not isinstance(path, CheckedJointPath) or path.digest != self._digest(path.joints, path.yaw, path.startup):
            raise SafetyStop("unchecked or modified joint path")
        self._require_yaw(path.yaw)
        try:
            self._checked(path.joints, path.yaw, path.startup)
        except PlanningFailure as exc:
            raise SafetyStop(f"joint-path revalidation failed before motion: {exc}") from exc
        joints = np.asarray(path.joints, float)
        st0 = self.transport.state()
        self._posture(st0, joints[0], startup=path.startup)
        offset = st0.q - joints[0]
        if np.max(np.abs(offset)) > self.end_tolerance:
            raise SafetyStop("robot moved since path preflight; replan before execution")
        try:
            # The first connector from the measured posture must clear the shelves too.
            self.geometry.check_joint_edge(st0.q, joints[1], min_clearance=self.clearance)
        except PlanningFailure as exc:
            raise SafetyStop(f"measured joint connector failed clearance checks: {exc}") from exc
        joint_space = path.startup or not self._streamable(path)
        profile = self.joint_profile(joints) if joint_space else self.stream_profile(joints, speed)
        total, ramp, q_of = profile["total"], profile["ramp"], profile["q_of"]
        if total + self.stream_settle + 1. > settings.timeout:
            raise SafetyStop("planned path exceeds the allowed timeout; shorten it or review the time budget")
        origin = st0.tip.copy()
        final_q = joints[-1]
        final_tip, _ = self.geometry.tip_pose(final_q)
        length = float(np.linalg.norm(final_tip - origin))
        direction = (final_tip - origin) / max(length, 1e-12)
        endpoint_limit = max(self.stream_end_tolerance, self.probe_end_tolerance if detect_contact else 0.)
        readings = []
        for _ in range(max(1, int(settings.bias_seconds / self.poll))):
            readings.append(self.transport.state().force)
            self.clock.sleep(self.poll)
        bias = np.mean(readings, axis=0)
        baseline, previous_tip = bias.copy(), st0.tip.copy()
        # Torque backstop margin 20 N m (not 10): under joint impedance, joint friction shows up
        # as fake external torque and ended a skill with no contact (2026-11-24 run 163112).
        thresholds = (np.abs(bias) + settings.safety_force).tolist() + (np.abs(st0.torque) + self.torque_margin).tolist()
        peak, over = 0., 0
        stats = dict(mode="stream", profile="joint" if joint_space else "tip", duration_s=total, ramp_s=ramp, cruise_speed_mps=profile["cruise_speed"],
                     path_length_m=profile["length"], knots=len(joints), max_cross_track_rad=0.,
                     max_tip_lag_m=0., ticks=0)
        self.last_stats["stream"] = stats

        def base_target(t):
            x = min(max(t / ramp, 0.), 1.)
            return q_of(t) + offset * (1. - (10*x**3 - 15*x**4 + 6*x**5))

        # Velocity feedforward through the position target. frankapy's joint impedance law
        # tau = k (q_target - q) - d dq damps ALL motion, so a joint moving at v trails its
        # target by (d/k) v; joints with different d/k then leave the checked path
        # (2026-11-24 run 162029: joint 6 trailed 17 mrad, tip left the path after 14.5 s).
        # Leading each target by (d/k) * reference velocity cancels that lag.
        lead = np.asarray(self.d_gains, float) / np.asarray(self.k_gains, float)

        def target(t, h=.02):
            a, b = max(t - h, 0.), t + h
            velocity = (base_target(b) - base_target(a)) / (b - a)
            return base_target(t) + lead * velocity

        try:
            self.active = True
            if self.transport.errors():
                raise SafetyStop("robot reports an active controller error before dispatch")
            self.transport.begin_stream(st0.q, total + self.stream_settle + 1., thresholds,
                                        self.k_gains, self.d_gains)
            started = t0 = self.clock.monotonic()
            integ, last_settle = np.zeros(7), None
            trace = stats.setdefault("trace", [])  # ~10 Hz: [t, q - reference(7)] for diagnosis
            while True:
                t = self.clock.monotonic() - t0
                goal = target(t) + integ
                self.transport.stream_target(goal)
                st = self.transport.state()
                now = self.clock.monotonic()
                self.last_stats["last_feedback"] = dict(
                    elapsed_s=now - t0, q=st.q.tolist(), target_q=goal.tolist(), tip=st.tip.tolist(),
                    force=st.force.tolist(), endpoint_error_rad=float(np.max(np.abs(st.q - final_q))))
                stats["ticks"] += 1
                if not trace or now - t0 - trace[-1][0] >= .1:
                    trace.append([round(now - t0, 3)] + np.round(st.q - base_target(t), 5).tolist())
                if self.transport.errors():
                    raise SafetyStop("robot reports an active controller error")
                if self.transport.done():
                    raise SafetyStop("dynamic joint skill ended early (backstop, virtual wall, or preemption)")
                reference, cross = self._nearest_on_path(st.q, joints)
                if t < ramp:
                    # While the start offset blends out, the arm follows the blended target.
                    cross = min(cross, float(np.max(np.abs(st.q - goal))))
                stats["max_cross_track_rad"] = max(stats["max_cross_track_rad"], cross)
                self._posture(st, reference, startup=path.startup)
                if self.path_checks and cross > self.joint_tolerance:
                    raise SafetyStop("joint tracking left the checked path")
                lag = float(np.linalg.norm(self.geometry.tip_pose(base_target(t) + integ)[0] - st.tip))
                stats["max_tip_lag_m"] = max(stats["max_tip_lag_m"], lag)
                residual = st.force - baseline
                fixed_residual = st.force - bias
                mag = float(np.linalg.norm(residual))
                self.last_stats["last_residual_n"] = mag
                peak = max(peak, mag)
                self.last_samples.append((now - started, *st.tip, *residual))
                self.last_raw_samples.append((now - started, *st.tip, *st.force))
                if mag > settings.safety_force or np.linalg.norm(fixed_residual) > settings.safety_force:
                    raise SafetyStop("probe exceeded the force safety limit")
                over = over + 1 if mag > settings.force_threshold else 0
                if detect_contact and over >= settings.debounce:
                    # Hold where the peg is so the impedance stops pushing, then end the skill.
                    self.transport.stream_target(st.q)
                    self.transport.stop()
                    self.active = False
                    self.last_records.append(dict(index=0, duration=now - t0, contact=True))
                    return ProbeResult(True, st.tip.copy(), residual.copy(), peak,
                                       float((st.tip - origin) @ direction), bias, self.last_samples)
                if self.path_checks and lag > self.max_lag:
                    raise SafetyStop("streamed joint target ran ahead of the arm (obstruction or tracking lag)")
                if mag <= settings.force_threshold:
                    moved = float(np.linalg.norm(st.tip - previous_tip))
                    baseline += min(1., moved / .005 + self.stream_poll / 5.) * residual
                previous_tip = st.tip.copy()
                if t >= total:
                    if last_settle is not None:
                        integ = np.clip(integ + self.settle_gain * (final_q - st.q) * (now - last_settle),
                                        -self.settle_max_offset, self.settle_max_offset)
                    last_settle = now
                    stats["settle_offset_rad"] = integ.tolist()
                    # Settle both joints and tool orientation: the next plan needs the wrist
                    # within 5 mrad of the fixed posture, so finish within 3 mrad.
                    _, end_R = self.geometry.tip_pose(final_q)
                    _, now_R = self.geometry.tip_pose(st.q)
                    orient = float(np.linalg.norm(Rotation.from_matrix(end_R @ now_R.T).as_rotvec()))
                    if np.max(np.abs(st.q - final_q)) <= endpoint_limit and orient <= .003:
                        break
                    if t > total + self.stream_settle:
                        raise SafetyStop("streamed path did not settle at its endpoint")
                if now - started > settings.timeout:
                    raise SafetyStop("streamed path timed out")
                self.clock.sleep(self.stream_poll)
            # Keep the settled target (with its friction offset) until the skill ends.
            self.transport.stream_target(goal)
            self.transport.stop()
            self.active = False
            st = self.transport.state()
            if np.max(np.abs(st.q - final_q)) > endpoint_limit:
                raise SafetyStop("arm left the endpoint when the streamed skill ended")
            self.last_records.append(dict(index=0, duration=self.clock.monotonic() - t0,
                                          endpoint_error=float(np.max(np.abs(st.q - final_q)))))
            return ProbeResult(False, st.tip.copy(), st.force - baseline, peak,
                               float((st.tip - origin) @ direction), bias, self.last_samples)
        except BaseException as exc:
            self.last_stats["failure"] = dict(type=type(exc).__name__, message=str(exc))
            if hasattr(self.transport, "diagnostics"):
                try:
                    self.last_stats["termination"] = dict(controller=self.transport.diagnostics())
                except Exception as diagnostic_error:
                    self.last_stats["termination"] = dict(diagnostic_error=str(diagnostic_error))
            try:
                if self.active:
                    try:
                        self.transport.stream_target(self.transport.state().q)
                    except Exception:
                        pass
                self.transport.stop()
            except Exception as stop_error:
                self.active = True
                raise SafetyStop(f"execution failed and stop request failed: {stop_error}; use the e-stop") from exc
            self.active = False
            if isinstance(exc, PlanningFailure):
                raise SafetyStop(f"measured joint connector failed clearance checks: {exc}") from exc
            raise

    def close(self):
        if self.active:
            self.transport.stop()
            self.active = False
