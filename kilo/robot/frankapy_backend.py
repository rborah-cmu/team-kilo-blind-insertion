"""frankapy backend (the course setup: User PC docker -> Control PC franka-interface at 1 kHz).

Free-space moves use static ``goto_pose`` Cartesian impedance skills by lab
configuration. Native guarded probes select static impedance or the internal controller;
streamed probes remain an option. Contact detection runs in Python on
``get_ee_force_torque()`` (libfranka's O_F_ext_hat_K, base frame).

Limits of this backend, measured by the `noise` experiment rather than assumed:
- every state getter is a ROS service call, so the loop rate (target 50 Hz) and
  the latency between contact and stop depend on the lab network;
- frankapy does not expose the per-joint external torques (tau_ext_hat_filtered).

Native sensing and free-air drift tracking were exercised on hardware Nov 22;
the drift-tracking detector still requires hardware contact validation.
"""

from __future__ import annotations

import time

import numpy as np

from .interface import RobotBackend, RobotState, SafetyStop, tool_rotation, yaw_of_tool
from .workspace import HandWorkspace, load_shelf_boxes


class FrankaPyBackend(RobotBackend):
    preferred_hz = 50.0

    def __init__(self, cfg: dict, robot_num: int | None = None):
        import rospy
        from autolab_core import RigidTransform
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy import FrankaArm, SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto import (CartesianImpedanceSensorMessage, PosePositionSensorMessage,
                                    ShouldTerminateSensorMessage)
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg

        self._rospy, self._RigidTransform = rospy, RigidTransform
        self._msg = dict(SensorDataMessageType=SensorDataMessageType, Pose=PosePositionSensorMessage,
                         Impedance=CartesianImpedanceSensorMessage, Terminate=ShouldTerminateSensorMessage,
                         group=make_sensor_group_msg, to_ros=sensor_proto2ros_msg)
        rc, mc = cfg["robot"], cfg["motion"]
        n = int(robot_num or rc.get("robot_num", 1))
        self.fa = FrankaArm(robot_num=n)
        topic = FC.DEFAULT_SENSOR_PUBLISHER_TOPIC if n == 1 else f"franka_ros_interface_{n}/sensor"
        self.pub = rospy.Publisher(topic, SensorDataGroup, queue_size=1000)
        self.probe_length = float(rc["probe_length"])
        self.force_sign = float(rc.get("force_sign", 1.0))
        self.k_trans = [float(k) for k in mc["stream_stiffness"][:3]]
        self.k_rot = [float(k) for k in mc["stream_stiffness"][3:]]
        self.stiffness = min(self.k_trans)
        self.stream_seconds = float(mc.get("stream_skill_seconds", 120.0))
        self.probe_mode = str(mc.get("frankapy_probe_mode", "native"))
        self.native_probe_controller = str(mc.get("native_probe_controller", "internal"))
        if self.native_probe_controller not in ("internal", "impedance"):
            raise ValueError("motion.native_probe_controller must be internal or impedance")
        self.probe_impedance = [float(k) for k in mc.get("native_probe_impedance", [600, 600, 600, 50, 50, 50])]
        self.min_move_seconds = float(mc.get("min_move_seconds", 2.0))
        self.move_mode = str(mc.get("frankapy_move_mode", "internal"))
        if self.move_mode not in ("internal", "impedance"):
            raise ValueError("motion.frankapy_move_mode must be internal or impedance")
        self.travel_impedance = [float(k) for k in mc.get("travel_impedance", self.probe_impedance)]
        self.min_travel_seconds = float(mc.get("min_travel_seconds", self.min_move_seconds))
        self.safety_force = float(cfg["contact"]["safety_force"])
        self.hand_workspace = HandWorkspace(cfg["workspace"])
        self.shelf_boxes = load_shelf_boxes(cfg["workspace"])
        self._id, self._t0, self._streaming, self._rates = 0, 0.0, False, {}

    # ---------------------------------------------------------------- helpers
    def _pose(self, tip, yaw):
        R = tool_rotation(yaw)
        return self._RigidTransform(rotation=R, translation=np.asarray(tip, float) - R[:, 2] * self.probe_length,
                                    from_frame="franka_tool", to_frame="world")

    def _now(self) -> float:
        return self._rospy.Time.now().to_time()

    def _motion_diagnostics(self):
        try:
            raw = self.fa._state_client._get_current_robot_state().robot_state
            active = [n for n in raw.current_errors.__slots__ if getattr(raw.current_errors, n)]
            last = [n for n in raw.last_motion_errors.__slots__ if getattr(raw.last_motion_errors, n)]
            return f"robot_mode={raw.robot_mode}, current_errors={active}, historical_last_errors={last}"
        except Exception:
            return "controller diagnostics unavailable"

    def _check_shelves(self, state):
        if self.shelf_boxes and self.fa.is_joints_in_collision_with_boxes(joints=state.q, boxes=self.shelf_boxes):
            raise SafetyStop("arm entered the inflated Lab 2 shelf keep-out geometry")

    # ---------------------------------------------------------------- interface
    def state(self) -> RobotState:
        pose = self.fa.get_pose()
        ft = np.asarray(self.fa.get_ee_force_torque(), dtype=float)
        R = np.asarray(pose.rotation)
        return RobotState(time=self._now(), tip=np.asarray(pose.translation) + R[:, 2] * self.probe_length,
                          yaw=yaw_of_tool(R), force=self.force_sign * ft[:3], torque=self.force_sign * ft[3:],
                          q=np.asarray(self.fa.get_joints(), dtype=float))

    def move(self, tip, yaw, speed) -> None:
        from .interface import SafetyStop

        st = self.state()
        self._check_shelves(st)
        self.hand_workspace.check(st.tip[:2], np.asarray(tip)[:2], st.yaw, yaw)
        current = st.tip
        duration = max(self.min_travel_seconds, float(np.linalg.norm(np.asarray(tip) - current)) / speed)
        thresholds = (np.abs(st.force) + self.safety_force).tolist() + (np.abs(st.torque) + 10.0).tolist()
        self.fa.goto_pose(self._pose(tip, yaw), duration=duration, use_impedance=self.move_mode == "impedance",
                          cartesian_impedances=self.travel_impedance, force_thresholds=thresholds,
                          block=False, ignore_errors=False, ignore_virtual_walls=False)
        t0 = time.monotonic()
        try:
            while True:
                self._check_shelves(self.state())
                if self.fa.is_skill_done(ignore_errors=False):
                    break
                if time.monotonic() - t0 > duration + 10.0:
                    raise SafetyStop("free-space travel did not finish in time")
                time.sleep(0.05)
        except BaseException:
            self.fa.stop_skill()
            raise
        miss = float(np.linalg.norm(self.state().tip - np.asarray(tip)))
        if miss > 0.015:
            raise SafetyStop(f"move stopped {miss * 1000:.0f} mm short of its target "
                             f"{np.round(tip, 3).tolist()}; {self._motion_diagnostics()}")

    def begin_stream(self) -> None:
        st = self.state()
        self.hand_workspace.check(st.tip[:2], st.tip[:2], st.yaw, st.yaw)
        # frankapy ends dynamic skills with a TimeTerminationHandler driven by buffer_time (not duration),
        # so buffer_time must cover the whole stream; end_stream() terminates it early.
        self.fa.goto_pose(self._pose(st.tip, st.yaw), duration=self.stream_seconds, dynamic=True,
                          buffer_time=self.stream_seconds, cartesian_impedances=self.k_trans + self.k_rot)
        self._t0, self._id, self._streaming = self._now(), 0, True

    def stream_target(self, tip, yaw) -> None:
        st = self.state()
        self.hand_workspace.check(st.tip[:2], np.asarray(tip)[:2], st.yaw, yaw)
        m = self._msg
        self._id += 1
        ts = self._now() - self._t0
        pose = self._pose(tip, yaw)
        traj = m["Pose"](id=self._id, timestamp=ts, position=pose.translation, quaternion=pose.quaternion)
        imp = m["Impedance"](id=self._id, timestamp=ts, translational_stiffnesses=self.k_trans,
                             rotational_stiffnesses=self.k_rot)
        T = m["SensorDataMessageType"]
        self.pub.publish(m["group"](trajectory_generator_sensor_msg=m["to_ros"](traj, T.POSE_POSITION),
                                    feedback_controller_sensor_msg=m["to_ros"](imp, T.CARTESIAN_IMPEDANCE)))

    def tick(self, dt: float) -> None:
        rate = self._rates.get(dt)
        if rate is None:
            rate = self._rates[dt] = self._rospy.Rate(1.0 / dt)
        rate.sleep()

    def end_stream(self) -> None:
        if not self._streaming:
            return
        m = self._msg
        term = m["Terminate"](timestamp=self._now() - self._t0, should_terminate=True)
        self.pub.publish(m["group"](termination_handler_sensor_msg=m["to_ros"](term, m["SensorDataMessageType"].SHOULD_TERMINATE)))
        time.sleep(0.2)
        try:
            self.fa.stop_skill()
        finally:
            self._streaming = False

    def move_and_record(self, tip, yaw, speed: float, hz: float) -> np.ndarray:
        """Non-blocking goto_pose (Franka's internal controller) while sampling the state."""
        if self.probe_mode != "native":
            return super().move_and_record(tip, yaw, speed, hz)
        st = self.state()
        self.hand_workspace.check(st.tip[:2], np.asarray(tip)[:2], st.yaw, yaw)
        start = st.tip
        duration = max(self.min_move_seconds, float(np.linalg.norm(np.asarray(tip) - start)) / speed)
        self.fa.goto_pose(self._pose(tip, yaw), duration=duration, use_impedance=False, block=False,
                          cartesian_impedances=self.probe_impedance, ignore_errors=True)
        rows, t0 = [], time.monotonic()
        while time.monotonic() - t0 < duration + 5.0:
            st = self.state()
            rows.append([st.time, *st.tip, *st.force])
            if self.fa.is_skill_done() and time.monotonic() - t0 > 0.5:
                break
            self.tick(1.0 / hz)
        self.fa.wait_for_skill()
        return np.asarray(rows)

    # ---------------------------------------------------------------- native guarded probe
    BASELINE_TRAVEL = 0.005  # m: the baseline catches up with the force over this much tip travel
    BASELINE_TAU = 5.0       # s: slow time-based catch-up (thermal drift at rest)
    SETTLE = 0.25          # s: controller-start jump settles within this
    MAX_SEGMENTS = 8       # resumes after drift trips frankapy's backstop

    def probe(self, target, yaw, cfg, stop_on_arrival: bool = True):
        """Default on frankapy: static goto_pose (selected controller) with Python-side
        contact detection at ~50 Hz. Set motion.frankapy_probe_mode: stream for the streamed probe.

        Measured on the lab robot (2026-11-22, runs/real/lab/):
          - starting the controller shifts the force estimate ~2.5 N within 0.25 s, before moving;
          - while moving, the estimate DRIFTS about 0.2 N per mm of travel with no contact
            (joint friction / model error), so an offset taken at the start trips any absolute
            threshold after a few cm (a false 'contact' ended the first full search).
        Drift grows per mm the tip actually moves; a contact force grows while the tip is blocked.
        So the baseline follows the force in proportion to tip travel (catching up over
        BASELINE_TRAVEL), and contact is the force exceeding it by ``cfg.force_threshold`` for
        ``cfg.debounce`` samples; Python then stops the skill. frankapy's
        own force_thresholds are only a high backstop (idle offset + ``cfg.safety_force``): if drift
        alone trips it, the probe re-zeroes and resumes from where it stopped."""
        if self.probe_mode != "native":
            return super().probe(target, yaw, cfg, stop_on_arrival)
        from .interface import ProbeResult, SafetyStop

        target = np.asarray(target, float)
        st = self.state()
        self._check_shelves(st)
        self.hand_workspace.check(st.tip[:2], target[:2], st.yaw, yaw)
        baseline = None
        samples, peak, origin, t_global = [], 0.0, None, time.monotonic()
        self.last_probe_samples = samples
        alpha_t = 0.02 / self.BASELINE_TAU
        for _segment in range(self.MAX_SEGMENTS):
            readings = []
            for _ in range(max(5, int(cfg.bias_seconds * 50))):
                readings.append(self.state())
                time.sleep(0.02)
            start = readings[-1].tip.copy()
            origin = start if origin is None else origin
            idle_bias = np.mean([r.force for r in readings], axis=0)
            length = float(np.linalg.norm(target - start))
            if length < 0.002:
                break
            direction = (target - start) / length
            raw_bias = idle_bias * self.force_sign  # franka-interface compares |O_F_ext_hat_K| per axis
            thresholds = (np.abs(raw_bias) + cfg.safety_force).tolist()
            thresholds += (np.abs(np.mean([r.torque for r in readings], axis=0)) + 10.0).tolist()
            duration = max(self.min_move_seconds, length / cfg.speed)
            self.fa.goto_pose(self._pose(target, yaw), duration=duration,
                              use_impedance=self.native_probe_controller == "impedance", block=False,
                              force_thresholds=thresholds, cartesian_impedances=self.probe_impedance,
                              ignore_errors=False, ignore_virtual_walls=False)
            t0 = time.monotonic()
            baseline, over, contact, prev_tip = None, 0, False, start.copy()
            while True:
                t = time.monotonic() - t0
                st = self.state()
                try:
                    self._check_shelves(st)
                except SafetyStop:
                    self.fa.stop_skill()
                    raise
                if t >= self.SETTLE:
                    baseline = st.force.copy() if baseline is None else baseline
                    residual = st.force - baseline
                    mag = float(np.linalg.norm(residual))
                    peak = max(peak, mag)
                    samples.append((time.monotonic() - t_global, *st.tip.tolist(), *residual.tolist()))
                    if mag > cfg.safety_force:
                        self.fa.stop_skill()
                        raise SafetyStop(f"contact force {mag:.1f} N exceeded the {cfg.safety_force:.0f} N safety limit")
                    if mag > cfg.force_threshold:
                        over += 1
                    else:
                        over = 0
                        moved = float(np.linalg.norm(st.tip - prev_tip))
                        w = min(1.0, moved / self.BASELINE_TRAVEL + alpha_t)
                        baseline = baseline + w * (st.force - baseline)  # follows drift while moving
                    prev_tip = st.tip.copy()
                    if over >= cfg.debounce:
                        self.fa.stop_skill()
                        contact = True
                        break
                if self.fa.is_skill_done() and t > 0.5:
                    break
                if t > duration + 10.0:
                    self.fa.stop_skill()
                    raise SafetyStop("native probe did not finish in time")
                time.sleep(0.02)
            t_done = time.monotonic() - t0
            time.sleep(0.1)  # let the state settle at the stop
            end = self.state()
            total = float(np.linalg.norm(target - origin))
            progress = float(np.dot(end.tip - origin, (target - origin) / max(total, 1e-9)))
            if contact:
                residual = end.force - baseline
                result = ProbeResult(True, end.tip.copy(), residual, max(peak, float(np.linalg.norm(residual))),
                                     progress, baseline, samples)
                if cfg.retreat > 0:
                    back = end.tip - direction * cfg.retreat
                    self.hand_workspace.check(end.tip[:2], back[:2], yaw, yaw)
                    self.move(back, yaw, cfg.retreat / self.min_move_seconds)
                return result
            if t_done < 0.6 and np.linalg.norm(end.tip - start) < 0.001:
                raise SafetyStop("frankapy ended the probe motion immediately without moving "
                                 f"(trajectory refused or safety backstop); {self._motion_diagnostics()}")
            if t_done < duration - 0.3:
                continue  # the backstop tripped on drift, not a detected contact: re-zero and resume
            break
        else:
            raise SafetyStop("probe kept tripping frankapy's force backstop without a detected contact")
        end = self.state()
        total = float(np.linalg.norm(target - origin))
        progress = float(np.dot(end.tip - origin, (target - origin) / max(total, 1e-9)))
        residual = end.force - (baseline if baseline is not None else end.force)
        return ProbeResult(False, end.tip.copy(), residual, peak, progress,
                           baseline if baseline is not None else end.force, samples,
                           stalled=total - progress > 0.015)

    def gripper(self, width: float, grasp: bool = False, force: float = 20.0) -> None:
        self.fa.goto_gripper(width, grasp=grasp, force=force, block=True)

    def guide_mode(self, seconds: float) -> None:
        self.fa.run_guide_mode(duration=seconds, block=False)

    def close(self) -> None:
        try:
            self.end_stream()
        finally:
            # Native nonblocking skills also need stopping on interruption/error.
            # Only stop a skill owned by this client, not another ROS client's work.
            if getattr(self.fa, "_in_skill", False):
                self.fa.stop_skill()
