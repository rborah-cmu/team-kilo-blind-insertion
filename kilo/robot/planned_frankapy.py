"""Franka adapter for the checked joint-path executor. No Cartesian motion fallback."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .frankapy_backend import FrankaPyBackend
from .interface import ProbeSettings, SafetyStop, tool_rotation
from .joint_executor import JointPathExecutor
from .planning import FrankaGeometry


class FrankaJointTransport:
    def __init__(self, backend):
        self.backend = backend

    def state(self):
        return self.backend.state()

    def send(self, q, duration, thresholds, impedances):
        self.backend.fa.goto_joints(np.asarray(q).tolist(), duration=float(duration),
                                    use_impedance=False, joint_impedances=impedances,
                                    force_thresholds=thresholds, block=False,
                                    ignore_errors=False, ignore_virtual_walls=False,
                                    skill_desc="Kilo checked joint segment")

    def done(self):
        return self.backend.fa.is_skill_done(ignore_errors=False)

    # Streamed execution: ONE frankapy dynamic joint skill per checked path.
    # frankapy runs it as an ImpedanceControlSkill (libfranka torque control with
    # franka-interface's joint impedance law, PassThroughJointTrajectoryGenerator),
    # so there is no libfranka motion generator and no per-segment motion start.
    # 2026-11-24: every hardware stop so far was joint_motion_generator_*_discontinuity
    # at the start of one of many short JointPositionSkill segments. Virtual walls,
    # joint limits and the force/torque backstop are still checked every 1 ms.
    def begin_stream(self, q, duration, thresholds, k_gains, d_gains):
        from frankapy.proto import JointPositionSensorMessage
        b = self.backend
        self._stream_msg = JointPositionSensorMessage
        b.fa.goto_joints(np.asarray(q, float).tolist(), duration=float(duration), dynamic=True,
                         buffer_time=1.0, force_thresholds=list(thresholds[:6]),
                         k_gains=[float(k) for k in k_gains], d_gains=[float(d) for d in d_gains],
                         ignore_errors=False, ignore_virtual_walls=False,
                         skill_desc="Kilo streamed joint path")
        self._stream_t0, self._stream_id = b._now(), 0

    def stream_target(self, q):
        b, m = self.backend, self.backend._msg
        self._stream_id += 1
        msg = self._stream_msg(id=self._stream_id, timestamp=b._now() - self._stream_t0,
                               joints=np.asarray(q, float).tolist())
        b.pub.publish(m["group"](trajectory_generator_sensor_msg=m["to_ros"](
            msg, m["SensorDataMessageType"].JOINT_POSITION)))

    def command_start(self):
        # The installed JointPositionSkill initializes from q_d, not q.
        raw = self.backend.fa._state_client._get_current_robot_state().robot_state
        return np.asarray(raw.q_d, float)

    def errors(self):
        raw = self.backend.fa._state_client._get_current_robot_state().robot_state
        errors = [n for n in raw.current_errors.__slots__ if getattr(raw.current_errors, n)]
        if not raw.is_fresh:
            errors.append("stale_robot_state")
        if raw.robot_mode not in (1, 2):
            errors.append(f"robot_mode_{raw.robot_mode}")
        return errors

    def diagnostics(self):
        raw = self.backend.fa._state_client._get_current_robot_state().robot_state
        result = {}
        for name in ("q", "q_d", "dq", "dq_d", "cartesian_contact", "cartesian_collision",
                     "joint_contact", "joint_collision", "robot_mode", "is_fresh",
                     "O_F_ext_hat_K", "K_F_ext_hat_K"):
            if hasattr(raw, name):
                result[name] = np.asarray(getattr(raw, name)).tolist()
        for name in ("current_errors", "last_motion_errors"):
            if hasattr(raw, name):
                flags = getattr(raw, name)
                result[name] = [n for n in flags.__slots__ if getattr(flags, n)]
        return result

    def stop(self):
        if getattr(self.backend.fa, "_in_skill", False):
            self.backend.fa.stop_skill()


class PlannedFrankaPyBackend(FrankaPyBackend):
    required_wrist_yaw = np.pi

    def __init__(self, cfg, robot_num=None):
        if (cfg["motion"].get("align_wrist", True)
                or abs(np.angle(np.exp(1j*(cfg["motion"].get("fixed_wrist_yaw", 0.)-np.pi)))) > 1e-5):
            raise ValueError("planned joint execution requires fixed wrist yaw pi and align_wrist=false")
        import rosnode
        if any("franka_arm_client" in n for n in rosnode.get_node_names()):
            raise SafetyStop("another arm client is present")
        super().__init__(cfg, robot_num=robot_num)
        root = Path(__file__).resolve().parents[2]
        options = cfg["joint_executor"]
        self.geometry = FrankaGeometry(root / options["model"], cfg)
        self.executor = JointPathExecutor(self.geometry, FrankaJointTransport(self), options)
        self.required_wrist_yaw = self.executor.yaw
        self.contact_cfg = cfg["contact"]
        self.path_audit = []
        self.motion_error_retries = int(options.get("motion_error_retries", 0))
        self.max_run_recoveries = int(options.get("max_run_recoveries", 0))
        self.recoveries = 0

    def _settings(self):
        c = self.contact_cfg
        return ProbeSettings(force_threshold=c["force_threshold"], safety_force=c["safety_force"],
                             debounce=c["debounce"], bias_seconds=c["bias_seconds"],
                             timeout=c["probe_timeout"], retreat=0)

    @property
    def last_probe_samples(self):
        if getattr(self, "_last_probe_samples", None) is not None:
            return self._last_probe_samples
        return self.executor.last_samples if hasattr(self, "executor") else []

    def _run(self, path, speed, settings, detect=False):
        record = dict(kind="startup" if path.startup else "probe" if detect else "travel",
                      digest=path.digest, joints=path.joints.tolist(), completed=False)
        self.path_audit.append(record)
        try:
            result = self.executor.execute(path, speed, settings, detect_contact=detect)
            record.update(completed=True, contact=result.contact, stop_tip=result.tip.tolist(),
                          peak_force_n=result.peak_force, distance_m=result.distance)
            return result
        except BaseException as exc:
            record["failure"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            record.update(segments=list(self.executor.last_records),
                          tracking=dict(self.executor.last_stats),
                          sample_count=len(self.executor.last_samples))

    # ---------------------------------------------------------------- recovery
    # 2026-11-24 (user wants full runs): frankapy JointPositionSkills on this robot raise
    # joint_motion_generator_{velocity,acceleration}_discontinuity about once per ~100
    # skill starts. The robot stops cleanly where it is. Such a stop - and ONLY such a stop:
    # active-error message, every flag in this set, no contact/collision flag, force
    # residual under the contact threshold - is waited out and the move/probe is replanned
    # from the measured state. Any other stop still ends the run.
    RECOVERABLE_FLAGS = frozenset({"joint_motion_generator_velocity_discontinuity",
                                   "joint_motion_generator_acceleration_discontinuity"})

    def _recoverable(self, exc):
        if not isinstance(exc, SafetyStop) or "active controller error" not in str(exc):
            return False
        stats = self.executor.last_stats
        ctrl = stats.get("error_diagnostics") or (stats.get("termination") or {}).get("controller") or {}
        flags = set(ctrl.get("current_errors") or []) | set(ctrl.get("last_motion_errors") or [])
        if not flags or not flags <= self.RECOVERABLE_FLAGS:
            return False
        for key in ("cartesian_contact", "cartesian_collision", "joint_contact", "joint_collision"):
            if any(abs(float(v)) > 0 for v in (ctrl.get(key) or [])):
                return False
        return float(stats.get("last_residual_n", 0.)) <= float(self.contact_cfg["force_threshold"])

    def _wait_ready(self, timeout=10.):
        import time as _time
        t0 = _time.monotonic()
        while _time.monotonic() - t0 < timeout:
            if not self.executor.transport.errors():
                _time.sleep(.5)
                return
            _time.sleep(.2)
        raise SafetyStop("controller error did not clear after a recoverable stop")

    def _with_recovery(self, what, fn):
        retries = int(getattr(self, "motion_error_retries", 0))
        for attempt in range(retries + 1):
            try:
                return fn()
            except SafetyStop as exc:
                used = getattr(self, "recoveries", 0)
                if (attempt >= retries or used >= int(getattr(self, "max_run_recoveries", 0))
                        or not self._recoverable(exc)):
                    raise
                self.recoveries = used + 1
                st = self.state()
                self.path_audit.append(dict(kind="recovery", completed=True, during=what, attempt=attempt + 1,
                                            reason=str(exc), tip=st.tip.tolist(),
                                            flags=(self.executor.last_stats.get("error_diagnostics") or {})
                                            .get("last_motion_errors")))
                print(f"  [recovered from a motion-generator discontinuity during {what}; "
                      f"replanning from the current pose ({self.recoveries} so far)]", flush=True)
                self._wait_ready()

    def move(self, tip, yaw, speed):
        return self._with_recovery("move", lambda: self._move_once(tip, yaw, speed))

    def _move_once(self, tip, yaw, speed):
        self.executor._require_yaw(yaw)
        st = self.state()
        self.hand_workspace.check(st.tip[:2], np.asarray(tip)[:2], st.yaw, yaw)
        _, current_rotation = self.geometry.tip_pose(st.q)
        orientation_error = np.linalg.norm(Rotation.from_matrix(
            tool_rotation(self.required_wrist_yaw) @ current_rotation.T).as_rotvec())
        # Planning requires 5 mrad orientation accuracy, tighter than the
        # 30 mrad emergency tracking envelope. Correct small startup residuals
        # at the high pose instead of attempting a mismatched Cartesian path.
        if orientation_error > .004:
            # Correct the wrist in place first (checked joint rotation, tip moves < 2 mm).
            # plan_startup itself refuses below the high-clearance height, so a low arm
            # still has to be repositioned by hand.
            if st.tip[2] < self.executor.startup_height and np.linalg.norm(np.asarray(tip) - st.tip) > .002:
                raise SafetyStop("prepare the planned wrist at a high pose before translating")
            self._run(self.executor.plan_startup(), speed, self._settings())
        # Travel is not a contact probe: merge checked samples into fewer skills.
        path = self.executor.plan(tip, yaw, max_step=self.executor.travel_step)
        self._run(path, speed, self._settings())

    def probe(self, target, yaw, cfg, stop_on_arrival=True):
        if not stop_on_arrival:
            raise SafetyStop("planned probing requires a bounded target")
        self._last_probe_samples = None

        def attempt():
            st = self.state()
            self.hand_workspace.check(st.tip[:2], np.asarray(target)[:2], st.yaw, yaw)
            # Entire ray checked before any motion. No replay of hypothetical contacts.
            p = self.executor.plan(target, yaw)
            return p, self._run(p, cfg.speed, cfg, detect=True)

        path, result = self._with_recovery("probe", attempt)
        samples = result.samples
        self._last_probe_samples = samples
        if result.contact and cfg.retreat > 0:
            direction = np.asarray(target) - path_start_tip(self.geometry, path)
            direction /= max(np.linalg.norm(direction), 1e-12)
            back = result.tip - direction * cfg.retreat
            self.move(back, yaw, cfg.speed)
            # Preserve contact samples rather than the retreat's samples.
            self.executor.last_samples = samples
        return result

    def move_and_record(self, tip, yaw, speed, hz):
        self.move(tip, yaw, speed)
        return np.asarray(self.executor.last_raw_samples)

    def begin_stream(self):
        raise SafetyStop("streamed Cartesian skills are disabled by the joint-path executor")

    def stream_target(self, tip, yaw):
        raise SafetyStop("streamed Cartesian skills are disabled by the joint-path executor")

    def close(self):
        try:
            if hasattr(self, "executor"):
                self.executor.close()
        finally:
            super().close()


def path_start_tip(geometry, path):
    return geometry.tip_pose(path.joints[0])[0]
