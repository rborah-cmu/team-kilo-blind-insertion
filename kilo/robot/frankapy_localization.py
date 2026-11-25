"""Run the simulated probe pattern on a Franka using the Lab 2 joint commands."""

import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from kilo.localization.common import (
    POS_EPS,
    PROBE_LOOKAHEAD,
    SEARCH_MODE_OFFSET,
    SEARCH_SPACE_CORNERS,
    SEARCH_SPEED,
    TRAVEL_MODE_OFFSET,
    TRAVEL_SPEED,
    probe_rotation,
)
from kilo.localization.search import generate_ray_probe_waypoints
from kilo.sim.control import ARM_DOF, solve_ik
from kilo.sim.scene import build_scene

JOINT_TOLERANCE = 0.01       # Lab 2 live waypoint tracking, rad
JOINT_PEAK_SPEED = 0.15      # Lab 2 CLI default, rad/s
TOOL_Z_PEAK_SPEED = 1.5 * JOINT_PEAK_SPEED  # Panda joint 7 rotates about tool Z
JOINT_PEAK_ACCEL = 0.30      # Lab 2 live timing, rad/s²
MIN_MOVE_SECONDS = 0.5       # Lab 2 live timing
IK_POSITION_TOLERANCE = 0.0005  # Lab 2 IK, m
IK_ROTATION_TOLERANCE = 0.003   # Lab 2 IK, rad
CONTACT_THRESHOLD = 5.0     # N, horizontal force above the probe baseline
FORCE_LIMIT = 20.0
FORCE_ZERO_SECONDS = 5.0
FORCE_ZERO_HZ = 20.0
MIN_FORCE_ZERO_HZ = 10.0
POLL_SECONDS = 0.02
BASELINE_TRAVEL = 0.005     # m, free-air drift tracking from the earlier bench test
BASELINE_SECONDS = 5.0      # s, slower drift tracking while nearly stationary
STATIONARY_SPEED = 0.01     # Lab 2 stationary joint speed, rad/s
SETTLE_TIMEOUT = 5.0
MOTION_RETRIES = 2
MOTION_DISCONTINUITIES = frozenset({
    "joint_motion_generator_velocity_discontinuity",
    "joint_motion_generator_acceleration_discontinuity",
})


class JointTrackingError(RuntimeError):
    pass


class FrankaLocalizer:
    """Map the simulation's peg-tip motions to FrankaPy joint skills."""

    def __init__(self, arm, table_z):
        self.arm = arm
        self.model = build_scene()
        self.data = mujoco.MjData(self.model)
        self.search_z = table_z + SEARCH_MODE_OFFSET
        self.travel_z = table_z + TRAVEL_MODE_OFFSET
        self.force_zero = None

    def zero_forces(self, log_path=None):
        """Average a stationary wrench recording before commanding any motion."""
        self.force_zero = None
        samples = []
        print(f"Recording the force zero for {FORCE_ZERO_SECONDS:g} s", flush=True)
        start = time.monotonic()
        deadline = start + FORCE_ZERO_SECONDS
        sample_time = start
        while sample_time < deadline:
            time.sleep(max(0., sample_time - time.monotonic()))
            force = np.asarray(self.arm.get_ee_force_torque(), dtype=float)[:3]
            measured_at = time.monotonic()
            if measured_at >= deadline:
                break
            if force.shape != (3,) or not np.all(np.isfinite(force)):
                raise RuntimeError("Invalid force sample while zeroing")
            samples.append((measured_at - start, *force))
            sample_time = max(sample_time + 1. / FORCE_ZERO_HZ, measured_at)
        time.sleep(max(0., deadline - time.monotonic()))
        rate = len(samples) / FORCE_ZERO_SECONDS
        if rate < MIN_FORCE_ZERO_HZ:
            raise RuntimeError(f"Force zero collected {rate:g} Hz, below the required 10 Hz")
        samples = np.asarray(samples)
        if log_path is not None:
            np.savetxt(log_path, samples, delimiter=",", fmt="%.6f",
                       header="elapsed_s,fx_n,fy_n,fz_n", comments="")
        self.force_zero = samples[:, 1:].mean(axis=0)
        print(f"Force zero at {rate:g} Hz from {len(samples)} samples: "
              f"Fx={self.force_zero[0]:+.3f}, Fy={self.force_zero[1]:+.3f}, "
              f"Fz={self.force_zero[2]:+.3f} N", flush=True)

    def _read_joints(self):
        joints = np.asarray(self.arm.get_joints(), dtype=float)
        if joints.shape != (ARM_DOF,) or not np.all(np.isfinite(joints)):
            raise ValueError("Expected seven finite joint angles from FrankaPy")
        return joints

    def _pose(self, joints):
        self.data.qpos[:ARM_DOF] = joints
        mujoco.mj_kinematics(self.model, self.data)
        site = self.data.site("peg_tip")
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def _check_joints(self, expected):
        joints = self._read_joints()
        error = float(np.max(np.abs(joints - expected)))
        if error > JOINT_TOLERANCE:
            raise JointTrackingError(f"Lab 2 joint tracking tolerance exceeded ({error:.4f} rad)")
        return joints

    @staticmethod
    def _flags(errors):
        return {name for name in errors.__slots__ if getattr(errors, name)}

    def _wait_stationary(self, *, retrying=False):
        # FrankaPy can acknowledge preemption before the real-time arm stops.
        deadline = time.monotonic() + SETTLE_TIMEOUT
        previous_time = None
        stationary_samples = 0
        while time.monotonic() < deadline:
            state = self.arm._state_client._get_current_robot_state().robot_state
            if state.is_fresh and state.robot_time != previous_time:
                previous_time = state.robot_time
                errors = self._flags(state.current_errors)
                collisions = any(state.joint_collision) or any(state.cartesian_collision)
                if collisions or errors - MOTION_DISCONTINUITIES:
                    raise RuntimeError(f"Franka reported collision or active faults: {sorted(errors)}")
                if retrying:
                    past_errors = self._flags(state.last_motion_errors)
                    if (past_errors - MOTION_DISCONTINUITIES
                            or any(state.joint_contact) or any(state.cartesian_contact)):
                        raise RuntimeError("Franka reported contact or another fault; motion cannot be retried")
                idle = (state.robot_mode == 1 and not errors
                        and np.max(np.abs(state.dq)) <= STATIONARY_SPEED)
                stationary_samples = stationary_samples + 1 if idle else 0
                if stationary_samples >= 3:
                    status = self.arm._get_current_franka_interface_status().franka_interface_status
                    if status.is_fresh and status.is_ready and not status.error_description:
                        return np.asarray(state.q, dtype=float)
            time.sleep(POLL_SECONDS)
        raise RuntimeError("Franka did not report fresh, stationary, ready feedback")

    @staticmethod
    def _retryable(error):
        if isinstance(error, JointTrackingError):
            return True
        from frankapy.exceptions import (FrankaArmCommException, FrankaArmException,
                                         FrankaArmFrankaInterfaceNotReadyException)
        if isinstance(error, (FrankaArmCommException, FrankaArmFrankaInterfaceNotReadyException)):
            return True
        message = str(error).lower()
        return (isinstance(error, FrankaArmException)
                and any(flag in message for flag in MOTION_DISCONTINUITIES)
                and not any(fault in message for fault in ("reflex", "collision", "virtual wall")))

    def _probe_force(self):
        wrench = np.asarray(self.arm.get_ee_force_torque(), dtype=float)
        # This lab's FrankaPy wrench sign is opposite to the force on the peg.
        force = self.force_zero - wrench[:3]
        if np.linalg.norm(force) > FORCE_LIMIT:
            raise RuntimeError(f"Probe exceeded {FORCE_LIMIT:g} N")
        return force[:2]

    def _poll_for_contact(self, baseline, previous_tip):
        previous_time = time.monotonic()
        while True:
            force = self._probe_force() - baseline
            if np.linalg.norm(force) > CONTACT_THRESHOLD:
                return force
            if self.arm.is_skill_done(ignore_errors=False):
                return None
            tip, _ = self._pose(self._read_joints())
            now = time.monotonic()
            travel = np.linalg.norm(tip - previous_tip)
            baseline += min(1., travel / BASELINE_TRAVEL
                            + (now - previous_time) / BASELINE_SECONDS) * force
            previous_tip, previous_time = tip, now
            time.sleep(POLL_SECONDS)

    def _move_to_waypoint(self, waypoint, rotation, joints, thresholds, *,
                          detect_contact=False, speed=SEARCH_SPEED):
        for attempt in range(MOTION_RETRIES + 1):
            goal, _ = solve_ik(self.model, "peg_tip", waypoint, rotation, q0=joints)
            goal_tip, goal_rotation = self._pose(goal)
            position_error = np.linalg.norm(waypoint - goal_tip)
            rotation_error = Rotation.from_matrix(rotation @ goal_rotation.T).magnitude()
            if (position_error > IK_POSITION_TOLERANCE
                    or rotation_error > IK_ROTATION_TOLERANCE):
                raise RuntimeError(f"Lab 2 IK tolerances exceeded for peg tip at {waypoint}")
            measured = self._check_joints(joints)
            current_tip, _ = self._pose(measured)
            joint_deltas = np.abs(goal - measured)
            joint_delta = float(np.max(joint_deltas))
            joint_speed_time = 1.875 * max(
                float(np.max(joint_deltas[:-1])) / JOINT_PEAK_SPEED,
                float(joint_deltas[-1]) / TOOL_Z_PEAK_SPEED,
            )
            distance = float(np.linalg.norm(waypoint - current_tip))
            duration = max(MIN_MOVE_SECONDS, joint_speed_time,
                           np.sqrt(5.774 * joint_delta / JOINT_PEAK_ACCEL),
                           1.875 * distance / speed)
            try:
                if detect_contact:
                    baseline = self._probe_force()
                self.arm.goto_joints(goal.tolist(), duration=float(duration), use_impedance=False,
                                     block=not detect_contact, ignore_errors=False,
                                     ignore_virtual_walls=False, force_thresholds=thresholds,
                                     skill_desc="Kilo plate probe" if detect_contact else "Kilo plate travel")
                force = self._poll_for_contact(baseline, current_tip) if detect_contact else None
                if force is None:
                    return self._check_joints(goal), None
            except Exception as error:
                if attempt == MOTION_RETRIES or not self._retryable(error):
                    raise
                if getattr(self.arm, "_in_skill", False):
                    self.arm.stop_skill()
                joints = self._wait_stationary(retrying=True)
                if isinstance(error, JointTrackingError) and np.max(np.abs(joints - goal)) <= JOINT_TOLERANCE:
                    return joints, None
                print(f"Retrying from measured joints ({attempt + 1}/{MOTION_RETRIES}): "
                      f"{error}; previous command was {joint_delta:.4f} rad, {duration:.2f} s", flush=True)
                continue

            # A contact always ends the probe. Complete cancellation and
            # settle the arm before dispatching the vertical lift.
            self.arm.stop_skill()
            stopped_joints = self._read_joints()
            tip, _ = self._pose(stopped_joints)
            return self._wait_stationary(), (tip[:2], force)

    def _move(self, target, detect_contact=False, *, vertical=False):
        if self.force_zero is None:
            raise RuntimeError("Zero the force readings before moving the arm")
        joints = self._read_joints()
        start, current_rotation = self._pose(joints)
        if vertical:
            target = np.r_[start[:2], target]
        delta = target - start
        length = float(np.linalg.norm(delta))
        if length < POS_EPS:
            return None
        direction = delta / length
        rotation = probe_rotation(direction, current_rotation)
        speed = SEARCH_SPEED if detect_contact else TRAVEL_SPEED
        steps = int(np.ceil(length / PROBE_LOOKAHEAD))
        # Franka-Interface uses per-axis limits in the raw wrench frame.
        thresholds = (np.abs(self.force_zero) + FORCE_LIMIT).tolist() + [10.] * 3
        for waypoint in np.linspace(start, target, steps + 1)[1:]:
            joints, contact = self._move_to_waypoint(waypoint, rotation, joints, thresholds,
                                                      detect_contact=detect_contact, speed=speed)
            if contact is not None:
                return contact
        return None

    def position_at(self, xy):
        """Lift, cross at travel height, then lower as in the simulation."""
        self._move(self.travel_z, vertical=True)
        self._move(np.r_[xy, self.travel_z])
        self._move(self.search_z, vertical=True)

    def travel(self, ray):
        start, end = generate_ray_probe_waypoints(ray, SEARCH_SPACE_CORNERS)
        if start is None or end is None:
            raise RuntimeError("Contact ray does not cross the search rectangle")
        self.position_at(start)
        return start, end

    def probe(self, xy):
        contact = self._move(np.r_[xy, self.search_z], detect_contact=True)
        if contact is not None:
            print(f"Contact at {np.linalg.norm(contact[1]):.2f} N XY. Lifting vertically.", flush=True)
            self._move(self.travel_z, vertical=True)
        return contact
