"""Stream the real-robot spiral insertion reference to the Franka (frankapy, no RL).

One frankapy dynamic goto_pose skill (ImpedanceControlSkill with Cartesian impedance,
pass-through pose targets) runs the whole search + insertion: torque control, so no
joint-motion-generator discontinuity stops. The target and xy stiffness come from
kilo.insertion.real_spiral.RealSpiralReference every tick (50 Hz).

Safety stops (the skill is ended and the arm holds where it is):
  any residual force axis > force_limit, xy excursion from the estimate, tip far from the
  target beyond the commanded preload, the skill ending on its own, or the timeout.
"""

from __future__ import annotations

import time

import numpy as np

from kilo.insertion.real_spiral import RealSpiralReference


class InsertionStop(RuntimeError):
    pass


class FrankaSpiralInserter:
    def __init__(self, arm, cfg, *, tip_fn, force_fn, view=None, record=None, clock=time):
        """``tip_fn()`` -> peg-tip xyz in the base frame (calibrated FK).
        ``force_fn()`` -> contact force ON THE PEG, xyz (N), already zeroed.
        ``record(t, tip, target, force, phase)`` is called every tick (logging)."""
        self.arm, self.cfg = arm, cfg
        self.tip_fn, self.force_fn = tip_fn, force_fn
        self.view, self.record, self.clock = view, record, clock
        self._pub = None

    # ---------------------------------------------------------------- frankapy plumbing
    def _publisher(self):
        if self._pub is None:
            import rospy
            from franka_interface_msgs.msg import SensorDataGroup
            from frankapy import FrankaConstants as FC
            self._rospy = rospy
            self._pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        return self._pub

    def _start_skill(self, pose, kxy, seconds):
        c = self.cfg
        self.arm.goto_pose(pose, duration=seconds, dynamic=True, buffer_time=seconds,
                           cartesian_impedances=[kxy, kxy, c["z_stiffness"]] + list(c["rotation_stiffness"]),
                           ignore_errors=False, ignore_virtual_walls=False, skill_desc="Kilo spiral insertion")

    def _send(self, msg_id, t, position, quaternion, kxy):
        from frankapy import SensorDataMessageType
        from frankapy.proto import CartesianImpedanceSensorMessage, PosePositionSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
        c = self.cfg
        pose_msg = PosePositionSensorMessage(id=msg_id, timestamp=t, position=list(position),
                                             quaternion=list(quaternion))
        imp_msg = CartesianImpedanceSensorMessage(id=msg_id, timestamp=t,
                                                  translational_stiffnesses=[kxy, kxy, c["z_stiffness"]],
                                                  rotational_stiffnesses=list(c["rotation_stiffness"]))
        self._publisher().publish(make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(pose_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(imp_msg, SensorDataMessageType.CARTESIAN_IMPEDANCE)))

    # ---------------------------------------------------------------- run
    def run(self, estimate_xy, expected_top_z):
        c = self.cfg
        ee = self.arm.get_pose()
        tip = np.asarray(self.tip_fn(), float)
        offset = tip - np.asarray(ee.translation, float)   # tip = ee + offset (fixed tool orientation)
        quaternion = np.asarray(ee.quaternion, float)
        ref = RealSpiralReference(c, estimate_xy, expected_top_z, tip)
        dt = ref.dt
        result = dict(status="running", estimate_xy=list(map(float, estimate_xy)),
                      expected_top_z=float(expected_top_z), phases=[])
        self._start_skill(ee, c["search_xy_stiffness"], c["timeout"] + 5.)
        t0 = self.clock.monotonic()
        msg_id, last_phase = 0, None
        xy_bias = np.zeros(2)  # integral xy correction (approach/search only)
        try:
            while True:
                now = self.clock.monotonic()
                t = now - t0
                tip = np.asarray(self.tip_fn(), float)
                force = np.asarray(self.force_fn(), float)
                target, kxy = ref.step(tip, float(force[2]))
                if ref.phase != last_phase:
                    result["phases"].append(dict(t=round(t, 3), phase=ref.phase, tip=tip.tolist()))
                    last_phase = ref.phase
                    print(f"  [{t:6.2f} s] phase {ref.phase}"
                          + (f", surface z {ref.surface_z:.4f} m" if ref.surface_z is not None else "")
                          + (f" (touch by {ref.touch_reason})" if ref.phase == ref.SEARCH and ref.touch_reason else ""),
                          flush=True)
                    if ref.phase not in (ref.APPROACH, ref.SEARCH):
                        xy_bias[:] = 0.  # capture/insert: soft xy at the locked point, no lean
                    if self.view is not None:
                        self.view.status(f"insertion: {ref.phase}")
                depth = ref.depth_reference - tip[2]
                if self.record is not None:
                    self.record(t, tip, target, force, ref.phase)
                if self.view is not None:
                    self.view.sample(t, tip, force)
                if ref.done:
                    result.update(status="seated", depth_m=float(depth), seconds=t, final_tip=tip.tolist(),
                                  seat_reason=ref.seat_reason)
                    print(f"Seated ({ref.seat_reason}): {depth * 1000:.1f} mm below the surface after {t:.1f} s",
                          flush=True)
                    break
                # ---- safety
                if np.any(np.abs(force) > c["force_limit"]):
                    raise InsertionStop(f"force {np.round(force, 1).tolist()} N exceeds {c['force_limit']} N")
                if np.linalg.norm(tip[:2] - np.asarray(estimate_xy)) > c["max_xy_excursion"]:
                    raise InsertionStop("peg left the search region around the hole estimate")
                lead = c["insert_z_lead"] if ref.phase in (ref.INSERT,) else c["search_z_lead"]
                xy_err = float(np.linalg.norm(target[:2] - tip[:2]))  # vs the reference, not the lean
                if xy_err > c["max_tracking_error"] or tip[2] - target[2] > lead + c["max_tracking_error"]:
                    raise InsertionStop(f"tip not following the target (xy {xy_err * 1000:.1f} mm)")
                if t > c["timeout"]:
                    raise InsertionStop(f"no insertion within {c['timeout']:.0f} s "
                                        f"(phase {ref.phase}, {ref.spiral_laps} spiral laps)")
                if msg_id > 5 and self.arm.is_skill_done(ignore_errors=False):
                    raise InsertionStop("the impedance skill ended early (virtual wall, joint limit or contact backstop)")
                command = target.copy()
                if ref.phase in (ref.APPROACH, ref.SEARCH) and c.get("xy_integral_gain", 0) > 0:
                    xy_bias = np.clip(xy_bias + c["xy_integral_gain"] * (target[:2] - tip[:2]) * dt,
                                      -c["xy_integral_max"], c["xy_integral_max"])
                    command[:2] += xy_bias
                msg_id += 1
                self._send(msg_id, t, command - offset, quaternion, kxy)
                spent = self.clock.monotonic() - now
                self.clock.sleep(max(0., dt - spent))
        except BaseException as exc:
            result.update(status="stopped", reason=f"{type(exc).__name__}: {exc}",
                          seconds=self.clock.monotonic() - t0, phase=ref.phase)
            raise
        finally:
            try:
                # Hold where the peg is, then end the skill.
                tip = np.asarray(self.tip_fn(), float)
                self._send(msg_id + 1, self.clock.monotonic() - t0, tip - offset, quaternion,
                           c["insert_xy_stiffness"])
            except Exception:
                pass
            try:
                self.arm.stop_skill()
            except Exception as stop_error:
                print(f"stop_skill failed: {stop_error}; use the e-stop if the arm moves", flush=True)
            result.update(surface_z=ref.surface_z, spiral_laps=ref.spiral_laps, spiral_skips=ref.skips,
                          path=ref.path[::5].tolist())
            self.last_result = result
        return result


def move_keep_wrist(robot, target, *, vertical=False):
    """Straight-line peg-tip move with the tool pointing down at its CURRENT yaw.

    The localizer's ``_move`` turns the tool to face each move's direction
    (probe_rotation), which for sideways moves can demand a joint-7 angle past its
    limit ("Joints not reachable!", 2026-11-24). Insertion does not care about yaw,
    so keep it fixed. Uses the localizer's checked waypoint routine (IK, retries,
    force backstop) unchanged.
    """
    from kilo.localization.common import PROBE_LOOKAHEAD, TRAVEL_SPEED, probe_rotation
    from kilo.robot.frankapy_localization import FORCE_LIMIT

    if robot.force_zero is None:
        raise RuntimeError("Zero the force readings before moving the arm")
    joints = robot._read_joints()
    start, current_rotation = robot._pose(joints)
    target = np.r_[start[:2], target] if vertical else np.asarray(target, float)
    length = float(np.linalg.norm(target - start))
    if length < 1e-6:
        return joints
    rotation = probe_rotation(np.zeros(3), current_rotation)  # tool down, same yaw
    thresholds = (np.abs(robot.force_zero) + FORCE_LIMIT).tolist() + [10.] * 3
    steps = int(np.ceil(length / PROBE_LOOKAHEAD))
    for waypoint in np.linspace(start, target, steps + 1)[1:]:
        joints, _ = robot._move_to_waypoint(waypoint, rotation, joints, thresholds, speed=TRAVEL_SPEED)
    return joints
