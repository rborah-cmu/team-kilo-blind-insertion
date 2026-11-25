"""Exercise the hardware probe and motion recovery with a fake FrankaPy arm."""

from types import SimpleNamespace

import numpy as np
import pytest

from kilo.robot import frankapy_localization as live
from kilo.sim.control import solve_ik


class Flags:
    __slots__ = ("joint_reflex",)

    def __init__(self, joint_reflex=False):
        self.joint_reflex = joint_reflex


def make_robot(monkeypatch):
    clock = SimpleNamespace(elapsed=0.)

    def sleep(dt):
        clock.elapsed += dt

    monkeypatch.setattr(live, "time", SimpleNamespace(monotonic=lambda: clock.elapsed, sleep=sleep))
    stopped = SimpleNamespace(joints=None, idle_after=0., reflex=False)

    def read_state():
        idle = clock.elapsed >= stopped.idle_after
        return SimpleNamespace(robot_state=SimpleNamespace(
            is_fresh=True, robot_time=clock.elapsed, robot_mode=1 if idle else 2,
            q=stopped.joints, dq=np.zeros(7) if idle else np.full(7, .02),
            current_errors=Flags(stopped.reflex), last_motion_errors=Flags(stopped.reflex),
            joint_collision=(), cartesian_collision=(), joint_contact=(), cartesian_contact=(),
        ))

    arm = SimpleNamespace(
        get_joints=lambda: stopped.joints.copy(),
        get_ee_force_torque=lambda: np.zeros(6),
        is_skill_done=lambda **kwargs: True,
        _state_client=SimpleNamespace(_get_current_robot_state=read_state),
        _get_current_franka_interface_status=lambda: SimpleNamespace(
            franka_interface_status=SimpleNamespace(is_fresh=True, is_ready=True, error_description=""),
        ),
    )
    robot = live.FrankaLocalizer(arm, .0072)
    start = np.array([.50, -.15, robot.search_z])
    rotation = live.probe_rotation(np.array([1., 0., 0.]), np.eye(3))
    stopped.joints, _ = solve_ik(robot.model, "peg_tip", start, rotation)
    robot.zero_forces()
    return robot, arm, stopped, clock, start, rotation


def test_probe_stops_once_then_lifts_from_the_measured_tip(monkeypatch):
    robot, arm, state, clock, start, rotation = make_robot(monkeypatch)
    surface = start + [.025, 0., 0.]
    surface_joints, _ = solve_ik(robot.model, "peg_tip", surface, rotation, q0=state.joints)
    commands = []
    stop_times = []

    def goto(goal, **kwargs):
        commands.append((np.asarray(goal), kwargs["block"], clock.elapsed))
        state.joints = np.asarray(goal) if kwargs["block"] else surface_joints.copy()

    def stop():
        stop_times.append(clock.elapsed)
        state.idle_after = clock.elapsed + .06

    arm.goto_joints = goto
    arm.stop_skill = stop
    arm.get_ee_force_torque = lambda: (
        [-6., 0., -6., 0., 0., 0.] if commands and not stop_times else np.zeros(6)
    )
    tip, force = robot.probe(np.array([.54, -.15]))

    assert [blocking for _, blocking, _ in commands] == [False, True]
    assert stop_times == [commands[0][2]]
    assert commands[1][2] >= state.idle_after
    lift_tip, _ = robot._pose(commands[1][0])
    np.testing.assert_allclose(lift_tip, [*surface[:2], robot.travel_z], atol=1e-4)
    np.testing.assert_allclose(tip, surface[:2], atol=1e-4)
    np.testing.assert_allclose(force, [6., 0.])


@pytest.mark.parametrize("reflex", [False, True])
def test_waypoint_retry_uses_measured_joints_and_never_retries_reflex(monkeypatch, reflex):
    robot, arm, state, clock, start, _ = make_robot(monkeypatch)
    durations = []

    def goto(goal, **kwargs):
        durations.append(kwargs["duration"])
        if len(durations) == 1:
            state.joints = (state.joints + np.asarray(goal)) / 2.
            state.reflex = reflex
        else:
            state.joints = np.asarray(goal)

    arm.goto_joints = goto
    if reflex:
        with pytest.raises(RuntimeError, match="active faults"):
            robot._move(start + [.04, 0., 0.])
        assert len(durations) == 1
    else:
        robot._move(start + [.04, 0., 0.])
        assert len(durations) == 2
        assert durations[1] < durations[0]
