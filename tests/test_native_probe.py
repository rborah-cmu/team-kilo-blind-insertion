"""Deterministic force traces exercise the real backend without ROS or hardware."""

from types import SimpleNamespace

import numpy as np
import pytest

import kilo.robot.frankapy_backend as backend
from kilo.robot.interface import ProbeSettings, RobotState, SafetyStop
from kilo.robot.workspace import HandWorkspace


def fake_robot(monkeypatch, face=None, force_jump=0):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(backend, "time", SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds)))
    robot = backend.FrankaPyBackend.__new__(backend.FrankaPyBackend)
    robot.probe_mode = "native"
    robot.native_probe_controller = "internal"
    robot.hand_workspace = HandWorkspace({})
    robot.shelf_boxes = []
    robot.force_sign, robot.min_move_seconds = -1, 2
    robot.probe_impedance = [1500] * 3 + [100] * 3
    robot._pose = lambda tip, yaw: np.array(tip)
    motion = SimpleNamespace(start=None, duration=0, origin=np.array([0.35, 0., 0.1]),
                             target=None, stopped=False, frozen=None, stop_count=0)

    def state():
        tip, penetration = motion.origin.copy(), 0.
        if motion.start is not None:
            fraction = min(1., (clock.now - motion.start) / motion.duration)
            commanded = motion.origin + fraction * (motion.target - motion.origin)
            tip = motion.frozen.copy() if motion.stopped else commanded.copy()
            if face is not None:
                tip[0] = min(tip[0], face)
                penetration = max(0., commanded[0] - face)
        # 0.2 N/mm drift while travelling, 2.5 N controller-start offset,
        # plus contact spring force that rises while the peg is blocked.
        force = -(200 * (tip[0] - 0.35) + 1500 * penetration)
        if motion.start is not None:
            force -= 2.5 + (force_jump if clock.now - motion.start > 0.5 else 0)
        return RobotState(clock.now, tip, 0., np.array([force, 0., 0.]), np.zeros(3))

    def goto(target, duration, **kwargs):
        motion.kwargs = kwargs
        motion.origin = state().tip.copy()
        motion.start, motion.duration, motion.target = clock.now, duration, target
        motion.stopped, motion.frozen = False, None

    def stop():
        motion.frozen = state().tip.copy()
        motion.stopped = True
        motion.stop_count += 1

    robot.state = state
    robot.fa = SimpleNamespace(goto_pose=goto, stop_skill=stop,
                               is_skill_done=lambda **kwargs: motion.stopped or clock.now - motion.start >= motion.duration)
    return robot, motion


def test_native_probe_tracks_long_free_air_drift(monkeypatch):
    robot, motion = fake_robot(monkeypatch)
    result = robot.probe([0.47, 0., 0.1], 0., ProbeSettings(retreat=0))
    assert not result.contact and not result.stalled
    assert result.distance == pytest.approx(0.12)
    assert result.peak_force < 4
    assert motion.stop_count == 0


@pytest.mark.parametrize("controller", ["internal", "impedance"])
def test_native_probe_controller_keeps_guarding_enabled(monkeypatch, controller):
    robot, motion = fake_robot(monkeypatch, face=0.39)
    robot.native_probe_controller = controller
    result = robot.probe([0.47, 0., 0.1], 0., ProbeSettings(force_threshold=4, retreat=0))
    assert result.contact
    assert motion.kwargs["use_impedance"] == (controller == "impedance")
    assert motion.kwargs["ignore_virtual_walls"] is False
    assert motion.kwargs["ignore_errors"] is False
    assert motion.kwargs["force_thresholds"][:3] == [20., 20., 20.]


def test_native_probe_detects_force_rise_when_tip_is_blocked(monkeypatch):
    robot, motion = fake_robot(monkeypatch, face=0.39)
    result = robot.probe([0.47, 0., 0.1], 0., ProbeSettings(force_threshold=4, retreat=0))
    assert result.contact
    assert result.tip[0] == pytest.approx(0.39)
    assert result.force[0] < -4
    assert result.peak_force < 20
    assert motion.stop_count == 1


def test_native_probe_stops_on_sudden_safety_force(monkeypatch):
    robot, motion = fake_robot(monkeypatch, force_jump=30)
    with pytest.raises(SafetyStop, match="safety limit"):
        robot.probe([0.47, 0., 0.1], 0., ProbeSettings(retreat=0))
    assert motion.stop_count == 1


def test_native_probe_already_at_target_returns_without_starting_skill(monkeypatch):
    robot, motion = fake_robot(monkeypatch)
    result = robot.probe([0.3505, 0., 0.1], 0., ProbeSettings(retreat=0))
    assert not result.contact and not result.stalled
    assert motion.start is None


def test_free_travel_uses_selected_controller_and_keeps_safety_checks(monkeypatch):
    robot, _ = fake_robot(monkeypatch)
    robot.move_mode = "impedance"
    robot.min_travel_seconds = 6.0
    robot.travel_impedance = [1500] * 3 + [100] * 3
    robot.safety_force = 20.
    target = np.array([0.36, 0., 0.1])
    calls = []
    def goto(pose, **kwargs):
        calls.append(kwargs)
        robot.state = lambda: RobotState(0, target, 0., np.zeros(3), np.zeros(3))
    robot.fa.goto_pose = goto
    robot.fa.is_skill_done = lambda **kwargs: True
    robot.move(target, 0., .02)
    assert calls[0]["duration"] == 6.
    assert calls[0]["use_impedance"] is True
    assert calls[0]["ignore_virtual_walls"] is False
    assert calls[0]["ignore_errors"] is False
    assert calls[0]["force_thresholds"] == [20., 20., 20., 10., 10., 10.]


def test_free_travel_stops_if_arm_enters_shelf_margin(monkeypatch):
    robot, motion = fake_robot(monkeypatch)
    robot.move_mode = "impedance"
    robot.min_travel_seconds = 6.
    robot.travel_impedance = [1500] * 3 + [100] * 3
    robot.safety_force = 20.
    robot.shelf_boxes = [[.4, .37, .14, 0, 0, 0, .49, .14, .005]]
    checks = []
    def collision(**kwargs):
        checks.append(kwargs)
        return len(checks) > 1
    robot.fa.is_joints_in_collision_with_boxes = collision
    with pytest.raises(SafetyStop, match="shelf keep-out"):
        robot.move([.36, 0, .1], 0, .02)
    assert motion.stop_count == 1


def test_close_stops_owned_native_skill_after_interruption():
    robot = backend.FrankaPyBackend.__new__(backend.FrankaPyBackend)
    robot._streaming = False
    stops = []
    robot.fa = SimpleNamespace(_in_skill=True, stop_skill=lambda: stops.append(True))
    robot.close()
    assert stops == [True]
