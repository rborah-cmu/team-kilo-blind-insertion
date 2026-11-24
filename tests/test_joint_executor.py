from dataclasses import replace

import numpy as np
import pytest
import yaml

from kilo.config import REPO_ROOT
from kilo.robot.interface import ProbeSettings, SafetyStop
from kilo.robot.joint_executor import JointPathExecutor
from kilo.robot.offline_joint_transport import OfflineJointTransport, VirtualClock
from kilo.robot.planning import FrankaGeometry
from kilo.robot.planned_frankapy import FrankaJointTransport


def setup(box=None, xyz=(.36, -.075, .12), startup=False):
    cfg = yaml.safe_load((REPO_ROOT / "configs/real_planned.yaml").read_text())
    g = FrankaGeometry(REPO_ROOT / "configs/frankapy_geometry.json", cfg)
    home = np.array([0., -.785, 0., -2.356, 0., 1.571, -.785])
    seed = home.copy()
    seed[6] += np.pi
    q = home if startup else g.inverse(xyz, np.pi, seed, elbow=0.)
    clock = VirtualClock()
    transport = OfflineJointTransport(g, q, clock, box=box)
    executor = JointPathExecutor(g, transport, dict(cfg["joint_executor"], execution="segments", path_checks=True), clock)
    settings = ProbeSettings(force_threshold=4, timeout=300, bias_seconds=.02, retreat=0)
    return g, transport, executor, clock, settings


def test_checked_joint_motion_reaches_target_with_posture_and_limits():
    g, t, e, clock, s = setup()
    target = t.state().tip + [.012, 0, 0]
    path = e.plan(target, np.pi)
    result = e.execute(path, .01, s, detect_contact=True)
    assert not result.contact
    assert np.linalg.norm(t.state().tip - target) < .0005
    assert abs(t.state().q[2]) < 1e-5
    assert abs(np.angle(np.exp(1j*(t.state().yaw-np.pi)))) < .005
    assert t.commands and t.stop_count == 0
    previous = path.joints[0]
    for command in t.commands:
        delta = np.max(np.abs(command["q"] - previous))
        assert 1.875 * delta / command["duration"] <= e.max_velocity + 1e-8
        assert 5.774 * delta / command["duration"]**2 <= e.max_acceleration + 1e-8
        previous = command["q"]


@pytest.mark.parametrize("fault,message", [("joint3", "joint 3"), ("yaw", "wrist yaw"),
                                           ("tip", "peg tip"), ("force", "force safety")])
def test_fault_stops_active_skill_and_never_dispatches_next_segment(fault, message):
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip + [.02, 0, 0], np.pi)
    t.fault, t.fault_after = fault, .10
    with pytest.raises(SafetyStop, match=message):
        e.execute(path, .01, s, detect_contact=True)
    assert len(t.commands) == 1 and t.stop_count == 1
    assert not e.active


def test_active_robot_error_blocks_dispatch():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip + [.006, 0, 0], np.pi)
    t.controller_errors = ["joint_reflex"]
    with pytest.raises(SafetyStop, match="active controller error"):
        e.execute(path, .01, s)
    assert not t.commands


def test_early_backstop_does_not_resume_even_for_small_waypoints():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip + [.01, 0, 0], np.pi)
    t.early_done = True
    with pytest.raises(SafetyStop, match="ended early or short"):
        e.execute(path, .01, s, detect_contact=True)
    assert len(t.commands) == 1 and t.stop_count == 1


def test_stalled_joint_skill_times_out_and_stops():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip + [.006, 0, 0], np.pi)
    t.hold = True
    # A held simulated skill must remain active to exercise timeout.
    t.done = lambda: False
    with pytest.raises(SafetyStop, match="timed out"):
        e.execute(path, .01, s)
    assert len(t.commands) == 1 and t.stop_count == 1


def test_modified_and_stale_paths_are_rejected_before_motion():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip + [.006, 0, 0], np.pi)
    wrong = path.joints.copy()
    wrong[-1, 2] += .05
    with pytest.raises(SafetyStop, match="unchecked or modified"):
        e.execute(replace(path, joints=wrong), .01, s)
    t.q[0] += .01
    with pytest.raises(SafetyStop, match="moved since|left the checked|wrist yaw|tool orientation"):
        e.execute(path, .01, s)
    assert not t.commands


def test_rejects_wrong_half_turn_before_dispatch():
    g, t, e, clock, s = setup()
    with pytest.raises(SafetyStop, match="180-degree"):
        e.plan(t.state().tip + [.01, 0, 0], 0.)
    assert not t.commands


def test_whole_unsafe_ray_is_refused_even_if_contact_would_be_early():
    g, t, e, clock, s = setup(box=(.477, -.106, 0), xyz=(.36, -.075, .0322))
    with pytest.raises(SafetyStop, match="preflight failed"):
        e.plan([.72, -.075, .0322], np.pi)
    assert not t.commands


def test_detects_contact_before_reaching_end_of_prechecked_ray():
    g, t, e, clock, s = setup(box=(.477, -.106, 0), xyz=(.36, -.075, .0322))
    start = t.state().tip.copy()
    path = e.plan(start + [.065, 0, 0], np.pi)
    result = e.execute(path, .01, s, detect_contact=True)
    assert result.contact and result.peak_force < 20
    assert result.tip[0] == pytest.approx(.477-.1267/2-.0125, abs=.0005)
    assert result.force[0] < -4
    assert t.stop_count == 1
    assert len(t.commands) < len(path.joints)-1


def test_startup_is_a_checked_high_clearance_joint_rotation():
    g, t, e, clock, s = setup(startup=True)
    start = t.state().tip.copy()
    path = e.plan_startup()
    assert path.startup and len(path.joints) == 2
    e.execute(path, .02, s)
    assert len(t.commands) == 1
    assert np.linalg.norm(t.state().tip - start) < .002
    assert abs(np.angle(np.exp(1j*(t.state().yaw-np.pi)))) < .005


def test_startup_refuses_low_pose_before_motion():
    g, t, e, clock, s = setup(xyz=(.36, -.075, .12))
    with pytest.raises(SafetyStop, match="high-clearance"):
        e.plan_startup()
    assert not t.commands


def test_transport_only_dispatches_guarded_joint_commands():
    class Arm:
        def goto_joints(self, *args, **kwargs):
            self.command = (args, kwargs)
        def goto_pose(self, *args, **kwargs):
            pytest.fail("Cartesian fallback must not be called")
    arm = Arm()
    adapter = FrankaJointTransport(type("Backend", (), {"fa": arm})())
    adapter.send(np.zeros(7), 1., [20]*6, [600]*7)
    args, kwargs = arm.command
    assert kwargs["ignore_virtual_walls"] is False
    assert kwargs["ignore_errors"] is False
    assert kwargs["block"] is False
    assert kwargs["force_thresholds"] == [20]*6


def test_keyboard_interrupt_stops_owned_joint_skill():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip+[.006, 0, 0], np.pi)
    original = clock.sleep
    def interrupt(seconds):
        if t.commands:
            raise KeyboardInterrupt
        original(seconds)
    clock.sleep = interrupt
    with pytest.raises(KeyboardInterrupt):
        e.execute(path, .01, s)
    assert len(t.commands) == 1 and t.stop_count == 1 and not e.active


def test_endpoint_lag_is_included_in_actual_command_timing():
    g, t, e, clock, s = setup()
    t.joint_lag[5] = -.0008
    path = e.plan(t.state().tip+[.012, 0, 0], np.pi)
    e.execute(path, .01, s)
    for command in t.commands:
        d = np.max(np.abs(command["q"]-command["start_q"]))
        assert 1.875*d/command["duration"] <= e.max_velocity + 1e-8
        assert 5.774*d/command["duration"]**2 <= e.max_acceleration + 1e-8


def test_timeout_budget_is_checked_before_dispatch():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip+[.02, 0, 0], np.pi)
    s.timeout = .01
    with pytest.raises(SafetyStop, match="exceeds the allowed timeout"):
        e.execute(path, .01, s)
    assert not t.commands


def test_failed_stop_remains_active_for_cleanup_retry():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip+[.006, 0, 0], np.pi)
    t.fault, t.fault_after = "force", .1
    original_stop = t.stop
    def failed_stop():
        raise RuntimeError("transport disconnected")
    t.stop = failed_stop
    with pytest.raises(SafetyStop, match="stop request failed"):
        e.execute(path, .01, s)
    assert e.active
    t.stop = original_stop
    e.close()
    assert not e.active and t.stop_count == 1


def test_glancing_corner_contact_can_build_force_with_bounded_tracking_lag():
    g, t, e, clock, s = setup(box=(.472, -.111, np.radians(-5)), xyz=(.565, -.21, .0322))
    path = e.plan([.32, -.21, .0322], np.pi)
    result = e.execute(path, .01, s, detect_contact=True)
    assert result.contact and 4 < result.peak_force < 20
    assert result.tip[0] == pytest.approx(.5361, abs=.001)
    assert t.stop_count == 1


def test_controller_desired_start_must_match_measured_posture():
    g, t, e, clock, s = setup()
    path = e.plan(t.state().tip+[.006, 0, 0], np.pi)
    t.command_start = lambda: t.q + np.array([0, 0, .03, 0, 0, 0, 0])
    with pytest.raises(SafetyStop, match="desired joints disagree"):
        e.execute(path, .01, s)
    assert not t.commands


def test_nearly_tangential_contact_aborts_instead_of_loosening_tracking():
    g, t, e, clock, s = setup(box=(.482, -.111, np.radians(5)), xyz=(.565, -.21, .0322))
    path = e.plan([.32, -.21, .0322], np.pi)
    with pytest.raises(SafetyStop, match="ended early or short"):
        e.execute(path, .01, s, detect_contact=True)
    assert t.stop_count == 1 and not e.active
    assert 0 < len(t.commands) < len(path.joints)-1
    assert e.last_stats["max_joint3_error_rad"] < .01
    assert e.last_stats["max_wrist_tracking_error_rad"] < .03
    assert e.last_stats["min_actual_model_clearance_m"] > 0


@pytest.mark.parametrize("fresh,mode,expected", [(False, 1, "stale_robot_state"), (True, 5, "robot_mode_5")])
def test_transport_rejects_stale_or_user_stopped_state(fresh, mode, expected):
    from types import SimpleNamespace
    class Errors:
        __slots__ = ("joint_reflex",)
        def __init__(self):
            self.joint_reflex = False
    raw = SimpleNamespace(is_fresh=fresh, robot_mode=mode, current_errors=Errors())
    client = SimpleNamespace(_get_current_robot_state=lambda: SimpleNamespace(robot_state=raw))
    adapter = FrankaJointTransport(SimpleNamespace(fa=SimpleNamespace(_state_client=client)))
    assert expected in adapter.errors()


def test_small_high_pose_wrist_residual_gets_checked_correction():
    from scripts.real.test_joint_sequence import make_offline_backend
    cfg = yaml.safe_load((REPO_ROOT / "configs/real_planned.yaml").read_text())
    g = FrankaGeometry(REPO_ROOT / "configs/frankapy_geometry.json", cfg)
    q = g.inverse([.36, 0., .30], np.pi+.0108, [0, -.5, 0, -2.3, 0, 1.8, 2.34], elbow=0.)
    robot, transport, clock = make_offline_backend(cfg, q, None)
    start = robot.state().tip.copy()
    robot.move(start, np.pi, .02)
    assert robot.path_audit[0]["kind"] == "startup"
    assert abs(np.angle(np.exp(1j*(robot.state().yaw-np.pi)))) < .004
    assert np.linalg.norm(robot.state().tip-start) < .002
    robot.close()


@pytest.mark.parametrize("early", [True, False])
def test_failed_skill_captures_termination_before_stop_resets_controller(early):
    import json
    g, t, e, clock, s = setup(startup=True)
    path = e.plan_startup()
    t.early_done = early
    if not early:
        t.joint_lag[6] = -.004
    t.diagnostics = lambda: {"q_d": path.joints[-1].tolist(), "stop_count": t.stop_count}
    with pytest.raises(SafetyStop, match="ended early or short"):
        e.execute(path, .02, s)
    termination = e.last_stats["termination"]
    assert termination["premature"] is early
    assert termination["controller"]["stop_count"] == 0
    assert termination["controller"]["q_d"] == path.joints[-1].tolist()
    assert e.last_stats["last_segment"]["index"] == 0
    assert termination["endpoint_error_rad"] > e.end_tolerance
    assert t.stop_count == 1 and len(t.commands) == 1
    json.dumps(e.last_stats, allow_nan=False)


def test_failed_diagnostics_does_not_prevent_stop_or_failure_audit():
    from scripts.real.test_joint_sequence import make_offline_backend
    cfg = yaml.safe_load((REPO_ROOT / "configs/real_planned.yaml").read_text())
    g, t, e, clock, s = setup(startup=True)
    robot, t, clock = make_offline_backend(cfg, t.q, None)
    robot.executor.mode = "segments"  # this test covers segment early-termination handling
    t.early_done = True
    def unavailable():
        raise RuntimeError("diagnostic service unavailable")
    t.diagnostics = unavailable
    with pytest.raises(SafetyStop, match="ended early or short"):
        robot._run(robot.executor.plan_startup(), .02, s)
    audit = robot.path_audit[-1]
    assert not audit["completed"] and audit["sample_count"] > 0
    assert audit["failure"]["type"] == "SafetyStop"
    assert "unavailable" in audit["tracking"]["termination"]["diagnostic_error"]
    assert t.stop_count == 1
    robot.close()


def test_transport_diagnostics_serializes_optional_ros_fields():
    import json
    from types import SimpleNamespace
    class Errors:
        __slots__ = ("joint_reflex",)
        def __init__(self):
            self.joint_reflex = True
    raw = SimpleNamespace(q=np.zeros(7), q_d=np.ones(7), robot_mode=1,
                          is_fresh=True, current_errors=Errors())
    client = SimpleNamespace(_get_current_robot_state=lambda: SimpleNamespace(robot_state=raw))
    adapter = FrankaJointTransport(SimpleNamespace(fa=SimpleNamespace(_state_client=client)))
    result = adapter.diagnostics()
    assert result["current_errors"] == ["joint_reflex"]
    assert result["q_d"] == [1.]*7
    json.dumps(result, allow_nan=False)


def test_travel_step_merges_checked_samples_into_fewer_skills():
    g, t, e, clock, s = setup()
    target = np.array([.36, -.075, .12]) + np.array([.12, 0., 0.])
    fine = e.plan(target, np.pi)
    coarse = e.plan(target, np.pi, max_step=e.travel_step)
    assert e.travel_step > e.step
    assert len(coarse.joints) < len(fine.joints) / 5
    tips = np.array([g.tip_pose(q)[0] for q in coarse.joints])
    assert np.max(np.linalg.norm(np.diff(tips, axis=0), axis=1)) <= e.travel_step + 1e-6
    result = e.execute(coarse, .02, s)
    assert not result.contact and np.linalg.norm(result.tip - target) < .002
    with pytest.raises(SafetyStop):
        e.plan(target, np.pi, max_step=e.travel_step * 3)


def stream_setup(**kw):
    g, t, e, clock, s = setup(**kw)
    e.mode = "stream"
    return g, t, e, clock, s


def test_stream_travel_is_one_skill_and_reaches_the_target():
    g, t, e, clock, s = stream_setup()
    target = np.array([.36, -.075, .12]) + np.array([.15, -.05, 0.])
    path = e.plan(target, np.pi, max_step=e.travel_step)
    result = e.execute(path, .02, s)
    assert not result.contact
    assert np.linalg.norm(result.tip - target) < .002
    assert [c.get("kind") for c in t.commands] == ["stream"]
    assert t.stream_targets > 100
    stats = e.last_stats["stream"]
    assert stats["max_cross_track_rad"] < e.joint_tolerance
    assert stats["max_tip_lag_m"] < e.max_lag
    assert stats["cruise_speed_mps"] <= .02 + 1e-9


def test_stream_profile_respects_joint_limits():
    g, t, e, clock, s = stream_setup()
    target = np.array([.36, -.075, .12]) + np.array([.15, -.05, 0.])
    path = e.plan(target, np.pi, max_step=e.travel_step)
    prof = e.stream_profile(path.joints, .5)  # speed far above the joint limits
    ts = np.arange(0, prof["total"], .001)
    q = np.array([prof["q_of"](x) for x in ts])
    v = np.abs(np.diff(q, axis=0)) / .001
    assert v.max() <= e.max_velocity * 1.02
    assert np.allclose(q[0], path.joints[0]) and np.allclose(prof["q_of"](prof["total"]), path.joints[-1])


def test_stream_probe_detects_contact_and_holds():
    g, t, e, clock, s = stream_setup(box=(.477, -.106, 0), xyz=(.36, -.075, .0322))
    start = t.state().tip.copy()
    path = e.plan(start + [.065, 0, 0], np.pi)
    result = e.execute(path, .01, s, detect_contact=True)
    assert result.contact and result.peak_force < 20
    assert result.tip[0] == pytest.approx(.477-.1267/2-.0125, abs=.001)
    assert result.force[0] < -4
    assert t.stop_count == 1 and len(t.commands) == 1


def test_stream_stops_on_controller_error():
    g, t, e, clock, s = stream_setup()
    t.controller_errors = []
    target = np.array([.36, -.075, .12]) + np.array([.10, 0., 0.])
    path = e.plan(target, np.pi, max_step=e.travel_step)
    original = t.stream_target
    calls = {"n": 0}

    def flaky(q):
        calls["n"] += 1
        if calls["n"] == 50:
            t.controller_errors = ["joint_reflex"]
        original(q)
    t.stream_target = flaky
    with pytest.raises(SafetyStop, match="controller error"):
        e.execute(path, .02, s)
    assert t.stop_count >= 1 and not e.active


def test_stream_stops_if_skill_ends_early():
    g, t, e, clock, s = stream_setup()
    target = np.array([.36, -.075, .12]) + np.array([.10, 0., 0.])
    path = e.plan(target, np.pi, max_step=e.travel_step)
    original = t.stream_target
    calls = {"n": 0}

    def ender(q):
        calls["n"] += 1
        if calls["n"] == 40:
            t.stream = None
        original(q)
    t.stream_target = ender
    with pytest.raises(SafetyStop, match="ended early"):
        e.execute(path, .02, s)


def test_startup_is_one_streamed_joint_skill_in_stream_mode():
    g, t, e, clock, s = stream_setup(startup=True)
    start_tip = t.state().tip.copy()
    path = e.plan_startup()
    e.execute(path, .02, s)
    assert [c.get("kind") for c in t.commands] == ["stream"]
    assert e.last_stats["stream"]["profile"] == "joint"
    st = t.state()
    assert np.max(np.abs(st.q - path.joints[-1])) <= e.stream_end_tolerance
    assert np.linalg.norm(st.tip - start_tip) < .003
    assert abs(np.angle(np.exp(1j*(st.yaw - np.pi)))) < .004


def test_stream_settle_offset_cancels_wrist_friction():
    g, t, e, clock, s = stream_setup(startup=True)
    t.stream_bias = np.array([0, 0, 0, 0, 0, 0, -.0036])  # run 161055: joint 7 stuck 3.6 mrad short
    path = e.plan_startup()
    e.execute(path, .02, s)
    offset = e.last_stats["stream"]["settle_offset_rad"]
    assert .0005 < offset[6] <= e.settle_max_offset  # crept past the endpoint just enough
    assert np.max(np.abs(t.state().q - path.joints[-1])) <= e.stream_end_tolerance


def test_stream_without_settle_offset_stops_short_safely():
    g, t, e, clock, s = stream_setup(startup=True)
    e.settle_gain = 0.
    t.stream_bias = np.array([0, 0, 0, 0, 0, 0, -.0036])
    with pytest.raises(SafetyStop, match="did not settle"):
        e.execute(e.plan_startup(), .02, s)
    assert not e.active


def test_discontinuity_stop_is_recovered_and_move_completes():
    from scripts.real.test_joint_sequence import make_offline_backend
    cfg = yaml.safe_load((REPO_ROOT / "configs/real_planned.yaml").read_text())
    g, t0, e0, clock, s = setup()
    robot, t, clock = make_offline_backend(cfg, t0.q, None)
    robot.executor.mode = "segments"
    robot.motion_error_retries, robot.max_run_recoveries, robot.recoveries = 3, 30, 0
    flags = ["joint_motion_generator_velocity_discontinuity", "joint_motion_generator_acceleration_discontinuity"]
    state = {"sent": 0, "fired": False}
    send = t.send

    def flaky_send(*a, **k):
        state["sent"] += 1
        send(*a, **k)
    t.send = flaky_send
    t.errors = lambda: (["joint_motion_generator_velocity_discontinuity"]
                        if state["sent"] == 3 and not state["fired"] and t.segment is not None else [])
    t.diagnostics = lambda: dict(current_errors=[], last_motion_errors=flags, cartesian_contact=[0.]*6,
                                 cartesian_collision=[0.]*6, joint_contact=[0.]*7, joint_collision=[0.]*7)
    original_stop = t.stop

    def stop():
        state["fired"] = state["sent"] >= 3
        original_stop()
    t.stop = stop
    import time as _t
    sleep = _t.sleep
    _t.sleep = lambda s: None
    try:
        target = t.state().tip + np.array([.10, 0., 0.])
        robot.move(target, np.pi, .02)
    finally:
        _t.sleep = sleep
    assert robot.recoveries == 1
    assert any(a.get("kind") == "recovery" for a in robot.path_audit)
    assert np.linalg.norm(t.state().tip - target) < .002


def test_non_discontinuity_error_is_not_recovered():
    from scripts.real.test_joint_sequence import make_offline_backend
    cfg = yaml.safe_load((REPO_ROOT / "configs/real_planned.yaml").read_text())
    g, t0, e0, clock, s = setup()
    robot, t, clock = make_offline_backend(cfg, t0.q, None)
    robot.executor.mode = "segments"
    robot.motion_error_retries, robot.max_run_recoveries, robot.recoveries = 3, 30, 0
    t.errors = lambda: ["joint_reflex"] if t.segment is not None else []
    t.diagnostics = lambda: dict(current_errors=["joint_reflex"], last_motion_errors=["joint_reflex"])
    with pytest.raises(SafetyStop, match="controller error"):
        robot.move(t.state().tip + np.array([.10, 0., 0.]), np.pi, .02)
    assert robot.recoveries == 0
