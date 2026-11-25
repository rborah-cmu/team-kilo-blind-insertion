"""Real-robot spiral reference against a simple plate-with-hole contact model."""

import numpy as np
import pytest
import yaml

from kilo.config import REPO_ROOT
from kilo.insertion.real_spiral import RealSpiralReference

CFG = yaml.safe_load((REPO_ROOT / "configs/real_insertion.yaml").read_text())["insertion_real"]
TOP = 0.0580
CAPTURE = 0.0002 + 0.001  # radial clearance + chamfer: where the shaft can start to sink


def run(hole_offset, cfg=CFG, max_s=120.):
    """Tip follows the impedance target; the top blocks it except over the hole."""
    hole = np.array([0.51, -0.07])
    estimate = hole - np.asarray(hole_offset)
    tip = np.r_[estimate, TOP + cfg["approach_height"]]
    ref = RealSpiralReference(cfg, estimate, TOP, tip)
    dt, force_z = ref.dt, 0.
    for k in range(int(max_s / dt)):
        target, kxy = ref.step(tip, force_z)
        if ref.done:
            return True, k * dt, ref
        tip[:2] += min(1., dt / (0.05 if kxy >= 600 else 0.15)) * (target[:2] - tip[:2])
        tip[2] += min(1., dt / 0.05) * (target[2] - tip[2])
        floor = -np.inf if np.linalg.norm(tip[:2] - hole) <= CAPTURE else TOP
        if tip[2] < floor:  # blocked by the top: the z spring presses the peg down on it
            force_z = cfg["z_stiffness"] * (floor - target[2])
            tip[2] = floor
        else:
            force_z = 0.
    return False, max_s, ref


@pytest.mark.parametrize("offset", [(0, 0), (0.002, 0), (0, -0.004), (0.005, 0.005), (-0.007, 0.003)])
def test_spiral_finds_and_seats_from_estimate_error(offset):
    ok, t, ref = run(offset)
    assert ok, f"not seated from offset {offset}: phase {ref.phase}, surface {ref.surface_z}"
    assert ref.surface_z == pytest.approx(TOP, abs=1e-6) or ref.phase == "seated"
    assert t < 90


def test_surface_measured_on_touch_and_preload_is_bounded():
    hole = np.array([0.51, -0.07])
    est = hole + [0.008, 0]
    tip = np.r_[est, TOP]
    ref = RealSpiralReference(CFG, est, TOP, tip)
    for _ in range(CFG["touch_samples"]):
        target, _ = ref.step(tip, CFG["touch_force"] + 1)
    assert ref.phase == "search" and ref.surface_z == pytest.approx(TOP)
    target, kxy = ref.step(tip, 5.)
    assert tip[2] - target[2] == pytest.approx(CFG["search_z_lead"])
    assert kxy == CFG["search_xy_stiffness"]


class _Pose:
    def __init__(self, t):
        self.translation, self.quaternion = np.asarray(t, float), np.array([0., 1., 0., 0.])


class _FakeArm:
    """Peg tip = ee + offset; the tip follows streamed targets; a plate with a hole below."""
    OFFSET = np.array([0., 0., -0.15])

    def __init__(self, hole, start_tip, block_everywhere=False):
        self.hole, self.tip = np.asarray(hole, float), np.asarray(start_tip, float)
        self.target, self.force, self.block = self.tip.copy(), np.zeros(3), block_everywhere
        self.started = self.stopped = False

    def get_pose(self):
        return _Pose(self.tip - self.OFFSET)

    def goto_pose(self, pose, **kw):
        assert kw["dynamic"] and kw["cartesian_impedances"][2] == CFG["z_stiffness"]
        self.started = True

    def is_skill_done(self, ignore_errors=False):
        return self.stopped

    def stop_skill(self):
        self.stopped = True

    def advance(self, dt):
        inside = self.tip[2] < TOP - 1e-6  # shaft already in the bore
        self.tip[:2] += min(1., dt / 0.05) * (self.target[:2] - self.tip[:2])
        if inside:  # the bore holds the shaft laterally (0.2 mm radial clearance)
            off = self.tip[:2] - self.hole
            n = np.linalg.norm(off)
            if n > 0.0002:
                self.tip[:2] = self.hole + off / n * 0.0002
        self.tip[2] += min(1., dt / 0.05) * (self.target[2] - self.tip[2])
        over = (np.linalg.norm(self.tip[:2] - self.hole) <= CAPTURE or inside) and not self.block
        if not over and self.tip[2] < TOP:
            self.force = np.array([0., 0., CFG["z_stiffness"] * (TOP - self.target[2])])
            self.tip[2] = TOP
        else:
            self.force = np.zeros(3)


class _Clock:
    def __init__(self, arm, dt):
        self.t, self.arm, self.dt = 0., arm, dt

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.arm.advance(self.dt)
        self.t += self.dt


def _inserter(arm, cfg=CFG):
    from kilo.robot.frankapy_insertion import FrankaSpiralInserter
    clock = _Clock(arm, 1 / cfg["policy_rate_hz"])
    ins = FrankaSpiralInserter(arm, cfg, tip_fn=lambda: arm.tip.copy(), force_fn=lambda: arm.force.copy(), clock=clock)
    ins._send = lambda mid, t, pos, quat, kxy: setattr(arm, "target", np.asarray(pos) + arm.OFFSET)
    return ins


def test_robot_loop_streams_spiral_and_seats():
    hole = np.array([0.51, -0.07])
    est = hole + [0.004, -0.003]
    arm = _FakeArm(hole, np.r_[est, TOP + CFG["approach_height"]])
    res = _inserter(arm).run(est, TOP)
    assert arm.started and arm.stopped
    assert res["status"] == "seated" and res["depth_m"] >= CFG["seat_depth"]
    assert [p["phase"] for p in res["phases"]][:2] == ["approach", "search"]


def test_robot_loop_stops_on_force_limit():
    from kilo.robot.frankapy_insertion import InsertionStop
    hole = np.array([0.51, -0.07])
    cfg = dict(CFG, search_z_lead=0.06)  # absurd preload -> > 20 N on touch
    arm = _FakeArm(hole, np.r_[hole + [0.008, 0], TOP + 0.002], block_everywhere=True)
    with pytest.raises(InsertionStop, match="force"):
        _inserter(arm, cfg).run(hole + [0.008, 0], TOP)
    assert arm.stopped


def test_robot_loop_times_out_when_no_hole():
    from kilo.robot.frankapy_insertion import InsertionStop
    hole = np.array([0.51, -0.07])
    arm = _FakeArm(hole, np.r_[hole, TOP + 0.005], block_everywhere=True)
    with pytest.raises(InsertionStop, match="no insertion within"):
        _inserter(arm, dict(CFG, timeout=20.)).run(hole, TOP)
    assert arm.stopped


def test_approach_keeps_the_wrist_yaw():
    """move_keep_wrist sends every waypoint with the starting yaw, tool pointing down."""
    from scipy.spatial.transform import Rotation
    from kilo.robot.frankapy_insertion import move_keep_wrist

    yaw0 = np.radians(125.8)
    R0 = Rotation.from_euler("z", yaw0).as_matrix() @ np.diag([1., -1., -1.])  # tool down

    class Robot:
        force_zero = np.zeros(3)

        def __init__(self):
            self.tip, self.rotations = np.array([0.5067, -0.0613, 0.1068]), []

        def _read_joints(self):
            return np.zeros(7)

        def _pose(self, joints):
            return self.tip.copy(), R0

        def _move_to_waypoint(self, waypoint, rotation, joints, thresholds, **kw):
            self.rotations.append(rotation)
            self.tip = np.asarray(waypoint, float)
            return joints, None

    r = Robot()
    move_keep_wrist(r, 0.20, vertical=True)
    move_keep_wrist(r, np.array([0.5119, -0.0685, 0.20]))  # sideways move
    assert len(r.rotations) >= 2 and np.allclose(r.tip, [0.5119, -0.0685, 0.20])
    for R in r.rotations:  # same yaw for every waypoint, including the sideways move
        assert np.arctan2(R[1, 0], R[0, 0]) == pytest.approx(yaw0, abs=1e-9)
        assert R[2, 2] == pytest.approx(1.0)  # probe_rotation convention


def test_seats_despite_under_read_force_and_xy_friction_offset():
    """2026-11-25 hardware: Fz read +1.2..2.4 N with ~4 N preload, tip held 3.8 mm off in xy."""
    hole = np.array([0.51, -0.07])
    est = hole + [0.003, 0.002]
    arm = _FakeArm(hole, np.r_[est, TOP + CFG["approach_height"]])
    offset = np.array([0.0038, 0.0])
    advance = arm.advance

    def advance_with_friction(dt):
        aim = arm.target.copy()
        arm.target[:2] = aim[:2] - offset  # where the friction-limited impedance actually settles
        advance(dt)
        arm.target = aim
    arm.advance = advance_with_friction
    ins = _inserter(arm)
    ins.force_fn = lambda: 0.4 * arm.force  # under-reading wrench estimate
    res = ins.run(est, TOP)
    assert res["status"] == "seated", res
    assert any(p["phase"] == "search" for p in res["phases"])


def test_without_stall_detection_the_under_read_touch_is_missed():
    hole = np.array([0.51, -0.07])
    est = hole + [0.006, 0.]
    arm = _FakeArm(hole, np.r_[est, TOP + CFG["approach_height"]], block_everywhere=True)
    from kilo.robot.frankapy_insertion import InsertionStop
    ins = _inserter(arm, dict(CFG, stall_lag=1.0, timeout=15.))  # stall detection disabled
    ins.force_fn = lambda: 0.3 * arm.force
    with pytest.raises(InsertionStop, match="phase approach"):
        ins.run(est, TOP)


def test_bottoming_out_in_the_hole_counts_as_seated():
    """2026-11-25: the peg stopped at 18.4 mm and pushed for 95 s waiting for 20 mm."""
    hole = np.array([0.51, -0.07])
    arm = _FakeArm(hole, np.r_[hole + [0.002, 0.], TOP + CFG["approach_height"]])
    advance = arm.advance

    def advance_with_floor(dt):
        advance(dt)
        arm.tip[2] = max(arm.tip[2], TOP - 0.0184)  # shoulder / fingers reach the plate
    arm.advance = advance_with_floor
    res = _inserter(arm).run(hole + [0.002, 0.], TOP)
    assert res["status"] == "seated" and res["seat_reason"] == "bottomed out"
    assert res["depth_m"] == pytest.approx(0.0184, abs=0.0005)


def test_search_skips_ahead_when_caught_on_an_edge():
    """2026-11-25: tip stuck 2.0-2.1 mm from its first waypoint for 117 s."""
    hole = np.array([0.51, -0.07])
    est = hole + [0.0027, -0.001]
    tip = np.r_[est, TOP]
    ref = RealSpiralReference(CFG, est, TOP, tip)
    for _ in range(CFG["touch_samples"]):
        ref.step(tip, 5.)
    assert ref.phase == "search"
    stuck_tip = tip + [0.003, 0., 0.]  # held 3 mm from every waypoint, never moving
    start_index = ref.index
    for _ in range(int(2.5 / ref.dt)):
        ref.step(stuck_tip, 2.)
    assert ref.skips >= 2 and ref.index > start_index


def test_insert_memory_prefers_where_the_peg_went_in(tmp_path):
    import json
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import insert_live
    run = tmp_path / "20261125-155355-1-insert"
    run.mkdir()
    (run / "result.json").write_text(json.dumps(dict(attempts=[
        dict(phases=[dict(phase="approach", tip=[.5069, -.0744, .07]), dict(phase="insert", tip=[.5088, -.0683, .06])])])))
    far = tmp_path / "20261125-100000-1-insert"
    far.mkdir()
    (far / "result.json").write_text(json.dumps(dict(attempts=[dict(phases=[dict(phase="insert", tip=[.40, .10, .06])])])))
    hits = insert_live.inserted_locations(tmp_path, np.array([.5119, -.0685]))
    assert len(hits) == 1 and np.allclose(hits[0][1], [.5088, -.0683])


def test_shallow_stall_in_insert_resumes_the_spiral():
    """Never park: a peg jammed a few mm in (not seated) goes back to searching, further along."""
    hole = np.array([0.51, -0.07])
    tip = np.r_[hole, TOP]
    ref = RealSpiralReference(CFG, hole, TOP, tip)
    for _ in range(CFG["touch_samples"]):
        ref.step(tip, 5.)
    jammed = np.r_[hole, TOP - 0.004]  # 4 mm in, then stuck
    phases = []
    for _ in range(int(4.0 / ref.dt)):
        ref.step(jammed, 2.)
        phases.append(ref.phase)
    assert "insert" in phases and ref.phase in ("search", "capture", "insert")
    assert ref.skips >= 1 and ref.index > 0 and not ref.done


def test_home_after_insertion_scales_duration_and_keeps_walls_on():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import insert_live

    class Arm:
        def get_joints(self):
            return np.array([-0.04, 0.17, -0.08, -2.36, 0.02, 2.53, -1.55])

        def goto_joints(self, joints, **kw):
            self.joints, self.kw = joints, kw
    arm = Arm()
    insert_live.home_arm(arm)
    assert np.allclose(arm.joints, [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    assert arm.kw["ignore_virtual_walls"] is False and arm.kw["ignore_errors"] is False
    assert arm.kw["duration"] >= 1.875 * 2.335 / 0.4 - 1e-6  # 2.335 rad wrist change -> ~11 s
