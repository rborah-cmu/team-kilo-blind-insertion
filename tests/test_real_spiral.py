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
        self.tip[:2] += min(1., dt / 0.05) * (self.target[:2] - self.tip[:2])
        self.tip[2] += min(1., dt / 0.05) * (self.target[2] - self.tip[2])
        over = np.linalg.norm(self.tip[:2] - self.hole) <= CAPTURE and not self.block
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
