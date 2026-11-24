import itertools

import mujoco
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from kilo.config import REPO_ROOT
from kilo.robot.planning import FrankaGeometry, PlanningFailure, box_separations
from kilo.sim.scene import build_scene


def geometry():
    cfg = yaml.safe_load((REPO_ROOT / "configs/real.yaml").read_text())
    return FrankaGeometry(REPO_ROOT / "configs/frankapy_geometry.json", cfg)


def independent_gap(A, ah, B, bh):
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    av = (signs * ah) @ A[:3, :3].T + A[:3, 3]
    bv = (signs * bh) @ B[:3, :3].T + B[:3, 3]
    axes = [*A[:3, :3].T, *B[:3, :3].T]
    axes += [np.cross(a, b) for a in A[:3, :3].T for b in B[:3, :3].T]
    gaps = []
    for axis in axes:
        if np.linalg.norm(axis) < 1e-7:
            continue
        axis = axis / np.linalg.norm(axis)
        pa, pb = av @ axis, bv @ axis
        gaps.append(max(pa.min() - pb.max(), pb.min() - pa.max()))
    return max(gaps)


def test_vector_sat_matches_independent_vertex_projection():
    rng = np.random.default_rng(16662)
    for _ in range(100):
        A, B = np.eye(4), np.eye(4)
        A[:3, :3], B[:3, :3] = Rotation.random(random_state=rng).as_matrix(), Rotation.random(random_state=rng).as_matrix()
        A[:3, 3], B[:3, 3] = rng.uniform(-.2, .2, 3), rng.uniform(-.2, .2, 3)
        ah, bh = rng.uniform(.01, .15, 3), rng.uniform(.01, .15, 3)
        actual = box_separations(A[None], ah[None], B[None], bh[None])[0, 0]
        assert actual == pytest.approx(independent_gap(A, ah, B, bh), abs=1e-12)


def test_planning_tool_frame_matches_independent_mujoco_chain():
    g = geometry()
    model = build_scene(peg_mode="hand", box_xy=(2., 2.))
    data = mujoco.MjData(model)
    qs = [[0, -.785, 0, -2.356, 0, 1.571, -.785],
          [.2, .5, -.4, -2.1, .3, 2.5, -1.3],
          [-.5, .2, .3, -2.5, -.2, 2.7, 2.]]
    for q in qs:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        p, R = g.tip_pose(q)
        assert np.linalg.norm(p - data.site("peg_tip").xpos) < 1e-6
        assert np.abs(R - data.body("hand").xmat.reshape(3, 3)).max() < 1e-6


def test_hardware_failed_posture_is_rejected_at_right_middle_shelf():
    g = geometry()
    q = [.02659789, .53234551, -.39060175, -2.07402474, .29952514, 2.50235303, -1.34332411]
    gap, detail = g.clearance(q, True)
    assert gap < 0
    assert detail["robot_box"] == 10 and detail["obstacle"] == "RShelfMiddle"


def test_half_turn_wrist_clears_failed_search_corner():
    g = geometry()
    seed = np.array([-.356, .535, 0., -2.015, 0., 2.55, 2.])
    unsafe = g.inverse([.565, -.21, .0322], 0, seed, elbow=0.)
    safe = g.inverse([.565, -.21, .0322], np.pi, seed, elbow=0.)
    assert g.clearance(unsafe) < 0
    assert g.clearance(safe) > .01
    assert np.linalg.norm(g.tip_pose(unsafe)[0] - g.tip_pose(safe)[0]) < 1e-6


def test_edge_check_rejects_blocked_midpoint_with_clear_endpoints():
    g = FrankaGeometry.__new__(FrankaGeometry)
    g.point_reach_bound = 2.
    def gap(q, detail=False):
        clearance = abs(q[0]) - .2
        return (clearance, {"obstacle": "test"}) if detail else clearance
    g.clearance = gap
    a, b = np.zeros(7), np.zeros(7)
    a[0], b[0] = -1., 1.
    with pytest.raises(PlanningFailure, match="joint edge blocked"):
        g.check_joint_edge(a, b)


def test_edge_check_reports_conservative_between_sample_clearance():
    g = FrankaGeometry.__new__(FrankaGeometry)
    g.point_reach_bound = 2.
    g.clearance = lambda q: .1
    a, b = np.zeros(7), np.zeros(7)
    b[0] = .01
    path, bound = g.check_joint_edge(a, b)
    assert len(path) == 2
    assert bound == pytest.approx(.09)
