import mujoco
import numpy as np
import pytest

from kilo.config import load_config
from kilo.sim.control import GRIPPER_ACTUATOR, HOME_QPOS, JointPD, solve_ik
from kilo.sim.geometry import base_wedges
from kilo.sim.scene import build_scene, site_pos

CFG = load_config()
SHAFT = CFG["peg"]["shaft_length"]


def test_wedges_leave_hole_at_least_nominal_size():
    r = CFG["hole"]["radius"]
    for w in base_wedges(CFG["box"]["size"], CFG["hole"]["cad_offset"], r, CFG["hole"]["chamfer"]):
        lateral = np.linalg.norm(w[:, :2] - np.array(CFG["hole"]["cad_offset"]), axis=1)
        assert lateral.min() >= r - 1e-9


@pytest.mark.parametrize("mode", ["hand", "free", "none"])
def test_scene_compiles(mode):
    m = build_scene(CFG, peg_mode=mode)
    assert m.opt.timestep == pytest.approx(0.001)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hole_top") >= 0


def test_aligned_free_peg_drops_fully_in():
    m = build_scene(CFG, peg_mode="free", peg_pos=(0, 0, 1))
    d = mujoco.MjData(m)
    d.qpos[:7] = HOME_QPOS
    mujoco.mj_forward(m, d)
    hole = site_pos(m, d, "hole_top")
    j = m.joint("peg_free").qposadr[0]
    d.qpos[j:j + 3] = hole + np.array([0, 0, SHAFT + 0.003])
    pd = JointPD(m)
    for _ in range(1500):
        d.ctrl[:7] = pd(d, HOME_QPOS)
        mujoco.mj_step(m, d)
    depth = hole[2] - site_pos(m, d, "peg_tip")[2]
    assert depth == pytest.approx(SHAFT, abs=3e-4)  # square top resting on the base


def _insert(lateral_offset):
    m = build_scene(CFG, peg_mode="hand")
    d = mujoco.MjData(m)
    d.qpos[:7] = HOME_QPOS
    mujoco.mj_forward(m, d)
    hole = site_pos(m, d, "hole_top")
    off = np.array([lateral_offset, 0, 0])
    q, err = solve_ik(m, "peg_tip", hole + off + [0, 0, 0.02], np.eye(3))
    assert err < 1e-4
    d.qpos[:7], d.qpos[7:9] = q, 0.0125
    d.ctrl[GRIPPER_ACTUATOR] = 0.0125
    mujoco.mj_forward(m, d)
    pd = JointPD(m)
    for z in np.arange(0.02, -0.0201, -0.0001):  # 10 cm/s down to 20 mm deep
        q, _ = solve_ik(m, "peg_tip", hole + off + [0, 0, z], np.eye(3), q0=q)
        for _ in range(10):
            d.ctrl[:7] = pd(d, q)
            mujoco.mj_step(m, d)
    return hole[2] - site_pos(m, d, "peg_tip")[2]


def test_robot_inserts_aligned_peg():
    assert _insert(0.0) > 0.018


def test_robot_jams_when_misaligned():
    # 2 mm off is beyond what the 1 mm chamfers absorb under a stiff joint PD:
    # this is the case Phase 2's learned corrections are for.
    assert _insert(0.002) < 0.005
