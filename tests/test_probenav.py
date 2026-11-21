"""Exercise probe motion against the actual torque-controlled MuJoCo scene."""

import mujoco
import numpy as np
import pytest

from kilo.localization import probenav
from kilo.localization.common import FORCE_THRESHOLD, POS_EPS
from kilo.sim.control import GRIPPER_ACTUATOR, solve_ik
from kilo.sim.scene import TABLE_TOP_Z, build_scene, site_pos


def scene_at(xy):
    model = build_scene()
    data = mujoco.MjData(model)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    q, error = solve_ik(model, "peg_tip", np.r_[xy, TABLE_TOP_Z + .03], rotation)
    assert error < 1e-4
    data.qpos[:7], data.qpos[7:9] = q, .0125
    data.ctrl[GRIPPER_ACTUATOR] = .0125
    mujoco.mj_forward(model, data)
    return model, data


def test_probe_reaches_clear_waypoint():
    model, data = scene_at([.35, -.15])
    target = np.array([.39, -.15])
    assert probenav.probemove(target, model, data) is None
    position = site_pos(model, data, "peg_tip")
    assert np.linalg.norm(position - np.r_[target, TABLE_TOP_Z + .03]) < POS_EPS
    rotation = data.site_xmat[model.site("peg_tip").id].reshape(3, 3)
    np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=.03)
    assert data.ctrl[GRIPPER_ACTUATOR] == .0125


def test_probe_stops_at_plate_with_force_away_from_surface():
    model, data = scene_at([.35, -.04])
    position, force = probenav.probemove(np.array([.65, -.04]), model, data)
    assert np.linalg.norm(force) > FORCE_THRESHOLD
    assert force[0] < 0  # plate pushes the probe back toward -x
    assert .41 < position[0] < .45
    np.testing.assert_allclose(position, site_pos(model, data, "peg_tip")[:2])


def test_travel_lifts_crosses_then_lowers(monkeypatch):
    model, data = scene_at([.60, .15])
    legs = []
    move = probenav._move

    def record(target, *args):
        result = move(target, *args)
        legs.append((target.copy(), site_pos(model, data, "peg_tip")))
        return result

    monkeypatch.setattr(probenav, "_move", record)
    start, end = probenav.travelmove(np.array([[.60, .15], [1., 0.]]), model, data)
    np.testing.assert_allclose(start, [.70, .15])
    np.testing.assert_allclose(end, [.20, .15])
    np.testing.assert_allclose([t[2] for t, _ in legs], TABLE_TOP_Z + np.array([.10, .10, .03]))
    for target, reached in legs:
        assert np.linalg.norm(target - reached) < POS_EPS
    np.testing.assert_allclose(legs[0][0][:2], [.60, .15], atol=1e-4)


def test_unreachable_waypoint_has_simulation_time_limit():
    model, data = scene_at([.35, -.15])
    with pytest.raises(TimeoutError):
        probenav.probemove(np.array([.70, -.15]), model, data, max_time=.01)
