import numpy as np
import pytest
import yaml

from kilo.config import REPO_ROOT, load_config
from kilo.localization.fit import box_faces, fit_box, point_gaps
from kilo.robot import ProbeSettings, make_backend, tool_rotation, yaw_of_tool

SCENE = load_config()
HALF = np.array(SCENE["box"]["size"][:2]) / 2
RADIUS = SCENE["peg"]["shaft_radius"]


def real_cfg():
    return yaml.safe_load(open(REPO_ROOT / "configs" / "real.yaml"))


def contacts_for(center, yaw, faces):
    """Tip centres touching the given faces' centres, with outward force directions."""
    pts, forces = [], []
    for i in faces:
        n, c = box_faces(center, yaw, HALF)[i]
        pts.append(c + n * RADIUS)
        forces.append(n)
    return np.array(pts), np.array(forces)


def test_tool_rotation_points_down_with_finger_axis_along_yaw():
    for yaw in np.linspace(-3, 3, 7):
        R = tool_rotation(yaw)
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[:, 2], [0, 0, -1])
        assert np.isclose(np.angle(np.exp(1j * (yaw_of_tool(R) - yaw))), 0)


def test_fit_recovers_pose_from_all_four_faces():
    center, yaw = np.array([0.5, -0.05]), np.radians(30)
    pts, forces = contacts_for(center, yaw, range(4))
    fit = fit_box(pts, forces, HALF, RADIUS)
    assert np.linalg.norm(fit.center - center) < 1e-4
    assert abs(((np.degrees(fit.yaw - yaw) + 90) % 180) - 90) < 0.1
    assert point_gaps(fit.center, fit.yaw, pts, HALF, RADIUS).max() < 1e-5


def test_sim_probe_detects_box_face():
    cfg = real_cfg()
    # Hold the simulator's initial posture fixed as live search bounds are tuned.
    cfg["workspace"]["search_corners"] = [[.32, -.2], [.62, -.2], [.62, .2], [.32, .2]]
    robot = make_backend("sim", cfg, box=(0.5, 0.0, 0.0), force_noise=0.3, force_bias=2.0)
    start = np.array([0.40, 0.0, robot.table_z() + 0.016])
    robot.move(np.r_[start[:2], 0.10], robot.state().yaw, 0.05)
    robot.move(start, robot.state().yaw, 0.05)
    s = ProbeSettings(stiffness=robot.stiffness, speed=0.01)
    res = robot.probe(start + [0.10, 0, 0], robot.state().yaw, s)
    face_x = 0.5 - HALF[0] - RADIUS  # tip centre when the shaft touches the -x face
    assert res.contact and not res.stalled
    assert abs(res.tip[0] - face_x) < 0.002
    assert res.force[0] < 0  # the face pushes the peg back
    assert res.peak_force < 2 * s.force_threshold + 1


def test_localizer_refuses_unmeasured_table():
    from kilo.localization.touch_localizer import TouchLocalizer

    cfg = real_cfg()
    cfg["workspace"]["table_z"] = None
    with pytest.raises(ValueError, match="table_z"):
        TouchLocalizer(object(), cfg, SCENE)
