from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from kilo.config import REPO_ROOT, load_config
from kilo.localization.touch_localizer import TouchLocalizer
from kilo.robot.interface import RobotState, SafetyStop
from kilo.robot.workspace import HandWorkspace, load_shelf_boxes


def config():
    cfg = yaml.safe_load((REPO_ROOT / "configs/real.yaml").read_text())
    # Fixed wall-only regression fixture; live lab bounds change between sessions.
    cfg["workspace"].pop("shelf_inner_y_limits", None)
    cfg["workspace"]["search_corners"] = [[.32, -.2], [.62, -.2], [.62, .2], [.32, .2]]
    return cfg


def test_wall_guard_depends_on_finger_orientation():
    guard = HandWorkspace(config()["workspace"])
    with pytest.raises(SafetyStop):
        guard.check([0.32, -0.2], [0.62, -0.2], 0, 0)
    guard.check([0.32, -0.2], [0.62, -0.2], np.pi / 2, np.pi / 2)


def test_rotation_checks_swept_corners_not_just_endpoints():
    guard = HandWorkspace(config()["workspace"])
    xy = [0.56, 0]
    guard.check(xy, xy, 0, 0)
    guard.check(xy, xy, np.pi / 2, np.pi / 2)
    with pytest.raises(SafetyStop):
        guard.check(xy, xy, 0, np.pi / 2)


def test_rotation_wraps_across_pi_by_shortest_path():
    guard = HandWorkspace(config()["workspace"])
    guard.check([0.55, 0], [0.55, 0], np.radians(179), np.radians(-179))


@pytest.mark.parametrize("axis,point", [(0, [-0.32, 0]), (1, [0.4, 0.4]), (1, [0.4, -0.4])])
def test_guard_checks_other_wall_faces(axis, point):
    with pytest.raises(SafetyStop):
        HandWorkspace(config()["workspace"]).check(point, point, 0, 0)


def localizer(tip, yaw):
    st = RobotState(0, np.asarray(tip, float), yaw, np.zeros(3), np.zeros(3))
    robot = SimpleNamespace(state=lambda: st)
    return TouchLocalizer(robot, config(), load_config(), log=lambda *args: None)


def test_localizer_uses_perpendicular_yaw_at_far_wall():
    loc = localizer([0.4, -0.2, 0.11], 0)
    yaw = loc._yaw_for([1, 0], at_xy=[0.62, -0.2])
    assert abs(np.cos(yaw)) < 1e-8
    # Verify actual travel rotates at the safe starting point, before translating.
    moves = []
    def move(tip, angle, speed):
        moves.append((np.array(tip), angle))
        st = loc.robot.state()
        st.tip, st.yaw = np.array(tip), angle
    loc.robot.move = move
    loc.robot.probe = lambda target, *args: SimpleNamespace(contact=False, stalled=False, tip=target, samples=[])
    assert loc.travel([0.62, -0.2])
    assert np.allclose(moves[0][0][:2], [0.4, -0.2])
    assert abs(np.cos(moves[0][1])) < 1e-8
    assert np.allclose(moves[-1][0][:2], [0.62, -0.2])


def test_localizer_refuses_unsafe_current_pose_before_any_motion():
    loc = localizer([0.587, -0.2, 0.106], 0)
    loc.robot.move = lambda *args: pytest.fail("must refuse before moving")
    with pytest.raises(SafetyStop):
        loc.travel([0.4, 0])


def test_fixed_wrist_mode_cannot_bypass_wall_guard():
    loc = localizer([0.4, 0, 0.11], 0)
    loc.align_wrist = False
    with pytest.raises(SafetyStop):
        loc._yaw_for([1, 0], at_xy=[0.62, 0])


def test_invalid_wall_geometry_is_rejected():
    ws = config()["workspace"]
    ws["hand_half_extents"] = [-0.1, 0.1]
    with pytest.raises(ValueError):
        HandWorkspace(ws)


def test_first_sweep_travel_chooses_yaw_for_upcoming_probe():
    loc = localizer([0.4, -0.075, 0.11], 0)
    class Planned(Exception):
        pass
    def travel(xy, yaw=None):
        assert np.allclose(xy, [0.62, -0.2])
        assert abs(np.cos(yaw)) < 1e-8
        raise Planned
    loc.travel = travel
    with pytest.raises(Planned):
        loc.run()


def test_oblique_crossing_from_hardware_uses_wall_safe_cardinal_wrist():
    loc = localizer([0.4677, -0.0048, 0.0617], np.radians(-67.8))
    start, end = np.array([0.62, -0.058]), np.array([0.32, -0.1815])
    yaw = loc._yaw_for(end - start, at_xy=end, start_xy=start)
    assert abs(np.cos(yaw)) < 1e-8
    loc.hand_workspace.check(start, end, yaw, yaw)


def test_descent_cannot_silently_continue_23mm_above_search_height():
    loc = localizer([0.38, 0.2, 0.11], 0)
    loc.robot.move = lambda *args: None
    loc.robot.probe = lambda *args: SimpleNamespace(contact=False, stalled=True,
                                                    tip=np.array([0.38, 0.2, .0552]), samples=[])
    with pytest.raises(SafetyStop, match="descent stalled"):
        loc.travel([0.38, 0.2], yaw=0)


def test_side_probe_refuses_bad_height_before_moving():
    loc = localizer([0.38, 0.2, .0552], 0)
    loc.robot.move = lambda *args: pytest.fail("invalid height must stop before moving")
    with pytest.raises(SafetyStop, match="outside valid probing height"):
        loc.probe_xy([.5, .2], "invalid")


def test_shelf_geometry_rejects_old_descent_with_y_aligned_hand():
    cfg = yaml.safe_load((REPO_ROOT / "configs/real.yaml").read_text())
    guard = HandWorkspace(cfg["workspace"])
    with pytest.raises(SafetyStop, match="shelf boundary"):
        guard.check([.378, .2], [.378, .2], np.radians(-67.8), np.radians(-67.8))
    # Across-finger clearance is narrower and can fit at that location.
    guard.check([.378, .15], [.378, .15], 0, 0)


def test_revised_search_corners_clear_shelves_with_x_aligned_hand():
    cfg = yaml.safe_load((REPO_ROOT / "configs/real.yaml").read_text())
    guard = HandWorkspace(cfg["workspace"])
    corners = cfg["workspace"]["search_corners"]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        guard.check(start, end, 0, 0)


def test_shelf_boxes_use_full_dimensions_and_explicit_margin():
    cfg = yaml.safe_load((REPO_ROOT / "configs/real.yaml").read_text())
    boxes = load_shelf_boxes(cfg["workspace"])
    assert len(boxes) == 30
    assert all(isinstance(box, np.ndarray) for box in boxes)
    board = next(b for b in boxes if np.allclose(b[:3], [.455, .37, .1375]))
    assert np.allclose(board[6:], [.51, .16, .025])


def test_verification_skips_unplannable_faces_without_moving():
    loc = localizer([.45, 0, .0322], 0)
    def no_path(*args, **kwargs):
        raise SafetyStop("no safe wrist")
    loc._yaw_for = no_path
    loc.travel = lambda *args, **kwargs: pytest.fail("unplannable verification must not move")
    fit = SimpleNamespace(center=np.array([.477, -.106]), yaw=0.)
    report = loc.verify(fit, 0)
    assert report["skipped"] == 4
    assert report["new_contacts"] == 0
