"""Find the plate by touch. Run with --headless for faster-than-real-time simulation."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilo.localization.common import SEARCH_MODE_OFFSET, SNAKE_SPACING
from kilo.localization.plate import localize_plate
from kilo.localization.probenav import probemove, travelmove
from kilo.localization.search import generate_snake_waypoints
from kilo.sim.control import GRIPPER_ACTUATOR, solve_ik
from kilo.sim.scene import TABLE_TOP_Z, build_scene


def run(model, data, viewer, waypoints):
    return localize_plate(waypoints,
                          lambda point: probemove(point, model, data, viewer),
                          lambda ray: travelmove(ray, model, data, viewer))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--box", nargs=3, type=float, default=(.50, 0., 0.),
                        metavar=("X", "Y", "YAW_RAD"))
    parser.add_argument("--seed", type=int, help="random plate pose from this seed (overrides --box)")
    args = parser.parse_args()
    if args.seed is not None:
        model = build_scene(seed=args.seed)
    else:
        model = build_scene(box_xy=args.box[:2], box_yaw=args.box[2])
    data = mujoco.MjData(model)
    points = generate_snake_waypoints(SNAKE_SPACING)
    q, error = solve_ik(model, "peg_tip", np.r_[points[0], TABLE_TOP_Z + SEARCH_MODE_OFFSET],
                        np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    if error > .005:
        raise RuntimeError("Initial search corner is outside IK tolerance")
    data.qpos[:7], data.qpos[7:9] = q, .0125
    data.ctrl[GRIPPER_ACTUATOR] = .0125
    mujoco.mj_forward(model, data)
    from mujoco import viewer
    with (nullcontext() if args.headless else viewer.launch_passive(model, data)) as window:
        center, angle = run(model, data, window, points)
        x, y = data.body("base").xpos[:2]
        print(f"Actual plate center ({x:.4f}, {y:.4f}) m")
        print(f"Estimated plate center ({center[0]:.4f}, {center[1]:.4f}) m, yaw {angle:.1f} deg (mod 180)")


if __name__ == "__main__":
    main()
