"""Open the MuJoCo scene in the interactive viewer, or render it to a PNG.

Examples:
    python scripts/view_scene.py                      # peg in hand, above the hole
    python scripts/view_scene.py --mode free          # loose peg dropped into the hole
    python scripts/view_scene.py --box 0.45 0.05 0.3  # base at x, y (m) and yaw (rad)
    python scripts/view_scene.py --random             # base at a random pose in the search area
    python scripts/view_scene.py --render scene.png   # save an image instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kilo.config import load_config  # noqa: E402
from kilo.sim.control import GRIPPER_ACTUATOR, HOME_QPOS, JointPD, solve_ik  # noqa: E402
from kilo.sim.scene import build_scene, sample_box_pose, site_pos  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["hand", "free", "none"], default="hand")
    ap.add_argument("--box", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=(0.5, 0.0, 0.0))
    ap.add_argument("--random", action="store_true", help="random base pose inside the search rectangle (overrides --box)")
    ap.add_argument("--height", type=float, default=0.03, help="peg tip height above the hole (m)")
    ap.add_argument("--render", metavar="PNG", help="save an image instead of opening the viewer")
    args = ap.parse_args()

    cfg = load_config()
    box_xy, box_yaw = args.box[:2], args.box[2]
    if args.random:
        box_xy, box_yaw = sample_box_pose(cfg)
        print(f"base at x={box_xy[0]:.3f} m, y={box_xy[1]:.3f} m, yaw={np.degrees(box_yaw):.1f} deg")
    model = build_scene(cfg, peg_mode=args.mode, box_xy=box_xy, box_yaw=box_yaw, peg_pos=(0, 0, 1))
    data = mujoco.MjData(model)
    data.qpos[:7] = HOME_QPOS
    mujoco.mj_forward(model, data)
    hole = site_pos(model, data, "hole_top")

    q_hold = HOME_QPOS
    if args.mode == "hand":
        q_hold, err = solve_ik(model, "peg_tip", hole + [0, 0, args.height], np.eye(3))
        print(f"IK: peg tip {args.height * 1000:.0f} mm above the hole, error {err * 1000:.3f} mm")
        data.qpos[:7], data.qpos[7:9] = q_hold, 0.0125
        data.ctrl[GRIPPER_ACTUATOR] = 0.0125
    elif args.mode == "free":
        j = model.joint("peg_free").qposadr[0]
        data.qpos[j:j + 3] = hole + [0, 0, cfg["peg"]["shaft_length"] + args.height]
    mujoco.mj_forward(model, data)
    pd = JointPD(model)

    if args.render:
        from PIL import Image

        renderer = mujoco.Renderer(model, 600, 900)
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = hole + [0, 0, 0.02], 0.35, 120, -20
        renderer.update_scene(data, cam)
        Image.fromarray(renderer.render()).save(args.render)
        print(f"saved {args.render}")
        return

    from mujoco import viewer

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            data.ctrl[:7] = pd(data, q_hold)  # hold the arm still (gravity-compensated PD)
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
