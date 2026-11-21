"""Find the plate by touch. Run with --headless for faster-than-real-time simulation."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilo.config import load_config
from kilo.localization.common import SEARCH_MODE_OFFSET
from kilo.localization.probenav import probemove, travelmove
from kilo.localization.search import generate_snake_waypoints
from kilo.sim.control import GRIPPER_ACTUATOR, solve_ik
from kilo.sim.scene import TABLE_TOP_Z, build_scene


def run(model, data, viewer, waypoints):
    for point in waypoints[1:]:
        hit = probemove(point, model, data, viewer)
        if hit is not None:
            break
    else:
        raise RuntimeError("Sweep ended without touching the plate")

    def touch_from(origin, outward):
        _, target = travelmove(np.array([origin, outward]), model, data, viewer)
        print("Crossing from", origin, "toward", target, flush=True)
        contact = probemove(target, model, data, viewer)
        if contact is None:
            raise RuntimeError("Crossing ended without touching the plate")
        print("Contact:", contact, flush=True)
        return contact

    first, force = hit
    print("First contact:", hit, flush=True)
    normal = force / np.linalg.norm(force)
    opposite, opposite_force = touch_from(first, -normal)
    midpoint = (first + opposite) / 2
    tangent = np.array([-normal[1], normal[0]])
    left, left_force = touch_from(midpoint, tangent)
    right, right_force = touch_from(midpoint, -tangent)

    # The tip centre sits one shaft radius outside the contacted face or corner.
    cfg = load_config()
    half_sizes = np.array(cfg["box"]["size"][:2]) / 2
    radius = cfg["peg"]["shaft_radius"]
    samples = np.array([first, opposite, left, right])
    forces = np.array([force, opposite_force, left_force, right_force])
    normals = forces / np.linalg.norm(forces, axis=1, keepdims=True)

    def residual(pose):
        center, angle = pose[:2], pose[2]
        c, s = np.cos(angle), np.sin(angle)
        axes = np.array([[c, -s], [s, c]])
        local = (samples - center) @ axes
        closest = np.clip(local, -half_sizes, half_sizes)
        offsets = local - closest
        distances = np.linalg.norm(offsets, axis=1)
        predicted = offsets @ axes.T / np.maximum(distances[:, None], 1e-9)
        # Forces can include friction, so use them primarily to resolve yaw.
        return np.r_[distances - radius, (.004 * (predicted - normals)).ravel()]

    fits = [least_squares(residual, np.r_[samples.mean(axis=0), yaw])
            for yaw in np.linspace(0, np.pi, 12, endpoint=False)]
    pose = min(fits, key=lambda fit: fit.cost).x
    return pose[:2], np.degrees(pose[2]) % 180


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
    points = generate_snake_waypoints(.06)
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
