"""Side-by-side animation: the MuJoCo arm pressing the peg on the base (left) and
the hole-search particle filter choosing those presses (right).

The filter run is the same as scripts/animate_hole_search.py (placeholder press
model). The arm is simulated with real contact: for each press the filter
chooses, the peg is lowered onto the base and the actual tip depth is recorded
and shown. The base is built with its hole at that run's true position. The
final step inserts the peg at the filter's estimate.

Usage: python scripts/animate_search_with_arm.py [--seed 37] [--out docs/experiments/hole_search_arm]
       (writes <out>.mp4 and <out>.gif)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from animate_hole_search import draw, record_run  # noqa: E402
from kilo.config import load_config  # noqa: E402
from kilo.sim.control import GRIPPER_ACTUATOR, JointPD, solve_ik  # noqa: E402
from kilo.sim.scene import build_scene, site_pos  # noqa: E402

FPS = 25
BOX_XY = np.array([0.5, 0.0])
HOVER, PRESS_DEPTH = 0.015, 0.002  # m above / commanded below the top surface
MOVE_SPEED, PRESS_SPEED = 0.08, 0.01  # m/s


class ArmSim:
    def __init__(self, hole_offset, width=520, height=440):
        cfg = load_config()
        self.m = build_scene(cfg, peg_mode="hand", box_xy=tuple(BOX_XY), hole_offset=tuple(hole_offset))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.top_z = site_pos(self.m, self.d, "hole_top")[2]
        # Softer joint gains than the default so contact does not spike the force.
        self.pd = JointPD(self.m, kp=(300, 300, 300, 300, 120, 80, 30), kd=(35, 35, 35, 35, 12, 9, 3))
        self.q, err = solve_ik(self.m, "peg_tip", self.world(np.zeros(2), 0.03), np.eye(3))
        self.d.qpos[:7], self.d.qpos[7:9] = self.q, 0.0125
        self.d.ctrl[GRIPPER_ACTUATOR] = 0.0125
        mujoco.mj_forward(self.m, self.d)
        self.renderer = mujoco.Renderer(self.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [BOX_XY[0], BOX_XY[1], self.top_z + 0.04]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.55, 145, -28
        self.t = 0.0
        self.frames = []  # (image, step index, caption)

    def world(self, p_box, height):
        return np.array([BOX_XY[0] + p_box[0], BOX_XY[1] + p_box[1], self.top_z + height])

    def tip_height(self):
        return site_pos(self.m, self.d, "peg_tip")[2] - self.top_z

    def move(self, p_box, h0, h1, speed, step, caption, p_from=None):
        """Track a straight tip path at ``speed``, rendering at FPS."""
        start = self.world(p_box if p_from is None else p_from, h0)
        end = self.world(p_box, h1)
        dist = np.linalg.norm(end - start)
        n = max(1, int(dist / speed / 0.01))
        for k in range(1, n + 1):
            target = start + (end - start) * k / n
            self.q, _ = solve_ik(self.m, "peg_tip", target, np.eye(3), q0=self.q, iters=50)
            self.run(0.01, step, caption)

    def hold(self, seconds, step, caption):
        self.run(seconds, step, caption)

    def run(self, seconds, step, caption):
        for _ in range(int(round(seconds / self.m.opt.timestep))):
            self.d.ctrl[:7] = self.pd(self.d, self.q)
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
            if len(self.frames) < self.t * FPS:
                self.renderer.update_scene(self.d, self.cam)
                self.frames.append((self.renderer.render().copy(), step, caption(self)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "hole_search_arm"))
    args = ap.parse_args()
    radius, detect = 0.02, 0.005
    hole, steps = record_run(args.seed, radius, detect, 3000)

    sim = ArmSim(hole)
    results = [i for i, s in enumerate(steps) if s["kind"] == "result"]
    chooses = [i for i, s in enumerate(steps) if s["kind"] == "choose"]
    depth_log = []
    prev = np.zeros(2)
    sim.hold(1.0, 0, lambda s: "Peg in hand above the box centre")
    for n, (ci, ri) in enumerate(zip(chooses, results), start=1):
        p = steps[ri]["press"]
        sim.move(p, HOVER, HOVER, MOVE_SPEED, ci, lambda s, n=n: f"Moving to press {n}", p_from=prev)
        sim.move(p, HOVER, 0.002, MOVE_SPEED, ci, lambda s, n=n: f"Moving to press {n}")
        sim.move(p, 0.002, -PRESS_DEPTH, PRESS_SPEED, ri,
                 lambda s, n=n: f"Press {n}: tip {max(0.0, -s.tip_height()) * 1000:.1f} mm below surface")
        sim.hold(0.25, ri, lambda s, n=n: f"Press {n}: tip {max(0.0, -s.tip_height()) * 1000:.1f} mm below surface")
        depth = -sim.tip_height()
        depth_log.append((n, float(np.linalg.norm(p - hole)) * 1000, depth * 1000))
        sim.move(p, -PRESS_DEPTH if depth < 0.0005 else -depth, HOVER, MOVE_SPEED, ri,
                 lambda s, n=n, dd=depth: f"Press {n}: {'dropped in ' + format(dd * 1000, '.1f') + ' mm' if dd > 0.0005 else 'no drop'}")
        prev = p
    final = len(steps) - 1
    est = steps[final]["mean"]
    sim.move(est, HOVER, HOVER, MOVE_SPEED, final, lambda s: "Moving over the estimated hole", p_from=prev)
    sim.move(est, HOVER, -0.02, PRESS_SPEED, final, lambda s: f"Inserting at the estimate: tip {max(0.0, -s.tip_height()) * 1000:.1f} mm deep")
    err_mm = float(np.linalg.norm(est - hole)) * 1000

    def final_caption(s):
        d = max(0.0, -s.tip_height()) * 1000
        if d > 15:
            return f"Inserted {d:.1f} mm at the estimate"
        return f"Jammed {d:.1f} mm deep (estimate {err_mm:.1f} mm off)"
    sim.hold(2.0, final, final_caption)

    # Filter panels, one per step, rendered once.
    fig = plt.figure(figsize=(11, 5), dpi=90)
    ax = fig.add_axes([0.05, 0.11, 0.42, 0.8])
    ax2 = fig.add_axes([0.70, 0.18, 0.28, 0.66])
    fig.text(0.01, 0.01, "Placeholder press model (finds the hole within 5 mm). Arm: MuJoCo with real contact.",
             fontsize=7.5, color="#898781")
    panels = []
    for i in range(len(steps)):
        draw(fig, (ax, ax2), hole, steps, i, radius, detect)
        fig.canvas.draw()
        panels.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))

    h = sim.frames[0][0].shape[0]
    pw = int(panels[0].width * h / panels[0].height)
    panels = [im.resize((pw, h), Image.LANCZOS) for im in panels]
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    W = sim.frames[0][0].shape[1] + pw
    video = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, h))
    gif = []
    for k, (img, step, caption) in enumerate(sim.frames):
        frame = Image.new("RGB", (W, h), "white")
        arm = Image.fromarray(img)
        dr = ImageDraw.Draw(arm)
        dr.rectangle([0, 0, arm.width, 30], fill=(255, 255, 255))
        dr.text((10, 6), caption, fill=(11, 11, 11), font=font)
        frame.paste(arm, (0, 0))
        frame.paste(panels[step], (arm.width, 0))
        video.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        if k % 3 == 0:
            gif.append(frame.resize((int(W * 0.6), int(h * 0.6)), Image.LANCZOS).quantize(colors=128))
    video.release()
    gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=int(3000 / FPS), loop=0, optimize=True)
    print(f"saved {out.with_suffix('.mp4')} and .gif: {len(sim.frames)} frames, {sim.t:.1f} s")
    for n, dist, depth in depth_log:
        print(f"press {n:2d}: {dist:5.1f} mm from hole, sim tip depth {depth:5.2f} mm")


if __name__ == "__main__":
    main()
