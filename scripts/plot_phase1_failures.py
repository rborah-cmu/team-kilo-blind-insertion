"""Replay Phase 1 localization for chosen seeds and plot what happened.

For each seed: the true box, the estimated box, the peg-tip path (search height
vs. travel height), each contact with its measured force direction, and which
true-box face each contact actually touched.

Usage: .venv/Scripts/python.exe scripts/plot_phase1_failures.py [--seeds 8 12 22 26 41 46 49 0]
Writes docs/experiments/phase1_failures.png and phase1_failures.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def replay(seed: int) -> dict:
    import mujoco

    import localize
    from kilo.config import load_config
    from kilo.localization.common import SEARCH_MODE_OFFSET
    from kilo.localization.search import generate_snake_waypoints
    from kilo.sim.control import GRIPPER_ACTUATOR, solve_ik
    from kilo.sim.scene import TABLE_TOP_Z, build_scene, sample_box_pose

    cfg = load_config()
    true_xy, true_yaw = sample_box_pose(cfg, seed=seed)
    model = build_scene(cfg, seed=seed)
    data = mujoco.MjData(model)
    tip_id = model.site("peg_tip").id
    path = []
    original_step = mujoco.mj_step

    def recording_step(m, d, *args):
        original_step(m, d, *args)
        if d.time * 1000 % 20 < 1:  # every 20 ms of simulated time
            path.append(d.site_xpos[tip_id].copy())

    mujoco.mj_step = recording_step
    points = generate_snake_waypoints(.06)
    q, _ = solve_ik(model, "peg_tip", np.r_[points[0], TABLE_TOP_Z + SEARCH_MODE_OFFSET],
                    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    data.qpos[:7], data.qpos[7:9] = q, .0125
    data.ctrl[GRIPPER_ACTUATOR] = .0125
    mujoco.mj_forward(model, data)
    log = io.StringIO()
    est, err = None, None
    try:
        with contextlib.redirect_stdout(log):
            center, yaw_deg = localize.run(model, data, None, points)
        est = {"xy": list(map(float, center)), "yaw_deg": float(yaw_deg)}
    except Exception as exc:
        err = repr(exc)
    finally:
        mujoco.mj_step = original_step

    contacts = []
    for line in log.getvalue().splitlines():
        if "ontact:" in line:
            v = list(map(float, re.findall(NUM, line)))
            contacts.append({"xy": v[0:2], "force": v[2:4], "first": line.startswith("First")})

    # Which face of the TRUE box did each contact touch? (tip centre is one shaft radius outside)
    half = np.array(cfg["box"]["size"][:2]) / 2
    r = cfg["peg"]["shaft_radius"]
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    R = np.array([[c, -s], [s, c]])
    for ct in contacts:
        local = (np.array(ct["xy"]) - true_xy) @ R
        over = np.abs(local) - half
        if over.max() > r + 0.004:
            ct["face"] = "nothing (>4 mm off any face)"
        elif (over > -0.003).all():
            ct["face"] = "corner"
        else:
            ax = int(np.argmax(over))
            ct["face"] = ("+x" if local[0] > 0 else "-x") if ax == 0 else ("+y" if local[1] > 0 else "-y")
        ct["gap_mm"] = float((np.linalg.norm(np.maximum(over, 0)) - r) * 1000)
    hole = data.site("hole_top").xpos[:2].copy()
    return {"seed": seed, "true": {"xy": list(map(float, true_xy)), "yaw_deg": float(np.degrees(true_yaw) % 180)},
            "hole_xy": list(map(float, hole)), "estimate": est, "error": err, "contacts": contacts,
            "sim_seconds": float(data.time), "path": np.asarray(path)[:, :3].round(4).tolist()}


def box_corners(xy, yaw_deg, half):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]) * half
    return pts @ R.T + np.asarray(xy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 22, 26, 41, 49, 12, 46, 0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "phase1_failures"))
    args = ap.parse_args()
    with Pool(min(8, len(args.seeds))) as pool:
        runs = pool.map(replay, args.seeds)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kilo.config import load_config
    cfg = load_config()
    half = np.array(cfg["box"]["size"][:2]) / 2
    corners = np.array(cfg["snake_search"]["corners"])
    INK, MUTED, GRID, BLUE, ORANGE, RED = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#eb6834", "#e34948"
    n = len(runs)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 4.4 * ((n + cols - 1) // cols)), dpi=110)
    for ax, run in zip(axes.ravel(), runs):
        rect = np.vstack([corners, corners[:1]])
        ax.plot(rect[:, 0] * 1000, rect[:, 1] * 1000, color=GRID, lw=1)
        tb = box_corners(run["true"]["xy"], run["true"]["yaw_deg"], half) * 1000
        ax.fill(tb[:, 0], tb[:, 1], color=BLUE, alpha=0.12, lw=0)
        ax.plot(tb[:, 0], tb[:, 1], color=BLUE, lw=1.5)
        p = np.array(run["path"])
        if len(p):
            low = p[:, 2] < np.median(p[:, 2]) + 0.01
            ax.plot(p[:, 0] * 1000, p[:, 1] * 1000, color=MUTED, lw=0.6, alpha=0.6)
            ax.scatter(p[low, 0] * 1000, p[low, 1] * 1000, s=0.6, color=INK, alpha=0.35, lw=0)
        ax.plot(*(np.array(run["hole_xy"]) * 1000), "*", color=INK, ms=9)
        if run["estimate"]:
            eb = box_corners(run["estimate"]["xy"], run["estimate"]["yaw_deg"], half) * 1000
            ax.plot(eb[:, 0], eb[:, 1], color=ORANGE, lw=1.8, ls="--")
            ax.plot(*(np.array(run["estimate"]["xy"]) * 1000), "x", color=ORANGE, ms=8, mew=2)
        for k, ct in enumerate(run["contacts"]):
            xy, f = np.array(ct["xy"]) * 1000, np.array(ct["force"])
            u = f / (np.linalg.norm(f) + 1e-9) * 22
            ax.annotate("", xy=xy + u, xytext=xy, arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
            ax.plot(*xy, "o", color=RED, ms=6, mec="white", mew=1)
            ax.text(xy[0] + 4, xy[1] + 4, f"{k + 1}:{ct['face']}", fontsize=7, color=RED)
        if run["estimate"]:
            err = np.linalg.norm(np.array(run["estimate"]["xy"]) - run["true"]["xy"]) * 1000
            dy = (run["estimate"]["yaw_deg"] - run["true"]["yaw_deg"] + 90) % 180 - 90
            title = f"seed {run['seed']}: centre off {err:.1f} mm, yaw off {abs(dy):.0f}°"
            color = INK if err < 5 else RED
        else:
            title, color = f"seed {run['seed']}: timed out ({len(run['contacts'])} contacts)", RED
        ax.set_title(title, fontsize=9.5, color=color, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(180, 720)
        ax.set_ylim(-220, 220)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, lw=1.5, label="true box"),
               plt.Line2D([], [], color=ORANGE, lw=1.8, ls="--", label="estimated box"),
               plt.Line2D([], [], marker="o", ls="", color=RED, label="contact (number: face actually hit)"),
               plt.Line2D([], [], color=RED, marker=r"$\rightarrow$", ls="", label="measured force direction"),
               plt.Line2D([], [], color=MUTED, lw=0.8, label="peg-tip path (dark = search height)"),
               plt.Line2D([], [], marker="*", ls="", color=INK, label="true hole")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Phase 1 box localization: failed seeds (and seed 0, a success, for reference). Axes in mm, robot base frame.",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = Path(args.out)
    fig.savefig(out.with_suffix(".png"))
    slim = [{k: v for k, v in r.items() if k != "path"} for r in runs]
    out.with_suffix(".json").write_text(json.dumps(slim, indent=1))
    for r in slim:
        print(f"seed {r['seed']}: true yaw {r['true']['yaw_deg']:.1f}, est {r['estimate']}, err {r['error']}")
        for k, ct in enumerate(r["contacts"]):
            f = np.array(ct["force"])
            print(f"   contact {k + 1}: face {ct['face']:<30} gap {ct['gap_mm']:6.1f} mm  |F| {np.linalg.norm(f):5.1f} N")


if __name__ == "__main__":
    main()
