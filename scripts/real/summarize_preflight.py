"""Create inspection plots and independent-verifier inputs from an offline plan."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from scipy.spatial import ConvexHull
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kilo.robot.planning import FrankaGeometry


def vertices(pose, half):
    import itertools
    return np.array(list(itertools.product([-1, 1], repeat=3))) * half @ pose[:3, :3].T + pose[:3, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--plot-only", action="store_true", help="do not rewrite verifier inputs or summary")
    args = ap.parse_args()
    folder = Path(args.directory)
    report = json.loads((folder / "report.json").read_text())
    if len(report["cases"]) != 10:
        raise RuntimeError("wait for all ten requested scenarios to finish")
    cfg = yaml.safe_load((ROOT / "configs/real.yaml").read_text())
    g = FrankaGeometry(ROOT / "configs/frankapy_geometry.json", cfg)
    all_q, counts = [], {}
    for case in report["cases"]:
        data = np.load(folder / f"{case['label']}-joints.npz")
        points = np.vstack([data[key] for key in data.files])
        counts[case["label"]] = len(points)
        all_q.append(points)
    all_q = np.vstack(all_q)
    _, idx = np.unique(np.round(all_q, 9), axis=0, return_index=True)
    unique = all_q[np.sort(idx)]
    tips = np.array([g.tip_pose(q)[0] for q in unique])
    if not args.plot_only:
        np.savez_compressed(folder / "verification_states.npz", q=unique, obstacles=g.obstacles,
                            obstacle_names=np.asarray(g.obstacle_names), tool_mount_correction=g.mount_correction,
                            probe_length=g.probe_length, expected_tip=tips)
    summary = dict(
        cases=len(report["cases"]), all_listed_cases_passed=report["all_listed_cases_passed"],
        all_box_cases_verified=report["all_box_cases_verified"],
        joint_states_including_repeated_paths=len(all_q), unique_states=len(unique),
        deduplication_rounding_rad=1e-9,
        minimum_joint_limit_margin_rad=float(min((unique - g.lo).min(), (g.hi - unique).min())),
        minimum_continuous_clearance_bound_m=min(c["minimum_certified_clearance_m"] for c in report["cases"]),
        self_collision_samples=sum(c["self_collision_check"]["samples"] for c in report["cases"]),
        cases_samples=counts, execution_ready=False,
        requirement=report["execution_requirement"],
    )
    if not args.plot_only:
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    ax = axes[0]
    ax.add_patch(Rectangle((.21, .30), .49, .14, color="slategray", alpha=.3, label="Shelves"))
    ax.add_patch(Rectangle((.21, -.46), .49, .14, color="slategray", alpha=.3))
    corners = np.array(cfg["workspace"]["search_corners"])
    ax.add_patch(Rectangle(corners.min(0), *np.ptp(corners, axis=0), fill=False, color="steelblue", ls="--"))
    case = report["cases"][0]
    x, y, yaw = case["hypothetical_box"]
    half = np.array([.1267, .169]) / 2
    ax.add_patch(Rectangle((x-half[0], y-half[1]), *2*half, color="tan", alpha=.8, label="Hypothetical block"))
    data = np.load(folder / "nominal-joints.npz")
    nominal_q = []
    seen = set()
    for motion in case["motions"]:
        path = data[motion["path"]]
        p = np.array([g.tip_pose(q)[0] for q in path])
        full_ray = motion["kind"] == "full_guarded_ray"
        label = "Full checked probe ray" if full_ray else "Nominal contact-stop path"
        ax.plot(p[:, 0], p[:, 1], color="darkorange" if full_ray else "seagreen",
                lw=1 if full_ray else 1.5, ls="--" if full_ray else "-", alpha=.7,
                label=label if label not in seen else None)
        seen.add(label)
        nominal_q.extend(path)
    ax.set(xlim=(.18, .76), ylim=(-.47, .45), xlabel="Robot x (m)", ylabel="Robot y (m)",
           title="Nominal peg-tip XY projection")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    seed = [0, .535, 0, -2.015, 0, 2.55, 2.0]
    for yaw, color, label in [(0., "firebrick", "Old wrist orientation"), (np.pi, "seagreen", "Planned half-turn")]:
        q = g.inverse([.565, -.21, .0322], yaw, seed, elbow=0.)
        pose = g.fk(q)[g.box_links[10]] @ g.box_offsets[10]
        p = vertices(pose, g.half_sizes[10])[:, 1:3]
        hull = ConvexHull(p)
        ax.add_patch(Polygon(p[hull.vertices], facecolor=color, edgecolor=color, alpha=.25, label=label))
    ax.add_patch(Rectangle((-.47, .325), .16, .025, facecolor="gray", alpha=.25, label="10 mm shelf margin"))
    ax.add_patch(Rectangle((-.46, .335), .14, .005, facecolor="slategray", label="Right middle shelf"))
    ax.set(xlim=(-.37, -.10), ylim=(.16, .43), xlabel="Robot y (m)", ylabel="Robot z (m)",
           title="Wrist housing at the same peg-tip pose")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    gaps = np.array([g.clearance(q) for q in nominal_q]) * 1000
    ax.plot(gaps, color="seagreen", lw=.8)
    ax.axhline(2., color="firebrick", ls="--", label="Additional certified minimum")
    ax.set(xlabel="Checked nominal joint-state index", ylabel="Model clearance (mm)",
           title="Shelves already inflated by 10 mm", ylim=(0, max(gaps) * 1.1))
    ax.legend(fontsize=8)
    fig.suptitle("Offline preflight — geometry only; no robot motion", fontsize=14)
    fig.savefig(folder / "overview.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
