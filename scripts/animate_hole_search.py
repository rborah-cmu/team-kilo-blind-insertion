"""Animate one run of the hole-search particle filter (kilo.localization.HoleSearchFilter).

Left: top view of the search disc around the estimated box centre, the
particles (hole hypotheses), each press and its outcome, and the estimate with
its 2-sigma ellipse. Right: estimate error and spread after each step.

Uses the placeholder press model in hole_filter.py (a press "drops in" within
detect_radius of the hole centre), so it shows how the filter behaves, not the
final hole-search strategy.

Usage: python scripts/animate_hole_search.py [--seed 37] [--out docs/experiments/hole_search.gif]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kilo.localization import HoleSearchFilter  # noqa: E402

BLUE, ORANGE, GRAY, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#898781", "#0ca30c", "#0b0b0b", "#898781", "#e1e0d9"
MM = 1000.0


def record_run(seed: int, radius: float, detect: float, n_particles: int):
    """Replay the same random run as scripts/run_experiments.py and record every step."""
    rng = np.random.default_rng(seed)
    r, a = radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    hole = np.array([r * np.cos(a), r * np.sin(a)])
    f = HoleSearchFilter([0.0, 0.0], radius, detect, num_particles=n_particles, rng=rng)

    def snap(kind, press=None, dropped=None):
        mean, cov = f.estimate()
        return {"kind": kind, "particles": f.particles.copy(), "weights": f.weights.copy(),
                "mean": mean, "cov": cov, "press": press, "dropped": dropped,
                "error": float(np.linalg.norm(mean - hole)), "spread": float(2 * np.sqrt(np.trace(cov) / 2))}

    steps = [snap("prior")]
    for _ in range(60):
        p = f.next_press()
        steps.append(dict(steps[-1], kind="choose", press=p, dropped=None))
        dropped = bool(np.linalg.norm(p - hole) < detect)
        f.update_press(p, dropped)
        steps.append(snap("result", p, dropped))
        if dropped:
            break
    seated = hole + rng.normal(scale=0.0005, size=2)
    f.update_seated(seated, std=0.001)
    steps.append(snap("seated", seated, True))
    return hole, steps


def draw(fig, axes, hole, steps, i, radius, detect):
    ax, ax2 = axes
    ax.clear()
    ax2.clear()
    s = steps[i]
    lim = radius * MM + 8

    # search region and box centre
    ax.add_patch(Circle((0, 0), radius * MM, fill=False, ls="--", lw=1, ec=MUTED))
    ax.plot(0, 0, "+", color=MUTED, ms=10, mew=1.2)
    ax.text(0, -radius * MM - 4, "search disc (R = 20 mm) around box centre", ha="center", fontsize=8, color=MUTED)

    # particles: size and opacity follow weight
    w = s["weights"] / s["weights"].max()
    pts = s["particles"] * MM
    ax.scatter(pts[:, 0], pts[:, 1], s=3 + 5 * w, c=BLUE, alpha=np.clip(0.15 + 0.6 * w, 0, 1), lw=0, zorder=2)

    # earlier presses
    presses = [st for st in steps[: i] if st["kind"] == "result"]
    for st in presses:
        c = GREEN if st["dropped"] else GRAY
        ax.add_patch(Circle(st["press"] * MM, detect * MM, color=c, alpha=0.12, lw=0, zorder=1))
        ax.plot(*(st["press"] * MM), "x" if not st["dropped"] else "o", color=c, ms=5, mew=1.2, zorder=3)

    # current press
    if s["kind"] in ("choose", "result") and s["press"] is not None:
        p = s["press"] * MM
        if s["kind"] == "choose":
            ax.add_patch(Circle(p, detect * MM, fill=False, ec=ORANGE, lw=1.6, ls="--", zorder=4))
            ax.plot(*p, "v", color=ORANGE, ms=8, zorder=5)
        else:
            c = GREEN if s["dropped"] else GRAY
            ax.add_patch(Circle(p, detect * MM, fill=True, color=c, alpha=0.25, lw=0, zorder=1))
            ax.add_patch(Circle(p, detect * MM, fill=False, ec=c, lw=1.6, zorder=4))

    # true hole centre (unknown to the filter)
    ax.plot(*(hole * MM), marker="*", color=INK, ms=12, mec="white", mew=1, zorder=6)

    # estimate and 2-sigma ellipse
    vals, vecs = np.linalg.eigh(s["cov"] * MM * MM)
    vals = np.maximum(vals, 1e-9)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(s["mean"] * MM, 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                         fill=False, ec=ORANGE, lw=1.8, zorder=6))
    ax.plot(*(s["mean"] * MM), "o", color=ORANGE, ms=6, mec="white", mew=1.2, zorder=7)

    n_press = len([st for st in steps[: i + 1] if st["kind"] == "result"])
    titles = {
        "prior": "Start: hole could be anywhere in the disc",
        "choose": f"Press {n_press + 1}: aim where the most probability is",
        "result": f"Press {n_press}: {'tip dropped in - hole found' if s['dropped'] else 'no drop - rule out this circle'}",
        "seated": "Tip slides to the hole centre: estimate pinned down",
    }
    ax.set_title(titles[s["kind"]], fontsize=11, color=INK, loc="left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", color=MUTED, fontsize=9)
    ax.set_ylabel("y (mm)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    # legend
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.6, ms=5, label="particles (hole guesses)"),
        plt.Line2D([], [], marker="*", ls="", color=INK, ms=10, label="true hole centre"),
        plt.Line2D([], [], marker="o", ls="-", color=ORANGE, ms=5, label="estimate + 2σ"),
        plt.Line2D([], [], marker="x", ls="", color=GRAY, ms=6, label="press: no drop"),
        plt.Line2D([], [], marker="o", ls="", color=GREEN, ms=6, label="press: dropped in"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 1.0))

    # right panel: error and spread over steps
    hist = [st for st in steps[: i + 1] if st["kind"] != "choose"]
    allh = [st for st in steps if st["kind"] != "choose"]
    xs = np.arange(len(hist))
    ax2.plot(xs, [st["error"] * MM for st in hist], color=ORANGE, lw=2, marker="o", ms=4, label="estimate error")
    ax2.plot(xs, [st["spread"] * MM for st in hist], color=BLUE, lw=2, ls="--", marker="s", ms=4, label="2σ spread")
    ax2.set_xlim(-0.5, len(allh) - 0.5)
    ax2.set_ylim(0, max(max(st["error"], st["spread"]) for st in allh) * MM * 1.1)
    labels = ["start"] + [str(k) for k in range(1, len(allh) - 1)] + ["seat"]
    ax2.set_xticks(range(len(allh)))
    ax2.set_xticklabels(labels, fontsize=7, color=MUTED)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.set_xlabel("press", color=MUTED, fontsize=9)
    ax2.set_ylabel("mm", color=MUTED, fontsize=9)
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    ax2.set_title("Estimate vs. true hole", fontsize=11, color=INK, loc="left")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "hole_search.gif"))
    args = ap.parse_args()
    radius, detect = 0.02, 0.005
    hole, steps = record_run(args.seed, radius, detect, 3000)

    fig = plt.figure(figsize=(11, 5), dpi=90)
    ax = fig.add_axes([0.05, 0.11, 0.42, 0.8])
    ax2 = fig.add_axes([0.70, 0.18, 0.28, 0.66])
    fig.text(0.01, 0.01, "Placeholder press model: a press finds the hole within 5 mm of its centre. "
             "Simulated, not real-robot data.", fontsize=7.5, color=MUTED)
    frames, durations = [], []
    for i, s in enumerate(steps):
        draw(fig, (ax, ax2), hole, steps, i, radius, detect)
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
        durations.append({"prior": 1800, "choose": 650, "result": 900, "seated": 3500}[s["kind"]])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    n = sum(st["kind"] == "result" for st in steps)
    print(f"saved {out}: {len(frames)} frames, {n} presses, final error {steps[-1]['error'] * MM:.2f} mm")


if __name__ == "__main__":
    main()
