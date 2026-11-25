"""Charts and a plain-text report for one Phase 1 localization run.

    python3 scripts/real/view_localization.py runs/real/<stamp>-localize
    python3 scripts/real/view_localization.py --latest            # newest *-localize run
    python3 scripts/real/view_localization.py --latest --show     # also open the windows

Writes into <run>/charts/ (or --out):
  1_overview.png      top-down map: search area, peg-tip paths, contacts with force
                      arrows, initial/final fitted box, truth (sim or --truth), and the
                      final belief map
  2_belief.png        how the pose belief sharpens as contacts arrive (small multiples)
  3_estimate.png      estimated centre x / y / yaw and worst contact gap vs contact count
  4_forces.png        contact-force residual over every probe, with threshold lines
  5_motion.png        per-path execution audit (streamed paths: duration, tracking, lag)
  6_hole.png          measured vertical presses and the final hole particle distribution
  report.txt          outcome, safety stop, controller errors, estimate and score

The plate localizer fits the box pose by least squares to the touch contacts
(kilo.localization.fit.fit_box). The plate belief map
shows the same evidence over candidate box centres c:
    rms_gap(c) = min over yaw of sqrt(mean_i gap_i(c, yaw)^2),
where gap_i is how far contact i sits from the box outline grown by the peg radius
(contact positions only; the fit also uses force directions). Dark = box poses that
explain every contact within a millimetre or two; the dark region shrinking as
contacts arrive is the estimate becoming certain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reference categorical palette (light surface), fixed order; status colours reserved.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                     "#e87ba4", "#008300", "#6250d6")
CRITICAL, WARNING = "#d03b3b", "#fab219"
GAP_LEVELS_MM = [0, 1, 2, 4, 8, 16]


# ----------------------------------------------------------------------------- loading
def latest_run(root: Path) -> Path:
    runs = sorted(p for p in root.glob("*-localize") if (p / "result.json").exists())
    if not runs:
        raise SystemExit(f"no *-localize runs with result.json under {root}")
    return runs[-1]


def load_run(run: Path, truth_path: str | None):
    data = json.loads((run / "result.json").read_text())
    arrays = {}
    if (run / "samples.npz").exists():
        z = np.load(run / "samples.npz")
        arrays = {k: z[k] for k in z.files}
    res = data.get("result") or {}
    truth = res.get("truth")
    if truth_path:
        truth = json.loads(Path(truth_path).read_text())
    return data, arrays, res, truth


def geometry(data):
    from kilo.config import load_config
    scene = load_config()
    half = np.asarray(scene["box"]["size"][:2], float) / 2
    radius = float(scene["peg"]["shaft_radius"])
    cfg = data.get("config") or {}
    corners = np.asarray((cfg.get("workspace") or {}).get("search_corners",
                                                          [[.32, -.21], [.565, -.21], [.565, .15], [.32, .15]]), float)
    fit_cfg = cfg.get("fit") or {}
    contact = cfg.get("contact") or {}
    return half, radius, corners, float(fit_cfg.get("force_weight", .004)), \
        float(contact.get("force_threshold", 4.)), float(contact.get("safety_force", 20.))


# ----------------------------------------------------------------------------- estimation
def side_contacts(data):
    return [c for c in (data.get("contacts") or []) if c.get("kind") == "side"]


def sequential_fits(contacts, half, radius, force_weight):
    """Refit with the first k contacts, k = 3..N (what the estimate was after each touch)."""
    from kilo.localization.fit import fit_box
    out = []
    for k in range(3, len(contacts) + 1):
        sub = contacts[:k]
        try:
            f = fit_box([c["xy"] for c in sub], [c["force_xy"] for c in sub], half, radius, force_weight)
            out.append(dict(k=k, label=sub[-1]["label"], center=np.asarray(f.center), yaw_deg=f.yaw_deg,
                            gap_mm=f.max_gap * 1000, ambiguous=bool(f.ambiguous())))
        except Exception as exc:  # a degenerate early subset should not kill the report
            out.append(dict(k=k, label=sub[-1]["label"], error=str(exc)))
    return out


def belief_map(points, half, radius, xs, ys, yaws=np.radians(np.arange(0, 180, 1.5))):
    """Best-fit RMS contact gap (mm) over box centre, minimised over yaw."""
    pts = np.asarray(points, float)
    gx, gy = np.meshgrid(xs, ys)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
    best = np.full(len(centers), np.inf)
    for yaw in yaws:
        c, s = np.cos(yaw), np.sin(yaw)
        rel = pts[None, :, :] - centers[:, None, :]                      # (G, N, 2)
        local = np.stack([rel[..., 0] * c + rel[..., 1] * s, -rel[..., 0] * s + rel[..., 1] * c], axis=-1)
        off = local - np.clip(local, -half, half)
        gap = np.abs(np.linalg.norm(off, axis=-1) - radius)
        best = np.minimum(best, np.sum(gap ** 2, axis=1))
    return (np.sqrt(best / len(pts)) * 1000).reshape(gx.shape)


def box_outline(center, yaw_deg, half):
    y = np.radians(yaw_deg)
    R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * half
    return np.asarray(center) + corners @ R.T


# ----------------------------------------------------------------------------- plotting
def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    })


def mm_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, robot base)")
    ax.set_ylabel("y (m)")


def draw_contacts(ax, contacts, arrow_scale=.004, labels=True):
    for c in contacts:
        xy, f = np.asarray(c["xy"]), np.asarray(c["force_xy"], float)
        verify = c["label"].startswith("verify")
        ax.plot(*xy, "o", ms=7, color=VIOLET if verify else ORANGE, mec=SURFACE, mew=1.5, zorder=6)
        if np.linalg.norm(f) > 0:
            ax.annotate("", xy=xy + f * arrow_scale, xytext=xy, zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
        if labels:
            ax.annotate(c["label"].replace("verify", "v"), xy, xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=INK2, zorder=7)


def fig_overview(plt, data, arrays, res, truth, geo, out):
    half, radius, corners, *_ = geo
    contacts = side_contacts(data)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    lo, hi = corners.min(0) - .05, corners.max(0) + .05
    if len(contacts) >= 3:
        xs, ys = np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 200)
        gap = belief_map([c["xy"] for c in contacts], half, radius, xs, ys)
        cs = ax.contourf(xs, ys, gap, levels=GAP_LEVELS_MM, cmap="Blues_r", alpha=.6, zorder=0)
        cb = fig.colorbar(cs, ax=ax, shrink=.4, pad=.02, anchor=(0, 0))
        cb.set_label("best-fit RMS contact gap (mm)\nif the box centre were here", fontsize=7)
    rect = np.vstack([corners, corners[:1]])
    ax.plot(rect[:, 0], rect[:, 1], "--", color=INK2, lw=1, label="search area")
    first = True
    for name in sorted(k for k in arrays if k.startswith("probe")):
        a = arrays[name]
        if a.ndim == 2 and len(a) and a.shape[1] >= 3:
            ax.plot(a[:, 1], a[:, 2], color=BLUE, lw=1, alpha=.6, label="peg-tip probe paths" if first else None)
            first = False
    if res.get("initial"):
        b = box_outline(res["initial"]["center"], res["initial"]["yaw_deg"], half)
        ax.plot(b[:, 0], b[:, 1], ":", color=AQUA, lw=2, label="initial fit (4 contacts)")
    if res.get("center"):
        b = box_outline(res["center"], res["yaw_deg"], half)
        ax.plot(b[:, 0], b[:, 1], color=AQUA, lw=2.5, label="final fit")
        ax.plot(*res["center"], "+", color=AQUA, ms=12, mew=2)
    if truth and "box_center" in truth:
        b = box_outline(truth["box_center"], truth["box_yaw_deg"], half)
        ax.plot(b[:, 0], b[:, 1], "--", color=INK, lw=1.2, label="ground truth")
    draw_contacts(ax, contacts)
    ax.plot([], [], "o", color=ORANGE, label="search contacts")
    ax.plot([], [], "o", color=VIOLET, label="verification contacts")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    mm_axes(ax)
    title = "Localization overview"
    if res.get("center"):
        title += f": centre ({res['center'][0]:.4f}, {res['center'][1]:.4f}) m, yaw {res['yaw_deg']:.1f}°"
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.text(1.01, .62, "arrows: measured contact force\nshading: where the box centre\ncan be, given the contacts",
            transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(out / "1_overview.png", dpi=150)
    return fig


def fig_belief(plt, data, res, truth, geo, out):
    half, radius, corners, *_ = geo
    contacts = side_contacts(data)
    if len(contacts) < 3:
        return None
    ks = list(range(3, len(contacts) + 1))
    cols = min(4, len(ks))
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows), squeeze=False)
    center = np.asarray(res.get("center") or np.mean([c["xy"] for c in contacts], 0))
    span = max(half) + .06
    xs = np.linspace(center[0] - span, center[0] + span, 140)
    ys = np.linspace(center[1] - span, center[1] + span, 140)
    fits = {f["k"]: f for f in sequential_fits(contacts, half, radius, geo[3])}
    for ax, k in zip(axes.ravel(), ks):
        gap = belief_map([c["xy"] for c in contacts[:k]], half, radius, xs, ys)
        cs = ax.contourf(xs, ys, gap, levels=GAP_LEVELS_MM, cmap="Blues_r")
        draw_contacts(ax, contacts[:k], arrow_scale=.003, labels=False)
        f = fits.get(k, {})
        if "center" in f:
            b = box_outline(f["center"], f["yaw_deg"], half)
            ax.plot(b[:, 0], b[:, 1], color=AQUA, lw=1.8)
        if truth and "box_center" in truth:
            b = box_outline(truth["box_center"], truth["box_yaw_deg"], half)
            ax.plot(b[:, 0], b[:, 1], "--", color=INK, lw=1)
        sub = f"after {k} contacts (+{contacts[k-1]['label']})"
        if "gap_mm" in f:
            sub += f"\nworst gap {f['gap_mm']:.2f} mm" + ("  AMBIGUOUS" if f.get("ambiguous") else "")
        ax.set_title(sub, fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(ks):]:
        ax.axis("off")
    fig.suptitle("Pose belief as contacts arrive (shading = best-fit RMS gap if the box centre were there;"
                 " green box = fit; dashed = truth)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, .93, 1))
    cb = fig.colorbar(cs, cax=fig.add_axes([.945, .2, .012, .6]))
    cb.set_label("RMS gap (mm)", fontsize=8)
    fig.savefig(out / "2_belief.png", dpi=150)
    return fig


def fig_estimate(plt, data, truth, geo, out):
    half, radius, corners, force_weight, *_ = geo
    fits = [f for f in sequential_fits(side_contacts(data), half, radius, force_weight) if "center" in f]
    if not fits:
        return None
    k = [f["k"] for f in fits]
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.5), sharex=True)
    series = [("centre x (mm)", [f["center"][0] * 1000 for f in fits], truth and truth.get("box_center", [None])[0]),
              ("centre y (mm)", [f["center"][1] * 1000 for f in fits], truth and (truth.get("box_center") or [None, None])[1]),
              ("yaw (deg, mod 180)", [f["yaw_deg"] for f in fits], truth and truth.get("box_yaw_deg")),
              ("worst contact gap (mm)", [f["gap_mm"] for f in fits], None)]
    for ax, (name, ys, true) in zip(axes, series):
        ax.plot(k, ys, "-o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
        if true is not None:
            tv = true * (1000 if "mm" in name and "gap" not in name else 1)
            ax.axhline(tv, color=INK, ls="--", lw=1)
            ax.annotate("truth", (k[-1], tv), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
        ax.set_ylabel(name)
    for f in fits:
        if f.get("ambiguous"):
            axes[0].annotate("ambiguous", (f["k"], f["center"][0] * 1000), xytext=(0, 8), textcoords="offset points",
                             fontsize=7, color=INK2, ha="center")
    axes[-1].set_xlabel("contacts used in the fit")
    axes[-1].set_xticks(k)
    axes[-1].set_xticklabels([f"{f['k']}\n{f['label'].replace('verify', 'v')}" for f in fits], fontsize=7)
    axes[0].set_title("Estimate after each contact (refit with the contacts so far)")
    fig.tight_layout()
    fig.savefig(out / "3_estimate.png", dpi=150)
    return fig


def fig_forces(plt, data, arrays, geo, out):
    *_, threshold, safety = geo
    names = sorted(k for k in arrays if k.startswith("probe"))
    if not names:
        return None
    labels = data.get("probe_labels") or names
    directional = ((data.get("config") or {}).get("contact") or {}).get("directional_residual")
    fig, ax = plt.subplots(figsize=(11, 4))
    offset = 0.
    for i, name in enumerate(names):
        a = arrays[name]
        if a.ndim != 2 or len(a) < 2 or a.shape[1] < 7:
            continue
        t = a[:, 0] - a[0, 0] + offset
        is_contact_probe = i < len(labels) and not labels[i].startswith("descent")
        if directional:
            mag = np.linalg.norm(a[:, 4:6], axis=1) if is_contact_probe else np.abs(a[:, 6])
        else:
            mag = np.linalg.norm(a[:, 4:7], axis=1)
        ax.plot(t, mag, color=BLUE if is_contact_probe else AQUA, lw=1.2)
        if mag.max() > threshold:
            j = int(np.argmax(mag > threshold))
            ax.plot(t[j], mag[j], "o", color=ORANGE, ms=6, mec=SURFACE, mew=1.2)
            ax.annotate(labels[i].replace("verify", "v") if i < len(labels) else name, (t[j], mag[j]),
                        xytext=(0, 8), textcoords="offset points", fontsize=7, color=INK2, ha="center")
        offset = t[-1] + 1.
    ax.axhline(threshold, color=WARNING, lw=1.2, ls="--")
    ax.annotate(f"contact threshold {threshold:g} N", (0, threshold), xytext=(2, 3), textcoords="offset points",
                fontsize=7, color=INK2)
    ax.axhline(safety, color=CRITICAL, lw=1.2, ls="--")
    ax.annotate(f"safety stop {safety:g} N", (0, safety), xytext=(2, 3), textcoords="offset points",
                fontsize=7, color=INK2)
    ax.plot([], [], color=BLUE, label="side probes"); ax.plot([], [], color=AQUA, label="guarded descents")
    ax.plot([], [], "o", color=ORANGE, label="first sample over threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("probe time, concatenated (s)")
    ax.set_ylabel("XY or Z force residual (N)" if directional else "force residual |F - baseline| (N)")
    ax.set_ylim(0, max(safety * 1.1, 1))
    ax.set_title("Contact-force residual during every probe")
    fig.tight_layout()
    fig.savefig(out / "4_forces.png", dpi=150)
    return fig


def fig_motion(plt, data, out):
    audit = data.get("joint_path_audit") or []
    if not audit:
        return None
    rows = []
    for i, a in enumerate(audit):
        st = (a.get("tracking") or {}).get("stream") or {}
        segments = a.get("segments") or []
        rows.append(dict(i=i, kind=a.get("kind"), ok=bool(a.get("completed")),
                         dur=st.get("duration_s") or sum(float(s.get("duration") or 0) for s in segments), cross=st.get("max_cross_track_rad"),
                         lag=st.get("max_tip_lag_m"), fail=(a.get("failure") or {}).get("message")))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    x = [r["i"] for r in rows]
    colors = [{"travel": BLUE, "probe": ORANGE, "startup": VIOLET}.get(r["kind"], INK2) for r in rows]
    for ax, key, name, scale in ((axes[0], "dur", "planned duration (s)", 1),
                                 (axes[1], "cross", "max off-path joint error (mrad)", 1000),
                                 (axes[2], "lag", "max tip lag behind target (mm)", 1000)):
        vals = [(r[key] or 0) * scale for r in rows]
        ax.bar(x, vals, color=colors, width=.7)
        ax.set_ylabel(name, fontsize=8)
    for r in rows:
        if not r["ok"]:
            for ax in axes:
                ax.axvline(r["i"], color=CRITICAL, lw=1.2, ls="--")
            axes[0].annotate("STOP: " + (r["fail"] or "failed"), (r["i"], axes[0].get_ylim()[1]), xytext=(4, -10),
                             textcoords="offset points", fontsize=7, color=CRITICAL, va="top")
    axes[1].axhline(10, color=WARNING, lw=1, ls="--"); axes[2].axhline(6, color=WARNING, lw=1, ls="--")
    from matplotlib.patches import Patch  # explicit handles: empty bar() fails on matplotlib 3.1
    axes[0].legend(handles=[Patch(color=c, label=n) for c, n in ((VIOLET, "startup"), (BLUE, "travel"),
                                                                 (ORANGE, "probe"))],
                   loc="upper left", fontsize=8, ncol=3)
    axes[-1].set_xlabel("executed path #")
    axes[0].set_title("Motion audit per path (dashed amber = stop limits)")
    fig.tight_layout()
    fig.savefig(out / "5_motion.png", dpi=150)
    return fig


def fig_hole(plt, data, arrays, out):
    hole = data.get("hole")
    if not hole or not hole.get("reference"):
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    if "hole_particles" in arrays:
        points, weights = arrays["hole_particles"], arrays["hole_weights"]
        ax.scatter(*points.T, s=4, c=weights, cmap="Blues", alpha=.4, label="hole particles")
    for dropped, color, marker, label in ((False, ORANGE, "x", "no drop"),
                                         (True, GREEN, "o", "drop")):
        points = [p["tip"][:2] for p in hole["presses"] if p["dropped"] == dropped]
        if points:
            ax.scatter(*np.asarray(points).T, color=color, marker=marker, label=label)
    ax.plot(*hole["reference"]["tip"][:2], "+", color=INK, label="plate height reference")
    if hole["status"] == "found":
        ax.plot(*hole["center"], "*", color=VIOLET, ms=10, label="hole estimate")
    mm_axes(ax)
    ax.set_title(f"Hole search ({hole['status'].replace('_', ' ')})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "6_hole.png", dpi=150)
    return fig


# ----------------------------------------------------------------------------- report
def report(run, data, res, truth, geo):
    half, radius, corners, force_weight, *_ = geo
    lines = [f"run: {run}", f"backend: {data.get('backend')}   started: {data.get('started')}"]
    stop = data.get("safety_stop") or data.get("error")
    lines.append("OUTCOME: " + (f"STOPPED - {stop}" if stop else
                                ("verified estimate" if res.get("verified") else
                                 "estimate NOT verified (provisional)" if res.get("center") else "no estimate")))
    if res.get("center"):
        lines.append(f"estimate: centre ({res['center'][0]:.4f}, {res['center'][1]:.4f}) m, yaw {res['yaw_deg']:.2f} deg,"
                     f" worst gap {res['max_gap_mm']:.2f} mm, {res.get('seconds', 0):.0f} s")
    if res.get("score"):
        lines.append("score vs truth: " + json.dumps(res["score"]))
    hole = data.get("hole") or {}
    if hole.get("status"):
        lines.append(f"hole search: {hole['status']} after {len(hole['presses'])} presses")
        if hole["status"] == "found":
            x, y = hole["center"]
            lines.append(f"  hole centre ({x:.4f}, {y:.4f}) m, particle std {1000 * hole['std_m']:.2f} mm")
    contacts = data.get("contacts") or []
    lines.append(f"contacts: {len(contacts)}")
    for c in contacts:
        lines.append(f"  {c['label']:<16} {c['kind']:<5} xy=({c['xy'][0]:.4f}, {c['xy'][1]:.4f})"
                     f" F=({c['force_xy'][0]:+.1f}, {c['force_xy'][1]:+.1f}) N  peak {c['peak_force']:.1f} N")
    fits = sequential_fits(side_contacts(data), half, radius, force_weight)
    if fits:
        lines.append("estimate after each contact:")
        for f in fits:
            if "center" in f:
                lines.append(f"  {f['k']:>2} (+{f['label']:<14}) centre ({f['center'][0]:.4f}, {f['center'][1]:.4f})"
                             f" yaw {f['yaw_deg']:6.2f}  gap {f['gap_mm']:.2f} mm" + ("  ambiguous" if f["ambiguous"] else ""))
    audit = data.get("joint_path_audit") or []
    if audit:
        n_ok = sum(bool(a.get("completed")) for a in audit)
        lines.append(f"motion: {len(audit)} paths, {n_ok} completed")
        for i, a in enumerate(audit):
            if a.get("failure"):
                tr = a.get("tracking") or {}
                ctrl = (tr.get("termination") or {}).get("controller") or {}
                lf = tr.get("last_feedback") or {}
                lines.append(f"  FAILED path #{i} ({a.get('kind')}): {a['failure'].get('message')}")
                if ctrl:
                    lines.append(f"    controller current_errors={ctrl.get('current_errors')}"
                                 f" last_motion_errors={ctrl.get('last_motion_errors')} robot_mode={ctrl.get('robot_mode')}")
                if lf:
                    lines.append(f"    at t={lf.get('elapsed_s', 0):.2f} s, tip={np.round(lf.get('tip', []), 4).tolist()}")
    events = data.get("events") or []
    if events:
        lines.append("events:")
        for e in events:
            extra = {k: v for k, v in e.items() if k not in ("t", "what")}
            lines.append(f"  {e['what']}: {extra}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="a runs/real/<stamp>-localize folder")
    ap.add_argument("--latest", action="store_true", help="use the newest *-localize run")
    ap.add_argument("--root", default=str(ROOT / "runs" / "real"))
    ap.add_argument("--truth", default=None, help="truth JSON (from the groundtruth command)")
    ap.add_argument("--out", default=None, help="output folder (default <run>/charts)")
    ap.add_argument("--show", action="store_true", help="open the figures in windows")
    args = ap.parse_args()
    if not args.run and not args.latest:
        ap.error("give a run folder or --latest")
    run = latest_run(Path(args.root)) if args.latest else Path(args.run)
    data, arrays, res, truth = load_run(run, args.truth)
    geo = geometry(data)
    out = Path(args.out) if args.out else run / "charts"
    try:
        out.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out = Path("/tmp") / f"kilo_charts_{run.name}"
        out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    text = report(run, data, res, truth, geo)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    made = []
    for name, fn in (("overview", lambda: fig_overview(plt, data, arrays, res, truth, geo, out)),
                     ("belief", lambda: fig_belief(plt, data, res, truth, geo, out)),
                     ("estimate", lambda: fig_estimate(plt, data, truth, geo, out)),
                     ("forces", lambda: fig_forces(plt, data, arrays, geo, out)),
                     ("motion", lambda: fig_motion(plt, data, out)),
                     ("hole", lambda: fig_hole(plt, data, arrays, out))):
        try:
            if fn() is not None:
                made.append(name)
        except Exception as exc:  # one broken chart should not hide the others
            print(f"[chart {name} skipped: {type(exc).__name__}: {exc}]")
    print(f"\ncharts ({', '.join(made) or 'none'}) and report.txt written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
