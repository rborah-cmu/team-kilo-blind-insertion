"""Live visualization of a running localization (box estimate, contacts, forces, hole search).

The robot loop must never wait on drawing, so all plotting happens in a separate process.
The runner calls the cheap, non-blocking methods of ``LiveView``; events go through a
queue and are dropped (never block) if the viewer falls behind.

    view = LiveView(out_dir, corners, half_sizes, radius, thresholds=(5, 20), window=True)
    view.sample(t, tip_xyz, residual_xyz)      # every force/tip sample (throttled here)
    view.contact(label, xy, force_xy)          # each side contact
    view.plate(center_xy, yaw_deg)             # final plate fit
    view.hole(presses, particles, weights, center, std, status)
    view.status("text")                        # a line shown in the title
    view.close()                               # final image, then the window stays open

Images: <out>/live.png (refreshed about once a second) and <out>/final.png at the end.
A window opens only when a display is available; otherwise the PNGs are the live view.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import time
from pathlib import Path

import numpy as np

# Reference categorical palette (light surface), fixed order; status colours reserved.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#008300"
CRITICAL, WARNING = "#d03b3b", "#fab219"
GAP_LEVELS_MM = [0, 1, 2, 4, 8, 16]


# ----------------------------------------------------------------------------- runner side
class LiveView:
    """Non-blocking front end. Every method swallows errors: the view is never a reason to stop."""

    def __init__(self, out_dir, corners, half_sizes, radius, *, thresholds=(5.0, 20.0), window=True,
                 sample_hz=20.0):
        self.min_dt = 1.0 / sample_hz
        self.last_sample = 0.0
        self.proc = None
        try:
            ctx = mp.get_context("spawn")  # clean interpreter: no inherited ROS/GUI state
            self.q = ctx.Queue(maxsize=5000)
            self.proc = ctx.Process(target=_viewer_main, daemon=True, args=(
                self.q, str(out_dir), np.asarray(corners, float).tolist(), np.asarray(half_sizes, float).tolist(),
                float(radius), tuple(float(t) for t in thresholds), bool(window)))
            self.proc.start()
        except Exception as exc:  # no viewer, but the localization still runs
            print(f"[live view disabled: {exc}]", flush=True)
            self.proc = None

    def _put(self, *event):
        if self.proc is None:
            return
        try:
            self.q.put_nowait(event)
        except Exception:
            pass

    def sample(self, t, tip, residual):
        now = time.monotonic()
        if now - self.last_sample >= self.min_dt:
            self.last_sample = now
            self._put("sample", float(t), np.asarray(tip, float)[:3].tolist(),
                      np.asarray(residual, float)[:3].tolist())

    def contact(self, label, xy, force_xy):
        self._put("contact", str(label), np.asarray(xy, float).tolist(), np.asarray(force_xy, float).tolist())

    def plate(self, center, yaw_deg):
        self._put("plate", np.asarray(center, float).tolist(), float(yaw_deg))

    def hole(self, presses, particles=None, weights=None, center=None, std=None, status=None):
        p = None if particles is None else np.asarray(particles, float)
        w = None if weights is None else np.asarray(weights, float)
        if p is not None and len(p) > 1500:  # keep queue messages small
            idx = np.random.default_rng(0).choice(len(p), 1500, replace=False, p=w / w.sum() if w is not None else None)
            p, w = p[idx], (w[idx] if w is not None else None)
        self._put("hole", [dict(tip=list(map(float, x["tip"])), dropped=bool(x["dropped"])) for x in presses or []],
                  None if p is None else p.tolist(), None if w is None else w.tolist(),
                  None if center is None else list(map(float, center)), std, status)

    def status(self, text):
        self._put("status", str(text))

    def path(self, points, label="spiral search", zoom=None):
        """A planned path to overlay on the map (e.g. the insertion spiral).
        ``zoom``: half-width (m) to zoom the map around the path's centre."""
        self._put("path", np.asarray(points, float)[:, :2].tolist(), str(label), zoom)

    def close(self, keep_open=True):
        """Draw the final picture. With keep_open and a window, wait until the user closes it."""
        if self.proc is None:
            return
        self._put("close", bool(keep_open))
        try:
            self.proc.join(None if keep_open else 10)
        except KeyboardInterrupt:
            pass
        if self.proc.is_alive():
            self.proc.terminate()


# ----------------------------------------------------------------------------- geometry
def box_outline(center, yaw_deg, half):
    y = np.radians(yaw_deg)
    R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * np.asarray(half)
    return np.asarray(center) + corners @ R.T


def belief_map(points, half, radius, xs, ys, yaws=np.radians(np.arange(0, 180, 3.0))):
    """Best-fit RMS contact gap (mm) if the box centre were at each grid point (min over yaw)."""
    pts = np.asarray(points, float)
    half = np.asarray(half, float)
    gx, gy = np.meshgrid(xs, ys)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
    best = np.full(len(centers), np.inf)
    for yaw in yaws:
        c, s = np.cos(yaw), np.sin(yaw)
        rel = pts[None, :, :] - centers[:, None, :]
        local = np.stack([rel[..., 0] * c + rel[..., 1] * s, -rel[..., 0] * s + rel[..., 1] * c], axis=-1)
        off = local - np.clip(local, -half, half)
        gap = np.abs(np.linalg.norm(off, axis=-1) - radius)
        best = np.minimum(best, np.sum(gap ** 2, axis=1))
    return (np.sqrt(best / len(pts)) * 1000).reshape(gx.shape)


# ----------------------------------------------------------------------------- viewer process
def _viewer_main(q, out_dir, corners, half, radius, thresholds, window):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    has_display = bool(os.environ.get("DISPLAY")) or os.name == "nt"
    interactive = False
    if window and has_display:
        for backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt  # noqa: F401
                plt.figure().clf()
                plt.close("all")
                interactive = True
                break
            except Exception:
                continue
    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch  # noqa: F401  (legend handles; mpl 3.1 safe)

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
    if interactive:
        plt.ion()
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_force = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    corners = np.asarray(corners, float)
    lo, hi = corners.min(0) - .05, corners.max(0) + .05
    xs, ys = np.linspace(lo[0], hi[0], 90), np.linspace(lo[1], hi[1], 110)
    contact_threshold, force_limit = thresholds
    st = dict(tips=[], forces=[], contacts=[], plate=None, provisional=None, hole=None, status="starting", path=None,
              t0=time.monotonic(), dirty=True, belief_n=-1, belief=None, done=False, keep_open=True)

    def provisional_fit():
        side = st["contacts"]
        if len(side) < 3:
            return None
        try:
            from kilo.localization.fit import fit_box
            f = fit_box([c[1] for c in side], [c[2] for c in side], np.asarray(half), radius)
            return np.asarray(f.center), float(f.yaw_deg), float(f.max_gap * 1000)
        except Exception:
            return None

    def draw():
        ax_map.clear(); ax_force.clear(); ax_info.clear()
        pts = [c[1] for c in st["contacts"]]
        if len(pts) >= 2:
            if st["belief_n"] != len(pts):
                st["belief"] = belief_map(pts, half, radius, xs, ys)
                st["belief_n"] = len(pts)
            ax_map.contourf(xs, ys, st["belief"], levels=GAP_LEVELS_MM, cmap="Blues_r", alpha=.55, zorder=0)
        rect = np.vstack([corners, corners[:1]])
        ax_map.plot(rect[:, 0], rect[:, 1], "--", color=INK2, lw=1, label="search area")
        if st["tips"]:
            tips = np.asarray(st["tips"])
            ax_map.plot(tips[:, 0], tips[:, 1], color=BLUE, lw=1, alpha=.7, label="peg-tip path")
            ax_map.plot(tips[-1, 0], tips[-1, 1], "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, label="peg tip now")
        for label, xy, f in st["contacts"]:
            xy, f = np.asarray(xy), np.asarray(f)
            ax_map.plot(*xy, "o", color=ORANGE, ms=8, mec=SURFACE, mew=1.5, zorder=6)
            if np.linalg.norm(f) > 0:
                ax_map.annotate("", xy=xy + f / np.linalg.norm(f) * .02, xytext=xy, zorder=6,
                                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
            ax_map.annotate(label, xy, xytext=(5, 5), textcoords="offset points", fontsize=7, color=INK2)
        title = "Box estimate: "
        if st["plate"] is not None:
            c, y = st["plate"]
            b = box_outline(c, y, half)
            ax_map.plot(b[:, 0], b[:, 1], color=AQUA, lw=2.5, label="plate fit (final)")
            ax_map.plot(*c, "+", color=AQUA, ms=14, mew=2)
            title += f"({c[0]:.4f}, {c[1]:.4f}) m, yaw {y:.1f}°"
        elif st["provisional"] is not None:
            c, y, gap = st["provisional"]
            b = box_outline(c, y, half)
            ax_map.plot(b[:, 0], b[:, 1], ":", color=AQUA, lw=2.2,
                        label=f"provisional fit ({len(st['contacts'])} contacts)")
            ax_map.plot(*c, "+", color=AQUA, ms=12, mew=2)
            title += f"provisional ({c[0]:.3f}, {c[1]:.3f}) m, yaw {y:.0f}°, worst gap {gap:.1f} mm"
        else:
            title += f"waiting for contacts ({len(st['contacts'])} so far)"
        if st["path"] is not None:
            pts, label = st["path"]
            pts = np.asarray(pts)
            ax_map.plot(pts[:, 0], pts[:, 1], "--", color=VIOLET, lw=.8, alpha=.8, label=label)
        hole = st["hole"]
        if hole:
            presses, particles, weights, center, std, status = hole
            if particles:
                p = np.asarray(particles)
                ax_map.scatter(p[:, 0], p[:, 1], s=3, c=weights, cmap="Purples", alpha=.5, zorder=3)
            for pr in presses:
                ax_map.plot(pr["tip"][0], pr["tip"][1], "o" if pr["dropped"] else "x",
                            color=GREEN if pr["dropped"] else ORANGE, ms=7, zorder=7)
            if center:
                ax_map.plot(*center, "*", color=VIOLET, ms=13, zorder=8, label="hole estimate")
        zlo, zhi = st.get("zoom") or (lo, hi)
        ax_map.set_xlim(zlo[0], zhi[0]); ax_map.set_ylim(zlo[1], zhi[1])
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x (m, robot base)"); ax_map.set_ylabel("y (m)")
        ax_map.set_title(title)
        ax_map.legend(loc="upper left", fontsize=7)

        if st["forces"]:
            fr = np.asarray(st["forces"])
            t = fr[:, 0]
            keep = t >= t[-1] - 30
            ax_force.plot(t[keep], np.linalg.norm(fr[keep, 1:3], axis=1), color=BLUE, lw=1.2, label="|Fxy| residual")
            ax_force.plot(t[keep], np.abs(fr[keep, 3]), color=AQUA, lw=1.2, label="|Fz| residual")
        ax_force.axhline(contact_threshold, color=WARNING, ls="--", lw=1.2)
        ax_force.annotate(f"contact {contact_threshold:g} N", (0, contact_threshold), xycoords=("axes fraction", "data"),
                          xytext=(2, 3), textcoords="offset points", fontsize=7, color=INK2)
        ax_force.axhline(force_limit, color=CRITICAL, ls="--", lw=1.2)
        ax_force.annotate(f"safety {force_limit:g} N", (0, force_limit), xycoords=("axes fraction", "data"),
                          xytext=(2, 3), textcoords="offset points", fontsize=7, color=INK2)
        ax_force.set_ylim(0, force_limit * 1.1)
        ax_force.set_xlabel("time (s), last 30 s"); ax_force.set_ylabel("force residual (N)")
        ax_force.set_title("Contact force")
        ax_force.legend(loc="center right", fontsize=7)

        ax_info.axis("off")
        lines = [f"status: {st['status']}", f"elapsed: {time.monotonic() - st['t0']:.0f} s",
                 f"contacts: {len(st['contacts'])}"]
        for label, xy, f in st["contacts"]:
            lines.append(f"  {label:<8} ({xy[0]:.4f}, {xy[1]:.4f})  F=({f[0]:+.1f}, {f[1]:+.1f}) N")
        if st["provisional"] is not None and st["plate"] is None:
            c, y, gap = st["provisional"]
            lines.append(f"provisional box: ({c[0]:.4f}, {c[1]:.4f}), yaw {y:.1f}°, worst gap {gap:.2f} mm")
        if st["plate"] is not None:
            c, y = st["plate"]
            lines.append(f"PLATE: centre ({c[0]:.4f}, {c[1]:.4f}) m, yaw {y:.2f}° (mod 180)")
        if hole:
            presses, _, _, center, std, status = hole
            lines.append(f"hole search: {status or 'running'}, {len(presses)} presses")
            if center:
                lines.append(f"  hole estimate ({center[0]:.4f}, {center[1]:.4f})"
                             + (f", std {1000 * std:.2f} mm" if std else ""))
        ax_info.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
        fig.tight_layout()

    last_draw = last_save = 0.0
    while True:
        try:
            while True:
                ev = q.get(timeout=.05)
                kind = ev[0]
                if kind == "sample":
                    _, t, tip, res = ev
                    st["tips"].append(tip[:2]); st["forces"].append([t] + res)
                    if len(st["tips"]) > 20000:
                        st["tips"] = st["tips"][-20000:]
                    if len(st["forces"]) > 3000:
                        st["forces"] = st["forces"][-3000:]
                elif kind == "contact":
                    st["contacts"].append((ev[1], ev[2], ev[3]))
                    st["provisional"] = provisional_fit()
                elif kind == "plate":
                    st["plate"] = (np.asarray(ev[1]), ev[2])
                elif kind == "hole":
                    st["hole"] = ev[1:]
                elif kind == "status":
                    st["status"] = ev[1]
                elif kind == "path":
                    st["path"] = (ev[1], ev[2])
                    if ev[3]:
                        centre = np.asarray(ev[1])[0]
                        st["zoom"] = (centre - ev[3], centre + ev[3])
                elif kind == "close":
                    st["done"], st["keep_open"] = True, ev[1]
                st["dirty"] = True
        except queue_mod.Empty:
            pass
        except (EOFError, OSError):
            st["done"] = True
        now = time.monotonic()
        if st["dirty"] and (now - last_draw > .25 or st["done"]):
            try:
                draw()
                if interactive:
                    fig.canvas.draw_idle()
                last_draw = now
            except Exception as exc:
                print(f"[live view draw error: {exc}]", flush=True)
            st["dirty"] = False
        if now - last_save > 1.0 or st["done"]:
            try:
                fig.savefig(out / "live.png", dpi=100)
            except Exception:
                pass
            last_save = now
        if interactive:
            plt.pause(.05)
        if st["done"]:
            try:
                fig.savefig(out / "final.png", dpi=150)
            except Exception:
                pass
            if interactive and st["keep_open"]:
                plt.ioff()
                plt.show()  # leave the final picture up until the window is closed
            return
