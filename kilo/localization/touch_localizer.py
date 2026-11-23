"""Phase 1 box localization by touch, for any RobotBackend (real arm or sim).

Strategy (same as scripts/localize.py, which drives the MuJoCo scene directly):
  1. Snake sweep over the search rectangle at search height until first contact.
  2. Cross from the far side along the first contact's force direction (opposite face).
  3. From the midpoint, probe both ways along the perpendicular.
  4. Fit the box pose to the 4 contacts (kilo.localization.fit).
Additions for the real arm:
  - every descent to search height is force-guarded (landing on the box top stops);
  - optional verification: probe straight at each fitted face's centre and refit
    with all contacts. Four contacts can fit two different poses (see
    docs/experiments/phase1_refit*), and face-normal probes break that tie.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from kilo.localization.fit import BoxFit, box_faces, fit_box
from kilo.localization.search import get_ray_intersection
from kilo.robot.interface import ProbeSettings, RobotBackend, SafetyStop, nearest_equivalent_yaw


@dataclass
class Contact:
    label: str
    kind: str          # "side": hit a box face; "top": landed on the box top while descending
    xy: list
    force_xy: list
    z: float
    peak_force: float


class TouchLocalizer:
    def __init__(self, robot: RobotBackend, cfg: dict, scene_cfg: dict, log=print):
        ws, mc, cc = cfg["workspace"], cfg["motion"], cfg["contact"]
        if ws.get("table_z") is None:
            raise ValueError("workspace.table_z is not set: measure it with the `table` experiment first")
        self.robot, self.log = robot, log
        self.table_z = float(ws["table_z"])
        self.search_z = self.table_z + float(ws["search_offset"])
        self.travel_z = self.table_z + float(ws["travel_offset"])
        self.corners = np.asarray(ws["search_corners"], float)
        self.margin = float(ws.get("margin", 0.03))
        self.spacing = float(ws.get("snake_spacing", 0.06))
        self.travel_speed = float(mc["travel_speed"])
        self.wrist_offset = float(cfg["robot"].get("wrist_yaw_offset", -2.356))
        self.align_wrist = bool(mc.get("align_wrist", False))
        self.fixed_yaw = None
        hz = float(getattr(robot, "preferred_hz", mc.get("stream_hz", 50.0)))
        common = dict(force_threshold=float(cc["force_threshold"]), debounce=int(cc["debounce"]),
                      bias_seconds=float(cc["bias_seconds"]), safety_force=float(cc["safety_force"]),
                      stiffness=float(getattr(robot, "stiffness", mc["stream_stiffness"][0])), stream_hz=hz,
                      timeout=float(cc.get("probe_timeout", 90.0)))
        self.side_cfg = ProbeSettings(speed=float(mc["search_speed"]), retreat=float(cc["retreat"]), **common)
        self.descent_cfg = ProbeSettings(speed=float(mc["descent_speed"]), retreat=0.002, **common)
        self.half = np.asarray(scene_cfg["box"]["size"][:2], float) / 2
        self.radius = float(scene_cfg["peg"]["shaft_radius"])
        self.force_weight = float(cfg.get("fit", {}).get("force_weight", 0.004))
        self.contacts: list[Contact] = []
        self.events: list[dict] = []
        self.probe_logs: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _yaw_for(self, direction_xy, at_xy=None) -> float:
        """Finger-axis yaw along ``direction_xy``, choosing between it and its half turn so joint 7
        stays mid-range. With the tool pointing down, q7 ~ yaw - azimuth + 2.36 rad, where azimuth
        is the arm's base angle to the tip; ``wrist_yaw_offset`` (-2.36) puts q7 near 0."""
        if not self.align_wrist:
            # Default: keep one wrist angle for the whole search (no joint-7 limit risk). The peg
            # is round, so contact geometry does not depend on yaw; the grasp holds it by friction.
            if self.fixed_yaw is None:
                self.fixed_yaw = self.robot.state().yaw
            return self.fixed_yaw
        at = self.robot.state().tip[:2] if at_xy is None else np.asarray(at_xy)
        reference = self.wrist_offset + float(np.arctan2(at[1], at[0]))
        return nearest_equivalent_yaw(float(np.arctan2(direction_xy[1], direction_xy[0])), reference)

    def _event(self, what, **info):
        self.events.append({"t": time.time(), "what": what, **info})
        self.log(f"  {what}: " + ", ".join(f"{k}={v}" for k, v in info.items()))

    def _check_xy(self, xy):
        lo, hi = self.corners.min(axis=0) - self.margin, self.corners.max(axis=0) + self.margin
        if np.any(np.asarray(xy) < lo) or np.any(np.asarray(xy) > hi):
            raise SafetyStop(f"target {np.round(xy, 3)} is outside the search area (+{self.margin} m margin)")

    def _record_probe(self, label, res):
        self.probe_logs.append({"label": label, "contact": res.contact, "samples": np.asarray(res.samples)})

    def snake(self) -> list[np.ndarray]:
        """Same pattern as kilo.localization.search.generate_snake_waypoints, for the configured rectangle."""
        (x0, y0), (x1, y1) = self.corners.min(axis=0), self.corners.max(axis=0)
        ys = list(np.arange(y0, y1 + 1e-9, self.spacing))
        if y1 - ys[-1] > 1e-9:
            ys.append(y1)
        pts = []
        for i, y in enumerate(ys):
            far, near = np.array([x1, y]), np.array([x0, y])
            pts += [far, near] if i % 2 == 0 else [near, far]
        return pts

    # ------------------------------------------------------------------ motions
    def lift(self):
        st = self.robot.state()
        if st.tip[2] < self.travel_z - 0.002:
            self.robot.move(np.r_[st.tip[:2], self.travel_z], st.yaw, self.travel_speed)

    def travel(self, xy, yaw=None) -> bool:
        """Lift, move over ``xy``, then guarded descent to search height.
        Returns False (and records a top contact) if the descent landed on something."""
        self._check_xy(xy)
        self.lift()
        yaw = self.robot.state().yaw if yaw is None else yaw
        self.robot.move(np.r_[xy, self.travel_z], yaw, self.travel_speed)
        res = self.robot.probe(np.r_[xy, self.search_z], yaw, self.descent_cfg)
        self._record_probe(f"descent to {np.round(xy, 3).tolist()}", res)
        if res.contact:
            c = Contact(f"top@{len(self.contacts)}", "top", res.tip[:2].tolist(), res.force[:2].tolist(),
                        float(res.tip[2]), res.peak_force)
            self.contacts.append(c)
            self._event("landed on top during descent", xy=np.round(res.tip[:2], 4).tolist(),
                        height_above_table_mm=round((res.tip[2] - self.table_z) * 1000, 1))
            self.lift()
            return False
        return True

    def probe_xy(self, xy, label):
        """Rotate the wrist so the finger axis follows the motion, then probe toward ``xy``."""
        self._check_xy(xy)
        st = self.robot.state()
        direction = np.asarray(xy) - st.tip[:2]
        yaw = self._yaw_for(direction)
        if abs(np.angle(np.exp(1j * (yaw - st.yaw)))) > np.radians(3):
            self.robot.move(st.tip, yaw, self.travel_speed)  # rotate in place about the peg axis
        res = self.robot.probe(np.r_[xy, self.search_z], yaw, self.side_cfg)
        self._record_probe(label, res)
        if res.contact:
            c = Contact(label, "side", res.tip[:2].tolist(), res.force[:2].tolist(), float(res.tip[2]), res.peak_force)
            self.contacts.append(c)
            heading = direction / max(np.linalg.norm(direction), 1e-9)
            opposing = float(np.dot(res.force[:2], heading)) < 0
            self._event(f"contact ({label})", xy=np.round(res.tip[:2], 4).tolist(),
                        force=np.round(res.force[:2], 1).tolist(), force_opposes_motion=opposing)
            if not opposing:
                self._event("WARNING force points along the motion: check robot.force_sign in the config")
        return res

    def touch_from(self, origin, direction, label):
        """Approach the box from the search-area wall along ``-direction`` (like localize.py's touch_from)."""
        ray = np.array([origin, direction], float)
        start = get_ray_intersection(ray, self.corners)
        end = get_ray_intersection(np.array([origin, -np.asarray(direction)], float), self.corners)
        if start is None or end is None:
            raise RuntimeError(f"{label}: crossing line does not span the search area")
        if not self.travel(start, yaw=self._yaw_for(end - start, at_xy=start)):
            raise RuntimeError(f"{label}: landed on the box while positioning; search area too small?")
        res = self.probe_xy(end, label)
        if not res.contact:
            raise RuntimeError(f"{label}: crossing ended without touching the box")
        return np.asarray(res.tip[:2])

    # ------------------------------------------------------------------ strategy
    def _fit(self, kinds=("side",)) -> BoxFit:
        side = [c for c in self.contacts if c.kind in kinds]
        return fit_box([c.xy for c in side], [c.force_xy for c in side], self.half, self.radius, self.force_weight)

    def verify(self, fit: BoxFit, round_index: int) -> dict:
        """Probe straight at each fitted face centre; record misses and top landings."""
        clearance = self.radius + 0.03
        report = {"misses": 0, "top_landings": 0, "new_contacts": 0, "skipped": 0}
        for i, (normal, center) in enumerate(box_faces(fit.center, fit.yaw, self.half)):
            start = center + normal * clearance
            target = center - normal * 0.02
            try:
                self._check_xy(start)
                self._check_xy(target)
            except SafetyStop:
                report["skipped"] += 1
                self._event("verify face skipped (outside search area)", face=i)
                continue
            if not self.travel(start, yaw=self._yaw_for(-normal, at_xy=start)):
                report["top_landings"] += 1
                continue
            res = self.probe_xy(target, f"verify{round_index}-face{i}")
            if res.contact:
                report["new_contacts"] += 1
            elif res.stalled:
                report["stalled"] = report.get("stalled", 0) + 1
                self._event("verify probe stalled short of its target without contact", face=i)
            else:
                report["misses"] += 1
                self._event("verify probe found no face where the fit predicted one", face=i)
        return report

    def run(self, verify_rounds: int = 1) -> dict:
        t0 = time.time()
        waypoints = self.snake()
        self._event("start snake sweep", waypoints=len(waypoints))
        if not self.travel(waypoints[0]):
            raise RuntimeError("landed on the box at the first sweep corner; move the box or the search area")
        first = None
        for wp in waypoints[1:]:
            self._first_heading = np.asarray(wp) - self.robot.state().tip[:2]
            res = self.probe_xy(wp, "first")
            if res.contact:
                first = np.asarray(res.tip[:2])
                break
        if first is None:
            raise RuntimeError("sweep ended without touching the box")
        normal = np.asarray(self.contacts[-1].force_xy)
        normal = normal / np.linalg.norm(normal)
        try:
            opposite = self.touch_from(first, -normal, "opposite")
        except RuntimeError as exc:
            if "without touching" not in str(exc):
                raise
            # The force direction can be skewed by friction or a corner; cross along the
            # direction the first probe was travelling instead.
            heading = np.asarray(self._first_heading)
            self._event("opposite crossing missed; retrying along the first probe's travel direction")
            normal = -heading / np.linalg.norm(heading)
            opposite = self.touch_from(first, -normal, "opposite (retry)")
        mid = (first + opposite) / 2
        tangent = np.array([-normal[1], normal[0]])
        self.touch_from(mid, tangent, "left")
        self.touch_from(mid, -tangent, "right")
        fits = [self._fit()]
        self._event("initial fit", center=np.round(fits[0].center, 4).tolist(), yaw_deg=round(fits[0].yaw_deg, 1),
                    max_gap_mm=round(fits[0].max_gap * 1000, 2), ambiguous=fits[0].ambiguous())
        reports = []
        for r in range(verify_rounds):
            reports.append(self.verify(fits[-1], r))
            fits.append(self._fit())
            self._event(f"fit after verify round {r}", center=np.round(fits[-1].center, 4).tolist(),
                        yaw_deg=round(fits[-1].yaw_deg, 1), max_gap_mm=round(fits[-1].max_gap * 1000, 2),
                        **reports[-1])
            clean = reports[-1]["misses"] == 0 and reports[-1]["top_landings"] == 0 and fits[-1].max_gap < 0.001
            if clean:
                break
        self.lift()
        final = fits[-1]
        return {"center": final.center.tolist(), "yaw_deg": final.yaw_deg, "max_gap_mm": final.max_gap * 1000,
                "initial": {"center": fits[0].center.tolist(), "yaw_deg": fits[0].yaw_deg,
                            "max_gap_mm": fits[0].max_gap * 1000},
                "verify_reports": reports, "contacts": [asdict(c) for c in self.contacts],
                "events": self.events, "seconds": time.time() - t0}
