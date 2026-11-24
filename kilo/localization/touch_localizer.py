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
from kilo.robot.interface import Q7_LIMIT, ProbeSettings, RobotBackend, SafetyStop, choose_wrist_yaw
from kilo.robot.workspace import HandWorkspace


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
        self.height_tolerance = float(ws.get("search_height_tolerance", 0.006))
        self.travel_z = self.table_z + float(ws["travel_offset"])
        self.corners = np.asarray(ws["search_corners"], float)
        self.margin = float(ws.get("margin", 0.03))
        self.spacing = float(ws.get("snake_spacing", 0.06))
        self.travel_speed = float(mc["travel_speed"])
        self.wrist_offset = float(cfg["robot"].get("wrist_yaw_offset", -2.356))
        self.align_wrist = bool(mc.get("align_wrist", False))
        self.hand_workspace = HandWorkspace(ws)
        self.fixed_yaw = mc.get("fixed_wrist_yaw")
        hz = float(getattr(robot, "preferred_hz", mc.get("stream_hz", 50.0)))
        common = dict(force_threshold=float(cc["force_threshold"]), debounce=int(cc["debounce"]),
                      bias_seconds=float(cc["bias_seconds"]), safety_force=float(cc["safety_force"]),
                      stiffness=float(getattr(robot, "stiffness", mc["stream_stiffness"][0])), stream_hz=hz,
                      timeout=float(cc.get("probe_timeout", 90.0)))
        self.side_cfg = ProbeSettings(speed=float(mc["search_speed"]), retreat=float(cc["retreat"]), **common)
        self.descent_cfg = ProbeSettings(speed=float(mc["descent_speed"]), retreat=0.002, **common)
        self.half = np.asarray(scene_cfg["box"]["size"][:2], float) / 2
        self.radius = float(scene_cfg["peg"]["shaft_radius"])
        # The gripper fingers' lower edge sits one shaft length above the tip. If it is not clear of
        # the box top, the fingers (wider than the shaft) hit the box before the shaft does, and the
        # contact geometry is wrong. Seen on the lab robot: fingers landed on the box top.
        finger_clearance = float(ws["search_offset"]) + float(scene_cfg["peg"]["shaft_length"]) \
            - float(scene_cfg["box"]["size"][2])
        self.min_search_z = self.table_z + float(scene_cfg["box"]["size"][2]) + 0.005 \
            - float(scene_cfg["peg"]["shaft_length"])
        if finger_clearance < 0.005:
            raise ValueError(f"workspace.search_offset leaves the fingers only {finger_clearance * 1000:.1f} mm "
                             "above the box top; raise it so the clearance is at least 5 mm")
        self.force_weight = float(cfg.get("fit", {}).get("force_weight", 0.004))
        self.contacts: list[Contact] = []
        self.events: list[dict] = []
        self.probe_logs: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _yaw_for(self, direction_xy, at_xy=None, start_xy=None) -> float:
        """Finger-axis yaw along ``direction_xy`` (so contact pushes the peg into a finger), choosing
        between it and its half turn by the predicted joint 7 at ``at_xy`` (see choose_wrist_yaw)."""
        st = self.robot.state()
        start = st.tip[:2] if start_xy is None else np.asarray(start_xy)
        end = start if at_xy is None else np.asarray(at_xy)
        if not self.align_wrist:
            # Default: keep one wrist angle for the whole search (no joint-7 limit risk). The peg
            # is round, so contact geometry does not depend on yaw; the grasp holds it by friction.
            if self.fixed_yaw is None:
                self.fixed_yaw = st.yaw
            self.hand_workspace.check(st.tip[:2], st.tip[:2], st.yaw, self.fixed_yaw)
            self.hand_workspace.check(st.tip[:2], start, self.fixed_yaw, self.fixed_yaw)
            self.hand_workspace.check(start, end, self.fixed_yaw, self.fixed_yaw)
            return self.fixed_yaw
        direction = np.asarray(direction_xy, float)
        candidates = (direction, np.array([-direction[1], direction[0]]),
                      np.array([1., 0.]), np.array([0., 1.]))
        for index, candidate in enumerate(candidates):
            yaw, q7 = choose_wrist_yaw(candidate, st, end)
            if q7 is not None and abs(q7) > Q7_LIMIT - 0.3:
                continue
            try:
                self.hand_workspace.check(st.tip[:2], st.tip[:2], st.yaw, yaw)
                self.hand_workspace.check(st.tip[:2], start, yaw, yaw)
                self.hand_workspace.check(start, end, yaw, yaw)
            except SafetyStop:
                continue
            if index >= 2:
                self._event("using wall-safe cardinal wrist", yaw_deg=round(float(np.degrees(yaw)), 1))
            return yaw
        raise SafetyStop("no wall-safe wrist orientation for this path")

    def _event(self, what, **info):
        self.events.append({"t": time.time(), "what": what, **info})
        self.log(f"  {what}: " + ", ".join(f"{k}={v}" for k, v in info.items()))

    def _check_xy(self, xy):
        lo, hi = self.corners.min(axis=0) - self.margin, self.corners.max(axis=0) + self.margin
        if np.any(np.asarray(xy) < lo) or np.any(np.asarray(xy) > hi):
            raise SafetyStop(f"target {np.round(xy, 3)} is outside the search area (+{self.margin} m margin)")

    def _record_probe(self, label, res):
        self.probe_logs.append({"label": label, "contact": res.contact, "samples": np.asarray(res.samples)})

    def _check_search_height(self, tip):
        if abs(float(tip[2]) - self.search_z) > self.height_tolerance or tip[2] < self.min_search_z:
            raise SafetyStop(f"tip z={tip[2]:.4f} is outside valid probing height "
                             f"{self.search_z:.4f} +/- {self.height_tolerance:.3f} m "
                             "or leaves insufficient finger clearance; do not fit this contact")

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
            # Request the planned wrist yaw, not the measured one: an impedance-held wrist
            # sits a fraction of a mrad off, and the planned executor accepts only its yaw.
            yaw = st.yaw if self.fixed_yaw is None else float(self.fixed_yaw)
            self.hand_workspace.check(st.tip[:2], st.tip[:2], st.yaw, yaw)
            self.robot.move(np.r_[st.tip[:2], self.travel_z], yaw, self.travel_speed)

    def travel(self, xy, yaw=None) -> bool:
        """Lift, move over ``xy``, then guarded descent to search height.
        Returns False (and records a top contact) if the descent landed on something."""
        self._check_xy(xy)
        st = self.robot.state()
        if yaw is None:
            yaw = self._yaw_for(np.asarray(xy) - st.tip[:2], at_xy=xy)
        self.hand_workspace.check(st.tip[:2], st.tip[:2], st.yaw, yaw)
        self.hand_workspace.check(st.tip[:2], xy, yaw, yaw)
        self.lift()
        st = self.robot.state()
        # Only re-orient in place for a real yaw change; sub-degree residuals are corrected
        # by the backend (and an extra in-place skill per travel is itself a failure risk).
        if abs(np.angle(np.exp(1j * (yaw - st.yaw)))) > np.radians(0.2):
            self.robot.move(st.tip, yaw, self.travel_speed)
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
        if res.stalled:
            raise SafetyStop(f"descent stalled at z={res.tip[2]:.4f}, target z={self.search_z:.4f}")
        self._check_search_height(res.tip)
        return True

    def probe_xy(self, xy, label):
        """Rotate the wrist so the finger axis follows the motion, then probe toward ``xy``."""
        self._check_xy(xy)
        st = self.robot.state()
        self._check_search_height(st.tip)
        direction = np.asarray(xy) - st.tip[:2]
        yaw = self._yaw_for(direction, at_xy=xy)
        if abs(np.angle(np.exp(1j * (yaw - st.yaw)))) > np.radians(3):
            self.robot.move(st.tip, yaw, self.travel_speed)  # rotate in place about the peg axis
            self._check_search_height(self.robot.state().tip)
        res = self.robot.probe(np.r_[xy, self.search_z], yaw, self.side_cfg)
        self._record_probe(label, res)
        self._check_search_height(res.tip)
        if res.stalled:
            self._event(f"probe stalled short of its target ({label})", reached=np.round(res.tip[:2], 4).tolist(),
                        target=np.round(np.asarray(xy), 4).tolist())
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
        # A contact can sit a hair outside the search rectangle (tracking error); clamp it inside so
        # the crossing can be planned (seen on the lab robot: y = -0.2004 vs -0.20).
        lo, hi = self.corners.min(axis=0) + 0.001, self.corners.max(axis=0) - 0.001
        origin = np.clip(np.asarray(origin, float), lo, hi)
        ray = np.array([origin, direction], float)
        start = get_ray_intersection(ray, self.corners)
        end = get_ray_intersection(np.array([origin, -np.asarray(direction)], float), self.corners)
        if start is None or end is None:
            raise RuntimeError(f"{label}: crossing line does not span the search area")
        if not self.travel(start, yaw=self._yaw_for(end - start, at_xy=end, start_xy=start)):
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
                yaw = self._yaw_for(-normal, at_xy=target, start_xy=start)
            except SafetyStop as exc:
                report["skipped"] += 1
                self._event("verify face skipped (no safe positioning path)", face=i, reason=str(exc))
                continue
            if not self.travel(start, yaw=yaw):
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
        first_yaw = self._yaw_for(waypoints[1] - waypoints[0], at_xy=waypoints[1], start_xy=waypoints[0])
        if not self.travel(waypoints[0], yaw=first_yaw):
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
        reports, verified = [], False
        for r in range(verify_rounds):
            reports.append(self.verify(fits[-1], r))
            fits.append(self._fit())
            self._event(f"fit after verify round {r}", center=np.round(fits[-1].center, 4).tolist(),
                        yaw_deg=round(fits[-1].yaw_deg, 1), max_gap_mm=round(fits[-1].max_gap * 1000, 2),
                        **reports[-1])
            clean = (reports[-1]["misses"] == 0 and reports[-1]["top_landings"] == 0
                     and reports[-1]["new_contacts"] >= 2 and fits[-1].max_gap < 0.001
                     and not fits[-1].ambiguous())
            if clean:
                verified = True
                break
        self.lift()
        final = fits[-1]
        return {"center": final.center.tolist(), "yaw_deg": final.yaw_deg, "max_gap_mm": final.max_gap * 1000,
                "ambiguous": final.ambiguous(), "verified": verified,
                "initial": {"center": fits[0].center.tolist(), "yaw_deg": fits[0].yaw_deg,
                            "max_gap_mm": fits[0].max_gap * 1000},
                "verify_reports": reports, "contacts": [asdict(c) for c in self.contacts],
                "events": self.events, "seconds": time.time() - t0}
