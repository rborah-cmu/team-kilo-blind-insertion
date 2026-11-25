"""Phase 1 experiments on the real Franka (or the MuJoCo stand-in with --backend sim).

Run in this order the first time (see docs/real_robot.md):
    check        read state, measure loop rate and force noise at rest (no motion)
    home         joint move to frankapy home (checked against the shelf model first)
    move         reposition the peg tip to --start X Y Z (lifts first)
    grasp        open, place the peg by hand, close on its square top
    table        guarded descent onto the table: measures table_z and the force sign
    noise        force noise at rest and while moving in free air -> contact threshold
    touch        repeated guarded probes at one box face -> contact repeatability
    groundtruth  hand-guide the peg into the hole (and/or onto box corners) -> truth file
    localize     full Phase 1 search + fit (+ verification probes), scored against truth
    refit        refit a logged localize run offline with other force weights

Real-robot motion needs --execute and typing RUN. Every run logs to runs/real/<time>-<command>/.

Examples (from the repo root, inside the course docker on the User PC):
    python3 scripts/real/phase1.py check
    python3 scripts/real/phase1.py table --execute
    python3 scripts/real/phase1.py touch --direction 1 0 --distance 0.08 --repeats 5 --execute
    python3 scripts/real/phase1.py localize --truth runs/real/groundtruth.json --execute
    python scripts/real/phase1.py localize --backend sim --seed 8          # offline rehearsal
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kilo.config import load_config  # noqa: E402
from kilo.localization.fit import fit_box, pose_distance  # noqa: E402
from kilo.robot import ProbeSettings, SafetyStop, make_backend  # noqa: E402

MAX_PROBE_DISTANCE = 0.15  # m, for the sideways single-probe experiments
MAX_TABLE_DESCENT = 0.60   # m, the guarded table descent may start from near the home pose


# ---------------------------------------------------------------------------- utilities
class RunLog:
    def __init__(self, command: str, root: Path):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"{stamp}-{command}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data: dict = {"command": command, "started": stamp}
        self.arrays: dict = {}

    def save(self):
        (self.dir / "result.json").write_text(json.dumps(self.data, indent=1, default=_jsonable))
        if self.arrays:
            np.savez_compressed(self.dir / "samples.npz", **self.arrays)
        print(f"\nlogged to {self.dir}")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    # Never lose a run log to one odd value (2026-11-24: a numpy.bool_ in the executor
    # diagnostics aborted the save and dropped the return-move failure details).
    return repr(o)


def confirm(args, what: str) -> None:
    if args.backend == "sim":
        return
    if not args.execute:
        print(f"Dry run: would {what}. Add --execute to move the robot.")
        raise SystemExit(0)
    print("Keep a partner's hand on the e-stop. Stay outside the table area.")
    if input(f"About to {what}. Type RUN to continue: ").strip() != "RUN":
        print("Cancelled.")
        raise SystemExit(1)


def probe_settings(cfg: dict, robot, speed_key: str = "search_speed") -> ProbeSettings:
    mc, cc = cfg["motion"], cfg["contact"]
    return ProbeSettings(speed=float(mc[speed_key]), force_threshold=float(cc["force_threshold"]),
                         debounce=int(cc["debounce"]), bias_seconds=float(cc["bias_seconds"]),
                         safety_force=float(cc["safety_force"]), retreat=float(cc["retreat"]),
                         stiffness=float(getattr(robot, "stiffness", mc["stream_stiffness"][0])),
                         stream_hz=float(getattr(robot, "preferred_hz", mc["stream_hz"])),
                         timeout=float(cc.get("probe_timeout", 90.0)))


def state_summary(st) -> dict:
    return {"tip": np.round(st.tip, 4).tolist(), "yaw_deg": round(float(np.degrees(st.yaw)), 1),
            "force": np.round(st.force, 2).tolist(), "torque": np.round(st.torque, 3).tolist(),
            "q": None if st.q is None else np.round(st.q, 3).tolist()}


def read_forces(robot, seconds: float, hz: float):
    t0, rows = time.monotonic(), []
    while time.monotonic() - t0 < seconds:
        st = robot.state()
        rows.append([st.time, *st.tip, *st.force])
        robot.tick(1.0 / hz)
    return np.asarray(rows)


def force_stats(rows: np.ndarray) -> dict:
    f = rows[:, 4:7]
    resid = f - f.mean(axis=0)
    span = rows[-1, 0] - rows[0, 0]
    return {"samples": len(rows), "rate_hz": round((len(rows) - 1) / span, 1) if span > 0 else None,
            "mean_n": np.round(f.mean(axis=0), 3).tolist(), "std_n": np.round(f.std(axis=0), 3).tolist(),
            "p99_abs_dev_n": round(float(np.percentile(np.linalg.norm(resid, axis=1), 99)), 3),
            "max_abs_dev_n": round(float(np.linalg.norm(resid, axis=1).max()), 3)}


# ---------------------------------------------------------------------------- experiments
def cmd_check(args, cfg, robot, log):
    st = robot.state()
    print("state:", json.dumps(state_summary(st), indent=1))
    rows = read_forces(robot, args.seconds, getattr(robot, "preferred_hz", 50.0))
    stats = force_stats(rows)
    print("force at rest:", json.dumps(stats, indent=1))
    print(f"\nState loop ran at {stats['rate_hz']} Hz (target {getattr(robot, 'preferred_hz', 50)}).")
    log.data.update(state=state_summary(st), rest=stats)
    log.arrays["rest"] = rows


def cmd_home(args, cfg, robot, log):
    """Joint move to frankapy's home pose; refuses if the straight joint path hits the shelf model."""
    fa = getattr(robot, "fa", None)
    if fa is None:
        raise SystemExit("home needs the frankapy backend")
    from frankapy import FrankaConstants as FC
    home = np.asarray(FC.HOME_JOINTS, float)
    st = robot.state()
    # Fail closed: always check the joint path against the shelf model before moving.
    geometry = getattr(robot, "geometry", None)
    if geometry is None:
        from kilo.robot.planning import FrankaGeometry
        model = (cfg.get("joint_executor") or {}).get("model", "configs/frankapy_geometry.json")
        try:
            geometry = FrankaGeometry(ROOT / model, cfg)
        except Exception as exc:
            raise SafetyStop(f"cannot load the shelf/planning model ({exc}); refusing to move home")
    try:
        samples, _ = geometry.check_joint_edge(np.asarray(st.q, float), home, min_clearance=.002)
    except Exception as exc:
        raise SafetyStop(f"the joint path to home fails the shelf check ({exc}); move the arm by hand first")
    print(f"joint path to home checked: {len(samples)} samples clear the shelf model")
    print(f"largest joint change {np.degrees(np.max(np.abs(home - st.q))):.1f} deg")
    confirm(args, "move the arm to frankapy home (one 8 s joint move)")
    fa.goto_joints(home.tolist(), duration=8, ignore_errors=False, ignore_virtual_walls=False, skill_desc="Kilo home")
    end = robot.state()
    print(f"at home: max joint error {np.max(np.abs(end.q - home)) * 1000:.2f} mrad, tip {np.round(end.tip, 4)}")
    log.data.update(start=state_summary(st), end=state_summary(end))


def cmd_grasp(args, cfg, robot, log):
    """--stage both (default): open, wait for Enter, close. Over SSH use --stage open, then --stage close."""
    g = cfg["gripper"]
    if args.stage in ("both", "open"):
        confirm(args, "open the gripper fully")
        robot.gripper(0.08)
    if args.stage == "both":
        input("Place the peg's square top between the fingers (shaft pointing down), then press Enter: ")
    if args.stage in ("both", "close"):
        if args.stage == "close":
            confirm(args, f"close the gripper on the peg with {g['grasp_force']} N")
        robot.gripper(float(g["peg_width"]), grasp=True, force=float(g["grasp_force"]))
    width = getattr(getattr(robot, "fa", None), "get_gripper_width", lambda: None)()
    if width is not None:
        print(f"gripper width now {width * 1000:.1f} mm (peg top is {g['peg_width'] * 1000:.1f} mm)")
    log.data.update(stage=args.stage, gripper_width=width, state=state_summary(robot.state()))


def cmd_table(args, cfg, robot, log):
    st = robot.state()
    floor = float(cfg["workspace"].get("descent_floor_z", -0.03))
    target = st.tip - np.array([0, 0, min(args.distance, MAX_TABLE_DESCENT)])
    target[2] = max(target[2], floor)  # frankapy refuses targets below its virtual floor wall (z = -0.05)
    confirm(args, f"lower the peg tip up to {args.distance * 100:.0f} cm straight down until it touches the table")
    s = probe_settings(cfg, robot, "descent_speed")
    res = robot.probe(target, st.yaw, s)
    log.arrays["probe"] = np.asarray(res.samples)
    if not res.contact:
        travelled = res.distance * 1000
        if res.stalled:
            print(f"No contact: the arm STALLED after {travelled:.0f} mm without touching anything "
                  "(it stopped following the targets). See samples.npz.")
        else:
            print(f"No contact after {travelled:.0f} mm: the table is further than --distance below the tip.")
        log.data.update(contact=False, stalled=res.stalled, travelled_mm=travelled, end_tip=res.tip)
        return
    fz = float(res.force[2])
    sign_ok = fz > 0
    table_z = float(res.tip[2])
    print(f"\nTable touched at tip z = {table_z:.4f} m (force {np.round(res.force, 2)} N, peak {res.peak_force:.1f} N).")
    print(f"Set workspace.table_z: {table_z:.4f} in configs/real.yaml.")
    if sign_ok:
        print("Force sign OK: the table pushes the peg up (+z). Keep robot.force_sign as is.")
    else:
        print(f"Force sign is FLIPPED (table force reads {fz:.1f} N in z): set robot.force_sign: "
              f"{-float(cfg['robot'].get('force_sign', 1.0)):.0f}")
    print("Note: an error in robot.probe_length shifts every height equally, so it cancels out "
          "as long as table_z is measured with the same value.")
    log.data.update(contact=True, table_z=table_z, force=res.force, peak=res.peak_force, sign_ok=sign_ok)
    log.arrays["probe"] = np.asarray(res.samples)


def cmd_noise(args, cfg, robot, log):
    hz = getattr(robot, "preferred_hz", 50.0)
    st = robot.state()
    d = np.r_[np.asarray(args.direction, float), 0.0]
    d /= np.linalg.norm(d)
    distance = min(args.distance, MAX_PROBE_DISTANCE)
    confirm(args, f"hold still {args.seconds:.0f} s, then move {distance * 100:.0f} cm along {d[:2]} in free air "
                  "(make sure nothing is in the way)")
    rest = read_forces(robot, args.seconds, hz)
    moving = robot.move_and_record(st.tip + d * distance, st.yaw, float(cfg["motion"]["search_speed"]), hz)
    moved = float(np.linalg.norm(moving[-1, 1:4] - st.tip)) if len(moving) else 0.0
    print(f"moved {moved * 1000:.1f} mm of the requested {distance * 1000:.0f} mm")
    rs, ms = force_stats(rest), force_stats(moving)
    # moving deviation measured against the at-rest mean (what the probe's bias removal does)
    dev = np.linalg.norm(moving[:, 4:7] - rest[:, 4:7].mean(axis=0), axis=1)
    ms["p99_dev_from_rest_bias_n"] = round(float(np.percentile(dev, 99)), 3)
    ms["max_dev_from_rest_bias_n"] = round(float(dev.max()), 3)
    suggested = max(3.0, round(1.5 * ms["max_dev_from_rest_bias_n"] + 1.0, 1))
    print("at rest:", json.dumps(rs, indent=1))
    print("moving in free air:", json.dumps(ms, indent=1))
    print(f"\nSuggested contact.force_threshold: {suggested} N (1.5x the largest free-air deviation + 1 N; "
          f"current {cfg['contact']['force_threshold']} N).")
    log.data.update(rest=rs, moving=ms, suggested_threshold=suggested)
    log.arrays.update(rest=rest, moving=moving)


def cmd_touch(args, cfg, robot, log):
    st0 = robot.state()
    d = np.r_[np.asarray(args.direction, float), 0.0]
    d /= np.linalg.norm(d)
    distance = min(args.distance, MAX_PROBE_DISTANCE)
    if hasattr(robot, "required_wrist_yaw"):
        yaw = robot.required_wrist_yaw
    elif cfg["motion"].get("align_wrist", False):
        from kilo.robot.interface import choose_wrist_yaw
        yaw, _ = choose_wrist_yaw(d[:2], st0, st0.tip[:2] + d[:2] * distance)
    else:
        yaw = st0.yaw  # same as localize's default: one wrist angle throughout
    confirm(args, f"probe {args.repeats}x up to {distance * 100:.0f} cm along {np.round(d[:2], 2)} at the current height, "
                  "returning to the start each time")
    s = probe_settings(cfg, robot)
    travel = float(cfg["motion"]["travel_speed"])
    robot.move(st0.tip, yaw, travel)
    results = []
    log.data.update(direction=d, distance=distance, probes=[])
    for k in range(args.repeats):
        res = robot.probe(st0.tip + d * distance, yaw, s)
        results.append(res)
        print(f"  probe {k + 1}: {'contact' if res.contact else 'no contact'} at {np.round(res.tip, 4)}, "
              f"travel {res.distance * 1000:.1f} mm, force {np.round(res.force, 1)} N, peak {res.peak_force:.1f} N")
        log.arrays[f"probe{k}"] = np.asarray(res.samples)
        log.data["probes"].append({"contact": res.contact, "tip": res.tip, "force": res.force,
                                   "peak": res.peak_force, "distance": res.distance, "stalled": res.stalled})
        log.save()  # preserve the completed probe even if the return move fails
        robot.move(st0.tip, yaw, travel)
    hits = [r for r in results if r.contact]
    summary = {"repeats": args.repeats, "contacts": len(hits)}
    if hits:
        along = np.array([r.distance for r in hits]) * 1000
        pts = np.array([r.tip for r in hits])
        summary.update(contact_distance_mm_mean=round(float(along.mean()), 2),
                       contact_distance_mm_std=round(float(along.std()), 3),
                       contact_point_spread_mm=round(float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() * 1000), 3),
                       peak_force_n_max=round(max(r.peak_force for r in hits), 2),
                       force_direction_deg=[round(float(np.degrees(np.arctan2(r.force[1], r.force[0]))), 1) for r in hits])
    print(json.dumps(summary, indent=1))
    log.data.update(direction=d, distance=distance, summary=summary,
                    probes=[{"contact": r.contact, "tip": r.tip, "force": r.force, "peak": r.peak_force,
                             "distance": r.distance, "stalled": r.stalled} for r in results])


def cmd_groundtruth(args, cfg, robot, log):
    scene = load_config()
    if args.backend == "sim":
        c, y = robot.true_box()
        truth = {"box_center": c.tolist(), "box_yaw_deg": float(np.degrees(y) % 180), "hole": robot.true_hole().tolist(),
                 "source": "simulator"}
    else:
        print("Guide mode: the arm will be compliant. Move it only by the gripper, gently.")
        confirm(args, "enter guide mode so you can position the peg by hand")
        robot.guide_mode(args.seconds)
        truth = {"source": "hand-guided"}
        input("Guide the peg into the hole until the tip drops in and the shaft is vertical. Press Enter: ")
        truth["hole"] = robot.state().tip[:2].tolist()
        print(f"  hole at {np.round(truth['hole'], 4)}")
        if args.corners:
            corners = []
            for name in ("A", "B (next corner along the 126.7 mm edge)", "C (next corner along the 169 mm edge)"):
                input(f"Hold the peg tip centred over top corner {name} of the box. Press Enter: ")
                corners.append(robot.state().tip[:2])
            A, B, C = map(np.asarray, corners)
            truth["box_center"] = ((A + C) / 2).tolist()
            truth["box_yaw_deg"] = float(np.degrees(np.arctan2(*(B - A)[::-1])) % 180)
            truth["edge_lengths_mm"] = [float(np.linalg.norm(B - A) * 1000), float(np.linalg.norm(C - B) * 1000)]
            print(f"  box centre {np.round(truth['box_center'], 4)}, yaw {truth['box_yaw_deg']:.1f} deg, "
                  f"edges {np.round(truth['edge_lengths_mm'], 1)} mm (expected 126.7, 169.0)")
    hole_offset = np.asarray(scene["hole"]["cad_offset"])
    truth["note"] = f"CAD hole offset from box centre: {np.round(hole_offset * 1000, 2).tolist()} mm (box frame)"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(truth, indent=1))
    print(f"truth saved to {out}")
    log.data.update(truth=truth)


def cmd_localize(args, cfg, robot, log):
    from kilo.localization.touch_localizer import TouchLocalizer
    scene = load_config()
    if args.backend == "sim":
        cfg["workspace"]["table_z"] = robot.table_z()
    rounds = int(cfg["fit"]["verify_rounds"] if args.verify_rounds is None else args.verify_rounds)
    confirm(args, f"run the full Phase 1 search ({rounds} verification round(s)) over "
                  f"{cfg['workspace']['search_corners']}")
    loc = TouchLocalizer(robot, cfg, scene)
    try:
        res = loc.run(verify_rounds=rounds)
    finally:
        for i, p in enumerate(loc.probe_logs):
            log.arrays[f"probe{i:02d}"] = p["samples"]
        log.data.update(contacts=[c.__dict__ for c in loc.contacts], events=loc.events,
                        probe_labels=[p["label"] for p in loc.probe_logs])
    print(f"\nestimate: box centre {np.round(res['center'], 4)}, yaw {res['yaw_deg']:.1f} deg (mod 180), "
          f"worst contact gap {res['max_gap_mm']:.2f} mm, {res['seconds']:.0f} s")
    if not res["verified"]:
        print("Verification incomplete: treat this estimate as provisional.")
    truth = None
    if args.backend == "sim":
        c, y = robot.true_box()
        truth = {"box_center": c.tolist(), "box_yaw_deg": float(np.degrees(y) % 180), "hole": robot.true_hole().tolist()}
    elif args.truth:
        truth = json.loads(Path(args.truth).read_text())
    if truth:
        score = {}
        for key, est in (("initial", res["initial"]), ("final", res)):
            if "box_center" in truth:
                mm, deg = pose_distance(est["center"], np.radians(est["yaw_deg"]), truth["box_center"],
                                        np.radians(truth["box_yaw_deg"]))
                score[f"{key}_box_error_mm"], score[f"{key}_yaw_error_deg"] = round(mm * 1000, 2), round(deg, 2)
            if "hole" in truth:
                score[f"{key}_hole_estimate_error_mm"] = round(float(np.linalg.norm(np.asarray(est["center"]) - truth["hole"]) * 1000), 2)
        print("score vs truth:", json.dumps(score, indent=1))
        res["score"], res["truth"] = score, truth
    log.data.update(result=res)


def cmd_refit(args, cfg, robot, log):
    run = json.loads(Path(args.run, "result.json").read_text())
    scene = load_config()
    side = [c for c in run["contacts"] if c["kind"] == "side"]
    half = np.asarray(scene["box"]["size"][:2]) / 2
    radius = scene["peg"]["shaft_radius"]
    truth = run.get("result", {}).get("truth")
    rows = []
    for subset, contacts in (("first 4", side[:4]), ("all", side)):
        for w in args.weights:
            f = fit_box([c["xy"] for c in contacts], [c["force_xy"] for c in contacts], half, radius, w)
            row = {"contacts": subset, "weight": w, "center": np.round(f.center, 4).tolist(), "yaw_deg": round(f.yaw_deg, 2),
                   "max_gap_mm": round(f.max_gap * 1000, 3), "ambiguous": f.ambiguous()}
            if truth and "box_center" in truth:
                mm, deg = pose_distance(f.center, f.yaw, truth["box_center"], np.radians(truth["box_yaw_deg"]))
                row.update(error_mm=round(mm * 1000, 2), yaw_error_deg=round(deg, 2))
            rows.append(row)
            print(row)
    log.data.update(source=args.run, refits=rows)


def cmd_move(args, cfg, robot, log):
    """Free-space reposition to --start X Y Z (lift first, then across, then a guarded descent)."""
    if args.start is None:
        raise SystemExit("move needs --start X Y Z")
    st = robot.state()
    print(f"peg tip now at {np.round(st.tip, 4)}")
    log.data.update(end=state_summary(st))


COMMANDS = {"check": cmd_check, "home": cmd_home, "move": cmd_move, "grasp": cmd_grasp, "table": cmd_table, "noise": cmd_noise, "touch": cmd_touch,
            "groundtruth": cmd_groundtruth, "localize": cmd_localize, "refit": cmd_refit}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default=str(ROOT / "configs" / "real.yaml"))
    ap.add_argument("--backend", choices=["frankapy", "pandapy", "sim"], default=None)
    ap.add_argument("--robot-num", type=int, default=None)
    ap.add_argument("--execute", action="store_true", help="allow real-robot motion")
    ap.add_argument("--seed", type=int, default=0, help="sim backend: box pose seed")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--distance", type=float, default=0.10)
    ap.add_argument("--direction", type=float, nargs=2, default=(1.0, 0.0))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--stage", choices=["both", "open", "close"], default="both", help="grasp: which part to run")
    ap.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="table/noise/touch: first move the peg tip here (free-space move)")
    ap.add_argument("--verify-rounds", type=int, default=None)
    ap.add_argument("--truth", default=None, help="truth JSON from the groundtruth command")
    ap.add_argument("--corners", action="store_true", help="groundtruth: also record three box corners")
    ap.add_argument("--out", default=str(ROOT / "runs" / "real" / "groundtruth.json"))
    ap.add_argument("--run", default=None, help="refit: a runs/real/<...>-localize folder")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.004, 0.001, 0.0])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    args.backend = args.backend or cfg["robot"]["backend"]
    log = RunLog(args.command, ROOT / "runs" / "real")
    log.data.update(backend=args.backend, config=cfg, args=vars(args))
    if args.command == "refit":
        if not args.run:
            ap.error("refit needs --run")
        cmd_refit(args, cfg, None, log)
        log.save()
        return
    if args.backend != "sim" and args.command not in ("check",) and not args.execute:
        print(f"Dry run of `{args.command}` on {args.backend}: no connection made, nothing moved. "
              "Add --execute (on the lab PC, inside the course docker) to run it.")
        return
    kwargs = {"seed": args.seed, "force_noise": 0.3, "force_bias": 2.0} if args.backend == "sim" else \
             ({"robot_num": args.robot_num} if args.backend == "frankapy" else {})
    try:
        robot = make_backend(args.backend, cfg, **kwargs)
    except ImportError as exc:
        print(f"Cannot load the {args.backend} backend here ({exc}). Run this on the lab User PC inside the "
              "course docker (frankapy), or use --backend sim to rehearse offline.")
        raise SystemExit(2)
    try:
        if args.start is not None and args.command in ("table", "noise", "touch", "move"):
            start = np.asarray(args.start, float)
            confirm(args, f"move the peg tip over {np.round(start[:2], 3)}, then lower it (force-guarded) to z={start[2]:.3f}")
            st = robot.state()
            yaw = getattr(robot, "required_wrist_yaw", st.yaw)
            high = max(st.tip[2], start[2])
            robot.move(np.r_[st.tip[:2], high], yaw, float(cfg["motion"]["travel_speed"]))
            robot.move(np.r_[start[:2], high], yaw, float(cfg["motion"]["travel_speed"]))
            if start[2] < high - 1e-4:
                res = robot.probe(start, yaw, probe_settings(cfg, robot, "descent_speed"))
                if res.contact:
                    raise SafetyStop(f"touched something while lowering to the start point at z={res.tip[2]:.3f}")
        COMMANDS[args.command](args, cfg, robot, log)
    except SafetyStop as exc:
        print(f"\nSAFETY STOP: {exc}")
        log.data["safety_stop"] = str(exc)
        aborted = getattr(robot, "last_probe_samples", None)
        if aborted is not None:
            log.arrays["last_probe"] = np.asarray(aborted)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        log.data["interrupted"] = True
    finally:
        try:
            if hasattr(robot, "path_audit"):
                log.data["joint_path_audit"] = robot.path_audit
            robot.close()
        finally:
            log.save()


if __name__ == "__main__":
    main()
