"""Insert the peg with the scripted spiral (main's simulation baseline, adapted for the real arm).

Hole estimate, in order of preference:
  --hole X Y            explicit estimate (m, robot base)
  --from-run RUN_DIR    a localize_live.py run: the deepest hole press (>= 0.5 mm, i.e. the
                        chamfer was felt) if any, else the CAD hole offset from the plate fit
                        (both orientations, since plate yaw is known modulo 180 deg)
  --localize            run the plate localization now (snake + four contacts), then CAD offset
Then: approach above the estimate with joint moves, and run ONE frankapy Cartesian-impedance
skill that spirals under a light downward preload, captures the hole, and pushes the peg in
(see kilo/insertion/real_spiral.py). A live window shows the spiral, tip path and forces.
Once seated (default): open the gripper, lift straight up clear of the peg, go home.
Use --keep-grip to keep holding it, --no-home to stay after lifting.

    python3 scripts/insert_live.py --from-run runs/real/<stamp>-localize --table-z 0.0072
    python3 scripts/insert_live.py --from-run runs/real/<stamp>-localize --table-z 0.0072 --execute
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kilo.config import load_config  # noqa: E402
from kilo.insertion.spiral import spiral_waypoints  # noqa: E402
from kilo.localization.common import SEARCH_SPACE_CORNERS, SNAKE_SPACING, TABLETOP_HEIGHT  # noqa: E402


def cad_candidates(center, yaw_deg, cfg):
    yaw = np.radians(yaw_deg)
    axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    off = axes @ np.asarray(cfg["hole"]["cad_offset"], float)
    return [np.asarray(center) + off, np.asarray(center) - off]


def inserted_locations(runs_root, near, radius=0.015):
    """Tips where earlier insertion attempts went into the hole (reached INSERT/SEATED),
    newest first, within ``radius`` of ``near`` (i.e. the same box placement)."""
    found = []
    for run in sorted(Path(runs_root).glob("*-insert"), reverse=True):
        try:
            data = json.loads((run / "result.json").read_text())
        except Exception:
            continue
        for att in data.get("attempts") or []:
            for ph in att.get("phases") or []:
                if ph.get("phase") in ("insert", "seated") and ph.get("tip"):
                    xy = np.asarray(ph["tip"][:2], float)
                    if np.linalg.norm(xy - np.asarray(near)) <= radius:
                        found.append((run.name, xy))
                    break
    return found


def estimate_from_run(run_dir, cfg, table_z):
    data = json.loads((Path(run_dir) / "result.json").read_text())
    res = data.get("result") or {}
    if "center" not in res:
        raise SystemExit(f"{run_dir} has no plate estimate (result.center)")
    hole = data.get("hole") or {}
    top = hole.get("surface_z_m") or (table_z + cfg["box"]["size"][2])
    candidates, notes = [], []
    presses = [p for p in hole.get("presses") or [] if p.get("depth_m", 0) >= 0.0005]
    if presses:
        best = max(presses, key=lambda p: p["depth_m"])
        candidates.append(np.asarray(best["tip"][:2]))
        notes.append(f"deepest hole press {best['depth_m'] * 1000:.1f} mm at {np.round(best['tip'][:2], 4).tolist()}")
    for c in cad_candidates(res["center"], res["yaw_deg"], cfg):
        if all(np.linalg.norm(c - k) > 0.003 for k in candidates):
            candidates.append(c)
            notes.append(f"CAD offset candidate {np.round(c, 4).tolist()}")
    return dict(plate=(res["center"], res["yaw_deg"]), contacts=data.get("contacts") or [],
                top_z=float(top), candidates=candidates, notes=notes)


def home_arm(arm, view=None, joint_speed=0.4):
    """Joint move to frankapy home from a high pose; duration scales with the largest joint change."""
    try:
        from frankapy import FrankaConstants as FC
        home = np.asarray(FC.HOME_JOINTS, float)
    except ImportError:  # same values as frankapy's HOME_JOINTS
        home = np.array([0., -np.pi / 4, 0., -3 * np.pi / 4, 0., np.pi / 2, np.pi / 4])
    delta = float(np.max(np.abs(home - np.asarray(arm.get_joints(), float))))
    duration = max(6.0, 1.875 * delta / joint_speed)
    if view:
        view.status("going home")
    print(f"Going home ({np.degrees(delta):.0f} deg largest joint change, {duration:.1f} s)", flush=True)
    arm.goto_joints(home.tolist(), duration=duration, ignore_errors=False, ignore_virtual_walls=False,
                    skill_desc="Kilo home after insertion")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-num", type=int, default=1)
    ap.add_argument("--table-z", type=float, default=TABLETOP_HEIGHT)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hole", type=float, nargs=2, metavar=("X", "Y"))
    src.add_argument("--from-run", type=Path)
    src.add_argument("--localize", action="store_true")
    ap.add_argument("--top-z", type=float, default=None, help="plate top height (default: measured or table+box)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "real_insertion.yaml")
    ap.add_argument("--retract", action="store_true",
                    help="(default after seating) kept for compatibility; a failed run always lifts clear")
    ap.add_argument("--keep-grip", action="store_true",
                    help="after seating, keep holding the peg (no release, no lift, no home)")
    ap.add_argument("--no-home", action="store_true", help="after release and lift, stay instead of homing")
    ap.add_argument("--no-memory", action="store_true",
                    help="do not try where an earlier insertion attempt went into the hole first")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--execute", action="store_true", help="move the arm")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = load_config()
    icfg = yaml.safe_load(args.config.read_text())["insertion_real"]
    plan = dict(plate=None, contacts=[], top_z=args.top_z or args.table_z + cfg["box"]["size"][2],
                candidates=[], notes=[])
    if args.hole:
        plan.update(candidates=[np.asarray(args.hole)], notes=[f"explicit estimate {args.hole}"])
    elif args.from_run:
        plan.update(estimate_from_run(args.from_run, cfg, args.table_z))
        if args.top_z:
            plan["top_z"] = args.top_z
    if not args.no_memory and plan["candidates"]:
        hits = inserted_locations(ROOT / "runs" / "real", plan["candidates"][0])
        if hits:
            name, xy = hits[0]
            plan["candidates"] = [xy] + [c for c in plan["candidates"] if np.linalg.norm(c - xy) > 0.003]
            plan["notes"] = [f"where the peg went in on {name}: {np.round(xy, 4).tolist()}"] + plan["notes"]
    print("Insertion plan:", flush=True)
    for n in plan["notes"] or ["plate localization first, then the CAD hole offset"]:
        print("  candidate:", n, flush=True)
    print(f"  plate top z {plan['top_z']:.4f} m; spiral radius {icfg['spiral_max_radius'] * 1000:.0f} mm, "
          f"pitch {icfg['spiral_pitch'] * 1000:.1f} mm, {icfg['search_speed'] * 1000:.0f} mm/s; "
          f"preload ~{icfg['z_stiffness'] * icfg['search_z_lead']:.1f} N", flush=True)
    if not args.execute:
        print("Dry run: nothing moved. Add --execute to run on the arm.")
        return

    from frankapy import FrankaArm
    from kilo.robot.frankapy_insertion import FrankaSpiralInserter, InsertionStop, move_keep_wrist
    from kilo.robot.frankapy_localization import CONTACT_THRESHOLD, FORCE_LIMIT, FrankaLocalizer

    started = datetime.now()
    out = args.out or ROOT / "runs" / "real" / f"{started:%Y%m%d-%H%M%S-%f}-insert"
    out.mkdir(parents=True, exist_ok=True)
    run = dict(started=started.isoformat(), source=str(args.from_run or args.hole or "localize"),
               config=dict(insertion=icfg, contact=dict(force_threshold=CONTACT_THRESHOLD, safety_force=FORCE_LIMIT)),
               plan=dict(top_z=plan["top_z"], candidates=[c.tolist() for c in plan["candidates"]], notes=plan["notes"]),
               attempts=[], contacts=list(plan["contacts"]))
    rows = []
    start = time.monotonic()
    view = None
    if not args.no_live:
        from kilo.localization.live_view import LiveView
        view = LiveView(out, SEARCH_SPACE_CORNERS, np.asarray(cfg["box"]["size"][:2]) / 2, cfg["peg"]["shaft_radius"],
                        thresholds=(icfg["touch_force"], icfg["force_limit"]), window=not args.no_window)
        for c in plan["contacts"]:
            view.contact(c["label"], c["xy"], c["force_xy"])
        if plan["plate"]:
            view.plate(*plan["plate"])
        view.status("zeroing forces (keep the arm still)")

    arm = FrankaArm(robot_num=args.robot_num, offline=False)
    robot = FrankaLocalizer(arm, args.table_z,
                            record_sample=(lambda now, tip, f: view.sample(now - start, tip, f)) if view else None)

    def tip_now():
        return robot._pose(robot._read_joints())[0]

    try:
        robot.zero_forces(out / "force-zero.csv")
        if args.localize:
            from kilo.localization.plate import localize_plate
            from kilo.localization.search import generate_snake_waypoints

            def probe(xy):
                hit = robot.probe(xy)
                if hit is not None:
                    c = dict(kind="side", label=f"side-{len(run['contacts']) + 1}", xy=hit[0].tolist(),
                             force_xy=hit[1].tolist())
                    run["contacts"].append(c)
                    if view:
                        view.contact(c["label"], c["xy"], c["force_xy"])
                return hit
            waypoints = generate_snake_waypoints(SNAKE_SPACING)
            if view:
                view.status("plate localization")
            robot.position_at(waypoints[0])
            center, yaw = localize_plate(waypoints, probe, robot.travel)
            plan["plate"] = (center.tolist(), float(yaw))
            plan["candidates"] = cad_candidates(center, yaw, cfg)
            run["plate"] = dict(center=center.tolist(), yaw_deg=float(yaw))
            if view:
                view.plate(center, yaw)
            print(f"Plate ({center[0]:.4f}, {center[1]:.4f}) m, yaw {yaw:.1f} deg", flush=True)

        result = None
        n_attempts = min(3, len(plan["candidates"]))
        for i, xy in enumerate(plan["candidates"][:n_attempts]):
            print(f"\nAttempt {i + 1}: hole estimate {np.round(xy, 4).tolist()}", flush=True)
            if view:
                view.status(f"approach attempt {i + 1}")
                view.path(spiral_waypoints(xy, icfg["spiral_pitch"], icfg["spiral_max_radius"], 0.0005),
                          zoom=0.03)
            # Fixed-wrist moves: the localizer's _move turns the tool to the move direction,
            # which ran joint 7 past its limit on this approach.
            move_keep_wrist(robot, robot.travel_z, vertical=True)
            move_keep_wrist(robot, np.r_[xy, robot.travel_z])
            move_keep_wrist(robot, plan["top_z"] + icfg["approach_height"], vertical=True)
            # Zero the residual at the hover point, peg in free air.
            readings = []
            for _ in range(25):
                readings.append(robot._probe_force())
                time.sleep(0.02)
            baseline = np.mean(readings, axis=0)

            def record(t, tip, target, force, phase, _i=i):
                rows.append([_i, t, *tip, *target, *force, ["approach", "search", "capture", "insert", "seated"].index(phase)])
            inserter = FrankaSpiralInserter(arm, icfg, tip_fn=tip_now, force_fn=lambda: robot._probe_force() - baseline,
                                            view=view, record=record)
            try:
                result = inserter.run(xy, plan["top_z"])
                run["attempts"].append(result)
                break
            except InsertionStop as stop:
                run["attempts"].append(getattr(inserter, "last_result", dict(status="stopped", reason=str(stop))))
                print(f"Attempt {i + 1} stopped: {stop}", flush=True)
                retryable = "no insertion within" in str(stop) and i + 1 < n_attempts
                move_keep_wrist(robot, robot.travel_z, vertical=True)
                if not retryable:
                    raise
        run["result"] = result
        if result and result["status"] == "seated":
            print(f"\nINSERTED: {result['depth_m'] * 1000:.1f} mm deep, {result['seconds']:.1f} s", flush=True)
            if view:
                view.status(f"INSERTED {result['depth_m'] * 1000:.1f} mm deep")
        if result and result["status"] == "seated" and not args.keep_grip:
            # Default ending (user, 2026-11-25): let go of the peg, lift clear, go home.
            if view:
                view.status("releasing the peg")
            print("Releasing the peg", flush=True)
            arm.open_gripper()
            time.sleep(1.0)
            run["released"] = True
            move_keep_wrist(robot, robot.travel_z, vertical=True)
            print("Lifted clear of the peg", flush=True)
            if not args.no_home:
                home_arm(arm, view)
                run["homed"] = True
        elif not (result and result["status"] == "seated"):
            move_keep_wrist(robot, robot.travel_z, vertical=True)
    except BaseException as failure:
        run["error"] = str(failure) or type(failure).__name__
        if view:
            view.status("stopped: " + run["error"])
        try:
            arm.stop_skill()
        except Exception:
            pass
        raise
    finally:
        (out / "result.json").write_text(json.dumps(run, indent=2, default=lambda o: o.tolist()
                                                    if hasattr(o, "tolist") else str(o)) + "\n")
        if rows:
            np.savez_compressed(out / "insertion.npz", rows=np.asarray(rows),
                                columns=np.array(["attempt", "t", "tip_x", "tip_y", "tip_z", "target_x", "target_y",
                                                  "target_z", "fx", "fy", "fz", "phase"]))
        print(f"Insertion log saved to {out}", flush=True)
        if view:
            view.close(keep_open=True)


if __name__ == "__main__":
    main()
