"""Localize the plate and hole using the physical Franka.

A live window shows the box estimate while the run is going (contacts, force arrows,
where the box can be, provisional then final plate fit, hole presses and particles,
and the force trace). Without a display it still writes <run>/live.png every second.
Use --no-live to turn it off.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kilo.config import load_config
from kilo.localization.common import SEARCH_SPACE_CORNERS, SNAKE_SPACING, TABLETOP_HEIGHT
from kilo.localization.fit import point_gaps
from kilo.localization.hole import localize_hole
from kilo.localization.plate import localize_plate
from kilo.localization.search import generate_snake_waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-num", type=int, default=1)
    parser.add_argument("--table-z", type=float, default=TABLETOP_HEIGHT,
                        help="tabletop height in robot base metres (default: Lab 2 scene)")
    parser.add_argument("--execute", action="store_true", help="send motion commands to the arm")
    parser.add_argument("--out", type=Path, help="run directory for measurements and plot inputs")
    parser.add_argument("--no-live", action="store_true", help="do not show the live box-estimate view")
    parser.add_argument("--no-window", action="store_true",
                        help="live view as <run>/live.png only (no window), e.g. over SSH")
    args = parser.parse_args()

    waypoints = generate_snake_waypoints(SNAKE_SPACING)
    if not args.execute:
        print(f"Dry run. The sweep has {len(waypoints)} waypoints at table z={args.table_z:g} m.")
        return

    from frankapy import FrankaArm
    from kilo.robot.frankapy_localization import CONTACT_THRESHOLD, FORCE_LIMIT, FrankaLocalizer

    cfg = load_config()
    started = datetime.now()
    start = time.monotonic()
    out = args.out or Path(__file__).resolve().parents[1] / "runs" / "real" / f"{started:%Y%m%d-%H%M%S-%f}-localize"
    out.mkdir(parents=True, exist_ok=True)
    run = dict(backend="frankapy", started=started.isoformat(), contacts=[], probe_labels=[],
               config=dict(workspace=dict(search_corners=SEARCH_SPACE_CORNERS.tolist()),
                           contact=dict(force_threshold=CONTACT_THRESHOLD, safety_force=FORCE_LIMIT,
                                        directional_residual=True)))
    samples = {}
    trace = []
    view = None
    if not args.no_live:
        from kilo.localization.live_view import LiveView
        view = LiveView(out, SEARCH_SPACE_CORNERS, np.asarray(cfg["box"]["size"][:2]) / 2,
                        cfg["peg"]["shaft_radius"], thresholds=(CONTACT_THRESHOLD, FORCE_LIMIT),
                        window=not args.no_window)
        view.status("zeroing forces (keep the arm still)")
        print(f"Live view: window if a display is available, and {out / 'live.png'}", flush=True)

    def record_sample(now, tip, residual):
        trace.append((now - start, *tip, *residual))
        if view is not None:
            view.sample(now - start, tip, residual)

    def record_probe(label, action):
        trace.clear()
        try:
            return action()
        finally:
            if trace:
                samples[f"probe{len(samples):04d}"] = np.asarray(trace)
                run["probe_labels"].append(label)

    arm = FrankaArm(robot_num=args.robot_num, offline=False)
    robot = FrankaLocalizer(arm, args.table_z, record_sample=record_sample)

    def probe(xy):
        label = f"side-{len(run['contacts']) + 1}"
        contact = record_probe(label, lambda: robot.probe(xy))
        if contact is not None:
            tip_xy, force_xy = contact
            run["contacts"].append(dict(kind="side", label=label, xy=tip_xy.tolist(),
                                        force_xy=force_xy.tolist(), peak_force=float(np.linalg.norm(force_xy))))
            if view is not None:
                view.contact(label, tip_xy, force_xy)
                view.status(f"{len(run['contacts'])} contact(s); searching")
        return contact

    def press(xy, z, label):
        return record_probe(label, lambda: robot.press(xy, z))

    try:
        zero_log = out / "force-zero.csv"
        robot.zero_forces(zero_log)
        print(f"Raw force zero samples saved to {zero_log}", flush=True)
        if view is not None:
            view.status("snake sweep for the first contact")
        robot.position_at(waypoints[0])
        center, angle = localize_plate(waypoints, probe, robot.travel)
        if view is not None:
            view.plate(center, angle)
            view.status("plate found; hole search")
        gaps = point_gaps(center, np.radians(angle), [c["xy"] for c in run["contacts"]],
                          np.asarray(cfg["box"]["size"][:2]) / 2, cfg["peg"]["shaft_radius"])
        run["result"] = dict(center=center.tolist(), yaw_deg=float(angle),
                             max_gap_mm=float(1000 * np.max(np.abs(gaps))),
                             seconds=time.monotonic() - start)
        print(f"Estimated plate center ({center[0]:.4f}, {center[1]:.4f}) m, yaw {angle:.1f} deg (mod 180)",
              flush=True)
        run["hole"] = {}
        def hole_update(belief, record):
            if view is not None:
                view.hole(record.get("presses", []), belief.particles, belief.weights,
                          record.get("center"), record.get("std_m"), record.get("status"))

        belief = localize_hole(center, angle, args.table_z + cfg["box"]["size"][2], cfg, press, run["hole"],
                               on_update=hole_update)
        hole_update(belief, run["hole"])
        samples.update(hole_particles=belief.particles, hole_weights=belief.weights)
        if run["hole"]["status"] == "found":
            x, y = run["hole"]["center"]
            print(f"Estimated hole center ({x:.4f}, {y:.4f}) m", flush=True)
        else:
            print(f"Hole search ended with status {run['hole']['status']}", flush=True)
    except BaseException as failure:
        run["error"] = str(failure) or type(failure).__name__
        try:
            arm.stop_skill()
        except Exception as error:
            print(f"Arm stop request failed: {error}", file=sys.stderr)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(out / "samples.npz", **samples)
        print(f"Localization measurements saved to {out}", flush=True)
        if view is not None:
            view.status("finished: " + (run.get("error") or "ok"))
            view.close(keep_open=True)
            print("Final live view saved to " + str(out / "final.png") + " (close the window to exit)", flush=True)


if __name__ == "__main__":
    main()
