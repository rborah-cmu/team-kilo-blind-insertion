"""Plan complete Phase 1 sequences offline. This script has NO robot execution option."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kilo.config import load_config
from kilo.localization.touch_localizer import TouchLocalizer
from kilo.robot.interface import tool_rotation
from kilo.robot.planning import FrankaGeometry, PlanningFailure
from kilo.robot.preflight import PreflightRobot


def check_self_collisions(paths):
    """Independent mesh check using project MuJoCo arm, ignoring held peg/finger contact."""
    import mujoco
    from kilo.sim.scene import build_scene
    model = build_scene(peg_mode="hand", box_xy=(2., 2.))
    data = mujoco.MjData(model)
    robot_names = {"hand", "left_finger", "right_finger", "peg", *[f"link{i}" for i in range(8)]}
    ids = {i for i in range(model.nbody) if model.body(i).name in robot_names}
    count, failures = 0, []
    for name, path in paths.items():
        for index, q in enumerate(path):
            data.qpos[:7], data.qpos[7:9] = q, .01214
            mujoco.mj_forward(model, data)
            count += 1
            for contact in data.contact:
                if contact.dist >= -.0003:
                    continue
                a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                if a not in ids or b not in ids:
                    continue
                names = {model.body(a).name, model.body(b).name}
                if "peg" in names and names & {"left_finger", "right_finger", "hand"}:
                    continue
                failures.append(dict(path=name, index=index, bodies=sorted(names), distance_m=float(contact.dist)))
                if len(failures) >= 10:
                    return dict(samples=count, collisions=failures, passed=False)
    return dict(samples=count, collisions=failures, passed=not failures)


def run_case(g, cfg, scene, snapshot, box, out, label, elbow, step, force_error=0., height_offset=0.):
    trial_cfg = json.loads(json.dumps(cfg))
    trial_cfg["motion"]["align_wrist"] = False
    trial_cfg["workspace"]["search_offset"] += height_offset
    robot = PreflightRobot(g, trial_cfg, scene, snapshot["q"], box, elbow=elbow, step=step,
                           force_angle_error=np.radians(force_error))
    events = []
    t0 = time.monotonic()
    def log(message):
        events.append(message)
        print(f"[{label}] {message}", flush=True)
    result = dict(label=label, hypothetical_box=list(box), elbow_joint3=elbow,
                  tool_yaw_deg=180., force_direction_error_deg=force_error,
                  probe_height_offset_m=height_offset, status="failed")
    try:
        robot.reorient()
        estimate = TouchLocalizer(robot, trial_cfg, scene, log=log).run(
            verify_rounds=cfg["fit"]["verify_rounds"])
        result.update(status="geometry_passed", localization=estimate)
    except (PlanningFailure, RuntimeError, ValueError) as exc:
        if label == "no_contact" and str(exc) == "sweep ended without touching the box":
            result.update(status="geometry_passed", no_box_detected=True)
        else:
            result["failure"] = str(exc)
    result.update(motions=robot.motions, events=events, planning_seconds=time.monotonic() - t0)
    if robot.motions:
        result["minimum_certified_clearance_m"] = min(m["certified_clearance_m"] for m in robot.motions)
    result["self_collision_check"] = check_self_collisions(robot.paths)
    if not result["self_collision_check"]["passed"]:
        result["status"] = "failed"
    np.savez_compressed(out / f"{label}-joints.npz", **robot.paths)
    (out / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{label}] {result['status']}: {result.get('failure', '')}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=str(ROOT / "runs/real/lab/planning_state.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/real.yaml"))
    ap.add_argument("--model", default=str(ROOT / "configs/frankapy_geometry.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=float, nargs=3, default=[.477, -.106, 0.], metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--elbow", type=float, default=0.)
    ap.add_argument("--step", type=float, default=.005)
    ap.add_argument("--variants", action="store_true", help="also check +/-5 mm translation and +/-5 deg yaw")
    ap.add_argument("--stress", action="store_true", help="also check no box, +/-25 deg force errors, +/-6 mm probe height")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    snapshot = json.loads(Path(args.snapshot).read_text())
    g = FrankaGeometry(args.model, cfg)
    q = np.asarray(snapshot["q"])
    p, R = g.tip_pose(q)
    raw_fk_error = float(np.abs(g.fk(q) - snapshot["fk_reference"]).max())
    actual_R = np.asarray(snapshot["tool_pose"])[:3, :3]
    position_error = float(np.linalg.norm(p - snapshot["tip"]))
    rotation_matrix_error = float(np.abs(R - actual_R).max())
    if raw_fk_error > 1e-6 or position_error > .002 or rotation_matrix_error > .001:
        raise PlanningFailure("offline model failed installed-library/tool-frame snapshot validation")
    if snapshot["robot_mode"] != 1 or snapshot["current_errors"]:
        raise PlanningFailure("snapshot is not idle with clear active errors")
    x, y, yaw_deg = args.box
    nominal = (x, y, np.radians(yaw_deg))
    cases = [("nominal", nominal, 0., 0.)]
    if args.variants:
        cases += [(f"variant{i}", (x + dx, y + dy, np.radians(yaw_deg + da)), 0., 0.)
                  for i, (dx, dy, da) in enumerate([(-.005, -.005, -5), (-.005, .005, 5),
                                                  (.005, -.005, 5), (.005, .005, -5)])]
    if args.stress:
        cases += [("no_contact", (2., 2., 0.), 0., 0.),
                  ("force_minus25", nominal, -25., 0.), ("force_plus25", nominal, 25., 0.),
                  ("height_low", nominal, 0., -.006), ("height_high", nominal, 0., .006)]
    report = dict(
        execution_ready=False,
        scope="Offline geometric plans with hypothetical contact branches; no robot motion or force dynamics.",
        execution_requirement="A joint-path executor must preserve these postures. Existing Cartesian impedance "
                              "localize does not enforce them and must not be declared validated by this plan.",
        contact_branch_limit="Each whole commanded probe ray is checked before nominal truncation. "
                             "Subsequent lifts, crossings and verification are checked for the listed contact "
                             "scenarios, not every possible unknown contact/force direction.",
        margins=dict(shelf_m=cfg["workspace"]["shelf_margin"], extra_certified_m=.002,
                     joint_rad=.04, maximum_cartesian_step_m=args.step, maximum_joint_edge_step_rad=.01),
        model_validation=dict(raw_fk_max_error=raw_fk_error, measured_tip_error_m=position_error,
                              measured_rotation_matrix_max_error=rotation_matrix_error),
        source_hashes={str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                       for f in [args.config, args.model, args.snapshot, ROOT / cfg["workspace"]["shelf_bounds_file"],
                                 __file__, ROOT / "kilo/robot/planning.py", ROOT / "kilo/robot/preflight.py",
                                 ROOT / "kilo/localization/touch_localizer.py"]},
        cases=[],
    )
    for label, box, force, height in cases:
        report["cases"].append(run_case(g, cfg, load_config(), snapshot, box, out, label,
                                         args.elbow, args.step, force, height))
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["all_listed_cases_passed"] = all(c["status"] == "geometry_passed" for c in report["cases"])
    report["all_box_cases_verified"] = all(c.get("localization", {}).get("verified", False)
                                            for c in report["cases"] if not c.get("no_box_detected"))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(out=str(out), all_listed_cases_passed=report["all_listed_cases_passed"],
                          execution_ready=False)), flush=True)


if __name__ == "__main__":
    main()
