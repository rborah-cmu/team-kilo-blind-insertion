"""Replay localization through the REAL joint executor with an OFFLINE plant.

No ROS imports are executed and no connection to the robot is made.
The contact spring and perfect joint plant are test doubles, not hardware dynamics.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kilo.config import load_config
from kilo.localization.touch_localizer import TouchLocalizer
from kilo.robot.joint_executor import JointPathExecutor
from kilo.robot.offline_joint_transport import OfflineJointTransport, VirtualClock
from kilo.robot.planned_frankapy import PlannedFrankaPyBackend
from kilo.robot.planning import FrankaGeometry
from kilo.robot.workspace import HandWorkspace


def make_offline_backend(cfg, q, box):
    geometry = FrankaGeometry(ROOT / cfg["joint_executor"]["model"], cfg)
    clock = VirtualClock()
    transport = OfflineJointTransport(geometry, q, clock, box=box,
                                       table_z=cfg["workspace"]["table_z"])
    robot = PlannedFrankaPyBackend.__new__(PlannedFrankaPyBackend)
    robot.geometry = geometry
    robot.executor = JointPathExecutor(geometry, transport, cfg["joint_executor"], clock)
    robot.contact_cfg, robot.path_audit = cfg["contact"], []
    robot.hand_workspace = HandWorkspace(cfg["workspace"])
    robot.state = transport.state
    robot._streaming, robot.fa = False, SimpleNamespace(_in_skill=False)
    return robot, transport, clock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/real_planned.yaml"))
    ap.add_argument("--scenarios", default=str(ROOT / "runs/real/preflight-certified/report.json"))
    ap.add_argument("--snapshot", default=str(ROOT / "runs/real/lab/planning_state.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cases = json.loads(Path(args.scenarios).read_text())["cases"]
    snapshot = json.loads(Path(args.snapshot).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(offline_only=True, hardware_tracking_validated=False, cases=[],
                  plant="Deterministic joint interpolation, 1.5 N skill-start jump, travel bias and 6000 N/m contact spring.",
                  source_hashes={str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                 for f in [args.config, args.scenarios, args.snapshot, __file__,
                                           ROOT / "kilo/robot/joint_executor.py", ROOT / "kilo/robot/planned_frankapy.py",
                                           ROOT / "kilo/robot/offline_joint_transport.py", ROOT / "kilo/robot/planning.py"]})
    for case in cases:
        label = case["label"]
        if args.cases and label not in args.cases:
            continue
        trial = copy.deepcopy(cfg)
        trial["workspace"]["search_offset"] += case["probe_height_offset_m"]
        box = None if label == "no_contact" else case["hypothetical_box"]
        robot, transport, clock = make_offline_backend(trial, snapshot["q"], box)
        transport.force_angle_error = np.radians(case["force_direction_error_deg"])
        result = dict(label=label, passed=False)
        t0 = time.monotonic()
        print(f"[{label}] starting joint-executor replay", flush=True)
        try:
            loc = TouchLocalizer(robot, trial, load_config(), log=lambda m: print(f"[{label}] {m}", flush=True))
            estimate = loc.run(verify_rounds=trial["fit"]["verify_rounds"])
            result.update(passed=bool(estimate["verified"]), localization=estimate)
        except Exception as exc:
            if label == "no_contact" and str(exc) == "sweep ended without touching the box":
                result.update(passed=True, no_box_detected=True)
            else:
                result["failure"] = f"{type(exc).__name__}: {exc}"
        finally:
            robot.close()
        result.update(joint_commands=len(transport.commands), stop_count=transport.stop_count,
                      simulated_seconds=clock.now, wall_seconds=time.monotonic()-t0,
                      actual_motion_called=False, final_joints=transport.q.tolist())
        maxv, maxa = 0., 0.
        for command in transport.commands:
            d = float(np.max(np.abs(command["q"]-command["start_q"])))
            maxv = max(maxv, 1.875*d/command["duration"])
            maxa = max(maxa, 5.774*d/command["duration"]**2)
        result.update(max_commanded_joint_velocity=maxv, max_commanded_joint_acceleration=maxa)
        audits = robot.path_audit
        result["max_joint3_error_rad"] = max((a.get("tracking", {}).get("max_joint3_error_rad", 0.) for a in audits), default=0.)
        result["max_wrist_tracking_error_rad"] = max((a.get("tracking", {}).get("max_wrist_tracking_error_rad", 0.) for a in audits), default=0.)
        result["minimum_actual_model_clearance_m"] = min((a.get("tracking", {}).get("min_actual_model_clearance_m", np.inf) for a in audits), default=np.inf)
        result["peak_contact_residual_n"] = max((a.get("peak_force_n", 0.) for a in audits if a["kind"] == "probe"), default=0.)
        (out / f"{label}-audit.json").write_text(json.dumps(audits, indent=2, default=lambda o: o.item() if hasattr(o, "item") else repr(o)) + "\n")
        report["cases"].append(result)
        report["all_passed"] = all(c["passed"] for c in report["cases"])
        (out / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else repr(o)) + "\n")
        print(json.dumps(result if "localization" not in result else {k:v for k,v in result.items() if k!="localization"}), flush=True)
    print(json.dumps(dict(out=str(out), all_passed=report.get("all_passed"), cases=len(report["cases"]))), flush=True)
    if not report.get("all_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
