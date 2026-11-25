"""Offline timing of representative localization moves; never imports a robot client.

Virtual durations include quintic motion, the simulated controller's 0.2 s
termination buffer, and bias sampling. Host timings cover planning and monitoring.
They are not measurements of the real robot or network.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kilo.robot.interface import ProbeSettings
from kilo.robot.planning import FrankaGeometry
from scripts.real.test_joint_sequence import make_offline_backend


def benchmark(cfg):
    cases = [
        ("empty_sweep_245mm", [.32, -.21, .0322], [.565, -.21, .0322], None, True),
        ("face_probe_65mm", [.36, -.075, .0322], [.425, -.075, .0322], [.477, -.106, 0.], True),
        ("travel_100mm", [.36, -.075, .12], [.46, -.075, .12], None, False),
    ]
    results = []
    for label, start, target, box, probe in cases:
        geometry = FrankaGeometry(ROOT / cfg["joint_executor"]["model"], cfg)
        q = geometry.inverse(start, np.pi, [0, -.785, 0, -2.356, 0, 1.571, 2.356], elbow=0.)
        robot, transport, clock = make_offline_backend(cfg, q, box)
        # Exercise strict tracking checks as well as geometry/force guards.
        robot.executor.path_checks = True
        settings = ProbeSettings(speed=cfg["motion"]["search_speed"], force_threshold=4,
                                 safety_force=20, debounce=3, bias_seconds=.5,
                                 timeout=300, retreat=0)
        t0 = time.perf_counter()
        try:
            if probe:
                result = robot.probe(target, np.pi, settings)
                contact, peak = result.contact, result.peak_force
            else:
                robot.move(target, np.pi, cfg["motion"]["travel_speed"])
                contact, peak = False, None
            results.append(dict(case=label, virtual_seconds=clock.now,
                                host_wall_seconds=time.perf_counter()-t0,
                                commands=len(transport.commands), contact=contact,
                                peak_force_n=peak, final_tip=robot.state().tip.tolist()))
            print(json.dumps(results[-1]), flush=True)
        finally:
            robot.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/real_planned.yaml"))
    parser.add_argument("--probe-step", type=float, help="Override command length in metres")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.probe_step is not None:
        cfg = copy.deepcopy(cfg)
        cfg["joint_executor"]["probe_step"] = args.probe_step
    report = dict(offline_only=True, options=cfg["joint_executor"], cases=benchmark(cfg))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
