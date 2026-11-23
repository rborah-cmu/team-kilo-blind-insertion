"""Run the real-robot TouchLocalizer against the simulated backend for many seeds.

Checks the experiment code offline before robot time, with sensor bias and
noise like the real force estimate. Reports the initial (4-contact) fit and the
fit after face-normal verification.

Usage: .venv/Scripts/python.exe scripts/real/sim_check_localizer.py --seeds 8 26 49 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(seed: int) -> dict:
    from kilo.config import load_config
    from kilo.localization.fit import pose_distance
    from kilo.localization.touch_localizer import TouchLocalizer
    from kilo.robot import make_backend

    cfg = yaml.safe_load(open(ROOT / "configs" / "real.yaml"))
    robot = make_backend("sim", cfg, seed=seed, force_noise=0.3, force_bias=2.0, noise_seed=seed)
    cfg["workspace"]["table_z"] = robot.table_z()
    center, yaw = robot.true_box()
    try:
        res = TouchLocalizer(robot, cfg, load_config(), log=lambda s: None).run(
            verify_rounds=int(cfg["fit"]["verify_rounds"]))
    except Exception as exc:
        return {"seed": seed, "error": repr(exc)}
    init = pose_distance(res["initial"]["center"], np.radians(res["initial"]["yaw_deg"]), center, yaw)
    final = pose_distance(res["center"], np.radians(res["yaw_deg"]), center, yaw)
    return {"seed": seed, "initial_mm": init[0] * 1000, "initial_deg": init[1], "final_mm": final[0] * 1000,
            "final_deg": final[1], "verify": res["verify_reports"], "sim_seconds": float(robot.data.time),
            "contacts": len(res["contacts"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with Pool(min(args.workers, len(args.seeds))) as pool:
        rows = pool.map(run, args.seeds)
    for r in rows:
        if "error" in r:
            print(f"seed {r['seed']:3d}: ERROR {r['error']}")
        else:
            print(f"seed {r['seed']:3d}: initial {r['initial_mm']:6.1f} mm {r['initial_deg']:5.1f} deg -> "
                  f"verified {r['final_mm']:6.2f} mm {r['final_deg']:5.2f} deg | {r['contacts']} contacts, "
                  f"{r['sim_seconds']:.0f} s robot time | {r['verify']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
