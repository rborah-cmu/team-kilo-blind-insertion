"""End-to-end simulation: Phase 1 box localization feeding Phase 2 insertion.

For each seed:
1. Build the scene with the base at a seeded random pose and run the Phase 1
   tactile localization (scripts/localize.py) to estimate the box centre and yaw.
2. Use the estimated box centre as the hole estimate (no separate hole search;
   the CAD hole is 4 mm off-centre) and run Phase 2 on the same base pose,
   once with the scripted baseline (zero residual) and once with the trained policy.

Usage:
    .venv/Scripts/python.exe scripts/run_end_to_end.py --seeds 20 --workers 8
Writes docs/experiments/end_to_end.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = ROOT / "artifacts" / "insertion_v4" / "best_model.zip"


def phase1(seed: int) -> dict:
    """Run localize.py's tactile search headless; return estimate and truth."""
    import mujoco

    import localize
    from kilo.config import load_config
    from kilo.localization.common import SEARCH_MODE_OFFSET
    from kilo.localization.search import generate_snake_waypoints
    from kilo.sim.control import GRIPPER_ACTUATOR, solve_ik
    from kilo.sim.scene import TABLE_TOP_Z, build_scene, sample_box_pose

    cfg = load_config()
    true_xy, true_yaw = sample_box_pose(cfg, seed=seed)
    model = build_scene(cfg, seed=seed)
    data = mujoco.MjData(model)
    points = generate_snake_waypoints(.06)
    q, err = solve_ik(model, "peg_tip", np.r_[points[0], TABLE_TOP_Z + SEARCH_MODE_OFFSET],
                      np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    data.qpos[:7], data.qpos[7:9] = q, .0125
    data.ctrl[GRIPPER_ACTUATOR] = .0125
    mujoco.mj_forward(model, data)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        center, yaw_deg = localize.run(model, data, None, points)
    sim_seconds = float(data.time)
    hole = data.site("hole_top").xpos[:2].copy()
    yaw_err = (yaw_deg - np.degrees(true_yaw) + 90) % 180 - 90
    return {"true_box_xy": list(map(float, true_xy)), "true_yaw": float(true_yaw),
            "est_box_xy": list(map(float, center)), "est_yaw_deg": float(yaw_deg),
            "true_hole_xy": list(map(float, hole)),
            "box_error_mm": float(np.linalg.norm(np.asarray(center) - true_xy) * 1000),
            "yaw_error_deg": float(abs(yaw_err)),
            "hole_estimate_error_mm": float(np.linalg.norm(np.asarray(center) - hole) * 1000),
            "phase1_sim_s": sim_seconds, "phase1_wall_s": time.time() - t0}


def phase2(seed: int, p1: dict, use_policy: bool) -> dict:
    import torch
    from stable_baselines3 import PPO

    from kilo.insertion.env import PegInsertionEnv

    torch.set_num_threads(1)
    env = PegInsertionEnv()
    model = PPO.load(str(MODEL), device="cpu") if use_policy else None
    obs, initial = env.reset(seed=seed, options={"box_xy": p1["true_box_xy"], "box_yaw": p1["true_yaw"],
                                                 "estimate_xy": p1["est_box_xy"]})
    for step in range(env.max_steps):
        obs["privileged"] = np.zeros_like(obs["privileged"])  # deployment: no privileged inputs
        action = model.predict(obs, deterministic=True)[0] if use_policy else np.zeros(3)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return {"success": bool(info["success"]), "seconds": (step + 1) / env.policy_hz,
            "depth_mm": float(info["depth"] * 1000), "peak_load_n": float(info["peak_load_n"]),
            "start_error_mm": float(np.linalg.norm(initial["estimate_error_xy"]) * 1000)}


def run_seed(seed: int) -> dict:
    try:
        p1 = phase1(seed)
    except Exception as exc:  # localization failure counts as an end-to-end failure
        return {"seed": seed, "phase1_ok": False, "error": repr(exc)}
    row = {"seed": seed, "phase1_ok": True, **p1}
    row["baseline"] = phase2(seed, p1, use_policy=False)
    row["policy"] = phase2(seed, p1, use_policy=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "end_to_end.json"))
    args = ap.parse_args()
    seeds = list(range(args.start, args.start + args.seeds))
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = []
        for row in pool.imap_unordered(run_seed, seeds):
            rows.append(row)
            if row["phase1_ok"]:
                print(f"seed {row['seed']:3d}: box err {row['box_error_mm']:5.2f} mm, yaw err {row['yaw_error_deg']:5.2f} deg, "
                      f"hole est err {row['hole_estimate_error_mm']:5.2f} mm | baseline "
                      f"{'OK ' if row['baseline']['success'] else 'FAIL'} {row['baseline']['seconds']:5.1f} s | policy "
                      f"{'OK ' if row['policy']['success'] else 'FAIL'} {row['policy']['seconds']:5.1f} s", flush=True)
            else:
                print(f"seed {row['seed']:3d}: Phase 1 failed: {row['error']}", flush=True)
    rows.sort(key=lambda r: r["seed"])
    ok = [r for r in rows if r["phase1_ok"]]

    def summary(key):
        s = [r[key] for r in ok]
        wins = [x for x in s if x["success"]]
        return {"successes": len(wins), "of": len(rows),
                "mean_seconds_successes": float(np.mean([x["seconds"] for x in wins])) if wins else None,
                "max_peak_load_n": float(max(x["peak_load_n"] for x in s)) if s else None}

    result = {"seeds": seeds, "phase1_successes": len(ok),
              "box_error_mm": {"median": float(np.median([r["box_error_mm"] for r in ok])),
                               "max": float(max(r["box_error_mm"] for r in ok))} if ok else None,
              "hole_estimate_error_mm": {"median": float(np.median([r["hole_estimate_error_mm"] for r in ok])),
                                         "max": float(max(r["hole_estimate_error_mm"] for r in ok))} if ok else None,
              "baseline": summary("baseline"), "policy": summary("policy"),
              "wall_seconds": time.time() - t0, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
