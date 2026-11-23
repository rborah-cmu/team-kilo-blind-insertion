"""Run the simulation experiments so far and save the results to JSON.

1. Robot insertion sweep: peg in hand, commanded straight down at a lateral
   offset from the hole, tracked with the gravity-compensated joint PD.
2. Loose-peg drop sweep: a free peg released 3 mm above the hole at a lateral offset.
3. Hole-search filter: presses needed to find the hole over many random hole
   positions (placeholder press model; see kilo/localization/hole_filter.py).

Usage: python scripts/run_experiments.py [--out docs/experiments/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kilo.config import load_config  # noqa: E402
from kilo.localization import HoleSearchFilter  # noqa: E402
from kilo.sim.control import GRIPPER_ACTUATOR, HOME_QPOS, JointPD, solve_ik  # noqa: E402
from kilo.sim.scene import build_scene, site_pos  # noqa: E402

CFG = load_config()
SHAFT = CFG["peg"]["shaft_length"]


def peg_contact_force(model, data) -> float:
    """Total normal force (N) on the peg from the base."""
    total = 0.0
    f = np.zeros(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        names = model.geom(c.geom1).name + model.geom(c.geom2).name
        if "peg" in names and "base" in names:
            mujoco.mj_contactForce(model, data, i, f)
            total += abs(f[0])
    return total


def robot_insertion(offset: float, record: bool = False) -> dict:
    model = build_scene(CFG, peg_mode="hand")
    data = mujoco.MjData(model)
    data.qpos[:7] = HOME_QPOS
    mujoco.mj_forward(model, data)
    hole = site_pos(model, data, "hole_top")
    off = np.array([offset, 0.0, 0.0])
    q, _ = solve_ik(model, "peg_tip", hole + off + [0, 0, 0.02], np.eye(3))
    data.qpos[:7], data.qpos[7:9] = q, 0.0125
    data.ctrl[GRIPPER_ACTUATOR] = 0.0125
    mujoco.mj_forward(model, data)
    pd = JointPD(model)
    trace, peak = [], 0.0
    z_cmds = np.arange(0.02, -0.0301, -0.0001)  # 10 cm/s down to 30 mm deep
    step = 0
    for z in list(z_cmds) + [z_cmds[-1]] * 50:  # then hold for 0.5 s
        q, _ = solve_ik(model, "peg_tip", hole + off + [0, 0, z], np.eye(3), q0=q)
        for _ in range(10):
            data.ctrl[:7] = pd(data, q)
            mujoco.mj_step(model, data)
            step += 1
        force = peg_contact_force(model, data)
        peak = max(peak, force)
        if record:
            tip = site_pos(model, data, "peg_tip")
            trace.append({"t": round(step * model.opt.timestep, 3),
                          "z_cmd_mm": round(z * 1000, 2),
                          "z_tip_mm": round((tip[2] - hole[2]) * 1000, 3),
                          "force_n": round(force, 2)})
    tip = site_pos(model, data, "peg_tip")
    return {"offset_mm": round(offset * 1000, 3),
            "depth_mm": round((hole[2] - tip[2]) * 1000, 3),
            "lateral_mm": round(float(np.linalg.norm(tip[:2] - hole[:2])) * 1000, 3),
            "peak_force_n": round(peak, 2),
            "trace": trace}


def free_drop(offset: float) -> dict:
    model = build_scene(CFG, peg_mode="free", peg_pos=(0, 0, 1))
    data = mujoco.MjData(model)
    data.qpos[:7] = HOME_QPOS
    mujoco.mj_forward(model, data)
    hole = site_pos(model, data, "hole_top")
    j = model.joint("peg_free").qposadr[0]
    data.qpos[j:j + 3] = hole + np.array([offset, 0.0, SHAFT + 0.003])
    pd = JointPD(model)
    for _ in range(2000):
        data.ctrl[:7] = pd(data, HOME_QPOS)
        mujoco.mj_step(model, data)
    tip = site_pos(model, data, "peg_tip")
    depth = (hole[2] - tip[2]) * 1000
    quat = data.qpos[j + 3:j + 7]
    tilt = float(np.degrees(2 * np.arccos(min(1.0, abs(quat[0])))))
    if depth > SHAFT * 1000 - 0.5:
        outcome = "inserted"
    elif tilt > 45:
        outcome = "fell over"
    elif depth > 0.5:
        outcome = "jammed"
    else:
        outcome = "on top"
    return {"offset_mm": round(offset * 1000, 3), "depth_mm": round(depth, 3),
            "tilt_deg": round(tilt, 1), "outcome": outcome}


def hole_search(seeds: int, radius: float = 0.02, detect: float = 0.005) -> list[dict]:
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r, a = radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        true_hole = np.array([r * np.cos(a), r * np.sin(a)])
        f = HoleSearchFilter([0.0, 0.0], radius, detect, num_particles=3000, rng=rng)
        presses = None
        for n in range(1, 61):
            p = f.next_press()
            dropped = np.linalg.norm(p - true_hole) < detect
            f.update_press(p, dropped)
            if dropped:
                presses = n
                break
        f.update_seated(true_hole + rng.normal(scale=0.0005, size=2), std=0.001)
        err = float(np.linalg.norm(f.estimate()[0] - true_hole))
        rows.append({"seed": seed, "hole_r_mm": round(r * 1000, 2), "presses": presses,
                     "final_error_mm": round(err * 1000, 3)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "results.json"))
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    traced = {0.0, 0.0005, 0.001, 0.002}
    offsets = np.round(np.arange(0.0, 0.00301, 0.00025), 5)
    results = {
        "robot_insertion": [robot_insertion(o, record=o in traced) for o in offsets],
        "free_drop": [free_drop(o) for o in np.round(np.arange(0.0, 0.00401, 0.00025), 5)],
        "hole_search": hole_search(args.seeds),
        "meta": {"clearance_mm": CFG["hole"]["clearance"] * 1000, "chamfer_mm": CFG["hole"]["chamfer"] * 1000,
                 "hole_search_R_mm": 20, "hole_search_detect_mm": 5},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out} in {time.time() - t0:.0f} s")
    for r in results["robot_insertion"]:
        print(f"robot  offset {r['offset_mm']:5.2f} mm  depth {r['depth_mm']:6.2f} mm  peak force {r['peak_force_n']:7.1f} N")
    for r in results["free_drop"]:
        print(f"drop   offset {r['offset_mm']:5.2f} mm  depth {r['depth_mm']:6.2f} mm  {r['outcome']}")
    presses = [r["presses"] for r in results["hole_search"]]
    print("search presses: found", sum(p is not None for p in presses), "/", len(presses),
          "median", np.median([p for p in presses if p]), "max", max(p for p in presses if p))


if __name__ == "__main__":
    main()
