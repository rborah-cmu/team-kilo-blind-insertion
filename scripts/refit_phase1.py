"""Test Phase 1's box fit with different force-direction weights on recorded contacts.

The probing (contacts) does not depend on the fit, so contacts are recorded once
per seed by replaying scripts/localize.py, then refit offline with several
weights for the force-direction term. Also reports a fit-quality flag: the best
fit's largest point gap, and whether a clearly different pose fits almost as well.

Usage: .venv/Scripts/python.exe scripts/refit_phase1.py --seeds 50 --workers 12
Writes docs/experiments/phase1_refit.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_phase1_failures import replay  # noqa: E402

HALF = np.array([0.1267, 0.169]) / 2
RADIUS = 0.0125
WEIGHTS = [0.004, 0.001, 0.0003, 0.0]


def fit_all(samples, forces, w):
    """Same residual as scripts/localize.py, with the force weight as a parameter.
    Returns every multi-start solution sorted by cost."""
    normals = forces / np.linalg.norm(forces, axis=1, keepdims=True)

    def residual(pose):
        center, angle = pose[:2], pose[2]
        c, s = np.cos(angle), np.sin(angle)
        axes = np.array([[c, -s], [s, c]])
        local = (samples - center) @ axes
        off = local - np.clip(local, -HALF, HALF)
        d = np.linalg.norm(off, axis=1)
        pred = off @ axes.T / np.maximum(d[:, None], 1e-9)
        return np.r_[d - RADIUS, (w * (pred - normals)).ravel()]

    fits = [least_squares(residual, np.r_[samples.mean(axis=0), y]) for y in np.linspace(0, np.pi, 12, endpoint=False)]
    return sorted(fits, key=lambda f: f.cost)


def point_gaps(pose, samples):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    axes = np.array([[c, -s], [s, c]])
    local = (samples - pose[:2]) @ axes
    return np.abs(np.linalg.norm(local - np.clip(local, -HALF, HALF), axis=1) - RADIUS)


def pose_distance(a, b):
    dyaw = (np.degrees(a[2] - b[2]) + 90) % 180 - 90
    return np.linalg.norm(a[:2] - b[:2]) * 1000, abs(dyaw)


def evaluate(run, w):
    S = np.array([c["xy"] for c in run["contacts"]])
    F = np.array([c["force"] for c in run["contacts"]])
    fits = fit_all(S, F, w)
    best = fits[0].x
    true = np.r_[run["true"]["xy"], np.radians(run["true"]["yaw_deg"])]
    err_mm, err_deg = pose_distance(best, true)
    # runner-up: the cheapest solution that is a clearly different pose
    rival = next((f for f in fits[1:] if pose_distance(f.x, best)[0] > 5 or pose_distance(f.x, best)[1] > 5), None)
    geo = lambda x: float(np.sum(point_gaps(x, S) ** 2))  # noqa: E731
    return {"error_mm": float(err_mm), "yaw_error_deg": float(err_deg),
            "max_gap_mm": float(point_gaps(best, S).max() * 1000),
            "rival_gap_ratio": (geo(rival.x) / max(geo(best), 1e-12)) if rival is not None else None,
            "rival_max_gap_mm": float(point_gaps(rival.x, S).max() * 1000) if rival is not None else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "phase1_refit.json"))
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        runs = pool.map(replay, range(args.start, args.start + args.seeds))
    rows = []
    for run in runs:
        row = {"seed": run["seed"], "contacts": len(run["contacts"]), "timeout": run["error"] is not None,
               "true": run["true"], "contact_list": run["contacts"]}
        if run["error"] is None:
            row["fits"] = {str(w): evaluate(run, w) for w in WEIGHTS}
        rows.append(row)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if not r["timeout"]]
    print(f"{len(rows)} seeds, {len(rows) - len(ok)} timeouts")
    print("weight    good(<5mm)  median_err_mm  bad seeds")
    for w in WEIGHTS:
        errs = {r["seed"]: r["fits"][str(w)]["error_mm"] for r in ok}
        bad = sorted(s for s, e in errs.items() if e >= 5)
        print(f"{w:<8}  {len(errs) - len(bad):3d}/{len(rows)}     {np.median(list(errs.values())):8.2f}     {bad}")


if __name__ == "__main__":
    main()
