"""Independently check saved joint states with installed frankapy, in OFFLINE mode.

No robot-state services or motion methods are called. Inputs are numeric arrays.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from frankapy import FrankaArm

    data = np.load(args.states, allow_pickle=False)
    qs, obstacles = data["q"], data["obstacles"]
    a = FrankaArm(offline=True)
    failures, t0, max_tip_error = [], time.monotonic(), 0.
    for i, q in enumerate(qs):
        if a.is_joints_in_collision_with_boxes(joints=q, boxes=obstacles):
            hits = [str(data["obstacle_names"][j]) for j, b in enumerate(obstacles)
                    if a.check_box_collision(b, joints=q)]
            failures.append(dict(index=i, obstacles=hits, q=q.tolist()))
        T = a.get_links_transforms(q)[11] @ data["tool_mount_correction"]
        tip = T[:3, 3] + T[:3, 2] * float(data["probe_length"])
        max_tip_error = max(max_tip_error, float(np.linalg.norm(tip - data["expected_tip"][i])))
        if (i + 1) % 1000 == 0:
            print(f"Checked {i + 1}/{len(qs)} joint states; failures={len(failures)}", flush=True)
        if len(failures) >= 10:
            break
    checked = i + 1 if len(qs) else 0
    result = dict(offline_mode=True, checked=checked, total=len(qs),
                  passed=checked == len(qs) and not failures and max_tip_error < 1e-5,
                  failures=failures, max_tip_model_difference_m=max_tip_error,
                  seconds=time.monotonic() - t0,
                  states_sha256=hashlib.sha256(Path(args.states).read_bytes()).hexdigest())
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
