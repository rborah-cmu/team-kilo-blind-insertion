"""Replay a saved localize_live.py run through the live box-estimate view (no robot).

    python scripts/replay_live_view.py runs/real/<stamp>-localize
    python scripts/replay_live_view.py runs/real/<stamp>-localize --speed 4 --no-window

Plays the recorded peg-tip/force samples, contacts, plate fit and hole presses in order,
so you can watch how the box estimate formed. Writes live.png / final.png into --out
(default: <run>/replay).
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kilo.config import load_config
from kilo.localization.common import SEARCH_SPACE_CORNERS
from kilo.localization.live_view import LiveView


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--speed", type=float, default=8.0, help="playback speed-up (default 8x)")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads((args.run / "result.json").read_text())
    z = np.load(args.run / "samples.npz") if (args.run / "samples.npz").exists() else {}
    cfg = load_config()
    corners = (data.get("config") or {}).get("workspace", {}).get("search_corners", SEARCH_SPACE_CORNERS)
    thresholds = (data.get("config") or {}).get("contact", {})
    view = LiveView(args.out or args.run / "replay", corners, np.asarray(cfg["box"]["size"][:2]) / 2,
                    cfg["peg"]["shaft_radius"], window=not args.no_window,
                    thresholds=(thresholds.get("force_threshold", 5.0), thresholds.get("safety_force", 20.0)))
    labels = data.get("probe_labels") or []
    contacts = {c["label"]: c for c in data.get("contacts") or []}
    hole = data.get("hole") or {}
    presses = hole.get("presses") or []
    shown_presses = []
    names = sorted(k for k in getattr(z, "files", []) if k.startswith("probe"))
    for i, name in enumerate(names):
        a = z[name]
        label = labels[i] if i < len(labels) else name
        view.status(f"replaying {label}")
        t_prev = None
        for row in a:
            if t_prev is not None:
                time.sleep(max(0., (row[0] - t_prev) / args.speed))
            t_prev = row[0]
            view.min_dt = 0.05 / args.speed
            view.sample(row[0], row[1:4], row[4:7])
        if label in contacts:
            c = contacts[label]
            view.contact(label, c["xy"], c["force_xy"])
        if label.startswith("descent-hole") and len(shown_presses) < len(presses):
            shown_presses.append(presses[len(shown_presses)])
            view.hole(shown_presses, status="replay")
        if len(contacts) and label == list(contacts)[-1] and data.get("result"):
            view.plate(data["result"]["center"], data["result"]["yaw_deg"])
    if data.get("result"):
        view.plate(data["result"]["center"], data["result"]["yaw_deg"])
    if hole:
        particles = z["hole_particles"] if "hole_particles" in getattr(z, "files", []) else None
        weights = z["hole_weights"] if "hole_weights" in getattr(z, "files", []) else None
        view.hole(presses, particles, weights, hole.get("center"), hole.get("std_m"), hole.get("status"))
    view.status("replay finished" + (f" (run error: {data['error']})" if data.get("error") else ""))
    view.close(keep_open=not args.no_window)


if __name__ == "__main__":
    main()
