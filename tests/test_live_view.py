"""Live view smoke test: a headless viewer process renders a run's events to PNG."""

import json

import numpy as np

from kilo.config import load_config
from kilo.localization.common import SEARCH_SPACE_CORNERS
from kilo.localization.live_view import LiveView, belief_map


def test_belief_map_is_lowest_at_the_true_box():
    half, radius = np.array([.06335, .0845]), .0125
    # Contacts just outside each face of a box centred at (0.47, -0.10), yaw 0.
    pts = [[.47 - half[0] - radius, -.10], [.47 + half[0] + radius, -.10],
           [.47, -.10 - half[1] - radius], [.47, -.10 + half[1] + radius]]
    xs, ys = np.linspace(.42, .52, 41), np.linspace(-.15, -.05, 41)
    gap = belief_map(pts, half, radius, xs, ys)
    iy, ix = np.unravel_index(np.argmin(gap), gap.shape)
    assert abs(xs[ix] - .47) < .003 and abs(ys[iy] + .10) < .003 and gap.min() < .5


def test_live_view_renders_png_without_a_window(tmp_path):
    cfg = load_config()
    view = LiveView(tmp_path, SEARCH_SPACE_CORNERS, np.asarray(cfg["box"]["size"][:2]) / 2,
                    cfg["peg"]["shaft_radius"], window=False)
    for i in range(50):
        view.min_dt = 0
        view.sample(i * .05, [.5 - i * .001, -.2, .03], [0.2, 0.1, 0.0])
    contacts = [("side-1", [.398, -.106], [-5., 0.]), ("side-2", [.556, -.106], [5., 0.]),
                ("side-3", [.477, -.203], [0., -5.]), ("side-4", [.477, -.009], [0., 5.])]
    for c in contacts:
        view.contact(*c)
    view.plate([.477, -.106], 0.)
    view.hole([dict(tip=[.48, -.10, .05], dropped=False)], np.random.default_rng(0).normal([.477, -.106], .003, (200, 2)),
              np.full(200, 1 / 200), [.477, -.106], .002, "in_progress")
    view.status("test done")
    view.close(keep_open=False)
    assert (tmp_path / "final.png").stat().st_size > 10000
    assert (tmp_path / "live.png").exists()


def test_replay_of_a_saved_run_renders(tmp_path):
    import subprocess
    import sys
    from kilo.config import REPO_ROOT
    run = tmp_path / "20261124-000000-localize"
    run.mkdir()
    contacts = [dict(kind="side", label=f"side-{i + 1}", xy=xy, force_xy=f, peak_force=5.)
                for i, (xy, f) in enumerate([([.398, -.106], [-5., 0.]), ([.556, -.106], [5., 0.]),
                                             ([.477, -.203], [0., -5.]), ([.477, -.009], [0., 5.])])]
    data = dict(contacts=contacts, probe_labels=["side-1", "side-2", "side-3", "side-4", "descent-hole-1"],
                config=dict(contact=dict(force_threshold=5., safety_force=20.)),
                result=dict(center=[.477, -.106], yaw_deg=0.), hole=dict(
                    status="unconfirmed", presses=[dict(tip=[.48, -.10, .05], dropped=False)]))
    (run / "result.json").write_text(json.dumps(data))
    probes = {f"probe{i:04d}": np.column_stack([np.linspace(i, i + .5, 20), np.full((20, 3), [.45, -.15, .03]),
                                                np.zeros((20, 3))]) for i in range(5)}
    np.savez_compressed(run / "samples.npz", **probes)
    out = tmp_path / "replay"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/replay_live_view.py"), str(run),
                        "--no-window", "--speed", "1000", "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert (out / "final.png").stat().st_size > 10000
