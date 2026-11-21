"""Gymnasium environment for peg insertion in MuJoCo.

TODO (Phase 2 lead), following proposal Table "Phase 2 design":
- reset(): build or reuse the scene from ``kilo.sim.scene``; sample the hole
  offset (uniform within R), the Phase 1 estimate error (configs:
  insertion.train_noise_xy, later the measured error x 1.5) and a small grasp
  offset; place the peg hovering above the *estimated* hole.
- Observation: last ``history_seconds`` of q, dq, tau_ext (stacked), plus the
  estimated hole centre relative to the peg tip.
- Action: end-effector (dx, dy, dz), clipped, added to the spiral waypoint
  (residual mode) and tracked by a Cartesian impedance controller at 1 kHz.
- Reward: dense tip-to-hole-bottom distance, insertion bonus, force and time
  penalties.
- Expose a flag to drop tau_ext from the observation for the E3 ablation.
"""

from __future__ import annotations


class PegInsertionEnv:
    def __init__(self, config: dict, use_tau_ext: bool = True, residual: bool = True):
        raise NotImplementedError

    def reset(self, seed: int | None = None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
