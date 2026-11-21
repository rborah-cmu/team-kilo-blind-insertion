"""Build the MuJoCo scene: Franka Panda + peg in the gripper + box with a hole.

TODO (Phase 2 lead):
- Robot: load the Franka Panda from MuJoCo Menagerie (``sim.menagerie_path``).
- Peg: a box geom for the square top between the fingers, and a capsule for the
  shaft (round section and rounded tip in one primitive). Weld it to the hand
  with a small randomizable offset rather than simulating the grasp.
- Box with hole: generate a plate with a circular hole using mjfit
  (https://github.com/younghyopark/mjfit; see its README for the API), with
  ``standalone=False`` so it can be attached to the robot scene via MjSpec.
  Shift the plate so the hole lands at a random offset within R of the box centre.
- Timestep 1 ms (mjfit's stiff contacts need it).
"""

from __future__ import annotations


def build_scene(config: dict):
    """Return a ``mujoco.MjModel`` for the given config."""
    raise NotImplementedError
