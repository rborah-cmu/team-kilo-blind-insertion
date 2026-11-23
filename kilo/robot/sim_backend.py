"""MuJoCo backend: the same probing interface as the real arm, for offline tests.

Uses the course scene (kilo.sim.scene) with the peg fixed in the hand, and the
Cartesian impedance law and contact-force readout from
kilo.localization.probenav. Optional sensor bias and noise emulate the real
robot's external-force estimate so the bias-removal logic gets exercised.
"""

from __future__ import annotations

import mujoco
import numpy as np

from kilo.config import load_config
from kilo.localization.probenav import _apply_target, _probe_force
from kilo.sim.control import GRIPPER_ACTUATOR, solve_ik
from kilo.sim.scene import TABLE_TOP_Z, build_scene, sample_box_pose

from .interface import RobotBackend, RobotState


def _site_rotation(yaw: float) -> np.ndarray:
    """Peg-tip site orientation whose finger axis points along ``yaw`` (see probenav)."""
    phi = yaw - np.pi / 2
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SimBackend(RobotBackend):
    preferred_hz = 50.0
    stiffness = 200.0  # translational stiffness of probenav._apply_target, N/m

    def __init__(self, cfg: dict, seed: int | None = None, box: tuple | None = None,
                 force_noise: float = 0.0, force_bias: float = 0.0, noise_seed: int = 0):
        scene_cfg = load_config()
        if box is not None:
            self.box_xy, self.box_yaw = np.asarray(box[:2], float), float(box[2])
        else:
            self.box_xy, self.box_yaw = sample_box_pose(scene_cfg, seed=seed)
        self.model = build_scene(scene_cfg, peg_mode="hand", box_xy=tuple(self.box_xy), box_yaw=self.box_yaw)
        self.data = mujoco.MjData(self.model)
        self.tip_site = self.model.site("peg_tip").id
        rng = np.random.default_rng(noise_seed)
        self.noise = float(force_noise)
        self.bias = rng.normal(scale=force_bias, size=3) if force_bias > 0 else np.zeros(3)
        self.rng = rng
        ws = cfg["workspace"]
        corners = np.asarray(ws["search_corners"], float)
        start = np.r_[corners.mean(axis=0), TABLE_TOP_Z + ws["travel_offset"]]
        self.yaw = float(cfg["robot"].get("wrist_yaw_offset", -2.356)) + float(np.arctan2(start[1], start[0]))  # joint 7 mid-range
        q, err = solve_ik(self.model, "peg_tip", start, _site_rotation(self.yaw))
        if err > 0.005:
            raise RuntimeError(f"sim start pose unreachable (IK error {err * 1000:.1f} mm)")
        self.data.qpos[:7], self.data.qpos[7:9] = q, 0.0125
        self.data.ctrl[GRIPPER_ACTUATOR] = 0.0125
        mujoco.mj_forward(self.model, self.data)
        self.target = self.data.site_xpos[self.tip_site].copy()

    # truth, for scoring experiments
    def true_box(self) -> tuple[np.ndarray, float]:
        return self.box_xy.copy(), self.box_yaw

    def true_hole(self) -> np.ndarray:
        return self.data.site("hole_top").xpos[:2].copy()

    def table_z(self) -> float:
        return TABLE_TOP_Z

    def state(self) -> RobotState:
        R = self.data.site_xmat[self.tip_site].reshape(3, 3)
        force = _probe_force(self.model, self.data) + self.bias
        if self.noise > 0:
            force = force + self.rng.normal(scale=self.noise, size=3)
        return RobotState(time=float(self.data.time), tip=self.data.site_xpos[self.tip_site].copy(),
                          yaw=float(np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2), force=force,
                          torque=np.zeros(3), q=self.data.qpos[:7].copy())

    def begin_stream(self) -> None:
        self.target = self.data.site_xpos[self.tip_site].copy()

    def stream_target(self, tip, yaw) -> None:
        self.target, self.yaw = np.asarray(tip, float).copy(), float(yaw)

    def tick(self, dt: float) -> None:
        rotation = _site_rotation(self.yaw)
        for _ in range(max(1, int(round(dt / self.model.opt.timestep)))):
            _apply_target(self.model, self.data, self.target, rotation)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def end_stream(self) -> None:
        pass

    def move(self, tip, yaw, speed) -> None:
        tip = np.asarray(tip, float)
        start = self.data.site_xpos[self.tip_site].copy()
        length = float(np.linalg.norm(tip - start))
        dt, t = 0.02, 0.0
        while True:
            frac = min(1.0, speed * t / max(length, 1e-9))
            self.stream_target(start + frac * (tip - start), yaw)
            self.tick(dt)
            t += dt
            error = float(np.linalg.norm(self.data.site_xpos[self.tip_site] - tip))
            if frac >= 1.0 and error < 0.001:
                return
            # The borrowed impedance law (200 N/m, non-dynamically-consistent null space) can settle
            # a few mm off at awkward wrist angles. Probes use measured tip positions, so accept it.
            if t > length / speed + 3.0:
                if error > 0.02:
                    raise RuntimeError(f"sim move did not converge: target {np.round(tip, 4)}, "
                                       f"tip {np.round(self.data.site_xpos[self.tip_site], 4)}")
                if error > 0.005:
                    print(f"  [sim] move settled {error * 1000:.1f} mm from its target")
                return
