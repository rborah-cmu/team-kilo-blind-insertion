"""Gymnasium environment for estimate-guided peg insertion in MuJoCo.

The actor sees 30 frames of joint states, external-torque estimates, nominal
tool FK, past actions and reference state. True peg/hole poses and contact
loads are training-only critic/reward inputs. Three bounded position
corrections modify a contact-responsive search and insertion reference.
"""

from __future__ import annotations

import mujoco
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kilo.config import load_config
from kilo.insertion.reference import InsertionReference
from kilo.sim.control import ARM_DOF, HOME_QPOS, solve_ik
from kilo.sim.scene import build_scene, HAND_TCP_Z

FRAME_FEATURES = 42
PRIVILEGED_FEATURES = 8


class PegInsertionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, use_tau_ext: bool = True,
                 residual: bool = True):
        super().__init__()
        self.cfg = config or load_config()
        self.ins = self.cfg["insertion"]
        self.contact_grasp = self.cfg.get("grasp", {}).get("mode") == "contact"
        self.use_tau_ext = use_tau_ext
        self.residual = residual
        self.dt = float(self.cfg["sim"]["timestep"])
        self.policy_hz = int(self.ins["policy_rate_hz"])
        self.frame_skip = max(1, round(1.0 / (self.policy_hz * self.dt)))
        self.history_frames = int(self.ins["history_frames"])
        self.max_steps = int(self.ins["max_episode_steps"])
        self.action_limit = np.asarray(self.ins["action_limit_xyz"], dtype=float)
        self.obs_history_size = self.history_frames * FRAME_FEATURES

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "history": spaces.Box(-np.inf, np.inf, shape=(self.obs_history_size,), dtype=np.float32),
            "privileged": spaces.Box(-np.inf, np.inf, shape=(PRIVILEGED_FEATURES,), dtype=np.float32),
        })

        # Compile once. Episode randomization changes the fixed box pose and
        # estimate error, avoiding a costly model rebuild on every reset.
        self.model = build_scene(self.cfg, peg_mode="hand")
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.peg_body_pos_nominal = self.model.body_pos[self.peg_body_id].copy()
        self.peg_body_quat_nominal = self.model.body_quat[self.peg_body_id].copy()
        self.finger_ids = [self.model.body(name).id for name in ("left_finger", "right_finger")]
        if self.contact_grasp:
            self.peg_body_pos_nominal = np.array([0., 0., HAND_TCP_Z + self.cfg['peg']['top_size'][2]/2])
            self.peg_body_quat_nominal = np.array([0., 1., 0., 0.])
        self.peg_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
        self.control_site_id = self.model.site("peg_tcp").id
        self.hole_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hole_top")
        self.rng = np.random.default_rng()
        self.history = np.zeros((self.history_frames, FRAME_FEATURES), dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.train_noise_xy = float(self.ins["train_noise_xy"])
        self.episode_noise_xy = self.train_noise_xy
        self.step_count = 0
        self.estimate_xy = np.zeros(2)
        self.true_hole_xy = np.zeros(2)
        self.top_z = 0.0
        self.goal_z = 0.0
        # The peg frame already points +z toward its square shoulder. With the
        # scene's fixed hand-to-peg transform, identity makes the shaft point
        # down into the hole; a 180-degree flip points it up and is unreachable.
        self.desired_rotation = np.eye(3)
        self.reference = None
        self.target_position = np.zeros(3)
        self.correction = np.zeros(3)
        self.success_streak = 0
        self.grip_loss_steps = 0
        self.peak_slip = 0.
        self.peak_load = 0.0
        self.reward_sums = {}
        self.contact_load = 0.0

    def _forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def _site_pos(self, site_id: int) -> np.ndarray:
        return self.data.site_xpos[site_id].copy()

    def set_estimate_noise(self, halfwidth: float) -> None:
        """Set per-axis localization-error range for the next episode reset."""
        self.train_noise_xy = max(0.0, float(halfwidth))

    def _current_frame(self) -> np.ndarray:
        q = self.data.qpos[:ARM_DOF] / np.pi
        dq = self.data.qvel[:ARM_DOF] / 5.0
        # MuJoCo's constraint forces projected into generalized coordinates
        # are the simulated equivalent of the robot's external-torque estimate.
        tau_ext = self.data.qfrc_constraint[:ARM_DOF].copy()
        tau_ext = tau_ext / 10.0
        if not self.use_tau_ext:
            tau_ext[:] = 0.0
        tip = self._site_pos(self.control_site_id)
        relative_hole = np.r_[self.estimate_xy - tip[:2], self.top_z - tip[2]] / 0.01
        current_rot = self.data.site_xmat[self.control_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (self.desired_rotation @ current_rot.T).ravel())
        orientation_error = np.zeros(3)
        mujoco.mju_quat2Vel(orientation_error, quat, 1.0)
        orientation_error /= 0.1
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.control_site_id)
        twist = np.r_[jacp[:, :ARM_DOF] @ self.data.qvel[:ARM_DOF],
                      jacr[:, :ARM_DOF] @ self.data.qvel[:ARM_DOF]]
        twist = twist / np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        frame = np.r_[q, dq, tau_ext, relative_hole, orientation_error, twist, self.previous_action,
                      (self.target_position - tip) / 0.01,
                      (self.reference.path[self.reference.index] - self.estimate_xy) / 0.01,
                      self.reference.phase / 2.0]
        return frame.astype(np.float32)

    def _observation(self):
        tip = self._site_pos(self.peg_site_id)
        axis = self.data.site_xmat[self.peg_site_id].reshape(3, 3)[:, 2]
        privileged = np.r_[(np.r_[self.true_hole_xy, self.top_z] - tip) / [0.01, 0.01, 0.0381],
                           axis, self.contact_load / 10.0, 1.0 - self.step_count / self.max_steps]
        return {"history": self.history.ravel().copy(), "privileged": privileged.astype(np.float32)}

    def _controller(self, target: np.ndarray) -> None:
        """Cartesian impedance with soft wrist alignment and bounded preload."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.control_site_id)
        jac = np.vstack((jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]))
        current_rot = self.data.site_xmat[self.control_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (self.desired_rotation @ current_rot.T).ravel())
        rot_error = np.zeros(3)
        mujoco.mju_quat2Vel(rot_error, quat, 1.0)
        error = np.r_[target - self._site_pos(self.control_site_id), rot_error]
        xy_kp = self.ins["search_xy_stiffness"] if self.reference.phase == 0 else self.ins["insert_xy_stiffness"]
        stiffness = np.r_[xy_kp, xy_kp, self.ins['z_stiffness'], self.ins['rotation_stiffness']]
        damping = np.asarray(self.ins['cartesian_damping'])
        velocity = self.data.qvel[:ARM_DOF]
        torque = jac.T @ (stiffness * error - damping * (jac @ velocity))
        null = np.eye(ARM_DOF) - np.linalg.pinv(jac) @ jac
        torque += null @ (10 * (HOME_QPOS - self.data.qpos[:ARM_DOF]) - 3 * velocity)
        torque += self.data.qfrc_bias[:ARM_DOF]
        limits = self.model.actuator_ctrlrange[:ARM_DOF]
        self.data.ctrl[:ARM_DOF] = np.clip(torque, limits[:, 0], limits[:, 1])

    def _goal_distance(self) -> float:
        tip = self._site_pos(self.peg_site_id)
        return float(np.linalg.norm(tip - np.r_[self.true_hole_xy, self.goal_z]))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.peg_body_id] = self.peg_body_pos_nominal
        self.model.body_quat[self.peg_body_id] = self.peg_body_quat_nominal

        box = self.cfg["box"]
        corners = np.asarray(self.cfg["snake_search"]["corners"], dtype=float)
        margin = 0.03
        box_yaw = self.rng.uniform(0.0, 2.0 * np.pi)
        hx, hy = np.asarray(box["size"][:2]) / 2
        c, s = abs(np.cos(box_yaw)), abs(np.sin(box_yaw))
        half_extent = np.array([hx * c + hy * s, hx * s + hy * c])
        low = corners.min(axis=0) + half_extent + margin
        high = corners.max(axis=0) - half_extent - margin
        if np.any(low > high):
            raise ValueError("The box does not fit inside the configured search region")
        box_xy = self.rng.uniform(low, high)
        # Optional overrides for chaining with Phase 1 (scripts/run_end_to_end.py).
        # The random draws above still happen, so seeded training is unchanged.
        options = options or {}
        if "box_xy" in options:
            box_xy = np.asarray(options["box_xy"], dtype=float)
        if "box_yaw" in options:
            box_yaw = float(options["box_yaw"])
        self.model.body_pos[self.base_body_id, :2] = box_xy
        self.model.body_quat[self.base_body_id] = [np.cos(box_yaw / 2), 0., 0., np.sin(box_yaw / 2)]
        self._forward()
        self.true_hole_xy = self._site_pos(self.hole_site_id)[:2]

        # Keep training and validation on the same full error distribution.
        noise = self.train_noise_xy
        self.episode_noise_xy = float(noise)
        estimate_error = self.rng.uniform(-noise, noise, size=2)
        if "estimate_xy" in options:
            estimate_error = np.asarray(options["estimate_xy"], dtype=float) - self.true_hole_xy
        self.estimate_xy = self.true_hole_xy + estimate_error
        top_z = float(self._site_pos(self.hole_site_id)[2])
        self.top_z = top_z
        self.goal_z = top_z - float(self.ins["success_depth"])

        start = np.r_[self.estimate_xy, top_z + float(self.ins["start_clearance"])]
        q_start, ik_error = solve_ik(
            self.model, "peg_tcp", start, target_mat=self.desired_rotation, q0=HOME_QPOS
        )
        if ik_error > float(self.ins["ik_tolerance"]):
            raise RuntimeError(f"Could not reach insertion start pose (IK error {ik_error:.4f} m)")
        self.data.qpos[:ARM_DOF] = q_start
        self.data.qvel[:] = 0.0
        self.data.qpos[7:9] = 0.0125
        self.data.ctrl[7] = 0.0125
        self._forward()

        # Small fixed-grasp perturbation applied in peg body coordinates.
        grasp_sigma = float(self.ins["grasp_offset_std"])
        if grasp_sigma > 0:
            self.model.body_pos[self.peg_body_id] += grasp_sigma * np.clip(
                self.rng.normal(size=3), -self.ins['grasp_clip_sigma'], self.ins['grasp_clip_sigma'])
        tilt_sigma = float(self.ins["grasp_tilt_std_rad"])
        if tilt_sigma > 0:
            tilt = np.r_[tilt_sigma * np.clip(self.rng.normal(size=2),
                         -self.ins['grasp_clip_sigma'], self.ins['grasp_clip_sigma']), 0.0]
            tilt_quat = np.zeros(4)
            mujoco.mju_euler2Quat(tilt_quat, tilt, "XYZ")
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, self.peg_body_quat_nominal, tilt_quat)
            self.model.body_quat[self.peg_body_id] = quat
        if grasp_sigma > 0 or tilt_sigma > 0:
            self._forward()

        self.reference = InsertionReference(self.ins, self.estimate_xy, self.top_z,
                                             self._site_pos(self.control_site_id))
        if self.contact_grasp:
            self._initialize_contact_grasp(start)
        self.target_position = start.copy()
        self.correction[:] = 0.0
        self.success_streak = 0
        self.grip_loss_steps = 0
        self.peak_slip = 0.
        self.contact_load = self.peak_load = 0.0
        self.reward_sums = {key: 0.0 for key in ("shaping", "time", "force", "smoothness", "success", "failure")}
        self.step_count = 0
        self.previous_action[:] = 0.0
        self.history[:] = 0.0
        initial_frame = self._current_frame()
        self.history[:] = initial_frame
        self.previous_potential = self._potential()
        return self._observation(), {"ik_error": ik_error, "estimate_error_xy": estimate_error.copy()}

    def _initialize_contact_grasp(self, target):
        grasp = self.cfg['grasp']
        hand = self.model.body('hand').id
        pos = self.data.xpos[hand] + self.data.xmat[hand].reshape(3, 3) @ self.model.body_pos[self.peg_body_id]
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, self.data.xquat[hand], self.model.body_quat[self.peg_body_id])
        self.model.body_pos[self.peg_body_id] = [0., 0., 1.]
        self.model.body_quat[self.peg_body_id] = [1., 0., 0., 0.]
        joint = self.model.joint('peg_free')
        qp, qv = joint.qposadr[0], joint.dofadr[0]
        self.data.qpos[qp:qp+7] = np.r_[pos, quat]
        self.data.qpos[7:9] = 0.018
        self.data.ctrl[7] = 0.005
        self.grip_force = self.rng.uniform(*grasp['force_range_n'])
        self.grip_friction = self.rng.uniform(*grasp['friction_range'])
        self.model.actuator_forcerange[7] = [-self.grip_force, self.grip_force]
        self.model.pair_friction[:, :2] = self.grip_friction
        self.model.pair_solref[:, 0] = self.rng.uniform(*grasp['contact_time_range'])
        self._forward()
        # Initial holder supports the peg while the jaws close; it is removed
        # before settling and before any policy observation or reward.
        for k in range(round(grasp['settle_seconds']/self.dt)):
            if k*self.dt < grasp['supported_seconds']:
                self.data.qpos[qp:qp+7] = np.r_[pos, quat]
                self.data.qvel[qv:qv+6] = 0.
                self._forward()
            self._controller(target)
            mujoco.mj_step(self.model, self.data)
        self._forward()
        self.initial_grasp_tip = self._relative_grasp_tip()
        if np.linalg.norm(self._site_pos(self.peg_site_id)-self._site_pos(self.control_site_id)) > .01:
            raise RuntimeError('Contact grasp failed to settle; inspect gripper parameters')

    def _grip_loads(self):
        loads = np.zeros(2)
        wrench = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.peg_body_id not in bodies:
                continue
            for j, finger in enumerate(self.finger_ids):
                if finger in bodies:
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    loads[j] += max(0., wrench[0])
        return loads

    def _relative_grasp_tip(self):
        hand = self.model.body('hand').id
        return self.data.xmat[hand].reshape(3, 3).T @ (self._site_pos(self.peg_site_id)-self.data.xpos[hand])

    def _potential(self):
        tip = self._site_pos(self.peg_site_id)
        radial = np.linalg.norm(tip[:2] - self.true_hole_xy)
        depth = np.clip((self.top_z - tip[2]) / self.ins["success_depth"], 0.0, 1.0)
        return float(2.0 * np.exp(-radial / 0.003) + 4.0 * depth)

    def _contact_load(self):
        # Training metric/reward only; never an actor observation.
        load = 0.0
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.peg_body_id in bodies and not any(body in self.finger_ids for body in bodies):
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                load += max(0.0, wrench[0])
        return load

    def _estimated_vertical_force(self):
        """Cartesian estimate from joint torques, available on the real arm too."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.control_site_id)
        jac = np.vstack((jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]))
        return float(np.linalg.lstsq(jac.T, self.data.qfrc_constraint[:ARM_DOF], rcond=None)[0][2])

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be three finite values")
        action = np.clip(action, -1.0, 1.0)
        dt = self.frame_skip * self.model.opt.timestep
        tip = self._site_pos(self.control_site_id)
        nominal = self.reference.step(tip, float(self.cfg["peg"]["shaft_length"]),
                                      self._estimated_vertical_force())
        desired_correction = action * self.action_limit if self.residual else np.zeros(3)
        slew = self.ins["residual_speed"] * dt
        self.correction += np.clip(desired_correction - self.correction, -slew, slew)
        self.target_position = nominal + self.correction
        # The residual cannot defeat the bounded preload.
        lead = self.ins["search_z_lead"] if self.reference.phase == 0 else self.ins["insert_z_lead"]
        self.target_position[2] = max(self.target_position[2], tip[2] - lead)
        loads = []
        for _ in range(self.frame_skip):
            self._controller(self.target_position)
            mujoco.mj_step(self.model, self.data)
            loads.append(self._contact_load())
        self._forward()
        self.step_count += 1
        self.contact_load = self._contact_load()
        self.peak_load = max(self.peak_load, max(loads))
        tip = self._site_pos(self.peg_site_id)
        axis = self.data.site_xmat[self.peg_site_id].reshape(3, 3)[:, 2]
        depth = self.top_z - tip[2]
        radial_error = float(np.linalg.norm(tip[:2] - self.true_hole_xy))
        entry = tip + axis * depth / max(axis[2], 1e-6)
        entry_error = float(np.linalg.norm(entry[:2] - self.true_hole_xy))
        tolerance = self.cfg["hole"]["radius"] - self.cfg["peg"]["shaft_radius"] + self.ins["success_xy_tolerance"]
        slip = float(np.linalg.norm(self._relative_grasp_tip() - self.initial_grasp_tip)) if self.contact_grasp else 0.
        held = not self.contact_grasp or bool(np.all(self._grip_loads() > .1))
        self.grip_loss_steps = 0 if held else self.grip_loss_steps + 1
        self.peak_slip = max(self.peak_slip, slip)
        dropped = self.contact_grasp and (slip > self.cfg['grasp']['loss_distance'] or self.grip_loss_steps*dt >= .2)
        seated = held and not dropped and depth >= self.ins["success_depth"] and max(radial_error, entry_error) <= tolerance and axis[2] > 0.99
        self.success_streak = self.success_streak + 1 if seated else 0
        success = self.success_streak >= round(self.ins["success_hold_seconds"] / dt)
        potential = self._potential()
        terms = {
            "shaping": self.ins["ppo_gamma"] * (0.0 if success or dropped else potential) - self.previous_potential,
            "time": -self.ins["time_cost_per_second"] * dt,
            "force": -self.ins["force_cost_per_second"] * float(np.maximum(0., np.asarray(loads) - self.ins["force_soft_limit"]).mean()) * dt,
            "smoothness": -self.ins["smoothness_cost_per_second"] * float(np.mean((action - self.previous_action)**2)) * dt,
            "success": self.ins["success_reward"] if success else 0.0,
            "failure": -self.cfg["grasp"]["drop_penalty"] if dropped else 0.0,
        }
        self.previous_potential = potential
        for key, value in terms.items():
            self.reward_sums[key] += value
        self.previous_action = action.astype(np.float32)
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._current_frame()
        info = {"success": bool(success), "depth": float(depth), "radial_error": radial_error,
                "entry_error": entry_error, "goal_distance": self._goal_distance(),
                "estimate_error_xy": self.estimate_xy - self.true_hole_xy,
                "estimate_noise_xy": self.episode_noise_xy, "phase": self.reference.phase,
                "peak_load_n": self.peak_load, "grasp_slip_m": slip, "peak_grasp_slip_m": self.peak_slip, "dropped": bool(dropped)}
        info.update({"reward_" + key: value for key, value in self.reward_sums.items()})
        return self._observation(), float(sum(terms.values())), bool(success or dropped), bool(self.step_count >= self.max_steps and not (success or dropped)), info
