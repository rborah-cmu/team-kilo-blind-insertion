"""Simple controllers for the torque-controlled Panda in simulation.

- :func:`solve_ik`: damped least-squares IK for a site's position and orientation.
- :class:`JointPD`: gravity-compensated joint PD, the minimum needed to hold or
  track a pose with torque actuators. The Phase 2 environment will replace this
  with Cartesian impedance control.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_DOF = 7
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRIPPER_ACTUATOR = 7  # actuator8: average finger opening (m), ctrlrange 0.005-0.04


def solve_ik(
    model: mujoco.MjModel,
    site: str,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    iters: int = 300,
    tol: float = 1e-5,
    damping: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Arm joint angles that put ``site`` at ``target_pos`` (and ``target_mat``).

    Returns (q, final position error in metres). Uses a scratch MjData, so the
    caller's simulation state is untouched.
    """
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    q = (HOME_QPOS if q0 is None else np.asarray(q0)).copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lo, hi = model.jnt_range[:ARM_DOF, 0], model.jnt_range[:ARM_DOF, 1]
    err_pos = np.inf
    for _ in range(iters):
        data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err_pos = np.asarray(target_pos) - data.site_xpos[sid]
        err = err_pos
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        jac = jacp[:, :ARM_DOF]
        if target_mat is not None:
            cur = data.site_xmat[sid].reshape(3, 3)
            rot = np.asarray(target_mat) @ cur.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.ravel())
            err_rot = np.zeros(3)
            mujoco.mju_quat2Vel(err_rot, quat, 1.0)
            err = np.concatenate([err_pos, err_rot])
            jac = np.vstack([jac, jacr[:, :ARM_DOF]])
        if np.linalg.norm(err) < tol:
            break
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(len(err)), err)
        q = np.clip(q + dq, lo, hi)
    return q, float(np.linalg.norm(err_pos))


class JointPD:
    """tau = gravity/Coriolis compensation + Kp (q* - q) - Kd dq, clipped to limits."""

    def __init__(self, model: mujoco.MjModel,
                 kp=(600, 600, 600, 600, 250, 150, 50), kd=(50, 50, 50, 50, 20, 15, 5)):
        self.model = model
        self.kp, self.kd = np.asarray(kp, float), np.asarray(kd, float)
        self.lo = model.actuator_ctrlrange[:ARM_DOF, 0]
        self.hi = model.actuator_ctrlrange[:ARM_DOF, 1]

    def __call__(self, data: mujoco.MjData, q_target: np.ndarray) -> np.ndarray:
        q, dq = data.qpos[:ARM_DOF], data.qvel[:ARM_DOF]
        tau = data.qfrc_bias[:ARM_DOF] + self.kp * (q_target - q) - self.kd * dq
        return np.clip(tau, self.lo, self.hi)
