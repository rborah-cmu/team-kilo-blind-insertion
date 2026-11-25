"""Contact-guided probe motion for the MuJoCo scene (world coordinates)."""

from __future__ import annotations

import time

import mujoco
import numpy as np

from kilo.sim.control import ARM_DOF, HOME_QPOS
from kilo.sim.scene import TABLE_TOP_Z

from .common import (
    FORCE_THRESHOLD, POS_EPS, PROBE_LOOKAHEAD, SEARCH_MODE_HEIGHT,
    SEARCH_SPACE_CORNERS, SEARCH_SPEED, TRAVEL_MODE_HEIGHT, TRAVEL_SPEED,
    VIRTUAL_PROBE_LENGTH, probe_rotation,
)
from .search import generate_ray_probe_waypoints


def _probe_force(model, data):
    """Sum external contact forces on the peg, expressed in world axes."""
    peg = model.body("peg").id
    force = np.zeros(3)
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        first = model.geom_bodyid[contact.geom1] == peg
        second = model.geom_bodyid[contact.geom2] == peg
        if first == second:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
        force += world_force if second else -world_force
    return force


def _apply_target(model, data, position, rotation):
    """Cartesian impedance at the peg tip, with a gentle null-space posture."""
    site = model.site("peg_tip").id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros_like(jacp)
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    jac = np.vstack((jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        quat, (rotation @ data.site_xmat[site].reshape(3, 3).T).ravel()
    )
    rotation_error = np.zeros(3)
    mujoco.mju_quat2Vel(rotation_error, quat, 1.0)
    error = np.r_[position - data.site_xpos[site], rotation_error]
    stiffness = np.array([200., 200., 400., 30., 30., 30.])
    damping = np.array([30., 30., 40., 5., 5., 5.])
    velocity = data.qvel[:ARM_DOF]
    torque = jac.T @ (stiffness * error - damping * (jac @ velocity))
    null = np.eye(ARM_DOF) - np.linalg.pinv(jac) @ jac
    torque += null @ (10 * (HOME_QPOS - data.qpos[:ARM_DOF]) - 3 * velocity)
    torque += data.qfrc_bias[:ARM_DOF]
    limits = model.actuator_ctrlrange[:ARM_DOF]
    data.ctrl[:ARM_DOF] = np.clip(torque, limits[:, 0], limits[:, 1])


def _move(target, model, data, viewer, detect_contact, max_time):
    site = model.site("peg_tip").id
    mujoco.mj_forward(model, data)
    start = data.site_xpos[site].copy()
    delta = target - start
    length = np.linalg.norm(delta)
    direction = delta / length if length > 0 else np.zeros(3)

    rotation = probe_rotation(direction, data.site_xmat[site].reshape(3, 3))
    deadline = data.time + max_time
    next_frame = data.time
    wall_start, sim_start = time.monotonic(), data.time
    speed = SEARCH_SPEED if detect_contact else TRAVEL_SPEED

    while data.time < deadline:
        position = data.site_xpos[site].copy()
        force = _probe_force(model, data)
        if detect_contact and np.linalg.norm(force[:2]) > FORCE_THRESHOLD:
            _apply_target(model, data, position, data.site_xmat[site].reshape(3, 3))
            return position[:2].copy(), force[:2].copy()
        if np.linalg.norm(target - position) < POS_EPS:
            _apply_target(model, data, target, rotation)
            return None

        # Project progress onto the original line to correct lateral errors.
        progress = np.clip(np.dot(position - start, direction), 0, length)
        advance = speed * (data.time - sim_start + model.opt.timestep)
        lookahead = start + min(progress + PROBE_LOOKAHEAD, advance, length) * direction
        _apply_target(model, data, lookahead, rotation)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if viewer is not None and data.time >= next_frame:
            viewer.sync()
            time.sleep(max(0., data.time - sim_start - (time.monotonic() - wall_start)))
            next_frame = data.time + 1 / 50

    _apply_target(model, data, data.site_xpos[site], data.site_xmat[site].reshape(3, 3))
    raise TimeoutError("Probe did not reach its target within the simulation time limit")


def probemove(
    waypoint: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData,
    viewer=None, *, max_time: float = 60.,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Probe toward an XY waypoint at search height until arrival or contact.

    Returns ``(position_xy, force_xy)`` on contact, otherwise None. Forces are
    on the probe in world coordinates, suitable for np.array([position, force]).
    Height constants describe the grasp centre; control uses the actual peg tip.
    Heights are relative to the simulation's TABLE_TOP_Z. Target advancement
    uses SEARCH_SPEED, capped by PROBE_LOOKAHEAD.
    The caller owns the scene, initial pose, gripper command, and viewer.
    """
    tip_height = TABLE_TOP_Z + SEARCH_MODE_HEIGHT - VIRTUAL_PROBE_LENGTH
    target = np.r_[np.asarray(waypoint, dtype=float), tip_height]
    return _move(target, model, data, viewer, True, max_time)


def travelmove(
    ray: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData,
    viewer=None, *, max_time: float = 60.,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift, travel to the ray's start wall, then lower to search height.

    Returns ``(start_xy, end_xy)``; pass end_xy to probemove for the next sweep.
    Each leg uses TRAVEL_SPEED and has its own simulation time limit.
    Travel ignores probe contact.
    """
    mujoco.mj_forward(model, data)
    position = data.site_xpos[model.site("peg_tip").id].copy()
    travel_z = TABLE_TOP_Z + TRAVEL_MODE_HEIGHT - VIRTUAL_PROBE_LENGTH
    _move(np.r_[position[:2], travel_z], model, data, viewer, False, max_time)
    start, end = generate_ray_probe_waypoints(ray, SEARCH_SPACE_CORNERS)
    _move(np.r_[start, travel_z], model, data, viewer, False, max_time)
    search_z = TABLE_TOP_Z + SEARCH_MODE_HEIGHT - VIRTUAL_PROBE_LENGTH
    _move(np.r_[start, search_z], model, data, viewer, False, max_time)
    return start, end
