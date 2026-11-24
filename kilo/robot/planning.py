"""Offline Franka posture planning with the installed lab's exact box model.

This module has no ROS imports or motion methods. A kinematic plan does not
certify that an unconstrained Cartesian impedance controller will follow it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .interface import tool_rotation
from .workspace import load_shelf_boxes


class PlanningFailure(RuntimeError):
    pass


def box_separations(poses, half_sizes, obstacle_poses, obstacle_half_sizes):
    """Signed separating-axis gaps for every OBB pair; <=0 means overlap.

    Positive values are conservative lower bounds on Euclidean clearance.
    Includes all 15 SAT axes, normalized including non-parallel edge axes.
    """
    A, B = poses[:, :3, :3], obstacle_poses[:, :3, :3]
    R = np.einsum("nwi,mwj->nmij", A, B)
    absR = np.abs(R)
    delta = obstacle_poses[None, :, :3, 3] - poses[:, None, :3, 3]
    t = np.einsum("nwi,nmw->nmi", A, delta)
    sep_a = np.abs(t) - half_sizes[:, None, :] - np.einsum("nmij,mj->nmi", absR, obstacle_half_sizes)
    sep_b = np.abs(np.einsum("nmi,nmij->nmj", t, R)) - obstacle_half_sizes[None, :, :] \
        - np.einsum("ni,nmij->nmj", half_sizes, absR)
    edges = []
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            norm = np.sqrt(np.maximum(0., 1. - R[:, :, i, j] ** 2))
            sep = (np.abs(t[:, :, i2] * R[:, :, i1, j] - t[:, :, i1] * R[:, :, i2, j])
                   - half_sizes[:, None, i1] * absR[:, :, i2, j]
                   - half_sizes[:, None, i2] * absR[:, :, i1, j]
                   - obstacle_half_sizes[None, :, j1] * absR[:, :, i, j2]
                   - obstacle_half_sizes[None, :, j2] * absR[:, :, i, j1])
            edges.append(np.where(norm > 1e-7, sep / np.maximum(norm, 1e-7), -np.inf))
    return np.maximum(np.maximum(sep_a.max(axis=2), sep_b.max(axis=2)), np.max(edges, axis=0))


class FrankaGeometry:
    def __init__(self, model_path, cfg):
        self.model_path = Path(model_path)
        model = json.loads(self.model_path.read_text())
        g = model["geometry"]
        self.dh = np.asarray(g["DH_PARAMS"], float)
        self.box_links = np.asarray(g["COLLISION_BOX_LINKS"], int)
        self.box_offsets = np.asarray(g["COLLISION_BOX_POSES"], float)
        self.half_sizes = np.asarray(g["COLLISION_BOX_SHAPES"], float) / 2
        self.tool_delta = np.asarray(model["tool_delta"], float)
        self.mount_correction = np.asarray(model["tool_mount_correction"], float)
        self.probe_length = float(cfg["robot"]["probe_length"])
        # Intersect library limits with the Control PC's installed termination limits.
        server_lo = [-2.88, -1.75, -2.88, -3.06, -2.88, -.0025, -2.88]
        server_hi = [2.88, 1.75, 2.88, -.06, 2.88, 3.74, 2.88]
        self.lo = np.maximum(g["JOINT_LIMITS_MIN"], server_lo)
        self.hi = np.minimum(g["JOINT_LIMITS_MAX"], server_hi)
        shelf_file = Path(cfg["workspace"]["shelf_bounds_file"])
        if not shelf_file.is_absolute():
            shelf_file = Path(__file__).resolve().parents[2] / shelf_file
        names = [r["name"] for r in json.loads(shelf_file.read_text())["boxes"]]
        shelves = load_shelf_boxes(cfg["workspace"])
        walls = np.asarray(g["WORKSPACE_WALLS"], float)
        # Table is the Lab 2/3 mounting-frame reference. Probe length/table_z
        # calibration uncertainty is a separate limit of this geometric model.
        table = np.array([.46, 0, -.005, 0, 0, 0, .55, 1.008, .0102])
        self.obstacle_names = names + [f"virtual_wall_{i}" for i in range(len(walls))] + ["table"]
        self.obstacles = np.vstack([*shelves, *walls, table])
        self.obstacle_poses = np.repeat(np.eye(4)[None], len(self.obstacles), axis=0)
        self.obstacle_poses[:, :3, 3] = self.obstacles[:, :3]
        # Supplied shelf and wall boxes are axis aligned; reject silently different models.
        if not np.allclose(self.obstacles[:, 3:6], 0):
            raise ValueError("expected axis-aligned lab obstacles")
        self.obstacle_halves = self.obstacles[:, 6:] / 2
        self.point_reach_bound = (np.abs(self.dh[:, :2]).sum()
                                  + np.linalg.norm(self.box_offsets[:, :3, 3], axis=1).max()
                                  + np.linalg.norm(self.half_sizes, axis=1).max())

    def fk(self, q):
        q = np.asarray(q, float)
        transforms = np.repeat(np.eye(4)[None], 12, axis=0)
        for i, (a, d, alpha, theta) in enumerate(self.dh):
            theta += q[i] if i < 7 else 0.
            ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
            delta = np.array([[ct, -st, 0, a], [ca * st, ca * ct, -sa, -d * sa],
                              [sa * st, sa * ct, ca, d * ca], [0, 0, 0, 1.]])
            transforms[i + 1] = transforms[i] @ delta
        transforms[10] = transforms[9]
        transforms[11] = transforms[10] @ self.tool_delta
        return transforms

    def tip_pose(self, q):
        tool = self.fk(q)[11] @ self.mount_correction
        return tool[:3, 3] + tool[:3, 2] * self.probe_length, tool[:3, :3]

    def clearance(self, q, detail=False):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)) or np.any(q <= self.lo) or np.any(q >= self.hi):
            return (-np.inf, {"kind": "joint_limits"}) if detail else -np.inf
        poses = self.fk(q)[self.box_links] @ self.box_offsets
        gaps = box_separations(poses, self.half_sizes, self.obstacle_poses, self.obstacle_halves)
        i, j = np.unravel_index(gaps.argmin(), gaps.shape)
        gap = float(gaps[i, j])
        info = dict(robot_box=int(i), link=int(self.box_links[i]), obstacle=self.obstacle_names[j])
        return (gap, info) if detail else gap

    def inverse(self, xyz, yaw, seed, elbow=None, joint_margin=.04):
        xyz, seed = np.asarray(xyz, float), np.asarray(seed, float)
        target_R = tool_rotation(yaw)
        def residual(q):
            p, R = self.tip_pose(q)
            error = np.r_[100 * (p - xyz), 5 * Rotation.from_matrix(target_R @ R.T).as_rotvec()]
            if elbow is not None:
                error = np.r_[error, .5 * (q[2] - elbow)]
            return error
        lo, hi = self.lo + joint_margin, self.hi - joint_margin
        result = least_squares(residual, np.clip(seed, lo + 1e-8, hi - 1e-8), bounds=(lo, hi),
                               max_nfev=90, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        p, R = self.tip_pose(result.x)
        pe = float(np.linalg.norm(p - xyz))
        re = float(np.linalg.norm(Rotation.from_matrix(target_R @ R.T).as_rotvec()))
        if pe > .0005 or re > .005 or (elbow is not None and abs(result.x[2] - elbow) > .01):
            raise PlanningFailure(f"IK residual: position={pe:.6f} m, rotation={re:.5f} rad")
        return result.x

    def check_joint_edge(self, start, end, *, resolution=.01, min_clearance=.002):
        start, end = np.asarray(start), np.asarray(end)
        gap0, gap1 = self.clearance(start), self.clearance(end)
        path, certificates = [start], []
        def visit(a, b, ga, gb, depth=0):
            if min(ga, gb) < min_clearance:
                bad = a if ga < gb else b
                raise PlanningFailure(f"joint edge blocked: {self.clearance(bad, True)}")
            # Any box vertex moves <= reach_bound * sum(|dq|). Each point
            # on a linear joint edge is within half that bound of an endpoint.
            bound = .5 * self.point_reach_bound * np.abs(b - a).sum()
            certificate = min(ga, gb) - bound
            if certificate >= min_clearance and np.max(np.abs(b - a)) <= resolution:
                path.append(b)
                certificates.append(certificate)
                return
            if depth >= 22:
                raise PlanningFailure("could not certify continuous joint-edge clearance")
            mid = (a + b) / 2
            gm = self.clearance(mid)
            visit(a, mid, ga, gm, depth + 1)
            visit(mid, b, gm, gb, depth + 1)
        visit(start, end, gap0, gap1)
        return np.asarray(path), float(min(certificates))

    def cartesian_path(self, start_q, target, yaw, *, elbow=None, step=.005, min_clearance=.002):
        start, _ = self.tip_pose(start_q)
        n = max(1, int(np.ceil(np.linalg.norm(np.asarray(target) - start) / step)))
        path, minimum = [np.asarray(start_q)], self.clearance(start_q)
        if minimum < min_clearance:
            raise PlanningFailure(f"start posture blocked: {self.clearance(start_q, True)}")
        for xyz in np.linspace(start, target, n + 1)[1:]:
            q = self.inverse(xyz, yaw, path[-1], elbow=elbow)
            edge, gap = self.check_joint_edge(path[-1], q, min_clearance=min_clearance)
            # Check the actual interpolated joint edge, not only Cartesian IK endpoints.
            for t, joint in zip(np.linspace(0, 1, len(edge)), edge):
                p, R = self.tip_pose(joint)
                a, _ = self.tip_pose(path[-1])
                if np.linalg.norm(p - ((1 - t) * a + t * xyz)) > .001:
                    raise PlanningFailure("joint interpolation deviates >1 mm from Cartesian segment")
                if np.linalg.norm(Rotation.from_matrix(tool_rotation(yaw) @ R.T).as_rotvec()) > .005:
                    raise PlanningFailure("joint interpolation violates tool orientation tolerance")
            path.extend(edge[1:])
            minimum = min(minimum, gap)
        return np.asarray(path), float(minimum)
