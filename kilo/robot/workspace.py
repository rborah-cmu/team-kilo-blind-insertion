"""Conservative planar hand clearance; frankapy's full-arm walls remain enabled."""

import json
from pathlib import Path

import numpy as np

from .interface import SafetyStop


def load_shelf_boxes(workspace):
    """Franka-compatible [xyz,rpy,full dimensions] boxes from the Lab 2 reference."""
    name = workspace.get("shelf_bounds_file")
    if name is None:
        return []
    path = Path(name)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    margin = float(workspace.get("shelf_margin", 0.01))
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("invalid shelf margin")
    boxes = []
    for item in json.loads(path.read_text())["boxes"]:
        box = np.asarray(item["box"], float)
        if box.shape != (9,) or not np.all(np.isfinite(box)) or np.any(box[6:] <= 0):
            raise ValueError("invalid shelf obstacle box")
        box[6:] += 2 * margin
        boxes.append(box)  # frankapy slices dimensions and divides them by two
    if not boxes:
        raise ValueError("shelf obstacle file contains no boxes")
    return boxes


class HandWorkspace:
    def __init__(self, workspace):
        limits = workspace.get("hand_wall_limits")
        self.enabled = limits is not None
        if not self.enabled:
            return
        self.lo = np.array([limits["x"][0], limits["y"][0]], float)
        self.hi = np.array([limits["x"][1], limits["y"][1]], float)
        # Conservative at every experiment height: keep the complete hand out of
        # both shelf columns, including shelf undersides and supports.
        shelf_y = workspace.get("shelf_inner_y_limits")
        if shelf_y is not None:
            self.lo[1] = max(self.lo[1], float(shelf_y[0]))
            self.hi[1] = min(self.hi[1], float(shelf_y[1]))
        # Half extents measured from the peg axis, not full hand dimensions.
        self.along, self.across = np.asarray(workspace["hand_half_extents"], float)
        self.margin = float(workspace.get("hand_wall_margin", 0.01))
        if (not np.all(np.isfinite([*self.lo, *self.hi, self.along, self.across, self.margin]))
                or np.any(self.lo >= self.hi) or min(self.along, self.across) <= 0 or self.margin < 0):
            raise ValueError("invalid hand wall limits, half extents, or margin")

    def check(self, start_xy, end_xy, start_yaw, end_yaw):
        """Enclose translation and the entire shortest wrist rotation in a safe rectangle.

        This deliberately overbounds simultaneous translation/rotation. It does not
        model elbow links or vertical obstacles; the course collision checks still apply.
        """
        if not self.enabled:
            return
        start, end = np.asarray(start_xy, float), np.asarray(end_xy, float)
        if not np.all(np.isfinite([*start, *end, start_yaw, end_yaw])):
            raise SafetyStop("non-finite hand workspace target")
        turn = float(np.angle(np.exp(1j * (end_yaw - start_yaw))))
        low, high = sorted((start_yaw, start_yaw + turn))
        angles = [low, high]
        # Extrema of a|cos(yaw)|+b|sin(yaw)| and its y counterpart.
        for offset in (np.arctan2(self.across, self.along), np.arctan2(self.along, self.across)):
            for sign in (-1, 1):
                first = int(np.ceil((low - sign * offset) / np.pi))
                last = int(np.floor((high - sign * offset) / np.pi))
                angles.extend(sign * offset + k * np.pi for k in range(first, last + 1))
        c, s = np.abs(np.cos(angles)), np.abs(np.sin(angles))
        extent = np.array([(self.along * c + self.across * s).max(),
                           (self.along * s + self.across * c).max()]) + self.margin
        if np.any(np.minimum(start, end) - extent < self.lo) or np.any(np.maximum(start, end) + extent > self.hi):
            raise SafetyStop(f"hand footprint would cross a workspace or shelf boundary: "
                             f"{np.round(start, 3).tolist()} -> {np.round(end, 3).tolist()}, "
                             f"yaw {np.degrees(start_yaw):.1f} -> {np.degrees(end_yaw):.1f} deg")
