"""Robot-independent interface for touch probing with the peg.

Every backend (frankapy on the real arm, panda-py on the real arm, MuJoCo for
offline testing) implements a few primitives in terms of the PEG TIP:

    state()               -> RobotState (tip position, yaw, contact force, joints)
    move(tip, yaw, speed)    blocking free-space move (no contact detection)
    begin_stream(), stream_target(tip, yaw), tick(dt), end_stream()
                             stream Cartesian-impedance targets at a fixed rate
    gripper(width, grasp, force), close()

The tool always points straight down. ``yaw`` is the direction of the gripper's
finger-closing axis in the base frame (radians). Probing moves along that axis,
so contact pushes the peg into a finger instead of out of the grasp.

``probe`` (contact-guarded straight-line move) is written once here, so the
real arm and the simulator run the same detection logic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobotState:
    time: float
    tip: np.ndarray              # (3,) peg-tip centre, base frame, m
    yaw: float                   # finger-axis direction, rad
    force: np.ndarray            # (3,) contact force ON the peg, base frame, N (sign-corrected)
    torque: np.ndarray           # (3,) base frame, Nm
    q: np.ndarray | None = None  # (7,) joint angles
    tau_ext: np.ndarray | None = None  # (7,) external joint torques, if the backend has them


@dataclass
class ProbeResult:
    contact: bool
    tip: np.ndarray              # tip position when contact was declared (or at the end)
    force: np.ndarray            # bias-removed contact force at detection, base frame
    peak_force: float
    distance: float              # travelled along the probe line
    bias: np.ndarray
    samples: list = field(default_factory=list)  # (t, tip xyz, force xyz) per tick
    stalled: bool = False        # stopped short of the target without contact


@dataclass
class ProbeSettings:
    speed: float = 0.010             # m/s
    force_threshold: float = 5.0     # N, on |F - bias|
    debounce: int = 3                # consecutive ticks over threshold
    bias_seconds: float = 0.5        # stationary averaging before each probe
    safety_force: float = 20.0       # N: abort the probe above this
    stiffness: float = 600.0         # N/m translational stiffness of the stream
    retreat: float = 0.003           # m backed off after contact
    stream_hz: float = 50.0
    timeout: float = 60.0            # s

    min_lead: float = 0.025          # m: enough pull to overcome joint friction on the real arm

    @property
    def max_lead(self) -> float:
        """Furthest the target may run ahead of the tip. Bounds the push (stiffness x lead) even if
        detection fails: at least 2x the threshold so contact is detectable, at least ``min_lead``
        so the arm keeps moving against friction, and below the safety force."""
        lead = max(2.0 * self.force_threshold / self.stiffness, self.min_lead)
        return min(lead, 0.9 * self.safety_force / self.stiffness)


class SafetyStop(RuntimeError):
    """Raised when a probe exceeds the safety force or leaves the workspace."""


def tool_rotation(yaw: float) -> np.ndarray:
    """Tool frame pointing down (z = -world z) with its y axis (finger axis) along ``yaw``."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])


def yaw_of_tool(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 1], rotation[0, 1]))


def nearest_equivalent_yaw(yaw: float, reference: float) -> float:
    """The gripper is symmetric under a half turn: pick yaw or yaw+pi, whichever is closer."""
    return float(yaw + np.pi * np.round((reference - yaw) / np.pi))


Q7_LIMIT = 2.8973


def choose_wrist_yaw(direction_xy, state: "RobotState", target_xy=None) -> tuple[float, float | None]:
    """Finger-axis yaw along ``direction_xy`` (or its half turn, the gripper is symmetric),
    choosing the one that keeps joint 7 nearest the middle of its range at ``target_xy``.

    With the tool pointing down, measured on the Panda model (2026-11-22):
        dq7 ~ -d(yaw) + d(azimuth)
    where azimuth is the base angle to the tip. The controller rotates the shortest way.
    Returns (yaw as an angle next to the current one, predicted q7 or None).
    """
    base = float(np.arctan2(direction_xy[1], direction_xy[0]))
    here = np.asarray(state.tip[:2])
    there = here if target_xy is None else np.asarray(target_xy, float)
    d_az = float(np.arctan2(there[1], there[0]) - np.arctan2(here[1], here[0]))
    best = None
    for cand in (base, base + np.pi):
        turn = float(np.angle(np.exp(1j * (cand - state.yaw))))  # shortest rotation
        q7 = None if state.q is None else float(state.q[6] - turn + d_az)
        score = abs(q7) if q7 is not None else abs(turn)
        if best is None or score < best[0]:
            best = (score, state.yaw + turn, q7)
    return best[1], best[2]


class RobotBackend(ABC):
    """Primitives each backend implements, plus the shared guarded probe."""

    @abstractmethod
    def state(self) -> RobotState: ...

    @abstractmethod
    def move(self, tip: np.ndarray, yaw: float, speed: float) -> None:
        """Blocking free-space move of the peg tip. No contact detection."""

    @abstractmethod
    def begin_stream(self) -> None: ...

    @abstractmethod
    def stream_target(self, tip: np.ndarray, yaw: float) -> None: ...

    @abstractmethod
    def tick(self, dt: float) -> None:
        """Wait (real arm) or simulate (sim) for one stream period."""

    @abstractmethod
    def end_stream(self) -> None: ...

    def gripper(self, width: float, grasp: bool = False, force: float = 20.0) -> None:
        pass

    def guide_mode(self, seconds: float) -> None:
        raise NotImplementedError("this backend has no guide mode")

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ shared logic
    def move_and_record(self, tip, yaw, speed: float, hz: float) -> np.ndarray:
        """Free-space straight move while sampling (time, tip xyz, force xyz) at ``hz``.
        Default: streamed targets. Backends with a better native motion override this."""
        start = self.state().tip.copy()
        tip = np.asarray(tip, float)
        length = float(np.linalg.norm(tip - start))
        rows, t = [], 0.0
        self.begin_stream()
        try:
            while t * speed < length:
                self.stream_target(start + (tip - start) * min(1.0, t * speed / length), yaw)
                self.tick(1.0 / hz)
                st = self.state()
                rows.append([st.time, *st.tip, *st.force])
                t += 1.0 / hz
        finally:
            self.end_stream()
        return np.asarray(rows)

    def measure_bias(self, seconds: float, hz: float) -> np.ndarray:
        """Average contact force while holding still (removes the posture-dependent offset)."""
        n = max(1, int(seconds * hz))
        readings = []
        for _ in range(n):
            readings.append(self.state().force)
            self.tick(1.0 / hz)
        return np.mean(readings, axis=0)

    def probe(self, target: np.ndarray, yaw: float, cfg: ProbeSettings,
              stop_on_arrival: bool = True) -> ProbeResult:
        """Move the tip in a straight line toward ``target`` until contact or arrival.

        Contact = |F - bias| above ``cfg.force_threshold`` for ``cfg.debounce``
        consecutive ticks. The target never leads the tip by more than
        ``cfg.max_lead``, so the push stays bounded even if detection fails.
        On contact the tip holds, then backs off ``cfg.retreat``.
        """
        target = np.asarray(target, dtype=float)
        dt = 1.0 / cfg.stream_hz
        self.begin_stream()
        try:
            start_state = self.state()
            start = start_state.tip.copy()
            self.stream_target(start, yaw)
            bias = self.measure_bias(cfg.bias_seconds, cfg.stream_hz)
            delta = target - start
            length = float(np.linalg.norm(delta))
            direction = delta / length if length > 1e-9 else np.zeros(3)
            samples, over, peak = [], 0, 0.0
            t0, t = time.monotonic(), 0.0
            best_progress, best_time = -np.inf, 0.0
            while True:
                st = self.state()
                residual = st.force - bias
                mag = float(np.linalg.norm(residual))
                peak = max(peak, mag)
                progress = float(np.dot(st.tip - start, direction))
                samples.append((t, *st.tip.tolist(), *residual.tolist()))
                if mag > cfg.safety_force:
                    self.stream_target(st.tip, yaw)
                    raise SafetyStop(f"contact force {mag:.1f} N exceeded the {cfg.safety_force:.0f} N safety limit")
                over = over + 1 if mag > cfg.force_threshold else 0
                if over >= cfg.debounce:
                    self.stream_target(st.tip, yaw)
                    result = ProbeResult(True, st.tip.copy(), residual, peak, progress, bias, samples)
                    break
                if stop_on_arrival and length - progress < 0.001:
                    result = ProbeResult(False, st.tip.copy(), residual, peak, progress, bias, samples)
                    break
                if progress > best_progress + 0.0005:
                    best_progress, best_time = progress, t
                target_at_end = cfg.speed * t >= length
                if stop_on_arrival and target_at_end and t - best_time > 2.0:
                    # Stalled short of the target with no contact (tracking error or joint limit).
                    result = ProbeResult(False, st.tip.copy(), residual, peak, progress, bias, samples, stalled=True)
                    break
                if t > cfg.timeout:
                    raise SafetyStop(f"probe did not finish within {cfg.timeout:.0f} s")
                advance = min(cfg.speed * t, progress + cfg.max_lead, length)
                self.stream_target(start + advance * direction, yaw)
                self.tick(dt)
                t += dt
            if result.contact and cfg.retreat > 0:
                back = result.tip - direction * cfg.retreat
                for _ in range(int(0.4 * cfg.stream_hz)):
                    self.stream_target(back, yaw)
                    self.tick(dt)
            return result
        finally:
            self.end_stream()
