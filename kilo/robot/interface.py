"""Abstract robot interface so Phase 1 and Phase 2 code runs on the real Franka and in sim.

Implement a subclass per backend: one wrapping the course control stack on the
real robot, one wrapping the MuJoCo scene. ``guarded_move`` is written once on
top of the abstract methods.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from kilo.localization.archive.contact import ContactDetector


@dataclass
class RobotState:
    q: np.ndarray          # joint positions (7,)
    dq: np.ndarray         # joint velocities (7,)
    tau_ext: np.ndarray    # estimated external joint torques (7,)
    tip_pos: np.ndarray    # peg tip position (3,) in the base frame, from forward kinematics


class RobotInterface(ABC):
    @abstractmethod
    def get_state(self) -> RobotState: ...

    @abstractmethod
    def set_tip_target(self, position: np.ndarray) -> None:
        """Command a Cartesian impedance target for the peg tip (orientation held fixed)."""

    @abstractmethod
    def set_stiffness(self, translational: float, rotational: float) -> None: ...

    @abstractmethod
    def grasp(self, width: float, force: float) -> None: ...

    @abstractmethod
    def release(self) -> None: ...

    def guarded_move(
        self,
        direction: np.ndarray,
        max_distance: float,
        speed: float,
        detector: ContactDetector,
        rate_hz: float = 100.0,
    ) -> tuple[bool, np.ndarray]:
        """Move the tip along ``direction`` until contact or ``max_distance``.

        Returns (contacted, tip position at stop). Not yet tested on hardware.
        """
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        start = self.get_state().tip_pos.copy()
        detector.reset()
        dt = 1.0 / rate_hz
        travelled = 0.0
        while travelled < max_distance:
            travelled = min(travelled + speed * dt, max_distance)
            self.set_tip_target(start + travelled * direction)
            time.sleep(dt)
            state = self.get_state()
            if detector.update(state.tau_ext):
                self.set_tip_target(state.tip_pos)  # stop where we are
                return True, state.tip_pos.copy()
        return False, self.get_state().tip_pos.copy()
