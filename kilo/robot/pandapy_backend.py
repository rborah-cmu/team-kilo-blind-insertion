"""panda-py backend (EXPERIMENTAL, untested): direct libfranka with a 1 kHz control loop.

Why consider it: libfranka's torque loop runs at 1 kHz in C++, the Python side
can update targets and read state every millisecond, and the full robot state
is exposed, including the per-joint external torques (tau_ext_hat_filtered)
that frankapy hides. Contact is detected ~20x sooner than at frankapy's 50 Hz.

Why it is not the default:
- libfranka allows one controlling client, so frankapy's Control PC server must
  be stopped first, and panda-py must run on a machine with a direct, low-latency
  link to the robot (normally the Control PC);
- the panda-py build must match the robot's system version (`pip install
  panda-python` ships libfranka 0.9.2, for Panda system >= 4.2.1);
- installing software on the lab machines needs the TAs' approval.

API names below are taken from panda-py's README/examples (Panda(hostname),
controllers.CartesianImpedance, start_controller, create_context, set_control,
get_state); others (get_pose, move_to_pose, stop_controller) follow its C++
method names and must be checked against the installed version.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .interface import RobotBackend, RobotState, tool_rotation, yaw_of_tool


class PandaPyBackend(RobotBackend):
    preferred_hz = 1000.0

    def __init__(self, cfg: dict, hostname: str | None = None):
        import panda_py
        from panda_py import controllers

        rc = cfg["robot"]
        self._controllers = controllers
        self.panda = panda_py.Panda(hostname or rc["pandapy_hostname"])
        self.probe_length = float(rc["probe_length"])
        self.force_sign = float(rc.get("force_sign", 1.0))
        self.ctrl = self.ctx = None

    def _tool(self, tip, yaw):
        R = tool_rotation(yaw)
        return np.asarray(tip, float) - R[:, 2] * self.probe_length, Rotation.from_matrix(R).as_quat()  # xyzw

    def state(self) -> RobotState:
        s = self.panda.get_state()
        T = np.asarray(self.panda.get_pose(), dtype=float)
        F = np.asarray(s.O_F_ext_hat_K, dtype=float)
        return RobotState(time=float(s.time.to_sec()) if hasattr(s.time, "to_sec") else 0.0,
                          tip=T[:3, 3] + T[:3, 2] * self.probe_length, yaw=yaw_of_tool(T[:3, :3]),
                          force=self.force_sign * F[:3], torque=self.force_sign * F[3:],
                          q=np.asarray(s.q, dtype=float), tau_ext=np.asarray(s.tau_ext_hat_filtered, dtype=float))

    def move(self, tip, yaw, speed) -> None:
        position, quat = self._tool(tip, yaw)
        self.panda.move_to_pose(position, quat, speed_factor=min(0.2, speed / 0.25))

    def begin_stream(self) -> None:
        self.ctrl = self._controllers.CartesianImpedance()
        self.panda.start_controller(self.ctrl)
        self.ctx = self.panda.create_context(frequency=self.preferred_hz)

    def stream_target(self, tip, yaw) -> None:
        position, quat = self._tool(tip, yaw)
        self.ctrl.set_control(position, quat)

    def tick(self, dt: float) -> None:
        if self.ctx is None:
            import time
            time.sleep(dt)
        elif not self.ctx.ok():
            raise RuntimeError("panda-py control context stopped")

    def end_stream(self) -> None:
        if self.ctrl is not None:
            self.panda.stop_controller()
        self.ctrl = self.ctx = None

    def close(self) -> None:
        self.end_stream()
