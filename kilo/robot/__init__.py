"""Robot backends for touch probing: frankapy and panda-py (real arm), MuJoCo (offline)."""

from .interface import (ProbeResult, ProbeSettings, RobotBackend, RobotState, SafetyStop,
                        nearest_equivalent_yaw, tool_rotation, yaw_of_tool)


def make_backend(name: str, cfg: dict, **kwargs) -> RobotBackend:
    """Create a backend by name: 'sim', 'frankapy' or 'pandapy'. Imports lazily so the
    real-robot libraries are only needed when that backend is used."""
    if name == "sim":
        from .sim_backend import SimBackend
        return SimBackend(cfg, **kwargs)
    if name == "frankapy":
        from .frankapy_backend import FrankaPyBackend
        return FrankaPyBackend(cfg, **kwargs)
    if name == "pandapy":
        from .pandapy_backend import PandaPyBackend
        return PandaPyBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["ProbeResult", "ProbeSettings", "RobotBackend", "RobotState", "SafetyStop",
           "make_backend", "nearest_equivalent_yaw", "tool_rotation", "yaw_of_tool"]
