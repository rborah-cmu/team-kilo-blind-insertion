"""MuJoCo scene: the course's torque-controlled Panda, table and shelves, plus the
peg and the base with a hole.

The robot, table and shelves come from the Lab 3 scene
(``assets/franka/panda_table_shelves.xml``, joints driven by torque actuators
``actuator1``-``actuator7``, gripper by ``actuator8``). This module adds:

- the base (``base`` body): a block fixed to the table, built from convex wedges
  around its through-hole (see ``geometry.py``), with a ``hole_top`` site at the
  centre of the hole's top opening;
- the peg (``peg`` body): a box for the square top and a convex chamfered shaft,
  with a ``peg_tip`` site at the centre of the tip face. It is either fixed in
  the gripper (``peg_mode="hand"``), a free body (``"free"``), or absent.

Frames: the peg frame's origin is where the square top meets the shaft, +z
toward the top. The base frame's origin is the centre of its bottom face.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from kilo.config import load_config
from kilo.sim.geometry import base_wedges, peg_shaft_vertices

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "franka"
ROBOT_XML = ASSET_DIR / "panda_table_shelves.xml"

TABLE_TOP_Z = 0.0001          # top of the Lab 3 TablePlane
HAND_TCP_Z = 0.1034           # fingertip centre along the hand's z axis
FINGER_BODIES = ("left_finger", "right_finger")


def _fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in np.ravel(values))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def build_scene_xml(
    cfg: dict | None = None,
    peg_mode: str = "hand",
    box_xy: tuple[float, float] = (0.5, 0.0),
    box_yaw: float = 0.0,
    hole_offset: tuple[float, float] | None = None,
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    peg_pos: tuple[float, float, float] = (0.5, 0.0, 0.2),
    n_wedges: int = 48,
) -> str:
    """Return the scene as an MJCF string.

    Args:
        cfg: configuration dict (defaults to configs/default.yaml).
        peg_mode: "hand" (fixed in the gripper), "free" (free body) or "none".
        box_xy, box_yaw: pose of the base's centre on the table.
        hole_offset: hole centre relative to the base centre, in the base frame;
            defaults to the CAD offset ``hole.cad_offset``.
        grasp_offset: peg offset in the hand frame from the nominal grasp
            (for randomizing the grasp in training).
        peg_pos: world position of the peg origin when ``peg_mode="free"``.
        n_wedges: number of convex wedges around the hole.
    """
    cfg = cfg or load_config()
    peg, box, hole = cfg["peg"], cfg["box"], cfg["hole"]
    if peg_mode not in ("hand", "free", "none"):
        raise ValueError(f"unknown peg_mode {peg_mode!r}")
    offset = hole["cad_offset"] if hole_offset is None else hole_offset

    xml = ROBOT_XML.read_text(encoding="utf-8")
    xml = xml.replace('meshdir="meshes"', f'meshdir="{(ASSET_DIR / "meshes").as_posix()}"')
    xml = xml.replace(
        '<option integrator="implicit" />',
        '<option integrator="implicit" timestep="0.001">\n    <flag multiccd="enable" />\n  </option>\n'
        '  <visual><global offwidth="1280" offheight="960" /></visual>',
    )

    # ---- meshes ----
    shaft = peg_shaft_vertices(peg["shaft_radius"], peg["shaft_length"], peg["chamfer"])
    wedges = base_wedges(box["size"], offset, hole["radius"], hole["chamfer"], n=n_wedges)
    meshes = [f'    <mesh name="peg_shaft" vertex="{_fmt(shaft)}" />']
    meshes += [f'    <mesh name="base_wedge{i}" vertex="{_fmt(w)}" />' for i, w in enumerate(wedges)]
    xml = xml.replace("</asset>", "\n".join(meshes) + "\n  </asset>")

    # ---- peg ----
    top_half = np.array(peg["top_size"]) / 2.0
    peg_geoms = (
        f'<geom name="peg_top" type="box" size="{_fmt(top_half)}" pos="0 0 {top_half[2]:.6g}" '
        f'rgba="0.62 0.81 0.93 1" contype="1" conaffinity="1" solref="0.002 1" condim="4" />'
        f'<geom name="peg_shaft" type="mesh" mesh="peg_shaft" rgba="0.62 0.81 0.93 1" '
        f'contype="1" conaffinity="1" solref="0.002 1" condim="4" />'
        f'<site name="peg_tip" pos="0 0 {-peg["shaft_length"]:.6g}" size="0.002" rgba="1 0 0 1" />'
    )
    peg_inertial = f'<inertial pos="0 0 0" mass="{peg["mass"]:.6g}" diaginertia="5e-6 5e-6 2e-6" />'
    if peg_mode == "hand":
        # Peg +z points back toward the flange (opposite the hand's z axis); the
        # square top's centre sits at the fingertip centre.
        pos = np.array([0.0, 0.0, HAND_TCP_Z + top_half[2]]) + np.asarray(grasp_offset)
        peg_body = (f'<body name="peg" pos="{_fmt(pos)}" quat="0 1 0 0">'
                    f'{peg_inertial}{peg_geoms}</body>\n                      ')
        xml = xml.replace('<body name="left_finger"', peg_body + '<body name="left_finger"', 1)
    elif peg_mode == "free":
        peg_body = (f'<body name="peg" pos="{_fmt(peg_pos)}"><freejoint name="peg_free" />'
                    f'{peg_inertial}{peg_geoms}</body>')
    # ---- base ----
    base_geoms = "".join(
        f'<geom name="base_wedge{i}" type="mesh" mesh="base_wedge{i}" rgba="0.85 0.65 0.35 1" '
        f'contype="1" conaffinity="1" solref="0.002 1" condim="4" />'
        for i in range(len(wedges))
    )
    base_body = (
        f'<body name="base" pos="{box_xy[0]:.6g} {box_xy[1]:.6g} {TABLE_TOP_Z:.6g}" '
        f'quat="{_fmt(_yaw_quat(box_yaw))}">{base_geoms}'
        f'<site name="hole_top" pos="{offset[0]:.6g} {offset[1]:.6g} {box["size"][2]:.6g}" '
        f'size="0.002" rgba="0 1 0 1" /></body>'
    )
    world_extra = base_body + (peg_body if peg_mode == "free" else "")
    xml = xml.replace("</worldbody>", f"    {world_extra}\n  </worldbody>")

    # The peg is fixed in the hand, so finger contacts with it would only fight
    # the weld; the gripper is closed on it visually.
    if peg_mode == "hand":
        excludes = "".join(f'<exclude body1="peg" body2="{b}" />' for b in FINGER_BODIES)
        xml = xml.replace("</mujoco>", f"  <contact>{excludes}</contact>\n</mujoco>")
    return xml


def build_scene(cfg: dict | None = None, seed: int | None = None, **kwargs) -> mujoco.MjModel:
    """Compile the scene; keyword arguments are passed to :func:`build_scene_xml`.

    With ``seed``, the base pose comes from ``sample_box_pose(cfg, seed=seed)``,
    so the same seed always builds the same scene; don't also pass ``box_xy``
    or ``box_yaw``.
    """
    if seed is not None:
        if "box_xy" in kwargs or "box_yaw" in kwargs:
            raise ValueError("pass either seed or box_xy/box_yaw, not both")
        kwargs["box_xy"], kwargs["box_yaw"] = sample_box_pose(cfg, seed=seed)
    return mujoco.MjModel.from_xml_string(build_scene_xml(cfg, **kwargs))


def site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def sample_box_pose(
    cfg: dict | None = None, margin: float = 0.03, seed: int | None = None
) -> tuple[np.ndarray, float]:
    """Random base pose whose whole footprint lies inside the snake-search rectangle.

    The yaw is uniform over a full turn (the off-centre hole makes every angle
    distinct). The centre is then uniform over the positions that keep the
    rotated block at least ``margin`` inside ``snake_search.corners``. The same
    ``seed`` always gives the same pose; ``None`` gives a fresh one each call.

    Returns:
        (box_xy, box_yaw), ready for ``build_scene(cfg, box_xy=..., box_yaw=...)``.
    """
    cfg = cfg or load_config()
    corners = np.asarray(cfg["snake_search"]["corners"], dtype=float)
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    hx, hy = cfg["box"]["size"][0] / 2.0, cfg["box"]["size"][1] / 2.0

    rng = np.random.default_rng(seed)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    half_extent = np.array([hx * c + hy * s, hx * s + hy * c])  # rotated footprint
    lo, hi = lower + margin + half_extent, upper - margin - half_extent
    if np.any(lo > hi):
        raise ValueError("block does not fit inside the search rectangle with this margin")
    return rng.uniform(lo, hi), float(yaw)
