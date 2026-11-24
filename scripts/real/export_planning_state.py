"""Export installed frankapy geometry and a stationary snapshot. Never commands motion."""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import rosnode
    from frankapy import FrankaConstants as FC
    from kilo.robot.frankapy_backend import FrankaPyBackend

    if any("franka_arm_client" in n for n in rosnode.get_node_names()):
        raise RuntimeError("another arm client is present")
    cfg = yaml.safe_load((ROOT / "configs/real.yaml").read_text())
    robot = FrankaPyBackend(cfg)
    try:
        st = robot.state()
        model = {name: np.asarray(getattr(FC, name)).tolist() for name in (
            "DH_PARAMS", "COLLISION_BOX_LINKS", "COLLISION_BOX_POSES", "COLLISION_BOX_SHAPES",
            "JOINT_LIMITS_MIN", "JOINT_LIMITS_MAX", "WORKSPACE_WALLS")}
        raw = robot.fa._state_client._get_current_robot_state().robot_state
        result = dict(
            source="installed lab frankapy.FrankaConstants and read-only robot snapshot",
            exported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            geometry=model,
            tool_delta=robot.fa._tool_delta_pose.matrix.tolist(),
            probe_length=robot.probe_length,
            q=st.q.tolist(), tip=st.tip.tolist(), yaw=st.yaw,
            tool_pose=robot.fa.get_pose().matrix.tolist(),
            fk_reference=robot.fa.get_links_transforms(st.q).tolist(),
            collision_poses_reference=np.asarray(robot.fa.get_collision_boxes_poses(st.q)).tolist(),
            robot_mode=int(raw.robot_mode),
            current_errors=[n for n in raw.current_errors.__slots__ if getattr(raw.current_errors, n)],
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(out=str(out), q=result["q"], tip=result["tip"],
                              robot_mode=result["robot_mode"], current_errors=result["current_errors"])))
    finally:
        robot.close()


if __name__ == "__main__":
    main()
