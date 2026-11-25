"""Run the shared plate localization search using the physical Franka."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kilo.localization.common import SNAKE_SPACING, TABLETOP_HEIGHT
from kilo.localization.plate import localize_plate
from kilo.localization.search import generate_snake_waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-num", type=int, default=1)
    parser.add_argument("--table-z", type=float, default=TABLETOP_HEIGHT,
                        help="tabletop height in robot base metres (default: Lab 2 scene)")
    parser.add_argument("--execute", action="store_true", help="send motion commands to the arm")
    args = parser.parse_args()

    waypoints = generate_snake_waypoints(SNAKE_SPACING)
    if not args.execute:
        print(f"Dry run. The sweep has {len(waypoints)} waypoints at table z={args.table_z:g} m.")
        return

    from frankapy import FrankaArm
    from kilo.robot.frankapy_localization import FrankaLocalizer

    arm = FrankaArm(robot_num=args.robot_num, offline=False)
    robot = FrankaLocalizer(arm, args.table_z)
    try:
        zero_log = Path(f"force-zero-{datetime.now():%Y%m%d-%H%M%S-%f}.csv")
        robot.zero_forces(zero_log)
        print(f"Raw force zero samples saved to {zero_log}", flush=True)
        robot.position_at(waypoints[0])
        center, angle = localize_plate(waypoints, robot.probe, robot.travel)
    except BaseException:
        try:
            arm.stop_skill()
        except Exception as error:
            print(f"Arm stop request failed: {error}", file=sys.stderr)
        raise
    print(f"Estimated plate center ({center[0]:.4f}, {center[1]:.4f}) m, yaw {angle:.1f} deg (mod 180)")


if __name__ == "__main__":
    main()
