#!/usr/bin/env bash
# Read-only robot status: container, Control PC server, controller errors, pose, gripper,
# other arm clients. Never moves the robot.   Usage: ./status.sh
set -u
. "$(dirname "$0")/common.sh"
require_container

bold "== Control PC (iam-sneezy) frankapy server"
if ssh -o BatchMode=yes -o ConnectTimeout=5 student@iam-sneezy "pgrep -f franka_interface >/dev/null"; then
    green "franka_interface running"
else
    red "franka_interface NOT running: start it with  cd ~/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-sneezy"
fi

bold "== Robot state"
docker exec "$KILO_CONTAINER" bash -c "$ROS_SETUP; timeout 8 rostopic echo -n1 /robot_state_publisher_node_1/robot_state 2>/dev/null" > /tmp/kilo_state.txt
if [ ! -s /tmp/kilo_state.txt ]; then
    red "no robot_state message (server down, or robot locked / e-stop pressed?)"
else
    grep -E '^q: |^robot_mode: ' /tmp/kilo_state.txt
    echo "robot_mode: 1 = idle (ok), 2 = moving/guiding, 4 = reflex/error, 5 = user stopped (e-stop)"
    active=$(sed -n '/^current_errors:/,/^last_motion_errors:/p' /tmp/kilo_state.txt | grep ': True' || true)
    last=$(sed -n '/^last_motion_errors:/,/^[a-z_]*: *[^T ]/p' /tmp/kilo_state.txt | grep ': True' || true)
    if [ -n "$active" ]; then red "ACTIVE controller errors:"; echo "$active"; else green "no active controller errors"; fi
    [ -n "$last" ] && { echo "last motion errors (history, cleared on next motion):"; echo "$last"; }
fi

bold "== Gripper"
docker exec "$KILO_CONTAINER" bash -c "$ROS_SETUP; timeout 6 rostopic echo -n1 /franka_gripper_1/joint_states 2>/dev/null" \
  | awk '/position:/ {gsub(/[\[\],]/," "); printf "finger gap %.1f mm (peg gripped reads about 24 mm, fully open 80 mm)\n", ($2+$3)*1000}'

bold "== Peg tip and planner posture"
kdocker "python3 scripts/real/phase1.py check --config $KILO_CONFIG --seconds 1" 2>/dev/null \
  | tr -d '\n ' | grep -o -E '"(tip|q)":\[[^]]*\]' | head -2 | sed 's/^/  /'
echo "  (tip = peg tip in the robot base frame, m; q = joint angles, rad)"

bold "== Other arm clients"
n=$(docker exec "$KILO_CONTAINER" bash -c "$ROS_SETUP; timeout 5 rosnode list 2>/dev/null | grep -c -E 'franka_arm_client|guide'" || true)
[ "${n:-0}" = "0" ] && green "none" || red "$n other client(s) connected"
