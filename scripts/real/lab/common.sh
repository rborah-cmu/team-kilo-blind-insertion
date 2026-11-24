# Shared helpers for the Kilo lab scripts. Sourced by the other scripts; not run directly.
# Runs on the lab User PC (iam-camilo), outside docker. Everything robot-side runs inside
# the frankapy_docker container, which mounts ~/16662_RobotAutonomy/src/devel_packages.

KILO_HOST_DIR="$HOME/16662_RobotAutonomy/src/devel_packages/kilo_project"
KILO_DOCKER_DIR="/home/ros_ws/src/devel_packages/kilo_project"
KILO_CONFIG="${KILO_CONFIG:-configs/real_planned.yaml}"
KILO_CONTAINER="${KILO_CONTAINER:-frankapy_docker}"
ROS_SETUP="source /opt/ros/noetic/setup.bash; source /home/ros_ws/devel/setup.bash"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

# Run a command inside the container from the project folder (interactive when a TTY exists).
kdocker() {
    local tty=""
    [ -t 0 ] && tty="-it"
    docker exec $tty "$KILO_CONTAINER" bash -c "$ROS_SETUP; cd $KILO_DOCKER_DIR && $*" 2> >(grep -v -E 'WARNING:root|autolab_core' >&2)
}

require_container() {
    if ! docker ps --format '{{.Names}}' | grep -qx "$KILO_CONTAINER"; then
        red "Container $KILO_CONTAINER is not running. Start it the usual lab way, then retry."
        exit 1
    fi
}

# Refuse to start motion while another arm client (guide mode, another script) is connected.
require_no_other_client() {
    local n
    n=$(docker exec "$KILO_CONTAINER" bash -c "$ROS_SETUP; timeout 5 rosnode list 2>/dev/null | grep -c -E 'franka_arm_client|guide'" || true)
    if [ "${n:-0}" != "0" ]; then
        red "Another arm client is connected (guide mode or another script). Close it first."
        exit 1
    fi
    if pgrep -f "scripts/real/phase1.py" >/dev/null; then
        red "A phase1.py run is already in progress."
        exit 1
    fi
}

latest_run() {  # latest_run <command>  -> host path of newest runs/real/*-<command>
    ls -1d "$KILO_HOST_DIR"/runs/real/*-"$1" 2>/dev/null | sort | tail -1
}
