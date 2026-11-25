# Shared helpers for the Kilo lab scripts. Sourced by the other scripts; not run directly.
# Works from the lab User PC (iam-camilo) terminal OR from a shell inside the
# frankapy_docker container (e.g. the run_docker.sh terminal). Robot-side commands always
# run inside the container, which mounts ~/16662_RobotAutonomy/src/devel_packages.

KILO_DOCKER_DIR="/home/ros_ws/src/devel_packages/kilo_project"
# The project folder is found from this file's own location (…/kilo_project/scripts/real/lab),
# so it works for any user, with or without sudo, and inside the container.
KILO_HOST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f /.dockerenv ] || ! command -v docker >/dev/null 2>&1; then
    KILO_IN_CONTAINER=1            # already inside frankapy_docker: run commands directly
else
    KILO_IN_CONTAINER=0
fi
KILO_CONFIG="${KILO_CONFIG:-configs/real_planned.yaml}"
KILO_CONTAINER="${KILO_CONTAINER:-frankapy_docker}"
ROS_SETUP="source /opt/ros/noetic/setup.bash; source /home/ros_ws/devel/setup.bash"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

# Run a command inside the container from the project folder (interactive when a TTY exists).
kdocker() {
    if [ "$KILO_IN_CONTAINER" = 1 ]; then
        bash -c "$ROS_SETUP; cd $KILO_DOCKER_DIR && $*" 2> >(grep -v -E 'WARNING:root|autolab_core' >&2)
        return
    fi
    local tty=""
    [ -t 0 ] && tty="-it"
    docker exec $tty "$KILO_CONTAINER" bash -c "$ROS_SETUP; cd $KILO_DOCKER_DIR && $*" 2> >(grep -v -E 'WARNING:root|autolab_core' >&2)
}

# Run a short non-interactive command in the container (used for checks).
kcheck() {
    if [ "$KILO_IN_CONTAINER" = 1 ]; then
        bash -c "$*"
    else
        docker exec "$KILO_CONTAINER" bash -c "$*"
    fi
}

require_container() {
    [ "$KILO_IN_CONTAINER" = 1 ] && return 0
    if ! docker ps --format '{{.Names}}' | grep -qx "$KILO_CONTAINER"; then
        red "Container $KILO_CONTAINER is not running. Start it the usual lab way, then retry."
        exit 1
    fi
}

# Refuse to start motion while another arm client (guide mode, another script) is connected.
require_no_other_client() {
    local n
    n=$(kcheck "$ROS_SETUP; timeout 5 rosnode list 2>/dev/null | grep -c -E 'franka_arm_client|guide'" || true)
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
