#!/usr/bin/env bash
# Start the course container from frankapy_docker:kilo (= frankapy_docker:latest + MuJoCo,
# built from docker/frankapy_kilo.Dockerfile). Same container name, mounts and options as
# ~/16662_RobotAutonomy/run_docker.sh, which is left unchanged; only the image differs.
#
# Rebuild the image if it is missing:
#   docker build -t frankapy_docker:kilo - < docker/frankapy_kilo.Dockerfile
set -e
COURSE_DIR="${COURSE_DIR:-$HOME/16662_RobotAutonomy}"
IMAGE="${KILO_IMAGE:-frankapy_docker:kilo}"
cd "$COURSE_DIR"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build it (see the header of this script) or use ./run_docker.sh."
    exit 1
fi
xhost +SI:localuser:root >/dev/null 2>&1 || true
docker container prune -f
docker run --privileged --rm -it \
    --name="frankapy_docker" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTH:$XAUTH" \
    --network host \
    -v "$(pwd)/src/devel_packages:/home/ros_ws/src/devel_packages" \
    -v "$(pwd)/data:/home/ros_ws/data" \
    -v "$(pwd)/guide_mode.py:/home/ros_ws/guide_mode.py" \
    -v "$(pwd)/reset_joints.py:/home/ros_ws/reset_joints.py" \
    -v "/etc/timezone:/etc/timezone:ro" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    --gpus all \
    "$IMAGE" bash
