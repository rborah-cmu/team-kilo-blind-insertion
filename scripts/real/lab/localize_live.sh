#!/usr/bin/env bash
# Run main2's plate + hole localization (scripts/localize_live.py) on the arm WITH the live
# box-estimate window. Run on the lab User PC (iam-camilo), outside docker.
#
#   ./localize_live.sh                    live window on the lab PC desktop + live.png
#   ./localize_live.sh --no-window        no window (e.g. over SSH); watch <run>/live.png
#   ./localize_live.sh --table-z 0.0072   extra args go to scripts/localize_live.py
#
# Before: peg grasped (./grasp.sh), box in the search area, someone at the e-stop.
# Keep the arm still for the first 5 s (force zeroing). Close the window to finish.
set -u
. "$(dirname "$0")/common.sh"
# main2's localization (plus the live view) is deployed in its own folder, next to the
# teammate's deploy/localizeDeploy (left untouched).
LIVE_DIR="${KILO_LIVE_DIR:-deploy/main2Live}"
if [ ! -f "$KILO_HOST_DIR/$LIVE_DIR/scripts/localize_live.py" ]; then
    red "Not found: $KILO_HOST_DIR/$LIVE_DIR/scripts/localize_live.py"
    red "(running $( [ "$KILO_IN_CONTAINER" = 1 ] && echo inside || echo outside ) the container as $(whoami), HOME=$HOME)"
    red "Deploy main2 + live view to deploy/main2Live, or set KILO_LIVE_DIR."
    exit 1
fi
require_container
require_no_other_client

args=("$@")
# Default to the measured lab tabletop height unless one was given.
printf '%s\n' "${args[@]:-}" | grep -q -- '--table-z' || args+=(--table-z 0.0072)

# Let the container's root user draw on the desktop session (:1). Harmless if already allowed.
if printf '%s\n' "${args[@]}" | grep -q -- '--no-window'; then
    :
elif [ "$KILO_IN_CONTAINER" = 1 ]; then
    # Inside the container DISPLAY is already :1; the window works if the desktop allowed it
    # (xhost +SI:localuser:root on the lab PC). Otherwise the view falls back to live.png.
    green "Live window: trying display ${DISPLAY:-:1} (falls back to live.png if not allowed)."
elif xhost_as_user +SI:localuser:root >/dev/null 2>&1; then
    trap 'xhost_as_user -SI:localuser:root >/dev/null 2>&1' EXIT
    green "Live window: on the lab PC desktop (display ${DISPLAY:-:1})."
else
    red "No access to the lab PC desktop display; continuing with --no-window (live.png only)."
    args+=(--no-window)
fi

# mujoco (used by the joint IK) is only installed in this extra folder inside the container,
# the same one the teammate's localizeDeploy runs use. It lives in the container's /tmp,
# so it disappears if the container is recreated.
if ! PYDEPS=$(kilo_pydeps); then
    red "mujoco is not importable in the container (not in the image, nor in deploy/pydeps)."
    red "Reinstall inside the container:  pip3 install --target $KILO_DOCKER_DIR/deploy/pydeps mujoco==3.2.3"
    red "then delete deploy/pydeps/numpy* so the image's numpy 1.23.5 stays in use."
    exit 1
fi
[ -n "$PYDEPS" ] && green "mujoco from $PYDEPS" || green "mujoco from the docker image"

# The IK starts from the current joints, so a wrist near its limit makes the first move
# unreachable (2026-11-24: joint 7 at -2.84 rad -> "Joints not reachable!"). Start from home.
q=$(kcheck "$ROS_SETUP; timeout 8 rostopic echo -n1 /robot_state_publisher_node_1/robot_state 2>/dev/null" | grep -m1 '^q: ' | tr -d 'q:[] ')
if [ -n "$q" ]; then
    far=$(python3 -c "
q=[float(v) for v in '$q'.split(',')]
home=[0,-0.785,0,-2.356,0,1.571,0.785]
lo=[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]; hi=[2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]
near=[i+1 for i,(a,l,h) in enumerate(zip(q,lo,hi)) if a-l<0.25 or h-a<0.25]
off=max(abs(a-b) for a,b in zip(q,home))
print('near-limit joints %s' % near if near else ('off-home %.2f rad' % off if off > 0.35 else ''))" 2>/dev/null)
    if [ -n "$far" ]; then
        red "The arm is not at home ($far). The first move would likely be unreachable."
        red "Run ./home.sh first (shelf-checked joint move), then ./localize_live.sh."
        exit 1
    fi
fi

console="/tmp/kilo_localize_live_$(date +%Y%m%d-%H%M%S).log"
bold "Someone at the e-stop, table area clear. Ctrl-C stops the script (the e-stop stops the arm)."
# Shell-escape every argument before it goes into the container's bash -c.
qargs=$(printf '%q ' "${args[@]}")
kdocker "cd $(printf %q "$LIVE_DIR") && export PYTHONPATH=$(printf %q "$PYDEPS"):\$PYTHONPATH && python3 -u scripts/localize_live.py --execute $qargs" 2>&1 | tee "$console"
status=${PIPESTATUS[0]}

run=$(ls -1d "$KILO_HOST_DIR/$LIVE_DIR"/runs/real/*-localize 2>/dev/null | sort | tail -1)
if [ -n "$run" ]; then
    name=$(basename "$run")
    if [ "$KILO_IN_CONTAINER" = 1 ]; then
        cp "$console" "$run/console.log" 2>/dev/null || true
    else
        docker cp "$console" "$KILO_CONTAINER:$KILO_DOCKER_DIR/$LIVE_DIR/runs/real/$name/console.log" 2>/dev/null || true
    fi
    echo
    bold "Run folder: $run"
    echo "  live.png / final.png : the live box-estimate view"
    echo "  result.json, samples.npz, force-zero.csv : measurements"
    echo "Full charts + report:   docker exec -it $KILO_CONTAINER bash -c 'cd $KILO_DOCKER_DIR/$LIVE_DIR && python3 scripts/real/view_localization.py runs/real/$name'"
    echo "Replay with the window: docker exec -it $KILO_CONTAINER bash -c 'cd $KILO_DOCKER_DIR/$LIVE_DIR && python3 scripts/replay_live_view.py runs/real/$name'"
fi
exit $status
