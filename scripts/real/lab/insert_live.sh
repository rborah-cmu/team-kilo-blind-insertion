#!/usr/bin/env bash
# Spiral insertion of the peg (main's scripted baseline, adapted for the real arm) with the
# live view. Run on the lab User PC or inside the container.
#
#   ./insert_live.sh                         use the latest localize run's estimate
#   ./insert_live.sh --from-run <run dir>    a specific localize run (deploy/main2Live/runs/real/...)
#   ./insert_live.sh --hole 0.5119 -0.0685   explicit hole estimate (m, robot base)
#   ./insert_live.sh --localize              run the plate localization first
#   add --retract to lift straight up afterwards, --no-window over SSH
#
# Before: peg grasped, someone at the e-stop. Keep the arm still for the first 5 s.
set -u
. "$(dirname "$0")/common.sh"
LIVE_DIR="${KILO_LIVE_DIR:-deploy/main2Live}"
if [ ! -f "$KILO_HOST_DIR/$LIVE_DIR/scripts/insert_live.py" ]; then
    red "Not found: $KILO_HOST_DIR/$LIVE_DIR/scripts/insert_live.py (deploy it, or set KILO_LIVE_DIR)"
    exit 1
fi
require_container
require_no_other_client

args=("$@")
joined=" ${args[*]:-} "
case "$joined" in *" --table-z "*) ;; *) args+=(--table-z 0.0072) ;; esac
case "$joined" in
    *" --hole "*|*" --from-run "*|*" --localize "*) ;;
    *)
        latest=$(ls -1d "$KILO_HOST_DIR/$LIVE_DIR"/runs/real/*-localize 2>/dev/null | sort | tail -1)
        if [ -z "$latest" ]; then
            red "No localize run found; pass --hole X Y or --localize."
            exit 1
        fi
        args+=(--from-run "runs/real/$(basename "$latest")")
        green "Hole estimate from the latest localize run: $(basename "$latest")"
        ;;
esac

PYDEPS="${KILO_PYDEPS:-/tmp/kilo-localize-pydeps-20261124}"
if ! kcheck "PYTHONPATH=$PYDEPS python3 -c 'import mujoco'" >/dev/null 2>&1; then
    red "mujoco is not importable in the container (expected under $PYDEPS)."
    exit 1
fi

# A wrist at its limit makes the first joint move unreachable: home first in that case.
q=$(kcheck "$ROS_SETUP; timeout 8 rostopic echo -n1 /robot_state_publisher_node_1/robot_state 2>/dev/null" | grep -m1 '^q: ' | tr -d 'q:[] ')
if [ -n "$q" ]; then
    near=$(python3 -c "
q=[float(v) for v in '$q'.split(',')]
lo=[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]; hi=[2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]
print(' '.join(str(i+1) for i,(a,l,h) in enumerate(zip(q,lo,hi)) if a-l<0.25 or h-a<0.25))" 2>/dev/null)
    if [ -n "$near" ]; then
        red "Joint(s) $near are near their limits; run ./home.sh first."
        exit 1
    fi
fi

case "$joined" in
    *" --no-window "*) ;;
    *)
        if [ "$KILO_IN_CONTAINER" != 1 ]; then
            { if [ -n "${SUDO_USER:-}" ]; then sudo -u "$SUDO_USER" DISPLAY="${DISPLAY:-:1}" xhost +local:root;
              else DISPLAY="${DISPLAY:-:1}" xhost +local:root; fi; } >/dev/null 2>&1 \
              || { red "No desktop display access; using --no-window (live.png only)."; args+=(--no-window); }
        fi
        ;;
esac

console="/tmp/kilo_insert_live_$(date +%Y%m%d-%H%M%S).log"
bold "Someone at the e-stop, table area clear. Ctrl-C stops the script (the e-stop stops the arm)."
kdocker "cd $LIVE_DIR && export PYTHONPATH=$PYDEPS:\$PYTHONPATH && python3 -u scripts/insert_live.py --execute ${args[*]}" 2>&1 | tee "$console"
status=${PIPESTATUS[0]}

run=$(ls -1d "$KILO_HOST_DIR/$LIVE_DIR"/runs/real/*-insert 2>/dev/null | sort | tail -1)
if [ -n "$run" ]; then
    if [ "$KILO_IN_CONTAINER" = 1 ]; then cp "$console" "$run/console.log" 2>/dev/null || true
    else docker cp "$console" "$KILO_CONTAINER:$KILO_DOCKER_DIR/$LIVE_DIR/runs/real/$(basename "$run")/console.log" 2>/dev/null || true; fi
    echo
    bold "Run folder: $run"
    echo "  live.png / final.png : spiral, tip path and forces; insertion.npz : every tick; result.json : phases and outcome"
fi
exit $status
