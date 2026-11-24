#!/usr/bin/env bash
# Run the full Phase 1 localization on the arm, keep a console log, then make the charts.
#   ./localize.sh                          normal run (asks you to type RUN)
#   ./localize.sh --verify-rounds 1        extra args go to phase1.py localize
#   ./localize.sh --truth runs/real/groundtruth.json   score against measured truth
# Before: arm at home (./home.sh), peg gripped (./grasp.sh), box in the search area,
# someone at the e-stop. Outputs: runs/real/<stamp>-localize/{result.json,samples.npz,
# console.log,charts/*.png,charts/report.txt}
set -u
. "$(dirname "$0")/common.sh"
require_container
require_no_other_client

console="/tmp/kilo_localize_$(date +%Y%m%d-%H%M%S).log"
bold "Someone at the e-stop, table area clear. Ctrl-C stops the script (the e-stop stops the arm)."
kdocker "python3 -u scripts/real/phase1.py localize --config $KILO_CONFIG --execute $*" 2>&1 | tee "$console"
status=${PIPESTATUS[0]}

run=$(latest_run localize)
if [ -z "$run" ]; then
    red "No run folder was written; console log: $console"
    exit 1
fi
# runs/real is owned by the container's root user, so copy through docker.
docker cp "$console" "$KILO_CONTAINER:$KILO_DOCKER_DIR/runs/real/$(basename "$run")/console.log" 2>/dev/null \
  || echo "(console log kept at $console)"

echo
bold "== Charts and report"
"$(dirname "$0")/view.sh" "$run"
grep -q "SAFETY STOP" "$console" && red "The run ended with a SAFETY STOP - see report.txt and charts/5_motion.png."
exit $status
