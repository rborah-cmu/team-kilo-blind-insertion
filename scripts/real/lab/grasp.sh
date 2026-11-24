#!/usr/bin/env bash
# Gripper: open, then (after you place the peg by its square top) close at 40 N.
#   ./grasp.sh          open, wait for you, close
#   ./grasp.sh open     open only
#   ./grasp.sh close    close only (peg already between the fingers, hands clear)
set -eu
. "$(dirname "$0")/common.sh"
stage="${1:-both}"
require_container
require_no_other_client
kdocker "python3 scripts/real/phase1.py grasp --stage $stage --config $KILO_CONFIG --execute"
echo "A seated peg reads about 24 mm. Check with ./status.sh if unsure."
