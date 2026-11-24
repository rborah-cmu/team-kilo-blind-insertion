#!/usr/bin/env bash
# Move the arm to frankapy home (one 8 s joint move). Checks the joint path against the shelf
# model first and asks you to type RUN.   Usage: ./home.sh
set -eu
. "$(dirname "$0")/common.sh"
require_container
require_no_other_client
bold "Someone at the e-stop, table area clear."
kdocker "python3 scripts/real/phase1.py home --config $KILO_CONFIG --execute"
