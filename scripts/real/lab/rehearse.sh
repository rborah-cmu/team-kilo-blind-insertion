#!/usr/bin/env bash
# Offline rehearsal on the lab PC: the full localization through the real executor against a
# virtual robot (no ROS, no motion). Use it after any code change, before touching the arm.
#   ./rehearse.sh                 nominal box pose
#   ./rehearse.sh nominal variant1 no_contact     named scenarios
set -u
. "$(dirname "$0")/common.sh"
require_container
cases="${*:-nominal}"
kdocker "python3 scripts/real/test_joint_sequence.py --config $KILO_CONFIG \
    --scenarios offline_inputs/report.json --snapshot offline_inputs/planning_state.json \
    --out /tmp/kilo_rehearsal --cases $cases" 2>&1 | grep -E '^\{|fit after|Error|Stop' \
  | python3 -c '
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        print(line); continue
    d = json.loads(line)
    if "label" in d:
        mark = "PASS" if d["passed"] else "FAIL"
        extra = "  (%s)" % d["failure"] if d.get("failure") else ""
        print("%s %s: %s skills, %.0f s simulated%s" % (mark, d["label"], d.get("joint_commands"),
                                                      d.get("simulated_seconds", 0), extra))
'
