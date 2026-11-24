#!/usr/bin/env bash
# Make the charts + report for a localize run and print the report. No robot motion.
#   ./view.sh                 newest localize run
#   ./view.sh <run folder>    a specific run (host path or runs/real/<stamp>-localize)
#   ./view.sh --open          also open the overview image (needs a desktop session)
set -u
. "$(dirname "$0")/common.sh"
require_container
open=0; run=""
for a in "$@"; do
    case "$a" in --open) open=1 ;; *) run="$a" ;; esac
done
[ -z "$run" ] && run=$(latest_run localize)
[ -z "$run" ] && { red "no localize runs yet"; exit 1; }
name=$(basename "$run")
kdocker "python3 scripts/real/view_localization.py runs/real/$name"
charts="$KILO_HOST_DIR/runs/real/$name/charts"
echo
bold "Charts: $charts"
ls -1 "$charts" 2>/dev/null | sed 's/^/  /'
ip=$(hostname -I | tr ' ' '\n' | grep -E '^172\.26\.' | head -1)
echo "Copy to your laptop:  scp -r student@${ip:-<lab-pc-ip>}:$KILO_HOST_DIR/runs/real/$name ."
if [ $open = 1 ]; then
    (xdg-open "$charts/1_overview.png" >/dev/null 2>&1 &) || echo "could not open a viewer here"
fi
