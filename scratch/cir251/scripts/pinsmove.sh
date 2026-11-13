#!/usr/bin/env bash
# At a head (run in a clone): pins.py --stale, and --regen with its diff when it names one; then move_check.py OLD HEAD.
# Usage: pinsmove.sh OLD
set -o pipefail
OLD="$1"
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
uv run --no-sync python tools/verity/check/pins.py --stale > "$OUT/pins_stale.txt" 2>&1
p=$?
echo "pins --stale rc $p" >> "$OUT/pins_stale.txt"
if [ $p -ne 0 ]; then
  uv run --no-sync python tools/verity/check/pins.py --regen > "$OUT/pins_regen.txt" 2>&1
  echo "pins --regen rc $?" >> "$OUT/pins_regen.txt"
  git status --short > "$OUT/pins_status.txt"
  git diff --binary > "$OUT/pins.patch"
fi
uv run --no-sync python tools/verity/check/move_check.py "$OLD" HEAD > "$OUT/move_check.txt" 2>&1
m=$?
echo "move_check rc $m" | tee "$OUT/rc.txt"
echo "pins rc $p" >> "$OUT/rc.txt"
exit 0
