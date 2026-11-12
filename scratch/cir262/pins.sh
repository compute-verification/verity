#!/usr/bin/env bash
# circuit-check's own pins update pass on the V32 module's sidecar, in a clone; what it wrote is saved
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
CC_ARGS=--update-pins SERVED=0 bash "$RESEARCH_RUN_DIR/inputs/cc2.sh"; rc=$?
git status --short > "$OUT/git-status-after.txt"
git diff > "$OUT/update-pins.diff"
git ls-files --others --exclude-standard | grep -v '^\.venv' > "$OUT/untracked-after.txt" || true
exit $rc
