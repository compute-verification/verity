#!/usr/bin/env bash
# Runs the head's own tools/verity/check/pins.py --regen at the root of the run's clone and, if it rewrites anything,
# commits that and bundles the commit as refs/pins/head into $OUT/pins.bundle (prerequisite: the head).  Nothing is pushed.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
ROOT=${SCRATCH_ROOT:-/workspace/research/scratch}; mkdir -p "$ROOT"
d=$(mktemp -d -p "$ROOT" pins.XXXXXX) || exit 3
trap 'rm -rf "$d"' EXIT; trap 'exit 143' TERM INT
export TMPDIR="$d"
export GIT_AUTHOR_NAME="Cursor Agent" GIT_AUTHOR_EMAIL="cursoragent@cursor.com" GIT_COMMITTER_NAME="Cursor Agent" GIT_COMMITTER_EMAIL="cursoragent@cursor.com"
H=$(git rev-parse HEAD)
echo "head $H" | tee "$O/job.log"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
.venv/bin/python tools/verity/check/pins.py --regen > "$O/pins.out" 2> "$O/pins.err"; rc=$?
echo "pins.py --regen rc $rc" | tee -a "$O/job.log"
git status --porcelain | tee "$O/status.txt"
if [ -n "$(git status --porcelain)" ]; then
  git add -A && git commit -q -m "pins: tools/verity/check/pins.py --regen at the carried head" && git update-ref refs/pins/head HEAD
  git bundle create "$O/pins.bundle" refs/pins/head "^$H" > "$O/bundle.log" 2>&1
  echo "committed $(git rev-parse HEAD); bundle rc $?" | tee -a "$O/job.log"
else
  echo "pins.py --regen rewrote nothing" | tee -a "$O/job.log"
fi
exit $rc
