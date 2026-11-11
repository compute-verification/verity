#!/usr/bin/env bash
# `cir223/quick.sh STAGED NAME=RESULT...`: for each crossed result, the quick tier of the suites its change from STAGED touches
# (`suites.py --changed STAGED --quick`), in a scratch worktree of RESULT outside the run directory. Under $OUT: quick.txt
# (one line a result), NAME.suites.json, NAME.suites.log and NAME.logs/. With ONLY=DIR in the environment, only the suite
# of the package at DIR (its pyproject's project name).
set -uo pipefail
STAGED=$1; shift
OUT=$(realpath "${OUT:-out}"); mkdir -p "$OUT"
T=$(mktemp -d)
for spec in "$@"; do
  name=${spec%%=*}; result=${spec#*=}; wt=$T/$name; t0=$(date +%s)
  if ! git worktree add -q --detach "$wt" "$result"; then
    echo "$name $result: no worktree" | tee -a "$OUT/quick.txt"; continue
  fi
  ( cd "$wt" && git diff --stat "$STAGED" HEAD | tail -1 \
    && only=() && { [ -z "${ONLY:-}" ] || only=("$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1] + "/pyproject.toml", "rb"))["project"]["name"])' "$ONLY")"); } \
    && uv sync -q --all-packages --extra torch-cpu > "$OUT/$name.sync.log" 2>&1 \
    && uv run tools/verity/check/suites.py "${only[@]}" --changed "$STAGED" --quick --json "$OUT/$name.suites.json" --logs "$OUT/$name.logs" \
       > "$OUT/$name.suites.log" 2>&1 )
  rc=$?
  echo "$name $result: suites.py --changed $STAGED --quick: exit $rc after $(( $(date +%s) - t0 )) s" | tee -a "$OUT/quick.txt"
  tail -n 60 "$OUT/$name.suites.log"
  git worktree remove --force "$wt" || true
done
