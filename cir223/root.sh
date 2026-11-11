#!/usr/bin/env bash
# `cir223/root.sh JOBS NAME=COMMIT...`: the repository's own suite (`tests/`, the quick tier: `suites.py repository --quick`)
# on each commit's tree, in a scratch worktree outside the run directory, JOBS at once. Under $OUT: root.txt (one line a
# commit), NAME.suites.json, NAME.suites.log and NAME.logs/.
set -uo pipefail
JOBS=$1; shift
OUT=$(realpath "${OUT:-out}"); mkdir -p "$OUT"
T=$(mktemp -d)
one() {
  local spec=$1 name result wt t0 rc
  name=${spec%%=*}; result=${spec#*=}; wt=$T/$name; t0=$(date +%s)
  if [ ! -d "$wt" ]; then
    echo "ROOT $name $result: no worktree" | tee -a "$OUT/root.txt"; return
  fi
  ( cd "$wt" && uv sync -q --all-packages --extra torch-cpu > "$OUT/$name.sync.log" 2>&1 \
    && uv run tools/verity/check/suites.py repository --quick --json "$OUT/$name.suites.json" --logs "$OUT/$name.logs" \
       > "$OUT/$name.suites.log" 2>&1 )
  rc=$?
  echo "ROOT $name $result tree $(git rev-parse "$result^{tree}"): suites.py repository --quick: exit $rc after $(( $(date +%s) - t0 )) s; $(python3 - "$OUT/$name.suites.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    print(f"no record ({type(e).__name__})"); raise SystemExit
for s in d:
    print(f"{s.get('name')} {s.get('status')} counts {json.dumps(s.get('counts'))} reused {s.get('reused')} since {s.get('since')} "
          f"seconds {s.get('seconds')} failed {json.dumps(s.get('failed_tests'))[:1500]}")
PY
)" | tee -a "$OUT/root.txt"
  git worktree remove --force "$wt" > /dev/null 2>&1 || true
}
for spec in "$@"; do  # one at a time: two `git worktree add` at once can meet on the clone's lock
  git worktree add -q --detach "$T/${spec%%=*}" "${spec#*=}" || true
done
for spec in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  one "$spec" &
done
wait
echo "== root.txt"; cat "$OUT/root.txt"
for f in "$OUT"/*.suites.log; do echo "== $f"; tail -n 25 "$f" | cut -c1-600; done
