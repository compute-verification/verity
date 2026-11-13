#!/usr/bin/env bash
# Carries one branch head across the moves --main has and it lacks, with main's tools/move/restack.py, and bundles the
# result as refs/carried/NAME into $OUT/carried.bundle.  Nothing is pushed.
# HEAD is a carrier commit whose tree is main's and whose parents are main and the head, so the clone holds both.
# Usage: carry.sh NAME HEAD MAIN
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
N=$1; H=$2; MAIN=$3
log "carry $N $H onto $MAIN; clone head $(git rev-parse HEAD)"
git cat-file -e "$H^{commit}" && git cat-file -e "$MAIN^{commit}" || { log "missing a commit"; exit 3; }
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
PY=$PWD/.venv/bin/python
ROOT=${SCRATCH_ROOT:-/workspace/research/scratch}; mkdir -p "$ROOT"
d=$(mktemp -d -p "$ROOT" carry.XXXXXX) || exit 3
trap 'rm -rf "$d"' EXIT; trap 'exit 143' TERM INT
export TMPDIR="$d"
export GIT_AUTHOR_NAME="Cursor Agent" GIT_AUTHOR_EMAIL="cursoragent@cursor.com" GIT_COMMITTER_NAME="Cursor Agent" GIT_COMMITTER_EMAIL="cursoragent@cursor.com"
"$PY" - > "$O/drivers.txt" 2>&1 <<'PY'
import sys
sys.path.insert(0, "tools/move")
import restack
print("missing before:", restack.missing_drivers())
print("drivers():", restack.drivers(sys.executable))
print("missing after:", restack.missing_drivers())
PY
log "drivers: $(tr '\n' ' ' < "$O/drivers.txt" | cut -c1-600)"
grep -q "missing after: \[\]" "$O/drivers.txt" || { log "merge drivers missing"; exit 3; }
s=$(date +%s)
"$PY" tools/move/restack.py "$H" --main "$MAIN" --kept --no-pins --json > "$O/$N.json" 2> "$O/$N.err"; rc=$?
log "restack rc $rc in $(( $(date +%s) - s )) s"
"$PY" - "$O/$N.json" "$H" "$MAIN" <<'PY' | tee -a "$O/summary.txt"
import json, subprocess, sys
f, h, main = sys.argv[1:]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
r = json.load(open(f))[0]
print(f"{h[:12]} {r['status']} step={r.get('step')} crossed={len(r.get('crossed', []))} skipped={len(r.get('skipped', []))} "
      f"move={str(r.get('move'))[:12]} result={r.get('result')} files={(r.get('files') or [])[:25]}")
if r.get("error"):
    print("   error:", r["error"][:1500].replace("\n", " | "))
if r.get("result"):
    print("   holds its old head:", subprocess.run(["git", "merge-base", "--is-ancestor", h, r["result"]]).returncode == 0,
          "| holds main:", subprocess.run(["git", "merge-base", "--is-ancestor", main, r["result"]]).returncode == 0,
          "| against main:", git("diff", "--shortstat", main, r["result"]))
    open(sys.argv[1] + ".result", "w").write(r["result"] + " " + r["status"] + "\n")
PY
if [ -f "$O/$N.json.result" ]; then
  read -r res st < "$O/$N.json.result"
  git update-ref "refs/carried/$N" "$res"
  git bundle create "$O/carried.bundle" "refs/carried/$N" "^$MAIN" "^$H" > "$O/bundle.log" 2>&1
  log "bundle rc $?: $res ($st)"
fi
exit $rc
