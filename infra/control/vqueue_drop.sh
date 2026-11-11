#!/bin/bash
# Drop a candidate by hand, as the loop's `_drop` does: ci-drop-candidate.sh SHA "WHY".  Finds processes with `pgrep -f`
# (PIDs only: never list command lines, the exec daemon's argv holds tokens) and kills them by pid, never its own session.
# Deployed as /data/vqueue/bin/vqueue_drop.sh, linked as /root/ci-drop-candidate.sh.
set -u
SHA=$1; WHY=$2
VQ=/data/vqueue/live
REPO=/root/fast/vqueue-verity
MY=$(ps -o sid= -p $$ | tr -d ' ')
cp -p $VQ/decisions.jsonl $VQ/decisions.jsonl.bak-ci-drop-$(date -u +%H%M)Z
touch $VQ/stopped
flock -w 300 $VQ/vqueue.lock true || { echo "vqueue.lock still held after 300 s: nothing done"; rm -f $VQ/stopped; exit 1; }
sids=$(for p in $(pgrep -f "${SHA:0:12}"); do ps -o sid= -p "$p"; done | tr -d ' ' | sort -u | grep -vx "$MY")
for sid in $sids; do
  pids=$(pgrep -s "$sid")
  echo "sid $sid: TERM $(echo $pids | tr '\n' ' ')"
  kill -TERM $pids 2>/dev/null
done
sleep 5
for sid in $sids; do
  pids=$(pgrep -s "$sid")
  [ -n "$pids" ] && { echo "sid $sid: KILL $(echo $pids | tr '\n' ' ')"; kill -KILL $pids 2>/dev/null; }
done
echo "left matching ${SHA:0:12} (pids only): $(pgrep -f "${SHA:0:12}" | grep -vx "$$" | tr '\n' ' ')"
H=()
if ! git -C $REPO config --get-urlmatch credential.helper "$(git -C $REPO remote get-url origin)" >/dev/null 2>&1 && command -v gh >/dev/null; then
  H=(-c credential.helper= -c "credential.helper=!$(command -v gh) auth git-credential")
fi
git -C $REPO "${H[@]}" push -q --force-with-lease=refs/heads/queue/${SHA:0:12}:$SHA origin :refs/heads/queue/${SHA:0:12} \
  && echo "deleted queue/${SHA:0:12}" || echo "BRANCH DELETE FAILED"
CANCEL=$(/root/g2remote.sh cancel --check ready:$SHA 2>&1); RC=$?
python3 - "$SHA" "$RC" "${CANCEL: -300}" "$WHY" <<'EOF'
import json, sys, time
sha, rc, said, why = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
log = "/data/vqueue/live/decisions.jsonl"
recs = [json.loads(l) for l in open(log) if l.strip()]
build = [r for r in recs if r.get("action") == "build" and r.get("sha") == sha][-1]
slim = [r for r in recs if r.get("action") == "slim" and r.get("sha") == sha]
rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tick": build["tick"], "shadow": False,
       "main": build["main"], "action": "drop", "sha": sha, "base": build["base"], "members": build["members"],
       "heads": build["heads"], "priority": None, "number": None, "why": "ci dropped it by hand: " + why,
       "outcome": None, "failed": [], "cancel": "cancelled" if rc == 0 else f"cancel exited {rc}: {said.strip()[-200:]}",
       "done": True, **({"beside_launching": slim[-1]["record"]} if slim and slim[-1].get("record") else {})}
with open(log, "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
print("logged drop", rec["at"], rec["members"])
EOF
echo "loop left stopped: dry-run (live.sh once --dry-run \$(grep -v '^#' $VQ/args)), then rm $VQ/stopped"
