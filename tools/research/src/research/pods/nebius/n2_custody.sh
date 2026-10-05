#!/usr/bin/env bash
# Node 2's custody: its stores (/workspace/jobs/store, /workspace/research/store) have no remote and the node holds no R2 credential,
# so every ended run under its run roots goes home through node 1: rsync to node 1, then a one-off Job there with the Secret
# research-r2 (the credential never leaves Kubernetes) runs `research data custody RUN --publish` on the copy: the attempt with its
# run record (logs and declared outputs, store.custody) into node 1's /workspace/jobs/store, pushed to R2 and verified there. The
# .custody that verify writes beside node 1's copy comes back into node 2's run dir, whose files it lists unchanged, so node 2's
# own `research data custody --triage` sees the run on the remote. The Job's research is VY_CUSTODY_SRC on node 1, which deploy.toml
# installs from the commit this file comes from. n2_commit.sh and n2_build.sh name this as their runs' custody (RESEARCH_LOCAL_ONLY),
# without which `research run` refuses to start on a store with no remote.
#
#   on node 2:  n2_custody.sh [--dry-run]   one round (the user timer n2-custody.timer, every 10 min); --dry-run lists what it would push
#
# A run has ended when its status says so or its process is gone: a runner killed before it recorded an end (an OOM kill of its
# scope) is published with --force --runner-gone (node 1 can't see node 2's pids) and keeps the state it last wrote. A run its own
# custody preserved (.custody "preserved": true) is only marked; any other waits VY_CUSTODY_SETTLE_S after any file in it last
# changed, and while its runner holds its custody key, until the key expires (a runner's custody writes nothing into the run dir
# until its .custody: 11 min once), so a runner's own custody finishes first.
# A run is marked in STATE/pushed/ once node 1's Job verified it on R2 and its .custody is back, which it is only while every file
# here is the one node 1's copy had. A failure is logged with its reason
# and retried every round, never given up, and the round exits 1, which systemd records (`systemctl --user status n2-custody`).
# A round is bounded: at most VY_CUSTODY_MAX_RUNS pushes, none started after VY_CUSTODY_PASS_S seconds, runs that failed before
# going last; the rest wait for the next. A failure no retry can mend is held, with the Job's line, for a person to decide: node 1
# holds the run's attempt without a run record (the stopgap published it), R2 holds another attempt for its id (its runner's own;
# node 1's vy-custody --reconcile takes it), or its attempt's run record lacks what the run dir holds now.
# Only run dirs go home, never a row directory (3 Oct: the stopgap's row sync mirrored 1.2M files to node 1 and overran its window),
# and a run dir over VY_CUSTODY_MAX_FILES files or VY_CUSTODY_MAX_GB GiB is held instead of mirrored: STATE/held/RUN says why, and the
# round exits 1 while it is there. Move the bulk out of the run dir (or declare it elsewhere), then remove STATE/held/RUN.
# The stopgap this replaces, STATE/push.sh in a tmux loop, shares STATE/pushed/ and the Jobs' names but takes no lock: while it runs,
# a round refuses (exit 3), as it does while another round holds STATE/.lock. Exit 0 ok, 1 a run not yet home, 2 usage, 3 refused.
# Node 2's timed windows (PoUW's measurements) run with nothing else on the node: a round starts, and starts each push, only while the
# fill runner's status is fresh and says no window runs or waits, and no booked window is on or starts within the round's bound
# (PASS_S + PUSH_S); else it stops there, with one log line and exit 0, and the next round asks again. Its Jobs run on node 1, so it
# keeps node 1's daily hold (VY_CUSTODY_QUIET, 12:10-13:30Z, as n1_custody.sh does) the same way, and no Job of it runs into node 1's
# quiet hour: a push starts only while its rsync and a Job of at least JOB_MIN_S end before 12:30Z, and the Job's
# activeDeadlineSeconds and the grace a stopped pod has (GRACE_S; on SIGTERM its bash exits, and research is killed) end it a minute before.
set -u
ROOTS=${VY_CUSTODY_ROOTS:-/workspace/jobs/runs=/workspace/jobs/runs,/workspace/research/runs=/workspace/jobs/runs-n2}
STATE=${VY_CUSTODY_STATE:-/workspace/research/custody-push}
SETTLE_S=${VY_CUSTODY_SETTLE_S:-900}
MAX_RUNS=${VY_CUSTODY_MAX_RUNS:-20}
PASS_S=${VY_CUSTODY_PASS_S:-1200}
MAX_FILES=${VY_CUSTODY_MAX_FILES:-20000}
MAX_GB=${VY_CUSTODY_MAX_GB:-50}
# on node 1, root's alone (as n1_custody.sh's): the research deploy.toml installs and the Python the Jobs run with the Secret
SRC=${VY_CUSTODY_SRC:-/workspace/custody}
PYHOME=${VY_CUSTODY_PYHOME:-/workspace/custody/python}
PY=${VY_CUSTODY_PY:-$PYHOME/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12}
LOOP=${VY_CUSTODY_LOOP:-$STATE/push.sh}
FILL=${VY_CUSTODY_FILL:-/workspace/pouw/fill}
PUSH_S=3600     # the most one push takes: rsync 20 min, the Job's wait 35 min, the calls around them
RSYNC_S=1200
JOB_MIN_S=300   # a Job left less time before node 1's quiet hour isn't started
GRACE_S=10      # a Job's pod stopped at its deadline: SIGTERM, then SIGKILL after this (terminationGracePeriodSeconds)
QUIET=${VY_CUSTODY_QUIET-1210-1330}     # node 1's daily hold, HHMM-HHMM UTC (dispatch's PACK_HOLD through its quiet hour's end)
BY=1230                                 # its quiet hour, which no Job runs into
N1=${VY_N1:-vy-n1}
DRY=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY=1 ;;
  *) echo "usage: n2_custody.sh [--dry-run]" >&2; exit 2 ;;
esac
KUBE="KUBECONFIG=\$HOME/.kube/config kubectl --request-timeout=30s"
# a dropped link ends the call within a minute; a short call has `timeout` around it too
SSH=(ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
mkdir -p "$STATE/pushed" "$STATE/failures" "$STATE/held"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATE/log"; }
exec 9> "$STATE/.lock"
flock -n 9 || { log "refused: another round holds $STATE/.lock"; exit 3; }
# the loop's own bash, not a tmux server whose first session started it (its argv keeps that session's command); asked again
# before every push, since the loop can be started while a round runs
stopgap() {
  local pids
  pids=$(pgrep -f -- "^[^ ]*bash $LOOP( |\$)") || return 0
  log "refused: the stopgap loop $LOOP still runs (pid $(echo $pids)) and takes no lock; stop it while it sleeps between rounds" \
    "(tmux custody-push-3), and the next round takes over"
  exit 3
}
# window: why no push may start now on node 2's account, or nothing. Fail closed, as dispatch's n2_room and write_probe do: the
# fill runner rewrites status.txt every 10 s, and a first line that doesn't read, is over 60 s old by this clock, or lacks
# `timed False, window waiting False` is a window; so is a window booked in FILL/windows that is on or starts before END
END=$(( $(date -u +%s) + PASS_S + PUSH_S ))
window() {
  python3 - "$FILL" "$(date -u +%s)" "$END" << 'EOF'
import sys
from datetime import datetime, timezone
fill, now, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])


def utc(s):
    t = datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


try:
    first = open(f"{fill}/status.txt").read().split("\n")[0]
    age = now - utc(first.split()[0])
except (OSError, IndexError, ValueError) as e:
    sys.exit(f"its fill status doesn't read ({type(e).__name__}: {e})")
if "timed False, window waiting False" not in first:
    sys.exit(f"its fill status: {first[:120]}")
if not -60 <= age <= 60:
    sys.exit(f"its fill status is {age:.0f} s old (the fill runner rewrites it every 10 s)")
try:
    lines = open(f"{fill}/windows").read().splitlines()
except FileNotFoundError:
    lines = []
except OSError as e:
    sys.exit(f"its booked windows don't read ({type(e).__name__}: {e})")
for line in lines:
    f = line.split("#")[0].split()
    try:
        s = utc(f[0])
        e = s + 60 * float(f[1] if len(f) > 1 else 30)
    except (IndexError, ValueError):
        continue
    if s < end and e > now:
        sys.exit(f"a window booked at {f[0]} for {f[1] if len(f) > 1 else 30} min is on or starts within the round's "
                 f"{(end - now) / 60:.0f} min")
EOF
}
if ! why=$(window 2>&1); then
  log "skipped: node 2's timed window: $why; the next round asks again"
  exit 0
fi
# n1_left: the seconds a Job started on node 1 now may run, to a minute before its quiet hour (BY) less its grace, to the second; 0 in
# its daily hold
n1_left() {
  [ -n "$QUIET" ] || { echo 86400; return; }
  local t q0 q1 by
  t=$(( $(date -u +%s) % 86400 )) by=$((10#${BY:0:2} * 3600 + 10#${BY:2:2} * 60))
  q0=$((10#${QUIET:0:2} * 3600 + 10#${QUIET:2:2} * 60)) q1=$((10#${QUIET:5:2} * 3600 + 10#${QUIET:7:2} * 60))
  if [ "$t" -ge "$q0" ] && [ "$t" -lt "$q1" ]; then echo 0; else echo $(( (by - t + 86400) % 86400 - 60 - GRACE_S )); fi
}
HOLD="node 1's daily hold ($QUIET UTC; no Job of it runs past ${BY}Z, where node 1's quiet hour starts)"
if [ "$DRY" = 0 ] && [ "$(n1_left)" -lt $((RSYNC_S + JOB_MIN_S)) ]; then
  log "skipped: $HOLD; the next round asks again"
  exit 0
fi
[ "$DRY" = 1 ] || stopgap

# "CAMPAIGN STATE AGE_S FILES BYTES": STATE is the status's, dead for a run still marked running whose process is gone, or custody
# while its runner holds its custody key (<root>/../requests/RUN/custody/cred.json) and the key hasn't expired; AGE_S since
# any file of the run dir last changed; FILES and BYTES of the run dir; all three counted no further than MAX_FILES + 1 files
info() {
  python3 - "$1" "$MAX_FILES" << 'EOF'
import calendar, json, os, re, sys, time
d, cap = sys.argv[1], int(sys.argv[2])
try:
    m = re.search(r'"campaign": ?"([^"]*)"', open(f"{d}/job.json").read())
except OSError:
    m = None
try:
    st = json.load(open(f"{d}/status.json"))
except (OSError, ValueError):
    st = {}
state = st.get("state") or "running"
if state == "running":
    # the runner's pid is status.json's own: `tele set-timeout` appends a `running` transition without one
    try:
        t = next(t for t in (st, *reversed(st.get("transitions") or [])) if t.get("pid") is not None)
        if int(t["pid"]) <= 0:
            raise ValueError(t["pid"])
        f = open(f"/proc/{int(t['pid'])}/stat").read().rsplit(")", 1)[1].split()
        alive = t.get("starttime_ticks") is None or int(f[19]) == int(t["starttime_ticks"])
    except FileNotFoundError:
        alive = False
    except Exception:           # no status yet, or no pid that parses: never taken for dead
        alive = True
    state = "running" if alive else "dead"
now = time.time()
# a `research run --custody-r2` runner writes nothing into the run dir until its .custody, and forgets its key after it
key = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(d))), "requests", os.path.basename(d), "custody", "cred.json")
try:
    exp = calendar.timegm(time.strptime(json.load(open(key))["expires_utc"], "%Y-%m-%dT%H:%M:%SZ"))
except FileNotFoundError:
    exp = 0
except Exception:               # a key that doesn't read: no custody key outlives 24 h
    exp = os.path.getmtime(key) + 86400
if state != "running" and now < exp:
    state = "custody"
mt, n, size = now, 0, 0
if state != "running":
    mt = 0
    for base, _, files in os.walk(d):
        for f in files:
            n += 1
            try:
                s = os.lstat(os.path.join(base, f))
            except OSError:
                continue
            size, mt = size + s.st_size, max(mt, s.st_mtime)
            if n > cap:
                break
        if n > cap:
            break
print(m.group(1) if m else "-", state, int(now - (mt or now)), n, size)
EOF
}

human() { awk -v b="$1" 'BEGIN { split("B KiB MiB GiB TiB", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'; }

# relay RUN_DIR: node 1's .custody (RUN_DIR/.custody.tmp) vouches for node 1's copy, on node 1's clock. It holds here only when every
# file here is in it with the same size and sha256, and from this node's clock read before hashing (node 1's if earlier), so a file
# written here after the rsync stops it now, and one written after this check voids it (store.custody.marker_problem); else says why
relay() {
  python3 - "$1" << 'EOF'
import hashlib, json, os, sys, time
d = sys.argv[1]
try:
    m = json.load(open(f"{d}/.custody.tmp"))
    t0, files = min(float(m["verified_from"]), time.time()), dict(m["files"])
except (OSError, ValueError, KeyError, TypeError) as e:
    sys.exit(f"it doesn't read ({type(e).__name__}: {e})")
if m.get("preserved") is not True:
    sys.exit(f"not preserved ({m.get('error') or '?'})")
markers = {".fetched", ".fetched.tmp", ".custody", ".custody.tmp", "preserved.json", "run-files.evicted.json", ".custody-trimmed.json"}
for base, _, names in os.walk(d):
    for f in names:
        p = os.path.join(base, f)
        rel = os.path.relpath(p, d)
        if rel in markers or os.path.islink(p) or not os.path.isfile(p):     # store.custody.run_files
            continue
        if rel not in files:
            sys.exit(f"{rel} is not in it: written here after the rsync")
        with open(p, "rb") as fh:
            if list(files[rel]) != [os.fstat(fh.fileno()).st_size, hashlib.file_digest(fh, "sha256").hexdigest()]:
                sys.exit(f"{rel} differs from node 1's copy: changed here after the rsync")
m["verified_from"] = t0
with open(f"{d}/.custody.tmp", "w") as fh:
    json.dump(m, fh)
EOF
}

push_run() {  # push_run ROOT DST RUN_ID STATE: 0 on R2 with .custody back, 4 held (hold= says what to decide), 6 no Job started
  # for node 1's daily hold (the copy waits there), else 1; why= says
  local root=$1 dst=$2 r=$3 name force=
  # dead: its pid is gone here, which node 1, where the Job runs, can't see
  [ "$4" = dead ] && force="--force --runner-gone"
  name=n2-custody-$(echo "$r" | tr -cd 'a-z0-9-' | tail -c 40)
  # a Job an earlier round left (its ssh dropped) may still be pushing node 1's copy: never rsync under it or delete it, wait on it
  job=$(timeout 60 "${SSH[@]}" -n "$N1" "$KUBE get job $name --ignore-not-found -o jsonpath='{.metadata.name}{range .status.conditions[*]} {.type}={.status}{end}'") \
    || { why="node 1's Job $name: its state not read ($?)"; return 1; }
  start="$KUBE delete job $name --ignore-not-found > /dev/null && $KUBE create -f - > /dev/null"
  case $job in
    "" | *Complete=True* | *Failed=True*)
      timeout "$RSYNC_S" rsync -a --timeout=300 --rsync-path="sudo -n rsync" --chown=1000:1000 --mkpath "$root/$r/" "$N1:$dst/$r/" \
        || { why="rsync to $N1 failed ($?)"; return 1; } ;;
    *)
      start=true
      log "$root/$r: node 1's Job $name, from an earlier round, has not finished: waiting on it" ;;
  esac
  deadline=$(n1_left)
  [ "$deadline" -gt 3600 ] && deadline=3600
  if [ "$start" != true ] && [ "$deadline" -lt "$JOB_MIN_S" ]; then
    why="$HOLD: ${deadline}s left, the copy waits on node 1"
    return 6
  fi
  # a failed Job never completes: wait for either condition, then name the failure from its log. The Job's bash, its container's init,
  # exits on SIGTERM (trapped) and the kernel SIGKILLs research; a rerun repeats it, as every write is a temp file, then a rename
  if ! timeout 2100 "${SSH[@]}" "$N1" "$start &&
    end=\$((\$(date +%s) + 1800)) && while [ \$(date +%s) -lt \$end ]; do
    c=\$($KUBE get job $name -o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}'); case \$c in
    *Complete=True*) exit 0 ;; *Failed=True*) $KUBE logs job/$name --tail=3 >&2; exit 1 ;; esac; sleep 5; done
    echo 'Job not done in 30 min' >&2; exit 1" 2> "$STATE/.job.err" << EOF; then
{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "$name", "labels": {"verity.dev/lane": "n2-custody", "verity.dev/kind": "custody"}},
 "spec": {"backoffLimit": 1, "activeDeadlineSeconds": $deadline, "ttlSecondsAfterFinished": 3600, "template": {"metadata": {"labels": {"verity.dev/lane": "n2-custody"}}, "spec": {
  "restartPolicy": "Never", "terminationGracePeriodSeconds": $GRACE_S, "nodeSelector": {"verity.dev/server": "vy-nebius-1"},
  "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
  "securityContext": {"runAsUser": 1000, "runAsGroup": 1000},
  "containers": [{"name": "main", "image": "${VY_DISPATCH_IMAGE:-us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216}",
   "imagePullPolicy": "IfNotPresent",
   "command": ["taskset", "-c", "96-127", "bash", "-c", "trap 'exit 143' TERM; cd / || exit 1; HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC/tools/research/src RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=$dst $PY -m research data custody $dst/$r --publish --keep-local $force & wait \$!"],
   "envFrom": [{"secretRef": {"name": "research-r2"}}],
   "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
   "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}]}],
  "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}}]}}}}
EOF
    why="node 1's Job: $(grep -v '^$' "$STATE/.job.err" | tail -1)"
    # what no retry can mend is held, with what a person decides (the line names the push's conflict beside verify's error)
    case $why in
      *RemoteConflict*|*"DIFFERENT attempt document"*)
        hold="R2 already holds another attempt for $r (an attempt is write-once), so node 1's can never replace it; when its runner's own"
        hold+=" custody pushed that one, $root/$r/.custody names its run record and what the record lacks. Take R2's copy into node 1's"
        hold+=" store: on node 1, \`sudo -u research vy-custody --reconcile $r --by WHO\` (its Job, with the R2 Secret, runs \`research"
        hold+=" data reconcile-attempts $r --apply\` and publishes node 1's copy; when that says it is on R2 with its run record, rm"
        hold+=" $STATE/held/$r instead, and the next round brings its .custody home). What R2's record lacks is only in this run dir:"
        hold+=" keep it, or put it in the store (\`research data put --kind K --file P --preserve\`)" ;;
      *"not in the run record"*|*"changed since it was taken"*)
        hold="its attempt's run record, taken from an earlier copy of the run dir, lacks what the run dir holds now, and an attempt never"
        hold+=" changes; the rest is on R2 with that record, the difference only in this run dir: keep it, or put it in the store"
        hold+=" (\`research data put --kind K --file P --preserve\`)" ;;
      *"names no run record"*)
        hold="node 1's store holds its attempt without a run record (published before this custody), so it can never have .custody; the"
        hold+=" Job pushed what node 1 holds. Label it \`custody waived\` (or fetch it whole)" ;;
      *) return 1 ;;
    esac
    return 4
  fi
  # verify wrote it only with the attempt and its run record verified on R2, over node 1's copy as the rsync left it
  if ! timeout 60 "${SSH[@]}" -n "$N1" "sudo -n cat $dst/$r/.custody" > "$root/$r/.custody.tmp"; then
    rm -f "$root/$r/.custody.tmp"
    why="no .custody at $N1:$dst/$r"
    return 1
  fi
  why=$(relay "$root/$r" 2>&1) && mv "$root/$r/.custody.tmp" "$root/$r/.custody" && return 0
  rm -f "$root/$r/.custody.tmp"
  why="node 1's .custody not taken home: $why"
  return 1
}

# runs that have not failed go first, so a run that keeps failing never takes a newer one's place in a bounded round
fresh=() retry=()
IFS=, read -ra pairs <<< "$ROOTS"
for pair in "${pairs[@]}"; do
  root=${pair%%=*} dst=${pair#*=}
  for d in "$root"/r*/; do
    [ -d "$d" ] || continue
    r=$(basename "$d")
    # node 1's shell and the Job's bash -c take the name unquoted
    [[ $r =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || continue
    [ -e "$STATE/pushed/$r" ] || [ -e "$STATE/held/$r" ] && continue
    if [ -e "$STATE/failures/$r" ]; then retry+=("$root|$dst|$r"); else fresh+=("$root|$dst|$r"); fi
  done
done

src_ok= paused=
failed=0 tried=0 left=0
for c in "${fresh[@]}" "${retry[@]}"; do
  IFS='|' read -r root dst r <<< "$c"
  d=$root/$r
  if [ "$DRY" = 0 ] && { [ -n "$paused" ] || [ "$tried" -ge "$MAX_RUNS" ] || [ "$SECONDS" -ge "$PASS_S" ]; }; then
    left=$((left + 1))
    continue
  fi
  read -r camp state age files bytes < <(info "$d")
  case $state in running|custody|"") continue ;; esac
  if grep -qs '"preserved": true' "$d/.custody"; then
    [ "$DRY" = 1 ] || touch "$STATE/pushed/$r"
    continue
  fi
  [ "$age" -lt "$SETTLE_S" ] && continue
  if [ "$files" -gt "$MAX_FILES" ] || [ "$bytes" -gt $((MAX_GB << 30)) ]; then
    why="over $( [ "$files" -gt "$MAX_FILES" ] && echo "$MAX_FILES files" || echo "$MAX_GB GiB ($(human "$bytes"))")"
    if [ "$DRY" = 1 ]; then
      echo "would hold $d ($camp, $state): $why"
    else
      echo "$why" > "$STATE/held/$r"
      log "$d ($camp, $state): held, $why: not mirrored to node 1; move the bulk out of the run dir, then rm $STATE/held/$r"
    fi
    continue
  fi
  if [ "$DRY" = 1 ]; then
    echo "would push $d ($camp, $state, $files files, $(human "$bytes"))"
    continue
  fi
  if [ -z "$src_ok" ]; then
    # python -m puts the working directory first on sys.path, so the Job runs from /, never the world-writable /workspace/jobs
    bad=$(timeout 60 "${SSH[@]}" -n "$N1" "test -f $SRC/tools/research/src/research/store/custody.py && test -f $SRC/tools/research/store.pod.toml && test -x $PY || exit 1
      bad=\$(find $SRC/tools/research $PYHOME -path $PYHOME/.lock -prune -o \\( \\( ! -uid 0 ! -uid \$(id -u) \\) -o \\( -perm /022 ! -type l \\) \\) -print -quit 2>&1)
      [ -z \"\$bad\" ] || { echo \"\$bad\"; exit 4; }")
    rc=$?
    [ "$rc" = 1 ] && { log "refused: no research tree at $N1:$SRC, or no $PY (deploy.toml installs the tree: research deploy install vy-nebius-1 --by WHO $SRC/tools/research/src/research $SRC/tools/research/store.pod.toml; the Python: n1_custody.sh)"; exit 3; }
    [ "$rc" = 4 ] && { log "refused: $N1:$bad could be changed by a uid other than root: the Jobs run $SRC's research and $PY with the Secret research-r2"; exit 3; }
    [ "$rc" = 0 ] || { log "refused: node 1 not reached (\`ssh $N1\` exited $rc)"; exit 3; }
    src_ok=1
  fi
  if ! why=$(window 2>&1); then
    paused="node 2's timed window: $why"
    left=$((left + 1))
    continue
  fi
  if [ "$(n1_left)" -lt $((RSYNC_S + JOB_MIN_S)) ]; then
    paused=$HOLD
    left=$((left + 1))
    continue
  fi
  stopgap
  tried=$((tried + 1))
  why= hold=
  push_run "$root" "$dst" "$r" "$state"
  rc=$?
  if [ "$rc" = 0 ]; then
    touch "$STATE/pushed/$r"
    rm -f "$STATE/failures/$r"
    log "$d ($camp, $state, $files files, $(human "$bytes")): on R2 with its run record, .custody back on node 2"
  elif [ "$rc" = 4 ]; then
    echo "$why" > "$STATE/held/$r"
    log "$d ($camp, $state): held: $why. $hold; then mv $STATE/held/$r $STATE/pushed/"
  elif [ "$rc" = 6 ]; then
    paused=$why
    tried=$((tried - 1)) left=$((left + 1))
  else
    n=$(( $(cat "$STATE/failures/$r" 2> /dev/null || echo 0) + 1 ))
    echo "$n" > "$STATE/failures/$r"
    log "$d ($camp, $state): push failed ($n so far): $why; the next round retries"
    failed=$((failed + 1))
  fi
done
if [ -n "$paused" ]; then
  log "paused after $tried push(es), for $paused: $left run dir(s) left for the next round"
elif [ "$left" != 0 ]; then
  log "round bounded ($tried push(es) in ${SECONDS}s, at most $MAX_RUNS or ${PASS_S}s): $left run dir(s) left for the next round"
fi
held=$(ls "$STATE/held" | wc -l)
[ "$held" = 0 ] || echo "$held run(s) held for a person ($STATE/held/RUN says why)"
[ "$failed" = 0 ] || log "$failed run(s) not yet on R2"
[ "$failed" = 0 ] && [ "$held" = 0 ]
