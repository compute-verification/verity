#!/usr/bin/env bash
# Node 1's custody: every ended run of its own under its runs roots that isn't on R2 once it has settled goes there through a one-off
# Job on this node with the Secret research-r2 (node 1's only R2 credential: neither host user has a store remote), running
# `research data custody RUN --publish --keep-local` from the research deploy.toml installs at VY_CUSTODY_SRC: the attempt with its run
# record (logs and declared outputs) into the run's own store, pushed to R2 and verified there, .custody beside the run. Nothing is
# copied: the run is here already. n2_custody.sh does the same for node 2's runs, through rsync and the same Jobs.
#
#   on node 1, as research (deploy.toml installs this file as vy-custody):
#     vy-custody [--dry-run]                one round (vy-custody.timer, every 10 min); --dry-run lists it
#     vy-custody --reconcile RUN --by WHO   a held conflict: the remote's attempt replaces node 1's, then the run publishes (node 2's
#                                           copies too, VY_CUSTODY_RELAY, when n2_custody.sh holds the conflict)
#
# Which runs: `research data custody --triage --older-min` over each runs root, run as its store's owner (VY_CUSTODY_ROOTS:
# runs=store=uid; the root's Jobs run as that uid too), the rows that need action and have settled, of runs this machine's kernel ran
# (job.json's boot id is one in STATE/boots): node 2's relay copies sit in /workspace/jobs/runs too, and n2_custody.sh owns those.
# A status that never became terminal (a pod preempted before its runner recorded an end) is published with --force --runner-gone,
# keeping the state it last wrote, only once its runner is seen gone: a host run's pid gone (force_refusal, here), a pod's
# `kubectl get pod POD` (job.json's hostname) NotFound, as backfill_attempts.sh asks. A pod Kubernetes still lists waits, however long
# its records have been quiet. A run that never ran its workload (a status before `running`, and none of ran_evidence's signs of a
# start: a pid, a started, span_start or sampler_start event, a sampler reading, output) is held, never published as evidence, until
# a `custody never-ran` label lifts it; any other is evidence, a killed run a failed attempt. `attempt-no-record` (an attempt
# published without its run record, which it can never gain) gets no Job: triage lists it until a person waives its logs. A local
# attempt published without its run record that never reached R2 (`attempt-unpreserved`) is held record-less: no Job, and each
# store's pending push skips every attempt without a run record (--skip-record-less), since R2's write-once key would keep it without
# its logs; one line counts them, and they fail the round until a person rules on them.
# Bounds as n2_custody.sh's: at most VY_CUSTODY_MAX_RUNS Jobs, none started after VY_CUSTODY_PASS_S, runs that failed before going
# last; a run dir over VY_CUSTODY_MAX_FILES files or VY_CUSTODY_MAX_GB GiB is held. A failed Job is logged with its last line and
# retried every round, except what no Job can mend, since an attempt key is write-once: a remote that holds another attempt document
# (triage's `conflict`), or a run dir that holds what its attempt's run record lacks (a run forced, then written to), each held after
# its one Job. A hold (STATE/held/RUN says why) fails the round, and lifts itself once triage no longer lists the run as
# needing action (a label, a fetch, --reconcile); remove it by hand to have the next round try again.
# No Job starts in node 1's daily hold, VY_CUSTODY_QUIET (UTC; dispatch.py's PACK_HOLD to the quiet hour's end), and none runs into
# the quiet hour (QUIET_AT, 12:30Z, which measurements own): each Job's activeDeadlineSeconds and the grace a stopped pod has (GRACE_S;
# on SIGTERM its bash exits, and research is killed) end it a minute before, to the second, and a Job that would get less than
# JOB_MIN_S isn't started (as n2_custody.sh's).
# Each store also gets `research data push --pending` in a Job of its own whenever none is left (a finished one stays
# VY_CUSTODY_PENDING_S): the labels, and the artifacts no run names, that no run's custody pushes. It is the store-push template's
# push on a schedule, for both stores.
# Exit 0 ok, 1 a run not yet on R2 or held, 2 usage, 3 refused.
set -u
ROOTS=${VY_CUSTODY_ROOTS:-/workspace/jobs/runs=/workspace/jobs/store=1000,/workspace/research/runs=/workspace/research/store=1001}
# node 2's copies, which n2_custody.sh's Jobs publish: never in a round, but --reconcile repairs a conflict there too
RELAY=${VY_CUSTODY_RELAY-/workspace/jobs/runs-n2=/workspace/jobs/store=1000}
STATE=${VY_CUSTODY_STATE:-/workspace/research/custody-n1}
SETTLE_S=${VY_CUSTODY_SETTLE_S:-900}
MAX_RUNS=${VY_CUSTODY_MAX_RUNS:-20}
PASS_S=${VY_CUSTODY_PASS_S:-1200}
MAX_FILES=${VY_CUSTODY_MAX_FILES:-20000}
MAX_GB=${VY_CUSTODY_MAX_GB:-50}
PENDING_S=${VY_CUSTODY_PENDING_S:-3600}
# the research and the Python the round runs on the host and its Jobs run with the Secret: root's (deploy.toml installs SRC; PYHOME
# is `uv python install --install-dir PYHOME --no-bin 3.12.14` as root), never under the world-writable /workspace/jobs
SRC=${VY_CUSTODY_SRC:-/workspace/custody}
PYHOME=${VY_CUSTODY_PYHOME:-/workspace/custody/python}
PY=${VY_CUSTODY_PY:-$PYHOME/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12}
QUIET=${VY_CUSTODY_QUIET-1210-1330}
QUIET_AT=1230   # node 1's quiet hour, which no Job runs into
JOB_MIN_S=300   # a Job left less time before it isn't started
GRACE_S=10      # a Job's pod stopped at its deadline: SIGTERM, then SIGKILL after this (terminationGracePeriodSeconds)
BOOT=${VY_CUSTODY_BOOT:-$(cat /proc/sys/kernel/random/boot_id)}
usage() { echo "usage: n1_custody.sh [--dry-run] | --reconcile RUN --by WHO" >&2; exit 2; }
DRY=0 REC= BY=
case "${1:-}" in
  "") ;;
  --dry-run) [ $# = 1 ] || usage; DRY=1 ;;
  --reconcile) [ $# = 4 ] && [ "$3" = --by ] && [ -n "$4" ] && [[ $2 =~ ^r[0-9a-z-]+$ ]] || usage; REC=$2 BY=$4 ;;
  *) usage ;;
esac
cd / || exit 3   # `python3 -` and `-m` put the working directory first on sys.path
mkdir -p "$STATE/failures" "$STATE/held"
exec 9> "$STATE/.lock"
if ! flock -n 9; then
  [ -z "$REC" ] || { echo "n1_custody.sh: a round holds $STATE/.lock; --reconcile after it" >&2; exit 3; }
  echo "n1_custody.sh: another round holds $STATE/.lock"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATE/log"; }
kube() { KUBECONFIG=${KUBECONFIG:-$HOME/.kube/config} kubectl "$@"; }   # as a root unit, root.env's k3s.yaml
# deadline MAX_S: the seconds a Job started now may run, at most MAX_S: to a minute before the quiet hour (QUIET_AT), less its grace,
# to the second; 0 in the daily hold
deadline() {
  [ -n "$QUIET" ] || { echo "$1"; return; }
  local t q0 q1 at s
  t=$(( $(date -u +%s) % 86400 )) at=$((10#${QUIET_AT:0:2} * 3600 + 10#${QUIET_AT:2:2} * 60))
  q0=$((10#${QUIET:0:2} * 3600 + 10#${QUIET:2:2} * 60)) q1=$((10#${QUIET:5:2} * 3600 + 10#${QUIET:7:2} * 60))
  if [ "$t" -ge "$q0" ] && [ "$t" -lt "$q1" ]; then s=0; else s=$(( (at - t + 86400) % 86400 - 60 - GRACE_S )); fi
  echo $(( s < $1 ? s : $1 ))
}
HOLD="node 1's daily hold ($QUIET UTC; no Job runs past ${QUIET_AT}Z, where its quiet hour starts)"
human() { awk -v b="$1" 'BEGIN { split("B KiB MiB GiB TiB", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ "$(deadline 1800)" -lt "$JOB_MIN_S" ]; then
  [ -z "$REC" ] || { echo "n1_custody.sh: $HOLD: no Job starts; --reconcile after it" >&2; exit 3; }
  echo "n1_custody.sh: $HOLD: no Job starts"
  exit 0
fi
if [ ! -f "$SRC/tools/research/src/research/store/custody.py" ] || [ ! -f "$SRC/tools/research/store.pod.toml" ]; then
  log "refused: no research tree at $SRC (deploy.toml installs it: research deploy install vy-nebius-1 --by WHO" \
    "$SRC/tools/research/src/research $SRC/tools/research/store.pod.toml)"
  exit 3
fi
# untrusted PATH...: the first of each PATH (resolved), a file under it, or a directory above it or above the path it resolves to
# that a uid other than root and this round's could change (owned by another, or writable by group or others; a sticky directory
# above aside), or nothing. uv's empty .lock is 0666.
untrusted() {
  local p q d o m bad lock
  lock=$(realpath -m "$PYHOME")/.lock
  for p; do
    q=$(realpath -e "$p" 2> /dev/null) || { echo "$p (missing)"; return; }
    for d in "$p" "$q"; do
      while [ "$d" != / ]; do
        d=$(dirname "$d")
        read -r o m <<< "$(stat -c '%u %a' "$d")"
        if { [ "$o" != 0 ] && [ "$o" != "$EUID" ]; } || (( (8#$m & 8#022) && !(8#$m & 8#1000) )); then echo "$d"; return; fi
      done
    done
    bad=$(find "$q" -path "$lock" -prune -o \( \( ! -uid 0 ! -uid "$EUID" \) -o \( -perm /022 ! -type l \) \) -print -quit 2>&1)
    [ -z "$bad" ] || { echo "$bad"; return; }
  done
}
case $(realpath -e "$PY" 2> /dev/null) in
  "$(realpath -e "$PYHOME" 2> /dev/null)"/*) ;;
  *) log "refused: $PY is no interpreter under $PYHOME (as root: uv python install --install-dir $PYHOME --no-bin 3.12.14)"; exit 3 ;;
esac
if bad=$(untrusted "$SRC/tools/research" "$PYHOME") && [ -n "$bad" ]; then
  log "refused: $bad could be changed by a uid other than root and $(id -un): the round runs $SRC's research and $PY on this host" \
    "and in Jobs with the Secret research-r2, so every file of them, and each directory above, is root's and writable by root alone"
  exit 3
fi

# job NAME UID TTL DEADLINE_S STORE ROOT KIND ARG...: a one-off Job on this node as UID with the Secret research-r2 that runs KIND's
# fixed script (custody RUN_DIR [--force --runner-gone], reconcile RUN RUN_DIR, push --skip-record-less): research from SRC
# with PY, RESEARCH_STORE=STORE and RESEARCH_RUNS=ROOT. A run dir's data reach it only as the script's arguments, never as shell
# text or JSON written by hand.
job() {
  kube delete job "$1" --ignore-not-found > /dev/null || return 1
  PYTHONDONTWRITEBYTECODE=1 python3 - "${VY_DISPATCH_IMAGE:-verity-sky-gpu:3dbe60044f88c462}" \
    "$SRC" "$PY" "$GRACE_S" "$@" << 'EOF' | kube create -f - > /dev/null
import json, re, sys

image, src, py, grace, name, uid, ttl, deadline, store, root, kind, *args = sys.argv[1:]
# bash is the container's init, which ignores SIGTERM unless trapped: on it bash exits and the kernel SIGKILLs research. A rerun
# repeats what that cuts short: .custody and every store write go to a temp file, then a rename
run = ('trap "exit 143" TERM; cd / && export HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$1/tools/research/src" RESEARCH_STORE="$3" '
       'RESEARCH_RUNS="$4" || exit 1; { ')
run += {"custody": '"$2" -m research data custody "$5" --publish --keep-local "${@:6}"',
        "reconcile": '"$2" -m research data reconcile-attempts "$5" --apply && "$2" -m research data custody "$6" --publish --keep-local',
        "push": '"$2" -m research data push --pending --jobs 8 "${@:5}"'}[kind]
run += '; } & wait $!'
# each kind's arguments, where _ is a run or a run dir, never an option
shapes = {"custody": [["_"], ["_", "--force", "--runner-gone"]], "reconcile": [["_", "_"]], "push": [["--skip-record-less"]]}[kind]
fits = any(len(s) == len(args) and all(a == w or (w == "_" and a[:1] not in ("", "-")) for a, w in zip(args, s)) for s in shapes)
if not re.fullmatch(r"n1-[a-z0-9-]{1,60}[a-z0-9]", name) or int(deadline) <= 0 or not fits:
    sys.exit(f"n1_custody.sh: no Job {name!r} with deadline {deadline!r} and {kind} {args!r}")
labels = {"verity.dev/lane": "n1-custody"}
json.dump({"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": name, "labels": {**labels, "verity.dev/kind": "custody"}},
           "spec": {"backoffLimit": 0, "activeDeadlineSeconds": int(deadline), "ttlSecondsAfterFinished": int(ttl), "template": {
               "metadata": {"labels": labels}, "spec": {
                   "restartPolicy": "Never", "terminationGracePeriodSeconds": int(grace), "nodeSelector": {"verity.dev/server": "vy-nebius-1"},
                   "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                   "securityContext": {"runAsUser": int(uid), "runAsGroup": int(uid)},
                   "containers": [{"name": "main", "image": image, "imagePullPolicy": "IfNotPresent",
                                   "command": ["taskset", "-c", "96-127", "bash", "-c", run, "n1-custody", src, py, store, root, *args],
                                   "envFrom": [{"secretRef": {"name": "research-r2"}}],
                                   "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
                                   "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}]}],
                   "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}}]}}}}, sys.stdout)
EOF
}

# a failed Job never completes: wait for either condition (its own deadline ends it), then keep its last lines in $tmp/job.err
wait_job() {
  local c
  for _ in $(seq 400); do
    c=$(kube get job "$1" -o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}')
    case $c in
      *Complete=True*) return 0 ;;
      *Failed=True*) kube logs "job/$1" --tail=3 > "$tmp/job.err" 2>&1; return 1 ;;
    esac
    sleep 5
  done
  echo "Job not done in 33 min" > "$tmp/job.err"
  return 1
}

custody_job() {  # custody_job RUN ROOT STORE UID DEADLINE_S KIND ARG...: 0 when the Job completed, else 1 with why=
  local name=n1-custody-$(echo "$1" | tr -cd 'a-z0-9-' | tail -c 40)
  : > "$tmp/job.err"
  if ! job "$name" "$4" 3600 "$5" "$3" "$2" "${@:6}" 2> "$tmp/job.err" || ! wait_job "$name"; then
    why="Job $name: $(grep -v '^$' "$tmp/job.err" | tail -1)"
    return 1
  fi
}

# pod_gone POD: 0 when Kubernetes says the run's pod is NotFound (dispatch's pods are in default), else 1 with seen= what it said
pod_gone() {
  seen=$(kube get pod "$1" --namespace=default --output=name --request-timeout=30s 2>&1)
  case $?:$seen in
    0:*) seen="its pod $1 still runs"; return 1 ;;
    *NotFound*) return 0 ;;
    *) seen="\`kubectl get pod $1\` failed: $(echo "$seen" | grep -v '^$' | tail -1)"; return 1 ;;
  esac
}

declare -A STORE UID_OF
IFS=, read -ra specs <<< "$ROOTS"
for spec in "${specs[@]}" ${RELAY:+"$RELAY"}; do
  IFS='=' read -r root store uid <<< "$spec"
  STORE[$root]=$store UID_OF[$root]=$uid
done

if [ -n "$REC" ]; then
  for spec in "${specs[@]}" ${RELAY:+"$RELAY"}; do
    root=${spec%%=*}
    [ -d "$root/$REC" ] || continue
    dl=$(deadline 1800)
    [ "$dl" -ge "$JOB_MIN_S" ] || { echo "n1_custody.sh: $HOLD: no Job starts; --reconcile after it" >&2; exit 3; }
    if custody_job "$REC" "$root" "${STORE[$root]}" "${UID_OF[$root]}" "$dl" reconcile "$REC" "$root/$REC"; then
      rm -f "$STATE/held/$REC" "$STATE/failures/$REC"
      boot=$(python3 -c 'import json, sys; print((json.load(open(sys.argv[1])).get("environment") or {}).get("boot_id") or "")' \
        "$root/$REC/job.json" 2> /dev/null)
      log "$root/$REC: reconciled by $BY: the remote's attempt replaced node 1's (kept under attempts/.superseded/); on R2 with its run record" \
        "$(grep -qsx -- "${boot:-?}" "$STATE/boots" || echo "(node 2's copy: remove n2_custody.sh's hold on node 2, and its next round brings the .custody home)")"
      exit 0
    fi
    if grep -qs "names no run record" "$tmp/job.err"; then
      rm -f "$STATE/held/$REC" "$STATE/failures/$REC"
      log "$root/$REC: reconciled by $BY, but the remote's attempt names no run record: its logs can only be waived (label custody waived)"
    else
      log "$root/$REC: --reconcile by $BY failed: $why"
    fi
    exit 1
  done
  echo "n1_custody.sh: no run dir $REC under ${!STORE[*]}" >&2
  exit 2
fi

[ "$DRY" = 1 ] || grep -qsx -- "$BOOT" "$STATE/boots" || echo "$BOOT" >> "$STATE/boots"
i=0
for spec in "${specs[@]}"; do
  IFS='=' read -r root store uid <<< "$spec"
  i=$((i + 1))
  # no remote (R2 is the Jobs' alone): never a store.toml under a HOME, such as /tmp, that another uid could write first
  (cd / && sudo -n -u "#$uid" env HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SRC/tools/research/src" RESEARCH_STORE="$store" \
    RESEARCH_STORE_CONFIG=/dev/null python3 -m research data custody --triage --runs-dir "$root" --older-min "$((SETTLE_S / 60))" --json) > "$tmp/triage.$i" 2> "$tmp/triage.err"
  rc=$?
  if [ "$rc" -gt 1 ]; then
    log "refused: triage of $root as uid $uid failed ($rc): $(grep -v '^$' "$tmp/triage.err" | tail -1)"
    exit 3
  fi
done

# What to do with each run that needs action, one line each: "job|ROOT|RUN|GONE|CAMPAIGN|STATUS|FILES|BYTES" (runs that never failed
# first; GONE, for a status that isn't terminal: `host` when this host saw its runner gone, else its pod's name to ask Kubernetes),
# "hold|ROOT|RUN|WHY" (a hold to write), "recordless|ROOT|RUN" (a record-less attempt to leave unpushed), "lift|RUN" (a hold to
# remove), and one "note|..." summary
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SRC/tools/research/src" python3 - "$tmp" "$STATE" "$BOOT" "$MAX_FILES" "$MAX_GB" "$ROOTS" > "$tmp/plan" << 'EOF' || { log "refused: no plan"; exit 3; }
import json, re, socket, sys
from collections import Counter
from pathlib import Path

from research.store.attempt import TERMINAL
from research.store.custody import NEEDS_ACTION, force_refusal, ran_evidence
from research.telemetry import STATES

tmp, state, boot, max_files, max_bytes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) << 30
boots = {boot, *((state / "boots").read_text().split() if (state / "boots").exists() else [])}
held = {p.name for p in (state / "held").iterdir()}
here = socket.gethostname()


def read(p):
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def field(s):  # a run dir's text, for a plan line and the log: its writer is any workload
    return re.sub(r"[^A-Za-z0-9._:@+/-]", "?", str(s))[:80] or "?"


fresh, retry, holds, recordless, seen, skipped = [], [], [], [], set(), Counter()
for i, (root, store) in enumerate(s.split("=")[:2] for s in sys.argv[6].split(",")):
    for row in json.loads((tmp / f"triage.{i + 1}").read_text())["rows"]:
        r, st = row["run"], row["state"]
        if st not in NEEDS_ACTION:
            continue
        seen.add(r)
        d = Path(root) / r
        # an attempt published without a run record that never reached R2: pushed, its write-once key would hold no logs for good
        if st == "attempt-unpreserved" and re.fullmatch(r"r[0-9a-z-]{1,63}", r) and not (
                (read(Path(store) / "attempts" / f"{r}.json").get("outputs") or {}).get("run_record")):
            recordless.append(f"recordless|{root}|{r}")
            continue
        job = read(d / "job.json")
        if (job.get("environment") or {}).get("boot_id") not in boots or not re.fullmatch(r"r[0-9a-z-]{1,63}", r):
            skipped["not run by this machine"] += 1
            continue
        if row["settling"] or r in held:
            skipped["settling" if row["settling"] else "held"] += 1
            continue
        if st == "attempt-no-record":
            skipped[st] += 1
            continue
        status = read(d / "status.json").get("state") or "no status"
        camp = ((job.get("store") or {}).get("campaign")) or "-"
        if status not in (*STATES, "no status"):
            holds.append(f"hold|{root}|{r}|its status.json says a state the runner never writes ({field(camp)})")
            continue
        # `running` is what the runner writes once it spawned the workload: a run killed after that is a failed attempt, and evidence
        if status in ("no status", "submitted", "admitted") and ran_evidence(d) is None:
            holds.append(f"hold|{root}|{r}|never ran its workload ({field(camp)}, status {status}: no pid, start event, sampler reading or output)")
            continue
        if row["files"] > max_files or row["bytes"] > max_bytes:
            holds.append(f"hold|{root}|{r}|over " + (f"{max_files} files" if row["files"] > max_files else f"{max_bytes >> 30} GiB"))
            continue
        gone, host = "", (job.get("environment") or {}).get("hostname") or ""
        if status not in TERMINAL:
            # a hostname of 63 characters may be a longer pod name cut short, which Kubernetes would call NotFound
            if force_refusal(d) is None:
                gone = "host"
            elif host != here and len(host) < 63 and re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", host):
                gone = host
            else:
                skipped["its runner not seen gone"] += 1
                continue
        line = f"job|{root}|{r}|{gone}|{field(camp)}|{status}|{int(row['files'])}|{int(row['bytes'])}"
        (retry if (state / "failures" / r).exists() else fresh).append(line)
for line in [*fresh, *retry, *holds, *recordless]:
    print(line)
for r in sorted(held - seen):
    print(f"lift|{r}")
print("note|" + ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())))
EOF

failed=0 tried=0 left=0 waiting=0 recordless=0
while IFS='|' read -r kind a b c camp status files bytes; do
  case $kind in
    note)
      [ -z "$a" ] || echo "no Job: $a (triage lists them)" ;;
    recordless)
      recordless=$((recordless + 1)) ;;
    lift)
      if [ "$DRY" = 1 ]; then echo "would lift the hold on $a: triage no longer lists it as needing action"; continue; fi
      rm -f "$STATE/held/$a" "$STATE/failures/$a"
      log "$a: hold lifted: triage no longer lists it as needing action" ;;
    hold)
      root=$a r=$b why=$c
      if [ "$DRY" = 1 ]; then echo "would hold $root/$r: $why"; continue; fi
      echo "$why" > "$STATE/held/$r"
      case $why in
        never*) log "$root/$r: held: $why: not published as evidence; label it (as uid ${UID_OF[$root]}, RESEARCH_STORE=${STORE[$root]}:" \
          "research data label $r custody never-ran --by WHO) and the hold lifts itself" ;;
        its*) log "$root/$r: held: $why: no round publishes it; a person reads the run dir (any workload may write it), then" \
          "rm $STATE/held/$r" ;;
        *) log "$root/$r: held, $why: no round publishes it; move the bulk out of the run dir (or declare it elsewhere), then rm $STATE/held/$r" ;;
      esac ;;
    job)
      root=$a r=$b gone=$c d=$a/$b force=() seen=
      dl=$(deadline 1800)
      if [ "$DRY" != 1 ] && { [ "$tried" -ge "$MAX_RUNS" ] || [ "$SECONDS" -ge "$PASS_S" ] || [ "$dl" -lt "$JOB_MIN_S" ]; }; then
        left=$((left + 1))
        continue
      fi
      if [ -n "$gone" ]; then
        if [ "$gone" = host ]; then
          seen="its runner's pid is gone"
        elif pod_gone "$gone"; then
          seen="its pod $gone is NotFound"
        else
          if [ "$DRY" = 1 ]; then echo "would leave $d ($camp, $status): $seen"; continue; fi
          case $seen in
            "its pod "*) waiting=$((waiting + 1)) ;;
            *) log "$d ($camp, $status): not published: $seen; the next round asks again"; failed=$((failed + 1)) ;;
          esac
          continue
        fi
        force=(--force --runner-gone)
      fi
      if [ "$DRY" = 1 ]; then echo "would publish $d ($camp, $status, $files files, $(human "$bytes"))${force:+ with ${force[*]} ($seen)}"; continue; fi
      tried=$((tried + 1))
      why=
      if custody_job "$r" "$root" "${STORE[$root]}" "${UID_OF[$root]}" "$dl" custody "$d" "${force[@]}"; then
        rm -f "$STATE/failures/$r"
        log "$d ($camp, $status, $files files, $(human "$bytes")): on R2 with its run record, .custody beside it"
      elif grep -qs "holds a DIFFERENT attempt document" "$tmp/job.err"; then
        rm -f "$STATE/failures/$r"
        echo "conflict: the remote holds another attempt document for this run id" > "$STATE/held/$r"
        log "$d ($camp, $status): held: R2 holds another attempt document for it (write-once; triage lists it as conflict, and" \
          "the store's pending push fails on it): vy-custody --reconcile $r --by WHO (the remote's copy is the first ever pushed, and canonical)"
      elif grep -qs "names no run record" "$tmp/job.err"; then
        rm -f "$STATE/failures/$r"
        log "$d ($camp, $status): its attempt is on R2 without a run record (published without one, and an attempt is immutable):" \
          "triage lists it as attempt-no-record until its logs are waived"
      elif grep -qs -e "not in the run record" -e "changed since it was taken" "$tmp/job.err"; then
        rm -f "$STATE/failures/$r"
        echo "changed: the run dir holds what its attempt's run record lacks" > "$STATE/held/$r"
        log "$d ($camp, $status): held: its attempt's run record, taken from the run dir as it was earlier, lacks what it holds now," \
          "and an attempt never changes: the rest is on R2 with that record, the difference only in this run dir. Keep it, or put it in" \
          "the store (research data put --kind K --tree DIR --preserve), then label it (as uid ${UID_OF[$root]}," \
          "RESEARCH_STORE=${STORE[$root]}: research data label $r custody waived --by WHO) and the hold lifts itself"
      else
        n=$(( $(cat "$STATE/failures/$r" 2> /dev/null || echo 0) + 1 ))
        echo "$n" > "$STATE/failures/$r"
        log "$d ($camp, $status): Job failed ($n so far): $why; the next round retries"
        failed=$((failed + 1))
      fi ;;
  esac
done < "$tmp/plan"

# each store's pending push: started when no earlier one is left, never waited on; a failed one is reported until its ttl removes it
declare -A done_store
for spec in "${specs[@]}"; do
  IFS='=' read -r root store uid <<< "$spec"
  [ -z "${done_store[$store]:-}" ] || continue
  done_store[$store]=1
  name=n1-store-push-$(basename "$(dirname "$store")" | tr -cd 'a-z0-9-')
  if c=$(kube get job "$name" -o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}' 2> /dev/null); then
    case $c in
      *Failed=True*) log "$store: the pending push failed (Job $name: $(kube logs "job/$name" --tail=1 2>&1)); the next starts once its ttl ends"
        failed=$((failed + 1)) ;;
    esac
  elif [ "$DRY" = 1 ]; then
    echo "would start $store's pending push (Job $name)"
  elif dl=$(deadline 7200) && [ "$dl" -lt "$JOB_MIN_S" ]; then
    echo "$store: no pending push: $HOLD"
  elif job "$name" "$uid" "$PENDING_S" "$dl" "$store" "$root" push --skip-record-less; then
    echo "$store: pending push started (Job $name, ${dl}s at most)"
  else
    log "$store: the pending push did not start (Job $name)"
    failed=$((failed + 1))
  fi
done
[ "$waiting" = 0 ] || echo "$waiting run(s) wait on their pod, which Kubernetes still lists: a status that isn't terminal is its end only once the pod is gone"
[ "$left" = 0 ] || log "round bounded ($tried Job(s) in ${SECONDS}s, at most $MAX_RUNS or ${PASS_S}s): $left run(s) left for the next round"
held=$(ls "$STATE/held" | wc -l)
[ "$held" = 0 ] || echo "$held run(s) held ($STATE/held/)"
[ "$recordless" = 0 ] || echo "$recordless record-less attempt(s) held (triage's attempt-unpreserved): published here without a run record" \
  "and never pushed, so R2's write-once key would keep each without its logs; no Job or pending push (--skip-record-less) sends them," \
  "and nothing replaces them, until a person rules (pushed as they are, or each replaced here by an attempt that names its run record)"
[ "$failed" = 0 ] || log "$failed run(s) not yet on R2"
[ "$failed" = 0 ] && [ "$held" = 0 ] && { [ "$recordless" = 0 ] || [ "$DRY" = 1 ]; }
