#!/usr/bin/env bash
# Node 1's custody: every ended run of its own under its runs roots that isn't on R2 once it has settled goes there through a one-off
# Job on this node with the Secret research-r2 (node 1's only R2 credential: neither host user has a store remote), running
# `research data custody RUN --publish --keep-local` from the research deploy.toml installs at VY_CUSTODY_SRC: the attempt with its run
# record (logs and declared outputs) into the run's own store, pushed to R2 and verified there, .custody beside the run. Nothing is
# copied: the run is here already. n2_custody.sh does the same for node 2's runs, through rsync and the same Jobs.
#
#   on node 1:  n1_custody.sh [--dry-run]                one round (vy-custody.timer, every 10 min, as research); --dry-run lists it
#               n1_custody.sh --reconcile RUN --by WHO   a held conflict: the remote's attempt replaces node 1's, then the run publishes
#
# Which runs: `research data custody --triage --older-min` over each runs root, run as its store's owner (VY_CUSTODY_ROOTS:
# runs=store=uid; the root's Jobs run as that uid too), the rows that need action and have settled, of runs this machine's kernel ran
# (job.json's boot id is one in STATE/boots): node 2's relay copies sit in /workspace/jobs/runs too, and n2_custody.sh owns those.
# A status that never became terminal (a pod preempted before its runner recorded an end: triage's dead, or stale after 24 h) is
# published with --force and keeps the state it last wrote, unless the run never ran its workload (empty stdout and stderr, no file
# but the runner's bookkeeping): that is held, never published as evidence. `attempt-no-record` (an attempt published without its run
# record, which it can never gain) gets no Job: triage lists it until a person waives its logs.
# Bounds as n2_custody.sh's: at most VY_CUSTODY_MAX_RUNS Jobs, none started after VY_CUSTODY_PASS_S, runs that failed before going
# last; a run dir over VY_CUSTODY_MAX_FILES files or VY_CUSTODY_MAX_GB GiB is held. A failed Job is logged with its last line and
# retried every round, except a remote that holds another attempt document (triage's `conflict`; an attempt key is write-once), which
# is held after its one Job. A hold (STATE/held/RUN says why) fails the round, and lifts itself once triage no longer lists the run as
# needing action (a label, a fetch, --reconcile); remove it by hand to have the next round try again.
# No Job starts in node 1's daily hold, VY_CUSTODY_QUIET (UTC; dispatch.py's PACK_HOLD to the quiet hour's end).
# Each store also gets `research data push --pending` in a Job of its own whenever none is left (a finished one stays
# VY_CUSTODY_PENDING_S): the labels, and the artifacts no run names, that no run's custody pushes. It is the store-push template's
# push on a schedule, for both stores.
# Exit 0 ok, 1 a run not yet on R2 or held, 2 usage, 3 refused.
set -u
ROOTS=${VY_CUSTODY_ROOTS:-/workspace/jobs/runs=/workspace/jobs/store=1000,/workspace/research/runs=/workspace/research/store=1001}
STATE=${VY_CUSTODY_STATE:-/workspace/research/custody-n1}
SETTLE_S=${VY_CUSTODY_SETTLE_S:-900}
MAX_RUNS=${VY_CUSTODY_MAX_RUNS:-20}
PASS_S=${VY_CUSTODY_PASS_S:-1200}
MAX_FILES=${VY_CUSTODY_MAX_FILES:-20000}
MAX_GB=${VY_CUSTODY_MAX_GB:-50}
PENDING_S=${VY_CUSTODY_PENDING_S:-3600}
SRC=${VY_CUSTODY_SRC:-/workspace/jobs/custody}
PY=${VY_CUSTODY_PY:-/workspace/jobs/cache/python/cpython-3.12-linux-x86_64-gnu/bin/python3}   # any uid runs it; uv's cache is ubuntu's
QUIET=${VY_CUSTODY_QUIET-1210-1330}
BOOT=${VY_CUSTODY_BOOT:-$(cat /proc/sys/kernel/random/boot_id)}
usage() { echo "usage: n1_custody.sh [--dry-run] | --reconcile RUN --by WHO" >&2; exit 2; }
DRY=0 REC= BY=
case "${1:-}" in
  "") ;;
  --dry-run) [ $# = 1 ] || usage; DRY=1 ;;
  --reconcile) [ $# = 4 ] && [ "$3" = --by ] && [ -n "$4" ] && [[ $2 =~ ^r[0-9a-z-]+$ ]] || usage; REC=$2 BY=$4 ;;
  *) usage ;;
esac
mkdir -p "$STATE/failures" "$STATE/held"
exec 9> "$STATE/.lock"
if ! flock -n 9; then
  [ -z "$REC" ] || { echo "n1_custody.sh: a round holds $STATE/.lock; --reconcile after it" >&2; exit 3; }
  echo "n1_custody.sh: another round holds $STATE/.lock"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATE/log"; }
kube() { KUBECONFIG=$HOME/.kube/config kubectl "$@"; }
quiet() { now=$(date -u +%H%M); [ -n "$QUIET" ] && [ "$now" -ge "${QUIET%-*}" ] && [ "$now" -lt "${QUIET#*-}" ]; }
human() { awk -v b="$1" 'BEGIN { split("B KiB MiB GiB TiB", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if quiet; then
  [ -z "$REC" ] || { echo "n1_custody.sh: node 1's daily hold ($QUIET UTC): no Job starts; --reconcile after it" >&2; exit 3; }
  echo "n1_custody.sh: node 1's daily hold ($QUIET UTC): no Job starts"
  exit 0
fi
if [ ! -f "$SRC/tools/research/src/research/store/custody.py" ] || [ ! -f "$SRC/tools/research/store.pod.toml" ]; then
  log "refused: no research tree at $SRC (deploy.toml installs it: research deploy install vy-nebius-1 --by WHO" \
    "$SRC/tools/research/src/research $SRC/tools/research/store.pod.toml)"
  exit 3
fi

# job NAME UID STORE ROOT TTL DEADLINE_S CMD: a one-off Job on this node as UID with the Secret research-r2, research from SRC
job() {
  kube delete job "$1" --ignore-not-found > /dev/null && kube create -f - > /dev/null << EOF
{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "$1", "labels": {"verity.dev/lane": "n1-custody", "verity.dev/kind": "custody"}},
 "spec": {"backoffLimit": 0, "activeDeadlineSeconds": $6, "ttlSecondsAfterFinished": $5, "template": {"metadata": {"labels": {"verity.dev/lane": "n1-custody"}}, "spec": {
  "restartPolicy": "Never", "nodeSelector": {"verity.dev/server": "vy-nebius-1"},
  "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
  "securityContext": {"runAsUser": $2, "runAsGroup": $2},
  "containers": [{"name": "main", "image": "${VY_DISPATCH_IMAGE:-us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216}",
   "imagePullPolicy": "IfNotPresent",
   "command": ["taskset", "-c", "96-127", "bash", "-c", "cd / && export HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC/tools/research/src RESEARCH_STORE=$3 RESEARCH_RUNS=$4 && $7"],
   "envFrom": [{"secretRef": {"name": "research-r2"}}],
   "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
   "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}]}],
  "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}}]}}}}
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

custody_job() {  # custody_job RUN ROOT STORE UID CMD: 0 when the Job completed, else 1 with why=
  local name=n1-custody-$(echo "$1" | tr -cd 'a-z0-9-' | tail -c 40)
  : > "$tmp/job.err"
  if ! job "$name" "$4" "$3" "$2" 3600 1800 "$5" 2> "$tmp/job.err" || ! wait_job "$name"; then
    why="Job $name: $(grep -v '^$' "$tmp/job.err" | tail -1)"
    return 1
  fi
}

declare -A STORE UID_OF
IFS=, read -ra specs <<< "$ROOTS"
for spec in "${specs[@]}"; do
  IFS='=' read -r root store uid <<< "$spec"
  STORE[$root]=$store UID_OF[$root]=$uid
done

if [ -n "$REC" ]; then
  for spec in "${specs[@]}"; do
    root=${spec%%=*}
    [ -d "$root/$REC" ] || continue
    research="$PY -m research data"
    if custody_job "$REC" "$root" "${STORE[$root]}" "${UID_OF[$root]}" \
      "$research reconcile-attempts $REC --apply && $research custody $root/$REC --publish --keep-local"; then
      rm -f "$STATE/held/$REC" "$STATE/failures/$REC"
      log "$root/$REC: reconciled by $BY: the remote's attempt replaced node 1's (kept under attempts/.superseded/); on R2 with its run record"
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
  (cd / && sudo -n -u "#$uid" env HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SRC/tools/research/src" RESEARCH_STORE="$store" \
    python3 -m research data custody --triage --runs-dir "$root" --older-min "$((SETTLE_S / 60))" --json) > "$tmp/triage.$i" 2> "$tmp/triage.err"
  rc=$?
  if [ "$rc" -gt 1 ]; then
    log "refused: triage of $root as uid $uid failed ($rc): $(grep -v '^$' "$tmp/triage.err" | tail -1)"
    exit 3
  fi
done

# What to do with each run that needs action, one line each: "job|ROOT|RUN|FORCE|CAMPAIGN|STATUS|FILES|BYTES" (runs that never failed
# first), "hold|ROOT|RUN|WHY" (a hold to write), "lift|RUN|STATE" (a hold to remove), and one "note|..." summary
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SRC/tools/research/src" python3 - "$tmp" "$STATE" "$BOOT" "$MAX_FILES" "$MAX_GB" "$ROOTS" > "$tmp/plan" << 'EOF' || { log "refused: no plan"; exit 3; }
import json, re, sys
from collections import Counter
from pathlib import Path

from research.store.attempt import TERMINAL
from research.store.custody import NEEDS_ACTION, run_files

tmp, state, boot, max_files, max_bytes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) << 30
boots = {boot, *((state / "boots").read_text().split() if (state / "boots").exists() else [])}
BOOKKEEPING = {"job.json", "status.json", "stdout.log", "stderr.log", "launcher.log", "sampler.log", "sampler.json", "resources.jsonl",
               "events.jsonl", "launch.json"}
held = {p.name for p in (state / "held").iterdir()}


def read(p):
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def field(s):
    return str(s).replace("|", "/").replace("\n", " ")


fresh, retry, holds, seen, skipped = [], [], [], set(), Counter()
for i, root in enumerate(s.split("=")[0] for s in sys.argv[6].split(",")):
    for row in json.loads((tmp / f"triage.{i + 1}").read_text())["rows"]:
        r, st = row["run"], row["state"]
        if st not in NEEDS_ACTION:
            continue
        seen.add(r)
        d = Path(root) / r
        job = read(d / "job.json")
        if (job.get("environment") or {}).get("boot_id") not in boots or not re.fullmatch(r"r[0-9a-z-]+", r):
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
        if status not in TERMINAL:
            files = run_files(d)
            if all(f in BOOKKEEPING for f in files) and not any((d / f).exists() and (d / f).stat().st_size for f in ("stdout.log", "stderr.log")):
                holds.append(f"hold|{root}|{r}|never ran its workload ({field(camp)}, status {status}, no output, only the runner's bookkeeping)")
                continue
        if row["files"] > max_files or row["bytes"] > max_bytes:
            holds.append(f"hold|{root}|{r}|over " + (f"{max_files} files" if row["files"] > max_files else f"{max_bytes >> 30} GiB"))
            continue
        line = f"job|{root}|{r}|{'--force' if status not in TERMINAL else ''}|{field(camp)}|{status}|{row['files']}|{row['bytes']}"
        (retry if (state / "failures" / r).exists() else fresh).append(line)
for line in [*fresh, *retry, *holds]:
    print(line)
for r in sorted(held - seen):
    print(f"lift|{r}")
print("note|" + ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())))
EOF

failed=0 tried=0 left=0
while IFS='|' read -r kind a b c camp status files bytes; do
  case $kind in
    note)
      [ -z "$a" ] || echo "no Job: $a (triage lists them)" ;;
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
        *) log "$root/$r: held, $why: no round publishes it; move the bulk out of the run dir (or declare it elsewhere), then rm $STATE/held/$r" ;;
      esac ;;
    job)
      root=$a r=$b force=$c d=$a/$b
      if [ "$DRY" = 1 ]; then echo "would publish $d ($camp, $status, $files files, $(human "$bytes"))${force:+ with --force}"; continue; fi
      if [ "$tried" -ge "$MAX_RUNS" ] || [ "$SECONDS" -ge "$PASS_S" ] || quiet; then
        left=$((left + 1))
        continue
      fi
      tried=$((tried + 1))
      why=
      if custody_job "$r" "$root" "${STORE[$root]}" "${UID_OF[$root]}" "$PY -m research data custody $d --publish --keep-local $force"; then
        rm -f "$STATE/failures/$r"
        log "$d ($camp, $status, $files files, $(human "$bytes")): on R2 with its run record, .custody beside it"
      elif grep -qs "holds a DIFFERENT attempt document" "$tmp/job.err"; then
        rm -f "$STATE/failures/$r"
        echo "conflict: the remote holds another attempt document for this run id" > "$STATE/held/$r"
        log "$d ($camp, $status): held: R2 holds another attempt document for it (write-once; triage lists it as conflict, and" \
          "the store's pending push fails on it): n1_custody.sh --reconcile $r --by WHO (the remote's copy is the first ever pushed, and canonical)"
      elif grep -qs "names no run record" "$tmp/job.err"; then
        rm -f "$STATE/failures/$r"
        log "$d ($camp, $status): its attempt is on R2 without a run record (published without one, and an attempt is immutable):" \
          "triage lists it as attempt-no-record until its logs are waived"
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
  elif quiet; then
    :
  elif job "$name" "$uid" "$store" "$root" "$PENDING_S" 7200 "$PY -m research data push --pending --jobs 8"; then
    echo "$store: pending push started (Job $name)"
  else
    log "$store: the pending push did not start (Job $name)"
    failed=$((failed + 1))
  fi
done
[ "$left" = 0 ] || log "round bounded ($tried Job(s) in ${SECONDS}s, at most $MAX_RUNS or ${PASS_S}s): $left run(s) left for the next round"
held=$(ls "$STATE/held" | wc -l)
[ "$held" = 0 ] || echo "$held run(s) held ($STATE/held/)"
[ "$failed" = 0 ] || log "$failed run(s) not yet on R2"
[ "$failed" = 0 ] && [ "$held" = 0 ]
