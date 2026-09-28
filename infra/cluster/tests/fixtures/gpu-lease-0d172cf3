#!/usr/bin/env bash
# Share one machine's GPUs between concurrent runs: one lock file per GPU, held with flock(1).
#
#   gpu-lease N [--wait] [--on LIST] [--max-min M] [--mem-gb G] [--preemptible] [--timed] -- CMD...
#                                                take N free GPUs and run CMD with CUDA_VISIBLE_DEVICES naming them
#   gpu-lease status                             one line per GPU: index, free or held, and the holder: minutes held, pid, run
#                                                id, who, UUID, start and command
#
# A GPU is held while any process has its lock ($GPU_LEASE_DIR/<index>.lock) open: CMD and every child it starts inherit the
# descriptors, and the kernel drops the lock when the last of them exits, so a killed run never leaks a GPU and there is nothing
# to clean up.  CUDA_DEVICE_ORDER=PCI_BUS_ID makes the indices nvidia-smi's.  Too few free GPUs exits 75 (EX_TEMPFAIL) at once,
# holding nothing; --wait retries every $GPU_LEASE_POLL_S seconds (default 15) instead.
# Holding nothing is also how a CPU-only run shares the machine: it simply doesn't call gpu-lease.
# GPU_LEASE_COUNT overrides the GPU count nvidia-smi reports (tests; a machine without nvidia-smi).
# On a machine whose other GPUs belong to Kubernetes (the Nebius cluster), $GPU_LEASE_ALLOWED_FILE (default /etc/vy/direct-gpus)
# names the only indices gpu-lease may hand out, comma-separated, or `none` (every GPU is Kubernetes'): it then refuses. The file
# is written by the cluster's bring-up (cluster_up.sh), never by hand. Absent, every GPU is gpu-lease's.
#
# --on LIST takes the N GPUs only from LIST, comma-separated indices or UUIDs (GPU-...), in that order; a GPU outside the allowed
# set is refused.  CMD gets GPU_LEASE_UUID, the UUIDs it holds (comma-separated), to record with its results.  The holder is named
# by $GPU_LEASE_WHO (else $RESEARCH_LANE, else the user), so `status` says who holds what and for how long.
# GPU_LEASE_UUID_TABLE overrides the UUIDs nvidia-smi reports, comma-separated in index order (tests).
#
# Waiting is first come, first served, so a whole-node request (`gpu-lease 8 --wait`) is not starved by shorter leases starting
# around it: a --wait request that can't be met at once queues as $GPU_LEASE_DIR/wait.<ns>.<pid>, locked while it waits, and
# every later request (waiting or not) leaves the GPUs to the older waiters: a request without --wait exits 75 naming them.
# `status` lists the waiters.  A waiter that dies drops its lock, and the next reader removes its file.
#
# --max-min M ends CMD after M minutes (SIGTERM, SIGKILL 20 s later; exit 124) and records `until=` in the lease, and it lets the
# request backfill: it may go ahead of the waiters when it ends before the oldest one's GPUs are sure to be free.  That time is
# the n-th earliest of the allowed GPUs' "free by": now for a free GPU, a lease's until=, else its start plus $GPU_LEASE_MAX_MIN
# (default 30, the servers' lease rule), else now for a lease past that.  So a timed window waiting for the last long lease to
# drain keeps the other GPUs busy with jobs that will be gone when it can start.
#
# Memory: CMD runs in a systemd user scope with MemoryMax=G GiB and no swap, so a job that runs away is OOM-killed inside its own
# cgroup (the scope `gpu-lease-<pid>`, named in the user journal) instead of the kernel choosing a victim for the whole machine.
# G is --mem-gb, else $GPU_LEASE_MEM_GB_PER_GPU (default 90% of MemTotal over the GPU count) per GPU leased; --mem-gb 0 runs
# uncapped.  Where this user's systemd can't make a scope (a container, no lingering), CMD runs uncapped and gpu-lease says so.
#
# --preemptible marks a lease that may be stopped for a waiter (fill work: a quality run, a sweep that resumes from its checkpoints).
# The oldest waiter, when the free GPUs and the preemptible leases together cover what it needs, stops the newest preemptible
# leases: SIGTERM to the whole scope (else the pid), SIGKILL 30 s later, each logged to $GPU_LEASE_DIR/preempted.log.  The stopped
# job exits 143 and restarts however its launcher restarts it.  A request without --wait preempts nothing.
#
# --timed (or GPU_LEASE_TIMED=1) tags a timed window: the lease records timed=1, and the server's ops daemon keeps the machine quiet
# while it is held (it pauses other CPU-heavy work; on vy-nebius-2, node_ops.py).  A lease of every GPU by one holder counts as one
# too.
#
# Usage, report only: when CMD exits (any status), gpu-lease releases the GPUs and then tells the holder on stderr, per GPU, how long
# it held it and how much of that the node's utilisation sampler saw busy (util > 0 or SM activity >= 1%), with a hint to prepare
# on the CPU outside the lease when the GPU was busy under half of at least 5 sampled minutes.  It appends a gpu-lease/usage/v1
# record to $RESEARCH_RUN_DIR/gpu-lease-usage.jsonl (published with the run) and to $GPU_LEASE_USAGE_LOG (default
# /workspace/research/lease-usage.jsonl).  The readings come from $GPU_LEASE_UTIL_LOG, a glob of JSONL lines {"t", "gpus": [{"i",
# "uuid", "util", "smact"}]} each covering $GPU_LEASE_UTIL_CADENCE_S seconds (default /workspace/pouw/infra/util/*.jsonl, every 10 s),
# never from NVML, so a timed window stays quiet; with no log a lease records its held time alone.  Nothing is refused or stopped
# on it, and CMD's exit status is gpu-lease's.

set -u
dir=${GPU_LEASE_DIR:-/run/gpu-lease}

allowed_file=${GPU_LEASE_ALLOWED_FILE:-/etc/vy/direct-gpus}

gpu_indices() {
  # the indices gpu-lease may use, one per line: the allowed file's list, else 0..count-1
  local total i a
  total=$(gpu_count)
  if [ -r "$allowed_file" ]; then
    a=$(tr -d ' \n' < "$allowed_file")
    [ "$a" = none ] && return 0
    echo "$a" | tr ',' '\n' | while read -r i; do [ -n "$i" ] && [ "$i" -lt "$total" ] && echo "$i"; done
  else
    for ((i = 0; i < total; i++)); do echo "$i"; done
  fi
}

gpu_count() {
  if [ -n "${GPU_LEASE_COUNT:-}" ]; then echo "$GPU_LEASE_COUNT"; else nvidia-smi -L 2>/dev/null | grep -c '^GPU '; fi
}

uuid_table() {
  # "index uuid" per line
  if [ -n "${GPU_LEASE_UUID_TABLE:-}" ]; then
    echo "$GPU_LEASE_UUID_TABLE" | tr ',' '\n' | awk 'NF {print NR - 1, $1}'
  else
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr ',' ' '
  fi
}

usage() {
  echo "usage: gpu-lease N [--wait] [--on INDEX|UUID,...] [--max-min M] [--mem-gb G] [--preemptible] [--timed] -- CMD... |" \
    "gpu-lease status" >&2
  exit 2
}

stop_lease() {
  # SIGTERM GPU $1's preemptible lease (its scope, else its pid); SIGKILL once 30 s have passed since
  local owner pid unit t sig=TERM
  owner=$(cat "$dir/$1.owner" 2>/dev/null) || return 0
  pid=$(echo "$owner" | sed -n 's/^pid=\([0-9]*\) .*/\1/p')
  unit=$(echo "$owner" | sed -n 's/.* scope=\([^ ]*\) .*/\1/p')
  t=$(echo "$owner" | sed -n 's/.* preempted=\([0-9]*\)$/\1/p')
  if [ -n "$t" ]; then
    [ $(($(date -u +%s) - t)) -ge 30 ] || return 0
    sig=KILL
  fi
  [ -n "$pid" ] || return 0
  if [ -z "$unit" ] || ! systemctl --user kill --signal="$sig" "$unit.scope" 2> /dev/null; then
    pkill -"$sig" -P "$pid" 2> /dev/null
    kill -s "$sig" "$pid" 2> /dev/null
  fi
  echo "${owner% preempted=*} preempted=$(date -u +%s)" > "$dir/$1.owner"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SIG$sig GPU $1 for pid $$ (n=$n, who=${who// /_}): $owner" >> "$dir/preempted.log"
}

preempt_for() {
  # the oldest waiter stops the newest preemptible leases when, with the free GPUs, they cover what it needs
  local want=$1 i fd owner free=0
  local -a pre=()
  for i in "${CANDIDATES[@]}"; do
    exec {fd}>>"$dir/$i.lock" || continue
    if flock -n -s "$fd"; then free=$((free + 1)); exec {fd}>&-; continue; fi
    exec {fd}>&-
    owner=$(cat "$dir/$i.owner" 2>/dev/null)
    case " $owner " in *" preempt=1 "*) pre+=("$(echo "$owner" | sed -n 's/.*since=\([^ ]*\).*/\1/p') $i") ;; esac
  done
  [ "$free" -lt "$want" ] && [ $((free + ${#pre[@]})) -ge "$want" ] || return 0
  printf '%s\n' "${pre[@]}" | sort -r | head -n $((want - free)) | while read -r _ i; do stop_lease "$i"; done
}

live_waiters() {
  # the queued --wait requests, oldest first; a file whose waiter is gone (its lock free) is removed
  local f fd
  for f in "$dir"/wait.*; do
    [ -e "$f" ] || continue
    exec {fd}<"$f" 2>/dev/null || continue
    if flock -n -s "$fd"; then exec {fd}>&-; rm -f "$f"; continue; fi
    exec {fd}>&-
    echo "$f"
  done
}

ahead() {
  # the waiters this request must leave the GPUs to: all of them, or for a queued waiter the older ones
  live_waiters | awk -v me="${MYWAIT:-}" 'me == "" || $0 < me'
}

free_by() {
  # when each allowed GPU is free at the latest (epoch seconds), one per line
  local i fd owner t now
  now=$(date -u +%s)
  for i in $(gpu_indices); do
    exec {fd}>>"$dir/$i.lock" || continue
    if flock -n -s "$fd"; then exec {fd}>&-; echo "$now"; continue; fi
    exec {fd}>&-
    owner=$(cat "$dir/$i.owner" 2>/dev/null)
    t=$(echo "$owner" | sed -n 's/.*until=\([^ ]*\).*/\1/p')
    if [ -n "$t" ]; then
      t=$(date -u -d "$t" +%s 2>/dev/null) || t=$now
    else
      t=$(echo "$owner" | sed -n 's/.*since=\([^ ]*\).*/\1/p')
      t=$(date -u -d "$t" +%s 2>/dev/null) && t=$((t + 60 * ${GPU_LEASE_MAX_MIN:-30})) || t=$now
    fi
    [ "$t" -gt "$now" ] && echo "$t" || echo "$now"
  done
}

may_backfill() {
  # a --max-min request that ends before the oldest waiter ahead of it can start
  local head need t_head
  [ -n "$maxmin" ] || return 1
  head=$(ahead | head -1)
  [ -n "$head" ] || return 0
  need=$(awk '{for (i = 1; i <= NF; i++) if ($i ~ /^n=[0-9]+$/) {print substr($i, 3); exit}}' "$head" 2>/dev/null)
  [ -n "$need" ] || return 1
  t_head=$(free_by | sort -n | sed -n "${need}p")
  [ -n "$t_head" ] && [ $(($(date -u +%s) + 60 * maxmin)) -le "$t_head" ]
}

minutes_since() {
  local since
  since=$(echo "$1" | sed -n 's/.*since=\([^ ]*\).*/\1/p')
  since=$(date -u -d "$since" +%s 2>/dev/null) && echo "$((($(date -u +%s) - since) / 60))m "
}

status() {
  local i fd state owner since age now f
  now=$(date -u +%s)
  [ -r "$allowed_file" ] && echo "# direct-run GPUs from $allowed_file: $(tr -d '\n' < "$allowed_file") (the rest are Kubernetes')"
  for i in $(gpu_indices); do
    owner=$(cat "$dir/$i.owner" 2>/dev/null)
    exec {fd}>>"$dir/$i.lock" || { echo "$i unreadable"; continue; }
    if flock -n -s "$fd"; then state=free; owner=""; else state=held; fi
    exec {fd}>&-
    age=""
    [ -n "$owner" ] && age=$(minutes_since "$owner")
    echo "$i $state${owner:+ $age$owner}"
  done
  for f in $(live_waiters); do
    owner=$(cat "$f" 2>/dev/null)
    echo "waiting $(minutes_since "$owner")$owner"
  done
}

take() {
  # take N from CANDIDATES: TAKEN = the indices, their lock descriptors left open; 1 (holding nothing) when fewer than N are free
  local want=$1 i fd
  TAKEN=() FDS=()
  for i in "${CANDIDATES[@]}"; do
    [ "${#TAKEN[@]}" -lt "$want" ] || break
    exec {fd}>>"$dir/$i.lock" || continue
    if flock -n -x "$fd"; then TAKEN+=("$i") FDS+=("$fd"); else exec {fd}>&-; fi
  done
  if [ "${#TAKEN[@]}" -lt "$want" ]; then
    for fd in "${FDS[@]}"; do exec {fd}>&-; done
    TAKEN=() FDS=()
    return 1
  fi
}

report_usage() {
  # report only (the lease's status never depends on it): each leased GPU's held time, and its busy time over the node sampler's
  # readings inside the lease; never NVML, so a timed window stays quiet
  command -v python3 > /dev/null 2>&1 || return 0
  GL_RC=$1 GL_END=$(date -u +%Y-%m-%dT%H:%M:%SZ) GL_SINCE=$started GL_IDX="${TAKEN[*]}" GL_UUID="${uuids[*]}" GL_WHO=$who \
    GL_PID=$$ GL_SCOPE=${scope:+gpu-lease-$$} GL_TIMED=$timed GL_PRE=$preempt GL_CMD=$cmdline timeout 30 python3 - <<'PY' || true
import glob, json, os, socket, sys
from datetime import datetime

e = os.environ
since, end = e["GL_SINCE"], e["GL_END"]
pattern = e.get("GPU_LEASE_UTIL_LOG", "/workspace/pouw/infra/util/*.jsonl")
cadence = float(e.get("GPU_LEASE_UTIL_CADENCE_S", "10"))
uuids = e["GL_UUID"].split()
gpus = [{"index": int(i), "uuid": uuids[k] if k < len(uuids) else "-", "busy": 0.0, "sampled": 0.0}
        for k, i in enumerate(e["GL_IDX"].split())]
files = sorted(glob.glob(pattern))
if not files and "GPU_LEASE_UTIL_LOG" in e:
    print(f"gpu-lease: no sampler log matches {pattern}, so the usage record has sampled_s 0", file=sys.stderr)
for f in files:
    name = os.path.basename(f)
    if name[:4].isdigit() and name[:10] < since[:10]:
        continue
    try:
        fh = open(f, errors="replace")
    except OSError:
        continue
    with fh:
        for line in fh:
            k = line.find('"t"')
            if k < 0:
                continue
            q = line.find('"', line.find(":", k) + 1)
            if not since <= line[q + 1:q + 21] <= end:
                continue
            try:
                s = json.loads(line)
            except ValueError:
                continue
            for g in s.get("gpus") or []:
                util, smact = g.get("util"), g.get("smact")
                if util is None and smact is None:
                    continue
                for v in gpus:
                    if (v["uuid"] != "-" and g.get("uuid") == v["uuid"]) or (v["uuid"] == "-" and g.get("i", g.get("index")) == v["index"]):
                        v["sampled"] += cadence
                        v["busy"] += cadence if (util or 0) > 0 or (smact or 0) >= 0.01 else 0
held = max(0, int((datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()))


def hms(x: float) -> str:
    x = int(x)
    return f"{x // 3600}h{x % 3600 // 60:02d}m{x % 60:02d}s" if x >= 3600 else f"{x // 60}m{x % 60:02d}s"


for v in gpus:
    share = f" ({100 * v['busy'] / v['sampled']:.0f}%)" if v["sampled"] else ""
    hint = "; prepare on the CPU outside the lease" if v["sampled"] >= 300 and v["busy"] < 0.5 * v["sampled"] else ""
    print(f"gpu-lease: GPU {v['index']} ({v['uuid'][:12]}) held {hms(held)}, busy {hms(v['busy'])} of {hms(v['sampled'])} sampled"
          f"{share}{hint}", file=sys.stderr)
rec = {"schema": "gpu-lease/usage/v1", "host": socket.gethostname(), "pid": int(e["GL_PID"]), "scope": e.get("GL_SCOPE") or None,
       "who": e["GL_WHO"], "run": e.get("RESEARCH_RUN_ID") or None, "timed": e.get("GL_TIMED") == "1",
       "preemptible": e.get("GL_PRE") == "1", "since": since, "end": end, "rc": int(e["GL_RC"]), "cmd": e.get("GL_CMD", ""),
       "gpus": [{"index": v["index"], "uuid": v["uuid"], "held_s": held, "busy_s": int(v["busy"]), "sampled_s": int(v["sampled"])}
                for v in gpus]}
outs = [os.path.join(e["RESEARCH_RUN_DIR"], "gpu-lease-usage.jsonl")] if e.get("RESEARCH_RUN_DIR") else []
for path in outs + [e.get("GPU_LEASE_USAGE_LOG", "/workspace/research/lease-usage.jsonl")]:
    try:
        with open(path, "a") as out:
            out.write(json.dumps(rec) + "\n")
    except OSError:
        pass
PY
}

[ $# -ge 1 ] || usage
if [ "$1" = status ]; then status; exit 0; fi
n=$1; shift
case $n in ''|*[!0-9]*|0) usage ;; esac
wait=0 on="" maxmin="" memgb="" preempt="" timed=${GPU_LEASE_TIMED:-}
[ "$timed" = 1 ] || timed=""
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case $1 in
    --wait) wait=1 ;;
    --preemptible) preempt=1 ;;
    --timed) timed=1 ;;
    --on) [ $# -ge 2 ] || usage; on=$2; shift ;;
    --max-min) [ $# -ge 2 ] || usage; maxmin=$2; shift; case $maxmin in ''|*[!0-9]*|0) usage ;; esac ;;
    --mem-gb) [ $# -ge 2 ] || usage; memgb=$2; shift; case $memgb in ''|*[!0-9]*) usage ;; esac ;;
    *) usage ;;
  esac
  shift
done
[ "${1:-}" = -- ] || usage
shift
[ $# -ge 1 ] || usage
mapfile -t CANDIDATES < <(gpu_indices)
total=${#CANDIDATES[@]}
if [ "$total" -eq 0 ] && [ -r "$allowed_file" ]; then
  echo "gpu-lease: no GPU is for direct runs here ($allowed_file: none): submit through the cluster's queues (sky)" >&2; exit 2
fi
if [ -n "$on" ]; then
  table=$(uuid_table)
  picked=()
  for g in $(echo "$on" | tr ',' ' '); do
    case $g in
      GPU-*) i=$(echo "$table" | awk -v u="$g" '$2 == u {print $1}') ;;
      *[!0-9]*) usage ;;
      *) i=$g ;;
    esac
    if [ -z "$i" ] || ! printf '%s\n' "${CANDIDATES[@]}" | grep -qx "$i"; then
      echo "gpu-lease: GPU $g is not one gpu-lease hands out here (${CANDIDATES[*]:-none})" >&2; exit 2
    fi
    picked+=("$i")
  done
  CANDIDATES=("${picked[@]}")
  total=${#CANDIDATES[@]}
fi
if [ "$n" -gt "$total" ]; then echo "gpu-lease: $n GPUs asked, $total are gpu-lease's here${on:+ in --on $on}" >&2; exit 2; fi
mkdir -p "$dir" 2>/dev/null
[ -w "$dir" ] || { echo "gpu-lease: $dir is not writable" >&2; exit 2; }

who=${GPU_LEASE_WHO:-${RESEARCH_LANE:-$(id -un)}}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MYWAIT=""
until { [ -z "$(ahead)" ] || may_backfill; } && take "$n"; do
  if [ "$wait" != 1 ]; then
    first=$(ahead | head -1)
    if [ -n "$first" ]; then
      echo "gpu-lease: a request is waiting first ($(cat "$first" 2>/dev/null)): add --wait to queue behind it, or a" \
        "--max-min short enough to end before it can start" >&2
    else
      echo "gpu-lease: fewer than $n of $total GPUs free ($(status | grep -v '^#' | tr '\n' ';'))" >&2
    fi
    exit 75
  fi
  [ -n "$(ahead)" ] || preempt_for "$n"
  if [ -z "$MYWAIT" ]; then
    exec {wfd}>"$dir/.wait.$$"
    flock -x "$wfd"
    echo "pid=$$ n=$n${on:+ on=$on} run=${RESEARCH_RUN_ID:--} who=${who// /_} since=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&"$wfd"
    MYWAIT="$dir/wait.$(date -u +%s%N).$$"
    mv "$dir/.wait.$$" "$MYWAIT"
    echo "gpu-lease: waiting for $n GPU(s)${on:+ in --on $on}, queued as $(basename "$MYWAIT")" >&2
  fi
  sleep "${GPU_LEASE_POLL_S:-15}"
done
if [ -n "$MYWAIT" ]; then rm -f "$MYWAIT"; exec {wfd}>&-; fi
scope=()
if [ "$memgb" != 0 ]; then
  if [ -z "$memgb" ]; then
    per=${GPU_LEASE_MEM_GB_PER_GPU:-$(awk -v n="$(gpu_count)" '/^MemTotal:/ {printf "%d", $2 * 0.9 / 1048576 / (n > 0 ? n : 1)}' /proc/meminfo)}
    memgb=$((per * n))
  fi
  if command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1; then
    scope=(systemd-run --user --scope --quiet --collect --unit "gpu-lease-$$" -p "MemoryMax=${memgb}G" -p MemorySwapMax=0 --)
  else
    echo "gpu-lease: this user's systemd can't make a scope here, so CMD runs without a memory cap" >&2
    memgb=""
  fi
fi
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ends=""
[ -n "$maxmin" ] && ends=" until=$(date -u -d "@$(($(date -u -d "$started" +%s) + 60 * maxmin))" +%Y-%m-%dT%H:%M:%SZ)"
[ -n "${table:-}" ] || table=$(uuid_table)
uuids=()
for i in "${TAKEN[@]}"; do
  u=$(echo "$table" | awk -v i="$i" '$1 == i {print $2}')
  uuids+=("${u:--}")
  echo "pid=$$ run=${RESEARCH_RUN_ID:--} who=${who// /_} uuid=${u:--} since=$started$ends${memgb:+ mem=${memgb}G}" \
    "${preempt:+preempt=1 }${timed:+timed=1 }${scope:+scope=gpu-lease-$$ }cmd=$(echo "$*" | cut -c1-80)" > "$dir/$i.owner"
done
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TAKEN[*]}")
GPU_LEASE_UUID=$(IFS=,; echo "${uuids[*]}")
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID GPU_LEASE_UUID
echo "gpu-lease: GPUs $CUDA_VISIBLE_DEVICES ($GPU_LEASE_UUID) for $who, pid $$${RESEARCH_RUN_ID:+ (run $RESEARCH_RUN_ID)}${ends:+,$ends}" >&2
cmdline=$(echo "$*" | cut -c1-200)
[ -n "$maxmin" ] && set -- timeout -k 20 "$((maxmin * 60))" "$@"
# CMD runs as a child, not exec'd, so the lease can report its usage once CMD is done.  `0<&0` keeps CMD's stdin (a background job's
# is /dev/null otherwise); a SIGTERM, SIGINT or SIGHUP to gpu-lease is passed on to CMD; gpu-lease exits with CMD's status.
"${scope[@]}" "$@" 0<&0 &
child=$!
trap 'kill -TERM "$child" 2> /dev/null' TERM INT HUP
rc=0; wait "$child" || rc=$?
while kill -0 "$child" 2> /dev/null; do rc=0; wait "$child" || rc=$?; done
trap - TERM INT HUP
for fd in "${FDS[@]}"; do exec {fd}>&-; done
report_usage "$rc"
exit "$rc"
