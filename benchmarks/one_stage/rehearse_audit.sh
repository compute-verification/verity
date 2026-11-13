#!/usr/bin/env bash
# A REHEARSAL, not an audit of record: a kept window's audit again, timed, with its negatives side by side.
# `records_audit.py audit` as served_verdicts.sh's v3c_audit calls it in a follow-on, on a kept serve dir's window read in
# place, into this job's own directory, on a local development challenger session (never the gate), with
# --negative-workers and --audit-mem-gb. The kept audit sampled this registration already; this one samples it again on a
# session whose key is its own, so its sample is no auditor's. It writes no verdict line of a record, and nothing reads its
# answer as one: $OUT holds no v3c-audit directory. Run it in a clone of the tree (`research run --cwd clone`), sent with
# rehearsal_read.py and main.txt (`git rev-list --first-parent origin/main`). Or with the tree as an input, where the run's
# source is another commit's (TREE_INPUT): the job then clones the tree into a directory of its own under WORK_PARENT,
# holds it to the commit and the tree id it was told, and runs there; the run's source is read for nothing.
#   KEPT                  the kept serve dir: read, never written (the job says so at its end, exit 90 if a file changed)
#   WORK_PARENT           a directory on the file system the composed circuits go to; the job's own directory under it
#                         (rehearse-audit.XXXXXXXX) is removed on every way out but a SIGKILL of this shell
#   V3C_NEGATIVE_WORKERS  default 7     AUDIT_MEM_GB  default 650
#   WARM_JOBS             N: `records_audit.py warm` composes the templates N at a time into the job's directory first, and
#                         the job ends there if the warm fails (exit 80); unset: the audit composes every template itself,
#                         one after another, as the kept run's did
#   JOB_SECONDS           default 12600: the warm and the audit together end by then, counted from this script's start
#                         (exit 124), which leaves the job's own limit room to remove its directory
#   MIN_FREE_GB           default 600: the job starts only with that much free under WORK_PARENT; rehearsal.json
#                         holds the value and what was free at the start
#   FLOOR_GB              default 15: the job ends (exit 71) once less than that is free there, read every 5 s
#   MEM_ROOM_GB           default 32 (GB of 10^9 bytes): the memory stop. The job ends (exit 72) once the processes of
#                         the memory scope that limits it hold more than the scope's limit less that much. The scope is
#                         the level of the job's cgroup, from its own up, whose memory.max is the tightest; what its
#                         processes hold is its memory.current less its memory.stat's file cache, which the kernel
#                         drops at the limit and which MemAvailable counts as free (shmem, part of `file` and dropped
#                         by no reclaim, stays counted). Read every 5 s beside the disk; where no level has a limit
#                         this stop reads nothing. At each stage's start one line says the scope's `max` events (the
#                         times it reached its limit), what its processes and its file cache hold, and MemAvailable
#   MEM_STOP_GB           (GB of 10^9 bytes): the second reading, the node's, and a stop of its own. The job ends (exit
#                         72) once the node's MemAvailable, read every 5 s, is more than that under its value at this
#                         script's start. Default: the memory scope's limit less MEM_ROOM_GB, the taking the first
#                         reading stops at, so it is under the limit by that room; 650 where no level has a limit. A
#                         value set at or over the scope's limit is refused (exit 64): the limit would end the job
#                         before that reading could. The stop's message, in the job's output and in rehearsal.json's
#                         stop_reason, names which reading stopped the job, the scope's or the node's, with both
#                         readings as they were then; rehearsal.json also holds the stage and the sampled member's
#                         circuit file's bytes, and on every way out MemAvailable's lowest value and its stage
#   MEMINFO               default /proc/meminfo: the file MemAvailable is read from (a test names its own)
#   CGROUP                default the job's own cgroup (v2), from /proc/self/cgroup: the directory the memory scope
#                         is looked for from (a test names its own)
#   TREE_INPUT            a file sent with the job: a tar of one bare git directory (NAME.git) holding the commit to audit
#                         and its history down to main's line. With it TREE_COMMIT, TREE_ID and TREE_PIN, 40 hex
#                         each: the job clones the directory (git recomputes each object's id as it receives it) and
#                         starts only where HEAD is TREE_COMMIT, its tree TREE_ID and `git status` says nothing changed
#                         (exit 64 otherwise), and the newest commit of main.txt in its history is TREE_PIN (exit 66
#                         otherwise: a local session pins that commit, and the verifier refuses a session pinned off
#                         main's line). Once the build has made the tree's environment, the tree's own on_main is held
#                         to TREE_PIN too, before the warm and the audit (exit 66). The clone is removed with the job's
#                         directory. Unset: the tree is the directory the job starts in
# The job drops the PYTHONPATH it was started with: the build, the warm, the audit, the public copy and the reader all run
# in the tree's own environment (`uv sync`'s), on the tree's own packages, whatever commit the run's source is. Once the
# build has made that environment, the imports the public copy and the reader make are held to the tree (exit 64).
# Once the audit has sampled, one line says the sampled member's circuit file's bytes and the seconds a table gives the
# audit from there to its end: the kept run's stages' seconds, those that follow the member scaled to its circuit file.
# The table is an estimate, no measurement, and nothing in the job acts on it.
# $OUT gets build.log, rehearsal-audit/ (the audit's public copy, `served_integrity.py public-copy`), and rehearsal.json,
# which says it is a rehearsal: the warm's and the audit's exits and seconds, the honest arm's outcome, the stages'
# seconds, `negative_workers`, `verifier_checks`, each negative's refusal beside the kept audit's by name, and the most
# disk the job's directory held. The audit's own directory, its logs and its composed circuits stay in the job's directory
# and go with it. For the memory it also keeps: each sampled member's circuit, program and partition files' bytes, each Lean
# command's exit and seconds, every negative with its codes (one whose code is disagree, or that Lean gave no verdict line
# for, as not answered, never as refused), and the job's cgroup's limit, kills and peak read at the start and at the end,
# with the memory scope's max events and what its processes held at the most.
#   AUDIT_WARNINGS        the audit's PYTHONWARNINGS (default::DeprecationWarning shows Python's warning of a fork with
#                         more than one thread alive, which it hides otherwise); unset, the audit runs as it did
# Where the public copy fails, or the reader's answer on the honest arm raises, the audit ended and no row of a record can
# be read of it here: the job says so, exits 73 whatever the audit's exit, and first keeps under $OUT/rehearsal-audit-kept
# the two files of the audit's directory those two steps read (summary.json and window/audit.json, as the audit wrote
# them: the audit's own files, which hold more than the public copy does), so both can be made again from them elsewhere.
# Exits: 64 an input is missing or not as this job reads it, 70 too little disk to start, 71 the disk ran low, 72 the
# memory ran low, 5/6/7 a build, 80 the warm failed, 124 the job's seconds ran out, 143 it was told to stop, 90 the kept
# dir changed, 73 the public copy or the reader's answer failed, else the audit's.
set -uo pipefail
: "${KEPT:?the kept serve dir}" "${WORK_PARENT:?a directory on the file system the composed circuits go to}"
K=${K:-0} CLASS=${V3C_CLASS:-8,32,4,4,16} LAW=${V3C_LAW:-subset:1}
WORKERS=${V3C_NEGATIVE_WORKERS:-7} MEM_GB=${AUDIT_MEM_GB:-650} JOB_S=${JOB_SECONDS:-12600}
MIN_FREE_GB=${MIN_FREE_GB:-600} FLOOR_GB=${FLOOR_GB:-15}
MEM_STOP_SET=${MEM_STOP_GB:-} MEMINFO=${MEMINFO:-/proc/meminfo} MEM_ROOM_GB=${MEM_ROOM_GB:-32}
# the table: a member of TABLE_BYTES took TABLE_MEMBER_S s in its session, its Lean check and the registered reads' four
# checks at once; the outcome and the negatives' workers take TABLE_WORKERS_S whatever the member: the outcome, then the
# negatives' twelve tasks in one wave, as six workers took them. The table has that stage at six workers alone: five
# or more read it, the line saying so where they are not six, and under five it has no seconds
TABLE_BYTES=3764130558 TABLE_MEMBER_S=1460 TABLE_WORKERS_S='' TABLE_AT=''
if [[ $WORKERS =~ ^[0-9]+$ ]] && [ "$WORKERS" -ge 5 ]; then
  TABLE_WORKERS_S=1600
  [ "$WORKERS" = 6 ] || TABLE_AT=" (its last stage as six workers took it, not $WORKERS)"
fi
TREE=$PWD OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out} IN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VW=$KEPT/v3c-w$K-${CLASS//,/-} WREG=$KEPT/weights_registration.json MAIN=${MAIN_FILE:-$IN/main.txt}
FW=${FLOCK_WORK:-/workspace/research/vllm-served-verdicts/flock-work}
UNPACK=''
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
mkdir -p "$OUT"
T0=$(date +%s)
step() { printf 'REHEARSAL +%ss %s\n' "$(($(date +%s) - T0))" "$*"; }
left() { echo $((T0 + JOB_S - $(date +%s))); }
free_gb() { df --output=avail -BG "$WORK_PARENT" | tail -1 | tr -dc 0-9; }
avail_kb() { awk '/^MemAvailable:/ {print $2; exit}' "$MEMINFO" 2>/dev/null; }
gb() { awk -v k="$1" 'BEGIN {printf "%.1f", k * 1024 / 1e9}'; }  # /proc/meminfo's kB (1024 bytes) -> GB of 10^9 bytes
MEM0_KB=$(avail_kb)
# the job's cgroup (v2): a scope of its own, or a login session's, which other jobs share and whose peak then is not this job's
CG=${CGROUP:-/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)}
case ${CG##*/} in check-*) CG_KIND="a scope of its own";; session-*) CG_KIND="a login session's scope, shared";; *) CG_KIND="another kind";; esac
kills() { local k; k=$(awk '$1 == "oom_kill" {print $2}' "$CG/memory.events" 2>/dev/null); echo "${k:-unread}"; }
scope() {  # the memory scope: the level from the job's cgroup up to the root whose memory.max is the tightest; empty where none has one
  local d=$CG m low='' at=''
  while [ -n "$d" ] && [ "$d" != /sys/fs/cgroup ] && [ "$d" != /sys/fs ]; do
    m=$(cat "$d/memory.max" 2>/dev/null)
    if [[ $m =~ ^[0-9]+$ ]] && { [ -z "$low" ] || [ "$m" -lt "$low" ]; }; then low=$m at=$d; fi
    d=${d%/*}
  done
  echo "$at"
}
limit() {  # the scope's memory.max, in bytes; none where no level has one
  local d; d=$(scope)
  if [ -n "$d" ]; then cat "$d/memory.max"; else echo none; fi
}
stat_of() { local v; v=$(awk -v k="$2" '$1 == k {print $2; exit}' "$1/memory.stat" 2>/dev/null); echo "${v:-unread}"; }
held() {  # held DIR: what the processes of the cgroup at DIR hold, in bytes: its memory.current less its file cache, shmem left in
  local cur file shmem
  cur=$(cat "$1/memory.current" 2>/dev/null) file=$(stat_of "$1" file) shmem=$(stat_of "$1" shmem)
  if [[ $cur =~ ^[0-9]+$ ]] && [[ $file =~ ^[0-9]+$ ]] && [[ $shmem =~ ^[0-9]+$ ]]; then echo $((cur - file + shmem)); else echo unread; fi
}
maxed() { local n; n=$(awk '$1 == "max" {print $2}' "${1:-$CG}/memory.events" 2>/dev/null); echo "${n:-unread}"; }  # the times a cgroup reached its limit
bgb() { if [[ $1 =~ ^[0-9]+$ ]]; then awk -v b="$1" 'BEGIN {printf "%.1f GB", b / 1e9}'; else echo "$1"; fi; }  # bytes -> GB of 10^9 bytes
KILLS0=$(kills) SCOPE0=$(scope)
MAXED0=$(maxed "${SCOPE0:-$CG}")
memory() {  # one JSON object: what the job reads of its cgroup and of the memory scope now
  local d peak='' peak_stage=''
  d=$(scope)
  IFS=$'\t' read -r peak peak_stage 2>/dev/null < "${WORK:-/nonexistent}/.held"
  printf '{"cgroup": "%s", "limit_bytes": "%s", "oom_kill_at_start": "%s", "oom_kill_now": "%s", "peak_bytes": "%s", "current_bytes": "%s", "audit_mem_gb": "%s", ' \
    "$CG_KIND" "$(limit)" "$KILLS0" "$(kills)" "$(cat "$CG/memory.peak" 2>/dev/null || echo unread)" \
    "$(cat "$CG/memory.current" 2>/dev/null || echo unread)" "$MEM_GB"
  printf '"scope": "%s", "scope_max_events_at_start": "%s", "scope_max_events_now": "%s", "scope_processes_bytes": "%s", "scope_file_bytes": "%s", "scope_processes_most_bytes": "%s", "scope_processes_most_stage": "%s", "stops_at_room_gb": "%s"}' \
    "$([ -z "$d" ] && echo 'no level of the cgroup has a limit' || { [ "$d" = "$CG" ] && echo "the job's own cgroup" || echo 'a level above the job'"'"'s cgroup'; })" \
    "$MAXED0" "$(maxed "${d:-$CG}")" "$(held "${d:-$CG}")" "$(stat_of "${d:-$CG}" file)" "${peak:-unread}" "${peak_stage:-unread}" "$MEM_ROOM_GB"
}

SENT=''
[ -z "${TREE_INPUT:-}" ] || SENT="; the tree audited is commit ${TREE_COMMIT:-?}, tree id ${TREE_ID:-?}, sent as an input: the run's source is the launcher's and is not the tree audited"
step "a rehearsal on a local development session: no audit of record, no verdict of a record$SENT"
[[ $WORKERS =~ ^[1-9][0-9]*$ ]] || { step "PREFLIGHT: V3C_NEGATIVE_WORKERS is a positive integer"; exit 64; }
[[ $JOB_S =~ ^[1-9][0-9]*$ ]] || { step "PREFLIGHT: JOB_SECONDS is a positive integer"; exit 64; }
[[ $MEM_ROOM_GB =~ ^[1-9][0-9]*$ ]] || { step "PREFLIGHT: MEM_ROOM_GB is a positive integer"; exit 64; }
[ -z "$MEM_STOP_SET" ] || [[ $MEM_STOP_SET =~ ^[1-9][0-9]*$ ]] || { step "PREFLIGHT: MEM_STOP_GB is a positive integer"; exit 64; }
# the node's reading stops the job under the scope's limit: at the first reading's taking where unset, and never set over it
LIMIT0=$(limit)
if [[ $LIMIT0 =~ ^[0-9]+$ ]]; then
  [ $((LIMIT0 / 1000000000)) -gt "$MEM_ROOM_GB" ] || { step "PREFLIGHT: MEM_ROOM_GB $MEM_ROOM_GB leaves nothing under the memory scope's limit $(bgb "$LIMIT0"): set it under the limit"; exit 64; }
  MEM_STOP_GB=${MEM_STOP_SET:-$((LIMIT0 / 1000000000 - MEM_ROOM_GB))}
  [ $((MEM_STOP_GB * 1000000000)) -lt "$LIMIT0" ] || { step "PREFLIGHT: MEM_STOP_GB $MEM_STOP_GB is not under the memory scope's limit $(bgb "$LIMIT0"), which would end the job before the node's reading could stop it: leave MEM_STOP_GB unset (it is then the limit less MEM_ROOM_GB, $((LIMIT0 / 1000000000 - MEM_ROOM_GB))) or set it under the limit"; exit 64; }
else
  MEM_STOP_GB=${MEM_STOP_SET:-650}
fi
[[ $MEM0_KB =~ ^[0-9]+$ ]] || { step "PREFLIGHT: no MemAvailable line in ${MEMINFO##*/}"; exit 64; }
MEM_STOP_KB=$((MEM_STOP_GB * 1000000000 / 1024))
[ -z "${WARM_JOBS:-}" ] || [[ $WARM_JOBS =~ ^[1-9][0-9]*$ ]] || { step "PREFLIGHT: WARM_JOBS is a positive integer"; exit 64; }
if [ -n "${TREE_INPUT:-}" ]; then  # the tree audited is a sent input: cloned, held to its commit and tree id, then the job's tree
  [[ ${TREE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] && [[ ${TREE_ID:-} =~ ^[0-9a-f]{40}$ ]] && [[ ${TREE_PIN:-} =~ ^[0-9a-f]{40}$ ]] \
    || { step "PREFLIGHT: TREE_INPUT needs TREE_COMMIT, TREE_ID and TREE_PIN, 40 hex each"; exit 64; }
  [ -s "$IN/$TREE_INPUT" ] || { step "PREFLIGHT: no $TREE_INPUT sent with the job"; exit 64; }
  [ -s "$MAIN" ] || { step "PREFLIGHT: no ${MAIN##*/} sent with the job"; exit 64; }
  [ -d "$WORK_PARENT" ] || { step "PREFLIGHT: no WORK_PARENT directory"; exit 64; }
  UNPACK=$(mktemp -d "$WORK_PARENT/rehearse-tree.XXXXXXXX") || exit 70
  trap 'rm -rf -- "$UNPACK"' EXIT
  trap 'exit 143' TERM INT HUP
  mkdir "$UNPACK/sent" && tar -xf "$IN/$TREE_INPUT" -C "$UNPACK/sent" \
    && sent=$(find "$UNPACK/sent" -mindepth 1 -maxdepth 1 -type d -name '*.git' | head -1) && [ -n "$sent" ] \
    && git clone -q "file://$sent" "$UNPACK/tree" 2> "$OUT/tree-clone.err" && git -C "$UNPACK/tree" checkout -q --detach 2>> "$OUT/tree-clone.err" \
    || { step "PREFLIGHT: $TREE_INPUT did not unpack to one git directory and clone (out/tree-clone.err)"; exit 64; }
  rm -rf -- "$UNPACK/sent"
  got_c=$(git -C "$UNPACK/tree" rev-parse HEAD) got_t=$(git -C "$UNPACK/tree" rev-parse 'HEAD^{tree}')
  changed=$(git -C "$UNPACK/tree" status --porcelain | wc -l)
  [ "$got_c" = "$TREE_COMMIT" ] && [ "$got_t" = "$TREE_ID" ] && [ "$changed" = 0 ] \
    || { step "PREFLIGHT: the tree sent is commit ${got_c:-unread}, tree id ${got_t:-unread}, $changed path(s) changed: not commit $TREE_COMMIT, tree id $TREE_ID, nothing changed"; exit 64; }
  pin=$(git -C "$UNPACK/tree" rev-list HEAD | grep -m1 -F -x -f - "$MAIN")
  [ "$pin" = "$TREE_PIN" ] || { step "PIN: the newest commit of ${MAIN##*/} in the history of the tree sent is ${pin:-none}, not $TREE_PIN: a local session would pin another commit than asked, or one off main's line, which the verifier refuses"; exit 66; }
  step "the tree: cloned from the input into the job's own directory under WORK_PARENT; HEAD is commit $got_c, its tree id $got_t, nothing changed in it: as asked; the newest commit of main's line in its history is $pin, as asked, by git; the tree's own on_main is held to it once the build has made its environment"
  TREE=$UNPACK/tree
  cd "$TREE" || exit 64
fi
PY=$TREE/.venv/bin/python
# Every interpreter the job starts is the tree's environment's and reads the tree's own packages there. The runner's
# PYTHONPATH names the packages of the run's source, which under TREE_INPUT is another commit, and an entry of it comes
# before the environment's install of the tree: with it, `import verity` in a script of the tree is the source's.
unset PYTHONPATH
for f in "$VW/window/registration.json" "$VW/window/circuit.json" "$KEPT/windows.jsonl" "$WREG"; do
  [ -s "$f" ] || { step "PREFLIGHT: the kept serve dir has no ${f#"$KEPT"/}"; exit 64; }
done
for f in "$MAIN" "$IN/rehearsal_read.py" "$TREE/benchmarks/prove_path/records_audit.py"; do
  [ -s "$f" ] || { step "PREFLIGHT: no ${f##*/} (sent with the job, or the tree's)"; exit 64; }
done
grep -q -- '--negative-workers' "$TREE/benchmarks/one_stage/a5.py" \
  || { step "PREFLIGHT: this tree's audit takes no --negative-workers"; exit 64; }
if [ -f "$KEPT/private/deferred/deferred.env" ]; then  # the settings the kept follow-on ran at: names only, no value printed
  bad=$(k=$K c=$CLASS l=$LAW; set -a; . "$KEPT/private/deferred/deferred.env" > /dev/null 2>&1; set +a
        [ -z "${H1_TIE:-}" ] || printf ' H1_TIE'
        [ "${K:-$k}" = "$k" ] || printf ' K'
        [ "${V3C_CLASS:-$c}" = "$c" ] || printf ' V3C_CLASS'
        [ "${V3C_LAW:-$l}" = "$l" ] || printf ' V3C_LAW')
  [ -z "$bad" ] || { step "PREFLIGHT: the kept follow-on's settings differ from this job's in:$bad"; exit 64; }
  step "the kept follow-on's settings (K, V3C_CLASS, V3C_LAW, no H1_TIE) are this job's"
else
  step "no private/deferred/deferred.env in the kept serve dir: K, V3C_CLASS and V3C_LAW are this job's defaults"
fi
[ -d "$WORK_PARENT" ] || { step "PREFLIGHT: no WORK_PARENT directory"; exit 64; }
free=$(free_gb)
[ "${free:-0}" -ge "$MIN_FREE_GB" ] || { step "PREFLIGHT: ${free:-?} GB free under WORK_PARENT, under MIN_FREE_GB $MIN_FREE_GB"; exit 70; }
step "disk: $free GB free under WORK_PARENT; memory: $(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo) GiB available of $(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo); $(nproc) cores"
step "memory limit: $(memory)"
if [ -n "$SCOPE0" ]; then
  step "memory stop: the memory scope's processes hold $(bgb "$(held "$SCOPE0")") and its file cache $(bgb "$(stat_of "$SCOPE0" file)") of its limit $(bgb "$(limit)"); the job ends (exit 72) once its processes hold more than the limit less $MEM_ROOM_GB GB; its max events so far: $MAXED0"
else
  step "memory stop: no level of the job's cgroup has a memory limit, so the stop on the scope's processes reads nothing"
fi
step "memory stop, the second reading (the node's): MemAvailable is $(gb "$MEM0_KB") GB now; the job ends (exit 72) once it is more than $MEM_STOP_GB GB under that ($([ -n "$MEM_STOP_SET" ] && echo 'MEM_STOP_GB as set' || { [ -n "$SCOPE0" ] && echo "the scope's limit less MEM_ROOM_GB $MEM_ROOM_GB" || echo 'the default where no level has a limit'; }))$([ "$MEM0_KB" -gt "$MEM_STOP_KB" ] || echo ', which it cannot be: less than that is available now')"

WORK=$(mktemp -d "$WORK_PARENT/rehearse-audit.XXXXXXXX") || exit 70
CHILD='' WATCH='' PEAK_GB=0 warm=() warn=() COPY=ok ANSWER=ok
NEG_STAGE="the audit, the outcome and the negatives' workers"
at() { echo "$1" > "$WORK/.stage"; }
stage() {  # the job's stage now: the one it last entered, and inside the audit what its directory holds so far
  local w=$WORK/audit/window s
  s=$(cat "$WORK/.stage" 2>/dev/null)
  if [ "$s" != "the audit" ]; then echo "${s:-the start}"
  elif [ -e "$w/negatives.json" ]; then echo "the audit, its negatives written"
  elif compgen -G "$w/session-m*/*.archive-put.log" > /dev/null; then echo "the audit, the registered reads' checks"
  elif compgen -G "$w/session-m*/lean-verify.log" > /dev/null; then echo "$NEG_STAGE"
  elif compgen -G "$w/session-m*/prove.log" > /dev/null; then echo "the audit, the sampled member's Lean check"
  elif compgen -G "$w/session-m*/serve.log" > /dev/null; then echo "the audit, the session"
  elif [ -e "$w/draw.json" ]; then echo "the audit, sampled, before the session"
  elif [ -e "$w/receipt.json" ]; then echo "the audit, the sample"
  else echo "the audit, before the sample"; fi
}
held_gb() { du -s -BG "$WORK" 2>/dev/null | cut -f1 | tr -dc 0-9; }
peak() { local g; g=$(held_gb); [ "${g:-0}" -le "$PEAK_GB" ] || PEAK_GB=$g; }
gone() {
  [ -z "$WATCH" ] || kill "$WATCH" 2>/dev/null
  rm -rf -- "$WORK" ${UNPACK:+"$UNPACK"}
  step "the job's directory$([ -z "$UNPACK" ] || echo " and the tree's clone") removed; $(free_gb) GB free under WORK_PARENT"
}
stop() {  # told to stop, or the disk or the memory ran low: the command in flight first, then out by the EXIT trap
  if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; fi
  step "stopping: exit $1$([ ! -s "$WORK/.stop" ] || echo ": $(cat "$WORK/.stop")")"
  [ ! -x "$PY" ] || read_out "${warm[@]}" --stopped "$1"
  exit "$1"
}
trap gone EXIT
trap 'stop 143' TERM INT HUP
trap 'stop 71' USR1
trap 'stop 72' USR2
touch "$WORK/.start"
at "the build"
printf '%s\t%s\n' "$MEM0_KB" "the preflight" > "$WORK/.mem"
(low=$MEM0_KB said='' neg=0 neg_t0=0 neg_m0=0 neg_low=0 was='' most=0
 while sleep 5; do
   for d in "$WORK"/audit/window/session-m*; do  # each sampled member, once, as its session's directory appears
     [ -d "$d" ] || continue
     case " $said " in *" ${d##*/} "*) continue;; esac
     b=$(stat -L -c %s "$WORK/audit/window/${d##*/session-}/circuit.txt" 2>/dev/null) || continue
     said="$said ${d##*/}"
     now=$(($(date +%s) - T0))
     if [ -n "$TABLE_WORKERS_S" ]; then
       more=$((TABLE_WORKERS_S + TABLE_MEMBER_S * b / TABLE_BYTES))
       printf "REHEARSAL +%ss sampled: the member's circuit file is %s bytes; by the table the audit ends about %s s from here, at about +%ss, %s the job's %s s%s\n" \
         "$now" "$b" "$more" "$((now + more))" "$([ $((now + more)) -le "$JOB_S" ] && echo inside || echo PAST)" "$JOB_S" "$TABLE_AT"
     else
       printf "REHEARSAL +%ss sampled: the member's circuit file is %s bytes; the table has no seconds for %s negative workers\n" "$now" "$b" "$WORKERS"
     fi
   done
   f=$(free_gb)
   [ "${f:-0}" -ge "$FLOOR_GB" ] || { printf 'REHEARSAL LOW DISK: %s GB free under WORK_PARENT, under FLOOR_GB %s\n' "${f:-?}" "$FLOOR_GB"; kill -USR1 $$; exit 0; }
   m=$(avail_kb)
   s=$(stage)
   sd=$(scope)
   h=$(held "${sd:-$CG}")
   if [ "$s" != "$was" ]; then  # a stage's start: the scope's max events, and what its processes and its file cache hold
     was=$s
     printf "REHEARSAL +%ss %s: the memory scope's max events %s, %s at the job's start; its processes hold %s, its file cache %s, of its limit %s; MemAvailable %s GB\n" \
       "$(($(date +%s) - T0))" "$s" "$(maxed "${sd:-$CG}")" "$MAXED0" "$(bgb "$h")" "$(bgb "$(stat_of "${sd:-$CG}" file)")" "$(bgb "$(limit)")" "$([[ $m =~ ^[0-9]+$ ]] && gb "$m" || echo unread)"
   fi
   # both readings as they are now: whichever stops the job, its message holds the two
   l=$([ -n "$sd" ] && cat "$sd/memory.max" 2>/dev/null)
   if [[ $h =~ ^[0-9]+$ ]] && [[ $l =~ ^[0-9]+$ ]]; then
     scope_now="the memory scope's processes hold $(bgb "$h") of its limit $(bgb "$l"), and stop the job over the limit less MEM_ROOM_GB $MEM_ROOM_GB"
   else
     scope_now="the memory scope's processes hold $(bgb "$h"), and $([ -n "$sd" ] && echo 'its limit is unread' || echo 'no level of the cgroup has a limit'), so that reading stops nothing"
   fi
   if [[ $m =~ ^[0-9]+$ ]]; then
     node_now="MemAvailable is $(gb "$m") GB, $(gb $((MEM0_KB - m))) GB under the $(gb "$MEM0_KB") GB at the start, and stops the job more than MEM_STOP_GB $MEM_STOP_GB under it"
   else
     node_now="MemAvailable is unread"
   fi
   if [[ $h =~ ^[0-9]+$ ]]; then
     if [ "$h" -gt "$most" ]; then most=$h; printf '%s\t%s\n' "$h" "$s" > "$WORK/.held.new" && mv -f "$WORK/.held.new" "$WORK/.held"; fi
     if [[ $l =~ ^[0-9]+$ ]] && [ "$h" -gt $((l - MEM_ROOM_GB * 1000000000)) ]; then
       why="the scope's reading stopped the job: $scope_now; the node's reading then: $node_now"
       echo "the memory ran low: $why" > "$WORK/.stop"
       printf 'REHEARSAL LOW MEMORY: %s, in %s\n' "$why" "$s"; kill -USR2 $$; exit 0
     fi
   fi
   [[ $m =~ ^[0-9]+$ ]] || continue
   if [ "$m" -lt "$low" ]; then low=$m; printf '%s\t%s\n' "$m" "$s" > "$WORK/.mem.new" && mv -f "$WORK/.mem.new" "$WORK/.mem"; fi
   if [ "$s" = "$NEG_STAGE" ]; then  # the stage's own seconds and its lowest MemAvailable, as read here every 5 s
     if [ "$neg" = 0 ]; then neg=1 neg_t0=$(date +%s) neg_m0=$m neg_low=$m; elif [ "$m" -lt "$neg_low" ]; then neg_low=$m; fi
   elif [ "$neg" = 1 ]; then
     neg=2
     printf "REHEARSAL +%ss the outcome and the negatives' workers ended: %s s in the stage; MemAvailable's lowest in it %s GB, %s GB under the job's start and %s GB under the stage's own start\n" \
       "$(($(date +%s) - T0))" "$(($(date +%s) - neg_t0))" "$(gb "$neg_low")" "$(gb $((MEM0_KB - neg_low)))" "$(gb $((neg_m0 - neg_low)))"
   fi
   if [ $((MEM0_KB - m)) -gt "$MEM_STOP_KB" ]; then
     why="the node's reading stopped the job: $node_now; the scope's reading then: $scope_now"
     echo "the memory ran low: $why" > "$WORK/.stop"
     printf 'REHEARSAL LOW MEMORY: %s, in %s\n' "$why" "$s"; kill -USR2 $$; exit 0
   fi
 done) &
WATCH=$!
limited() {  # limited STDOUT STDERR CMD...: CMD with the job's seconds left as its limit, stopped with the job
  local o=$1 e=$2 t rc; shift 2
  t=$(left)
  [ "$t" -gt 0 ] || return 124
  env -u PYTHONPATH ${warn[@]+"${warn[@]}"} RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$t" "$@" > "$o" 2> "$e" &
  CHILD=$!
  wait "$CHILD"; rc=$?
  CHILD=''
  return "$rc"
}
read_out() {  # rehearsal.json, from whatever the job reached
  local low_kb='' low_stage='' r
  peak
  IFS=$'\t' read -r low_kb low_stage 2>/dev/null < "$WORK/.mem"
  "$PY" "$IN/rehearsal_read.py" "$WORK/audit" "$VW/audit" "$OUT/rehearsal.json" --tree "$TREE" \
    --registration "$VW/window/registration.json" --workers "$WORKERS" --warm-jobs "${WARM_JOBS:-0}" --disk-gb "$PEAK_GB" \
    --min-free-gb "$MIN_FREE_GB" --free-at-start-gb "${free:-0}" --audit-err "$WORK/audit.err" \
    --warnings "${AUDIT_WARNINGS:-}" --memory "$(memory)" --stage "$(stage)" --mem-start-kb "$MEM0_KB" \
    --mem-low-kb "${low_kb:-$MEM0_KB}" --mem-low-stage "${low_stage:-unread}" --mem-stop-gb "$MEM_STOP_GB" \
    --stop-reason "$(cat "$WORK/.stop" 2>/dev/null)" \
    ${TREE_INPUT:+--tree-commit "$TREE_COMMIT" --tree-id "$TREE_ID"} "$@"
  r=$?
  case $r in
    0) ;;
    3) ANSWER=failed; step "THE ANSWER on the honest arm raised (rehearsal_read.py exit 3; rehearsal.json's honest.answer.error): no row of a record was read of this audit";;
    *) step "rehearsal.json: not whole (rehearsal_read.py exit $r)";;
  esac
}
keep_read() {  # the audit's own two files the public copy and the answer read, kept before the job's directory goes
  local d=$OUT/rehearsal-audit-kept f said=''
  for f in summary.json window/audit.json; do
    if [ -f "$WORK/audit/$f" ] && mkdir -p "$(dirname "$d/$f")" && cp -p "$WORK/audit/$f" "$d/$f"; then
      said="$said $f ($(stat -c %s "$d/$f") bytes)"
    else
      said="$said $f (NOT KEPT: $([ -f "$WORK/audit/$f" ] && echo 'the copy failed' || echo 'the audit left none'))"
    fi
  done
  step "kept under out/rehearsal-audit-kept, as the audit wrote them and not as a public copy:$said"
}

build() {  # what V3c's audit runs, as served_verdicts.sh's build
  uv sync -q --all-packages --extra torch-cpu || return 5
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) || return 7
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) || return 7
  mkdir -p "$OUT/m0-build" "$FW"
  RESEARCH_RUN_DIR=$OUT/m0-build FLOCK_WORK=$FW MODE=build GPU=0 flock "$FW/.build.lock" bash verity/ml/flock/pod/60-circuit.sh || return 6
}
(unset PYTHONPATH; cd "$TREE" && build) > "$OUT/build.log" 2>&1 &
CHILD=$!
wait "$CHILD"; rc=$?
CHILD=''
[ "$rc" = 0 ] || { step "build: exit $rc (out/build.log)"; exit "$rc"; }
leanb=$TREE/verity/core/service/flock/.lake/build/bin
bin=$FW/flock-gpu-link/flock/target/release/flock-circuit
for f in "$PY" "$leanb/flock-verify" "$leanb/flock-firewall" "$bin"; do
  [ -x "$f" ] || { step "build: no ${f##*/} after it"; exit 6; }
done
step "build done"
# what the public copy and the reader import after the audit, held before the warm and the audit: the tree's interpreter,
# started outside the tree's directory as they start it, reads served_integrity.py's imports and `verity` from the tree
imp=$(cd / && "$PY" -c 'import sys
sys.path.insert(0, sys.argv[1] + "/benchmarks/one_stage")
import served_integrity, verity
print(verity.__file__)' "$TREE" 2> "$OUT/imports.err" | tail -1)
[ -n "$imp" ] && [ "$imp" -ef "$TREE/verity/__init__.py" ] \
  || { step "IMPORTS: the tree's interpreter does not read served_integrity.py's imports from the tree (verity is ${imp:-not imported}; out/imports.err): the public copy and the reader's answer would fail after the audit, so the job ends before the warm and the audit"; exit 64; }
step "the imports: the tree's interpreter reads served_integrity.py's imports and verity from the tree"
if [ -n "${TREE_INPUT:-}" ]; then  # the pin a local session will take, by the tree's own function, held before the warm and the audit
  pin=$("$PY" -c 'import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "benchmarks" / "one_stage"))
import a5_coins
print(a5_coins.on_main(Path(sys.argv[2]).read_text().split()))' "$TREE" "$MAIN" 2> "$OUT/on-main.err" | tail -1)
  [ "$pin" = "$TREE_PIN" ] || { step "PIN: the tree's on_main gives ${pin:-nothing} (out/on-main.err), not $TREE_PIN: the session would be refused, so the job ends before the warm and the audit"; exit 66; }
  step "the pin: the tree's on_main gives $pin, as asked"
fi

run=$("$PY" -c 'import json, sys; print(next(json.loads(l)["run_root"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["window"] == int(sys.argv[2])))' \
  "$KEPT/windows.jsonl" "$K") || { step "PREFLIGHT: no run root for window $K in the kept windows.jsonl"; exit 64; }
m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/ml/flock/python:$TREE/integrations/vllm
cd "$TREE" || exit 64
cache=()
if [ -n "${WARM_JOBS:-}" ]; then
  w0=$(date +%s)
  at "the warm"
  limited "$WORK/warm.log" "$WORK/warm.err" "$PY" benchmarks/prove_path/records_audit.py warm --window "$VW/window" \
    --law "$LAW" --compose-cache "$WORK/compose-cache" --jobs "$WARM_JOBS"
  wrc=$?
  peak
  warm=(--warm-exit "$wrc" --warm-seconds "$(($(date +%s) - w0))")
  step "warm: exit $wrc, $WARM_JOBS at a time, $(($(date +%s) - w0)) s; the job's directory holds $PEAK_GB GB"
  if [ "$wrc" != 0 ]; then
    tail -3 "$WORK/warm.err" | cut -c1-600
    read_out "${warm[@]}"
    step "the warm failed: no audit starts"
    [ "$wrc" = 124 ] && exit 124
    exit 80
  fi
  cache=(--compose-cache "$WORK/compose-cache")
fi
a0=$(date +%s)
at "the audit"
[ -z "${AUDIT_WARNINGS:-}" ] || warn=(PYTHONWARNINGS="$AUDIT_WARNINGS")
step "audit: starts, $WORKERS negative workers, --audit-mem-gb $MEM_GB, $(left) s of the job left"
limited "$WORK/audit.log" "$WORK/audit.err" "$PY" benchmarks/prove_path/records_audit.py audit --window "$VW/window" \
  --law "$LAW" --circuit "$(sha512sum "$VW/window/circuit.json" | cut -c1-128)" --run "$run" --registered "$WREG" "${cache[@]}" \
  ${V3C_LEAN_WORKERS:+--lean-workers "$V3C_LEAN_WORKERS"} --audit-mem-gb "$MEM_GB" --negative-workers "$WORKERS" \
  --challenger local --main "$MAIN" --m0-pythonpath "$m0" --bin "$bin" --lean "$leanb/flock-verify" \
  --firewall "$leanb/flock-firewall" --ports "${V3C_PORTS:-16}" --out "$WORK/audit"
rc=$?
at "after the audit"
step "audit: exit $rc after $(($(date +%s) - a0)) s"
[ "$rc" = 0 ] || tail -3 "$WORK/audit.err" | cut -c1-600
"$PY" "$TREE/benchmarks/one_stage/served_integrity.py" public-copy "$WORK/audit" "$OUT/rehearsal-audit" \
  || { crc=$?; COPY=failed; step "NO PUBLIC COPY of the audit (served_integrity.py public-copy exit $crc)"; }
read_out "${warm[@]}" --exit "$rc" --audit-seconds "$(($(date +%s) - a0))"
[ "$COPY$ANSWER" = okok ] || keep_read
if [ -n "$(find "$KEPT" -newer "$WORK/.start" -print -quit 2>/dev/null)" ]; then
  step "THE KEPT SERVE DIR CHANGED while this job ran: $(find "$KEPT" -newer "$WORK/.start" 2>/dev/null | wc -l) path(s) newer than its start"
  exit 90
fi
step "the kept serve dir holds nothing newer than this job's start"
if [ "$COPY$ANSWER" != okok ]; then
  step "exit 73: the public copy $COPY, the answer $ANSWER; the audit's own exit was $rc"
  exit 73
fi
exit "$rc"
