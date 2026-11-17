#!/usr/bin/env bash
# The proof-layers table's three benchmarks on served values, from a clone, as one command: Flock (a plain audit), Flock
# with zero knowledge, and with sampling, each a run of a5_llama1b_h200.sh on Llama-3.2-1B int8 served on one H200, then
# `verity bench prove layers` over the three. Its steps are README.md's "The three benchmarks of the proof-layers table":
# the bootstrap, the three partition files, the three runs and the printer. (layers.sh measures layers on stand-in
# values.) The recursion benchmark is not run here: its job wants 32 cores of a CPU node, and the printer takes it as a
# fourth input.
#
#   A5_CARD=N A5_CORES=CPUS WORK=DIR GPU_LEASE_WHO=NAME bash benchmarks/one_stage/proof_layers.sh [--out DIR] [--dry-run]
#
# from the tree's root. Settings:
#   A5_CARD        the H200's index, as nvidia-smi numbers the cards (gpu-lease's --on)
#   A5_CORES       the CPUs every step runs on (taskset -c, gpu-lease's --cores), e.g. 40-59
#   WORK           a new or empty directory for the bootstrap's environment, the partition files and the three runs,
#                  removed when the script ends, success or failure, after their records are copied to --out
#   GPU_LEASE_WHO  who every lease is recorded for, where gpu-lease is installed
#   --out DIR      where the results go (default $RESEARCH_RUN_DIR/out in a research run, else ./out): the three runs'
#                  records (out-flock, out-zk, out-sampling), the bootstrap's (bootstrap), the printer's table
#                  (proof-layers.txt, and proof-layers.json in a research run), each step's output (STEP.log) and GNU
#                  time's report (STEP.time), and steps.log
#   --dry-run      every check, then each step's line; nothing runs and nothing is made
# Refused before anything is downloaded or built, each in one line with its fix: a setting missing; not the tree's root;
# nvidia-smi, taskset, python3 or cargo missing; card A5_CARD not an H200 (the runs serve on one, sm_90); under 96 GB of
# host memory available (MemAvailable: on a shared node the others' memory is not this job's); A5_CORES not a CPU list
# this job may run on; under 25 GB free under WORK; WORK not empty; results already in --out. Each run's steps.log has
# what the job used.
# steps.log has each step's start (UTC), command, exit code and seconds, and three peaks over its samples (every 10 s):
# the job's own memory (the sum of PSS over the processes of this script's session), the machine's memory in use
# (MemTotal less MemAvailable, every process on the machine, other jobs included) and WORK's size (du, with each run's
# own files under WORK/a5-llama1b before it removes them); and the largest process's peak resident memory (GNU time's,
# where /usr/bin/time is installed).
set -uo pipefail
ME=benchmarks/one_stage/proof_layers.sh
refuse() { echo "proof_layers.sh: $1" >&2; exit 2; }
DRY=0 RESULTS=
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1; shift ;;
    --out) [ $# -ge 2 ] || refuse "--out needs a directory"; RESULTS=$2; shift 2 ;;
    *) refuse "$1: the settings are A5_CARD, A5_CORES, WORK and GPU_LEASE_WHO in the environment, then --out DIR and --dry-run" ;;
  esac
done
for s in A5_CARD A5_CORES WORK; do
  [ -n "${!s:-}" ] || refuse "no $s: A5_CARD=N A5_CORES=CPUS WORK=DIR GPU_LEASE_WHO=NAME bash $ME (the H200's index, the CPUs every step runs on, a new or empty work directory)"
done
LEASE=; command -v gpu-lease >/dev/null && LEASE=1
[ -z "$LEASE" ] || [ -n "${GPU_LEASE_WHO:-}" ] || refuse "no GPU_LEASE_WHO: this machine leases its cards through gpu-lease, and every lease is recorded for GPU_LEASE_WHO (unset, a5_llama1b_h200.sh records its leases as vllm-sha512); set it to your name"
[[ $A5_CARD =~ ^[0-9]+$ ]] || refuse "A5_CARD=$A5_CARD: a card's index, as nvidia-smi numbers them"
[ -f "$ME" ] || refuse "run from the tree's root: cd to the clone, then bash $ME"
for t in nvidia-smi taskset python3; do
  command -v "$t" >/dev/null || refuse "no $t on PATH: the runs need it; install it and rerun"
done
command -v cargo >/dev/null || [ -x "$HOME/.cargo/bin/cargo" ] \
  || refuse "no cargo: Flock's prover builds with it (verity/protocols/proofs/flock/prover/pod/60-circuit.sh); install Rust (https://rustup.rs) and rerun"
card=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$A5_CARD" 2>/dev/null | head -1)
[[ $card == *H200* ]] || refuse "card $A5_CARD is ${card:-not one nvidia-smi names}, not an H200: the runs serve on one H200 (sm_90); set A5_CARD to an H200's index"
mem_gb=$(awk '/^MemAvailable:/ {print int($2 * 1024 / 1e9)}' /proc/meminfo)
[ "${mem_gb:-0}" -ge 96 ] || refuse "${mem_gb:-0} GB of host memory available, not 96: the bootstrap's native builds need most of that of their own; wait for other jobs to end or run on another machine"
python3 - "$A5_CORES" <<'EOF' || exit 2
import os, re, sys
spec = sys.argv[1]
if not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", spec):
    sys.exit(f"proof_layers.sh: A5_CORES={spec}: a CPU list as taskset -c takes it, e.g. 40-59")
cpus = {c for part in spec.split(",") for a, _, b in [part.partition("-")] for c in range(int(a), int(b or a) + 1)}
if outside := sorted(cpus - os.sched_getaffinity(0)):
    sys.exit(f"proof_layers.sh: A5_CORES={spec} names CPUs this job may not run on ({', '.join(map(str, outside[:8]))}"
             f"{', ...' if len(outside) > 8 else ''}): give CPUs from `taskset -cp $$`'s list")
EOF
WORK=$(realpath -m "$WORK")
[ ! -e "$WORK" ] || { [ -d "$WORK" ] && [ -z "$(ls -A "$WORK")" ]; } \
  || refuse "WORK=$WORK is not empty: the script removes WORK when it ends, so it takes only a new or empty directory"
d=$WORK; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
[ -w "$d" ] || refuse "can't write $d: give a WORK you can make"
free_gb=$(df -Pk "$d" | awk 'NR == 2 {print int($4 * 1024 / 1e9)}')
[ "${free_gb:-0}" -ge 25 ] || refuse "${free_gb:-0} GB free under $d, not 25: the job needs 25 GB there; free some or give another WORK"
RESULTS=$(realpath -m "${RESULTS:-${RESEARCH_RUN_DIR:-.}/out}")
case $RESULTS/ in "$WORK"/*) refuse "--out $RESULTS is inside WORK, which is removed at the end: give a directory outside it" ;; esac
for f in out-flock out-zk out-sampling bootstrap proof-layers.txt proof-layers.json steps.log; do
  [ ! -e "$RESULTS/$f" ] || refuse "$RESULTS/$f exists, an earlier run's: move it or give --out another directory"
done

q() { printf '%q' "$1"; }
PART=benchmarks/one_stage/llama32_1b_rope_d64_l0.json
if [ -n "$LEASE" ]; then ON="gpu-lease 1 --wait --on $A5_CARD --cores $(q "$A5_CORES") --"; else ON="taskset -c $(q "$A5_CORES")"; fi
part() {  # LAW NAME: the tree's partition file with its sampling design (its sampling design key) set to LAW
  echo "python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); p[\"law\"] = sys.argv[2]; json.dump(p, open(sys.argv[3], \"w\"))' $PART $1 $(q "$WORK/partition-$2.json")"
}
run() {  # SETTINGS NAME: one run of a5_llama1b_h200.sh on NAME's partition file, its record in WORK/out-NAME
  echo "A5_COINS=challenger $1A5_PARTITION=$(q "$WORK/partition-$2.json") OUT=$(q "$WORK/out-$2") taskset -c $(q "$A5_CORES") bash benchmarks/one_stage/a5_llama1b_h200.sh"
}
STEPS=(bootstrap partition-flock partition-zk partition-sampling flock zk sampling layers)
declare -A CMD=(
  [bootstrap]="cd adapters/vllm && $ON bash verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B_INT8 --out $(q "$WORK/bootstrap")"
  [partition-flock]=$(part stratified-rate:880/880 flock)
  [partition-zk]=$(part subset:880 zk)
  [partition-sampling]=$(part subset:256 sampling)
  [flock]=$(run "A5_PLAIN=1 " flock)
  [zk]=$(run "" zk)
  [sampling]=$(run "" sampling)
  [layers]="PATH=\$HOME/.local/bin:\$PATH uv run verity bench prove layers --flock $(q "$WORK/out-flock") --zk $(q "$WORK/out-zk") --sampling $(q "$WORK/out-sampling") | tee $(q "$RESULTS/proof-layers.txt")"
)
HOW="card $A5_CARD taken directly (no gpu-lease)"; [ -z "$LEASE" ] || HOW="leases recorded for $GPU_LEASE_WHO"
echo "proof_layers.sh: card $A5_CARD ($card), $mem_gb GB of host memory available, CPUs $A5_CORES, $free_gb GB free under $d, $HOW; WORK=$WORK, results in $RESULTS$([ "$DRY" = 1 ] && echo '; a dry run, which runs nothing')"
if [ "$DRY" = 1 ]; then
  i=0; for n in "${STEPS[@]}"; do i=$((i + 1)); echo "STEP $i/${#STEPS[@]} $n: ${CMD[$n]}"; done
  [ -z "$LEASE" ] || echo "(each run of a5_llama1b_h200.sh leases card $A5_CARD with CPUs $A5_CORES itself, from A5_CARD and A5_CORES)"
  exit 0
fi

export A5_CARD A5_CORES WORK GPU_LEASE_WHO
mkdir -p "$WORK" "$RESULTS" || refuse "can't make WORK=$WORK or --out $RESULTS"
log() { echo "$*" | tee -a "$RESULTS/steps.log"; }
MEM=$RESULTS/.samples
SID=$(ps -o sid= -p $$ | tr -d ' ')
sample() {  # one line: the job's own PSS, the machine's memory in use and WORK's size, in KiB
  local own k
  own=$(for k in $(ps -o pid= -s "$SID"); do awk '/^Pss:/ {print $2}' "/proc/$k/smaps_rollup" 2>/dev/null; done | awk '{t += $1} END {print t + 0}')
  echo "$own $(awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2} END {print t - a}' /proc/meminfo) $(du -sk "$WORK" 2>/dev/null | cut -f1)" >> "$MEM"
}
peak() { tail -n +"$((from + 1))" "$MEM" | awk -v c="$1" '$c > m {m = $c} END {printf "%.1f GB", m * 1024 / 1e9}'; }
( while :; do sample; sleep 10; done ) &
SAMPLER=$!
finish() {  # at exit, success or failure: the records copied to --out, then WORK removed
  local rc=$? o
  kill "$SAMPLER" 2>/dev/null
  for o in out-flock out-zk out-sampling bootstrap; do [ ! -e "$WORK/$o" ] || cp -a "$WORK/$o" "$RESULTS/"; done
  log "WORK=$WORK at the end: $(du -sh "$WORK" 2>/dev/null | cut -f1); removing it"
  rm -rf "$WORK" && log "removed: $WORK"
  rm -f "$MEM"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TIME=; /usr/bin/time -v true >/dev/null 2>&1 && TIME=/usr/bin/time
i=0
for n in "${STEPS[@]}"; do
  i=$((i + 1)); log "STEP $i/${#STEPS[@]} $n at $(date -u +%H:%M:%SZ): ${CMD[$n]}"
  sample; t0=$SECONDS from=$(($(wc -l < "$MEM") - 1))
  if [ -n "$TIME" ]; then
    { $TIME -v -o "$RESULTS/$n.time" bash -c "set -o pipefail; ${CMD[$n]}"; } 2>&1 | tee "$RESULTS/$n.log"; rc=${PIPESTATUS[0]}
    rss=$(awk -F': ' '/Maximum resident set size/ {if ($2 < 976563) printf "%d MB", $2 * 1024 / 1e6; else printf "%.1f GB", $2 * 1024 / 1e9}' "$RESULTS/$n.time")
  else
    bash -c "set -o pipefail; ${CMD[$n]}" 2>&1 | tee "$RESULTS/$n.log"; rc=${PIPESTATUS[0]}
    rss="not measured (no /usr/bin/time)"
  fi
  sample
  log "STEP $i/${#STEPS[@]} $n: rc $rc in $((SECONDS - t0)) s; peak memory: this job's $(peak 1), the machine's in use $(peak 2); largest process ${rss:-not measured}; WORK's peak $(peak 3), $(du -sh "$WORK" | cut -f1) after"
  [ "$rc" = 0 ] || { log "stopped at $n (rc $rc): its output is $RESULTS/$n.log"; exit "$rc"; }
done
log "all ${#STEPS[@]} steps passed: the table is $RESULTS/proof-layers.txt"
