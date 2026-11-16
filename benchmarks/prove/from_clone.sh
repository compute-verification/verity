#!/usr/bin/env bash
# The proof benchmark by layer from a clone, as one command, on one H200 (benchmarks/prove/README.md, "On a machine of
# your own"): benchmarks/one_stage/README.md's section "The three benchmarks of the proof-layers table" (the adapter's
# bootstrap, the three partition files, the Flock, zero-knowledge and sampling runs), then the recursion job, then
# `verity bench prove layers` on the four runs' OUT directories. Each step's command is the README's; this adds their
# order, the checks before any download or build, and the print.
#
#   bash benchmarks/prove/from_clone.sh WORK [--dry-run]        from the tree's root
#
# WORK holds the bootstrap's environment, the partition files and the four runs' OUT (WORK/out-LAYER); made here, it is
# removed at the end of a run that printed, and kept otherwise. OUT (default ./out) gets the printed benchmark results
# (layers.txt, and proof-layers.json from the printer), the four runs' OUT and this script's log (from_clone.log), whose
# lines say each step's name, command and seconds. --dry-run makes the checks and prints each step's command, and runs
# none.
# env: A5_CARD (0)  CPUS (the one-stage jobs' cores, default every core)  REC_CPUS (the recursion job's 32, default 0-31)
#      GPU_LEASE_WHO (each lease's name where gpu-lease exists, default proof-layers)
#      MEMINFO (/proc/meminfo)
# Checks, each refused in one line with its fix, before any download or build:
#   the card: A5_CARD is an H200 (a5_llama1b_h200.sh serves on one H200, sm_90)
#   memory: 448 GB (a5_llama1b_h200.sh asks --mem-gb 448, at its peak about 400 GB; recursion_e2e/job.sh asks 128)
#   cores: REC_CPUS names 32 cores this machine has (job.sh runs under taskset -c 0-31)
#   disk: 4 GB free under WORK for each one-stage run (a5_llama1b_h200.sh: about 3 GB under A5_ROOT and 0.3 GB for M0's Flock build)
#   tools: nvidia-smi, cargo (flock-circuit's build), uv, taskset and python3; a5 installs uv and elan where they're missing
set -uo pipefail
DRY=0; WORK=
for a in "$@"; do case $a in --dry-run) DRY=1 ;; -*) echo "from_clone.sh: $a: give WORK, and --dry-run or nothing" >&2; exit 2 ;; *) WORK=$a ;; esac; done
refuse() { echo "from_clone.sh: $1" >&2; exit 2; }
[ -n "$WORK" ] || refuse "no WORK: bash benchmarks/prove/from_clone.sh WORK [--dry-run], WORK a directory for the bootstrap and the runs"
[ -f benchmarks/one_stage/a5_llama1b_h200.sh ] || refuse "run from the tree's root: cd to the clone, then bash benchmarks/prove/from_clone.sh WORK"
export A5_CARD=${A5_CARD:-0} CPUS=${CPUS:-0-$(($(nproc --all) - 1))} REC_CPUS=${REC_CPUS:-0-31}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-proof-layers} A5_CORES=${CPUS:-}
OUT=$(realpath -m "${OUT:-./out}")

cores() {  # LIST: how many cores a taskset list names
  local n=0 part; IFS=, read -ra parts <<< "$1"
  for part in "${parts[@]}"; do [[ $part == *-* ]] && n=$((n + ${part#*-} - ${part%-*} + 1)) || n=$((n + 1)); done; echo "$n"
}
for t in nvidia-smi cargo uv taskset python3; do
  command -v "$t" >/dev/null || case $t in
    nvidia-smi) refuse "no nvidia-smi: the three one-stage runs serve on one H200 (sm_90); run this on an H200 machine with its driver" ;;
    cargo) refuse "no cargo: flock-circuit builds with it; install Rust (https://rustup.rs) and rerun" ;;
    uv) refuse "no uv: the bootstrap and the printer run with it; install it (https://docs.astral.sh/uv/getting-started/installation/) and rerun" ;;
    *) refuse "no $t on PATH: install it and rerun" ;;
  esac
done
card=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$A5_CARD" 2>/dev/null | head -1)
[[ $card == *H200* ]] || refuse "card $A5_CARD is ${card:-not one nvidia-smi names}, not an H200: the one-stage runs serve on one H200 (sm_90); set A5_CARD to an H200's index"
mem_kb=$(awk '/^MemTotal:/ {print $2}' "${MEMINFO:-/proc/meminfo}")
[ "${mem_kb:-0}" -ge $((448 * 1024 * 1024)) ] || refuse "$((${mem_kb:-0} / 1024 / 1024)) GB of memory, not 448: a one-stage run peaks at about 400 GB (a5_llama1b_h200.sh); run on a machine with 448 GB"
[ "$(cores "$REC_CPUS")" -ge 32 ] || refuse "REC_CPUS=$REC_CPUS names $(cores "$REC_CPUS") cores, not 32: the recursion job runs on 32 (job.sh's taskset -c 0-31); give REC_CPUS 32 of this machine's cores"
[ "${REC_CPUS##*[-,]}" -lt "$(nproc --all)" ] || refuse "REC_CPUS=$REC_CPUS goes past this machine's $(nproc --all) cores: give REC_CPUS 32 cores it has"
MADE=0; [ -d "$WORK" ] || MADE=1
[ "$DRY" = 1 ] || mkdir -p "$WORK" || refuse "can't make WORK=$WORK: give a directory you can write"
WORK=$(realpath -m "$WORK"); export WORK
d=$WORK; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
free_gb=$(($(df -Pk "$d" | awk 'NR == 2 {print $4}') / 1024 / 1024))
[ "$free_gb" -ge 4 ] || refuse "$free_gb GB free under $d, not 4: a one-stage run takes about 3.3 GB there while it runs (a5_llama1b_h200.sh); free some or give another WORK"

# benchmarks/one_stage/README.md's lines, in order, then the recursion job and the printer; on a machine with gpu-lease the
# bootstrap runs under a lease of the same card and cores in place of taskset (that README)
LEASE=; command -v gpu-lease >/dev/null && LEASE="gpu-lease 1 --wait --on \$A5_CARD --cores \$CPUS --"
STEPS=$(cat <<'EOF'
bootstrap	(cd adapters/vllm && taskset -c $CPUS bash verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B_INT8)
partitions	for b in flock=stratified-rate:880/880 zk=subset:880 sampling=subset:256; do python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); p["law"] = sys.argv[2]; json.dump(p, open(sys.argv[3], "w"))' benchmarks/one_stage/llama32_1b_rope_d64_l0.json "${b#*=}" "$WORK/partition-${b%%=*}.json"; done
flock	A5_COINS=challenger A5_PLAIN=1 A5_PARTITION=$WORK/partition-flock.json OUT=$WORK/out-flock taskset -c $CPUS bash benchmarks/one_stage/a5_llama1b_h200.sh
zk	A5_COINS=challenger A5_PARTITION=$WORK/partition-zk.json OUT=$WORK/out-zk taskset -c $CPUS bash benchmarks/one_stage/a5_llama1b_h200.sh
sampling	A5_COINS=challenger A5_PARTITION=$WORK/partition-sampling.json OUT=$WORK/out-sampling taskset -c $CPUS bash benchmarks/one_stage/a5_llama1b_h200.sh
recursion	OUT=$WORK/out-recursion taskset -c $REC_CPUS bash benchmarks/recursion_e2e/job.sh K=2048 N=2 GPU=0 FLOCK_WORK=$WORK/rec-flock
layers	OUT=$OUT uv run verity bench prove layers --flock $WORK/out-flock --zk $WORK/out-zk --sampling $WORK/out-sampling --recursion $WORK/out-recursion
EOF
)
[ -n "$LEASE" ] && STEPS=${STEPS/taskset -c \$CPUS bash verity_vllm/$LEASE bash verity_vllm}
HOW="no gpu-lease: the card directly"; [ -n "$LEASE" ] && HOW="gpu-lease as $GPU_LEASE_WHO"

[ "$DRY" = 1 ] || { mkdir -p "$OUT" && exec > >(tee -a "$OUT/from_clone.log") 2>&1; }
echo "from_clone.sh: card $A5_CARD $card, $((mem_kb / 1024 / 1024)) GB of memory, $free_gb GB free under WORK=$WORK, CPUS=$CPUS, REC_CPUS=$REC_CPUS, $HOW$([ "$DRY" = 1 ] && echo '; a dry run, which runs no step')"
echo "ran with: coins challenger (each one-stage run's local challenger session, a development stand-in for the auditor's device); parameters not stated (the printer states each run's)"
while IFS=$'\t' read -r name cmd; do
  echo "STEP $name: $cmd"
  [ "$DRY" = 1 ] && continue
  t0=$SECONDS
  if [ "$name" = layers ]; then (eval "$cmd") | tee "$OUT/layers.txt"; else (eval "$cmd"); fi && rc=0 || rc=$?
  echo "STEP $name: rc $rc in $((SECONDS - t0)) s"
  if [ "$rc" != 0 ] && [ "$name" != layers ]; then
    echo "from_clone.sh: step $name failed (rc $rc): its log is above; WORK=$WORK is kept for a rerun"; exit "$rc"
  fi
done <<< "$STEPS"
[ "$DRY" = 1 ] && exit 0
for l in flock zk sampling recursion; do cp -a "$WORK/out-$l" "$OUT/out-$l"; done
if [ "$MADE" = 1 ] && [ "${rc:-1}" = 0 ]; then rm -rf "$WORK" && echo "from_clone.sh: removed WORK=$WORK, which it made"; fi
exit "${rc:-1}"
