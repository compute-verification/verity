#!/usr/bin/env bash
# Replay determinism on one GPU (README.md).  Under `research run`, which sets OUT:
#   run.sh fetch   download the model to MODEL_DIR, refusing unless DISK_FLOOR_GB and INODE_FLOOR stay free, and register
#                  it with `research keep` (KEEP_OWNER, KEEP_EXPIRES, KEEP_PRIORITY)
#   run.sh         build the window, then under one `gpu-lease 1`: serve it, replay it (twice captured, once plain),
#                  serve each request alone; then compare.  MODEL_DIR may be any local checkpoint (a dense baseline), and
#                  BATCH_INVARIANT=1 serves every phase with VLLM_BATCH_INVARIANT=1.  The fingerprints go to SCRATCH, by
#                  default a `mktemp -d` under WORK that run.sh removes when it exits; a SCRATCH you give is left in place
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
: "${OUT:?run under research run, which sets OUT}"
PY=${PY:-/workspace/cp/venv312/bin/python}
WORK=${WORK:-/workspace/research/scratch/replay-determinism}
MODEL_DIR=${MODEL_DIR:-$WORK/moonlight}
N=${N:-48}
PHASES=${PHASES:-serve replay1 replay2 replay_plain alone}
LEASE_MIN=${LEASE_MIN:-60}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-50}
INODE_FLOOR=${INODE_FLOOR:-100000}
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 PYTHONPATH="$root/adapters/vllm${PYTHONPATH:+:$PYTHONPATH}"

floor() {   # refuse unless the filesystem holding WORK keeps its floors after NEED_GB more
  local need=$1 gb inodes
  gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  inodes=$(df --output=iavail /workspace | tail -1 | tr -dc 0-9)
  echo "/workspace: ${gb} GB and ${inodes} inodes free; needs ${need} GB more, floors ${DISK_FLOOR_GB} GB and ${INODE_FLOOR} inodes"
  (( gb - need >= DISK_FLOOR_GB && inodes - 1000 >= INODE_FLOOR )) || { echo "refused: below the floor"; exit 3; }
}

if [[ ${1:-} == fetch ]]; then
  floor 35
  mkdir -p "$WORK"
  HF_HUB_OFFLINE=0 HF_HOME="$WORK/hf" "$PY" "$here/replay_det.py" fetch --dir "$MODEL_DIR"
  rm -rf "$WORK/hf" "$MODEL_DIR/.cache"
  PYTHONPATH="$root/tools/research/src" "$PY" -m research keep "$MODEL_DIR" --owner "${KEEP_OWNER:-@vllm}" \
    --expires "${KEEP_EXPIRES:-never}" --priority "${KEEP_PRIORITY:-keep}" \
    --regenerate "MODEL_DIR=$MODEL_DIR bash benchmarks/replay_determinism/run.sh fetch" \
    --why "replay_determinism's model at its pinned revision, read by every phase" \
    ${RESEARCH_AGENT_URL:+--by "$RESEARCH_AGENT_URL"} || { echo "no retention record on $MODEL_DIR"; exit 4; }
  floor 0
  exit 0
fi

phase() {   # NAME: one phase in its own process; its fingerprints stay in SCRATCH, the rest goes to OUT
  local name=$1 args
  case $name in
    serve) args=(serve --capture) ;;
    replay_plain) args=(replay --record "$SCRATCH/serve") ;;
    replay*) args=(replay --capture --record "$SCRATCH/serve") ;;
    alone) args=(alone --capture) ;;
    *) echo "unknown phase $name"; exit 2 ;;
  esac
  [[ ${BATCH_INVARIANT:-0} == 1 ]] && args+=(--batch-invariant)
  echo "== $name $(date -u +%FT%TZ)"
  "$PY" "$here/replay_det.py" "${args[@]}" --model "$MODEL_DIR" --workload "$OUT/workload.json" --out "$SCRATCH/$name" \
    > "$OUT/$name.log" 2>&1 || { tail -n 120 "$OUT/$name.log"; exit 1; }
  tail -n 3 "$OUT/$name.log"
  mkdir -p "$OUT/$name"
  for f in steps.jsonl tokens.json timing.json labels.json arrivals.json whole.jsonl keys.npy logit_keys.npy \
           logit_digest.npy logit_top.npy logit_topi.npy; do
    [[ -e $SCRATCH/$name/$f ]] && cp "$SCRATCH/$name/$f" "$OUT/$name/"
  done
  return 0
}

if [[ ${1:-} == leased ]]; then
  t0=$(date +%s.%N)
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.sm --format=csv
  for p in $PHASES; do phase "$p"; done
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN { printf "{\"lease_wall_s\": %.1f, \"gpus\": 1}\n", b - a }' > "$OUT/lease.json"
  exit 0
fi

[[ -d $MODEL_DIR ]] || { echo "no model at $MODEL_DIR: run.sh fetch first"; exit 2; }
floor 3
mkdir -p "$WORK"
if [[ -z ${SCRATCH:-} ]]; then SCRATCH=$(mktemp -d "$WORK/scratch.XXXXXX"); trap 'rm -rf -- "$SCRATCH"' EXIT; fi
export SCRATCH OUT PY MODEL_DIR PHASES BATCH_INVARIANT
echo "scratch $SCRATCH"
"$PY" "$here/replay_det.py" workload --model "$MODEL_DIR" --corpus "$root" --n "$N" --out "$OUT/workload.json"
GPU_LEASE_WHO=${GPU_LEASE_WHO:-replay-determinism} gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash "$here/run.sh" leased
"$PY" "$here/compare.py" --runs "$SCRATCH" --workload "$OUT/workload.json" --out "$OUT"
