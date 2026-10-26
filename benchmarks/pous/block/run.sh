#!/usr/bin/env bash
# PoUS's block run on a node (block.py): the sessions with C* held (under gpu-lease for --memory hbm, when the node has
# it), then their checks outside the lease, each against the table in rows/. cwd = a clone of the tree:
#
#   research run --on NODE --project verity --source . --cwd clone --timeout 60m -- \
#     bash benchmarks/pous/block/run.sh --store DIR [--smoke] [BLOCK_OPTIONS]
#
#   --smoke        hold C*, check 64 random blocks against rows/ and print the certificate at the store's point: no
#                  session and no challenger (block.py --phase smoke), under the lease
#   --max-min M    the lease's bound (default 30)
#   BLOCK_OPTIONS  block.py's (--memory, --honest, --drop, --challenger/--challenger-address/--challenger-command,
#                  --trusted, --node, --cpus); --out is $OUT (a run's out/; else ./out)
#
# Outputs under $OUT: env.txt, gpu.txt (hbm), summary.json, row.json, sessions/. Exit 0 when every check passes.
set -euo pipefail
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd "$(dirname "$SELF")/../../.."
export O=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$O"
if [ "${1:-}" = leased ]; then
  shift
  [ -n "${GPU_LEASE_UUID:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
  if [ "$BLOCK_MEMORY" = hbm ]; then
    nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv > "$O/gpu.txt" 2>&1 || true
  fi
  exec uv run --no-sync python benchmarks/pous/block/block.py --out "$O" "$@"
fi
SMOKE=0 MAXMIN=30 MEMORY=hbm A=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --max-min) MAXMIN=$2; shift ;;
    --memory) MEMORY=$2; A+=("$1" "$2"); shift ;;
    --out|--phase) echo "run.sh: $1 is run.sh's own"; exit 2 ;;
    *) A+=("$1") ;;
  esac
  shift
done
A+=(--memory "$MEMORY")
{ echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"; echo "args: ${A[*]}"; date -u +%FT%TZ; hostname
  lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"; free -g; uptime
  for i in /sys/class/net/*; do echo "$(basename "$i") $(cat "$i/speed" 2>/dev/null || echo -) Mb/s"; done
} > "$O/env.txt" 2>&1
uv sync -q --all-packages --extra torch-cpu
export BLOCK_MEMORY=$MEMORY
leased() {
  if [ "$MEMORY" = hbm ] && [ -z "${GPU_LEASE_UUID:-}" ] && command -v gpu-lease >/dev/null 2>&1; then
    GPU_LEASE_WHO=memory-accounting gpu-lease 1 --wait --max-min "$MAXMIN" -- bash "$SELF" leased "$@"
  else
    bash "$SELF" leased "$@"
  fi
}
if [ "$SMOKE" = 1 ]; then
  date -u +smoke-%H:%M:%SZ
  leased --phase smoke "${A[@]}"
  exit $?
fi
date -u +sessions-%H:%M:%SZ
leased --phase sessions "${A[@]}"
date -u +check-%H:%M:%SZ
rc=0
uv run --no-sync python benchmarks/pous/block/block.py --out "$O" --phase check --sessions "$O/sessions" "${A[@]}" || rc=$?
date -u +done-%H:%M:%SZ
exit $rc
