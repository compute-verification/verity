#!/usr/bin/env bash
# verity bench tc_probe [ARGS]: tc_probe.py with ARGS (default --instruction sm120.mma.m16n8k32.e4m3 --canary) into $OUT,
# on a GPU of the instruction's architecture; exit 2 where none is visible.
set -euo pipefail
[ $# -gt 0 ] || set -- --instruction sm120.mma.m16n8k32.e4m3 --canary
ins=
for ((i = 1; i <= $#; i++)); do
  case ${!i} in
    --instruction) j=$((i + 1)); ins=${!j:-} ;;
    --instruction=*) ins=${!i#*=} ;;
  esac
done
[[ $ins =~ ^sm([0-9]+)([0-9]) ]] || { echo "tc_probe: --instruction names its architecture, as sm120.mma.m16n8k32.e4m3" >&2; exit 2; }
cc=${BASH_REMATCH[1]}.${BASH_REMATCH[2]}
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -qxF "$cc" \
  || { echo "tc_probe: refused: $ins needs an sm_${BASH_REMATCH[1]}${BASH_REMATCH[2]} GPU (compute capability $cc), and none is visible" >&2; exit 2; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-./out}}
mkdir -p "$OUT"
exec python benchmarks/tc_probe/tc_probe.py --out "$OUT" "$@"
