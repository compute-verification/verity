#!/usr/bin/env bash
# verity bench tc_probe_fp4 [ARGS]: probe.py with ARGS (default --instruction nvf4 --sass-check --sweep) into $OUT, on a GPU of
# its --arch (default sm_120a, an RTX PRO 6000 or RTX 5090; sm_100a for the tcgen05 instructions); exit 2 where none is
# visible.
set -euo pipefail
[ $# -gt 0 ] || set -- --instruction nvf4 --sass-check --sweep
arch= ins=nvf4
for ((i = 1; i <= $#; i++)); do
  j=$((i + 1))
  case ${!i} in
    --arch) arch=${!j:-} ;;
    --arch=*) arch=${!i#*=} ;;
    --instruction) ins=${!j:-} ;;
    --instruction=*) ins=${!i#*=} ;;
  esac
done
[[ -n $arch ]] || { [[ $ins == tcgen05_* ]] && arch=sm_100a || arch=sm_120a; }
[[ $arch =~ ^sm_([0-9]+)([0-9])a?$ ]] || { echo "tc_probe_fp4: --arch is an sm_NNa architecture, not $arch" >&2; exit 2; }
cc=${BASH_REMATCH[1]}.${BASH_REMATCH[2]}
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -qxF "$cc" \
  || { echo "tc_probe_fp4: refused: needs an $arch GPU (compute capability $cc), and none is visible" >&2; exit 2; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-./out}}
mkdir -p "$OUT"
exec python benchmarks/tc_probe_fp4/probe.py --out "$OUT" "$@"
