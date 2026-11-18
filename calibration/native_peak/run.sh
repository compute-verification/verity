#!/usr/bin/env bash
# verity bench native_peak [ARGS]: measure.py with ARGS (default --dtype bf16 --sizes 8192) into $OUT, on a CUDA GPU
# (nvfp4: an sm_120 one); exit 2 where none is visible.
set -euo pipefail
[ $# -gt 0 ] || set -- --dtype bf16 --sizes 8192
need="a CUDA GPU" cc='[0-9.]+'
[[ " $* " == *nvfp4* ]] && need="an sm_120 GPU (RTX PRO 6000, RTX 5090)" cc=12.0
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -qxE "$cc" \
  || { echo "native_peak: refused: needs $need, and none is visible" >&2; exit 2; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
exec python calibration/native_peak/measure.py --out "$OUT" "$@"
