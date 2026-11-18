#!/usr/bin/env bash
# verity bench native_peak [ARGS]: measure.py with ARGS (default --dtype bf16 --sizes 8192) into $OUT, on a CUDA GPU
# (nvfp4: one of measure.py's NVFP4_ARCH); exit 2 where none is visible.
set -euo pipefail
[ $# -gt 0 ] || set -- --dtype bf16 --sizes 8192
need="a CUDA GPU" cc='[0-9.]+'
if [[ " $* " == *nvfp4* ]]; then
  read -r arch cc < <(python3 -c 'import sys, tomllib; sys.path.insert(0, "calibration/native_peak"); import measure as m
print(m.NVFP4_ARCH, tomllib.loads((m.CARDS.parent / "architectures" / f"{m.NVFP4_ARCH}.toml").read_text())["capability"])') \
    && [ -n "$cc" ] || { echo "native_peak: refused: nvfp4's architecture file is unreadable" >&2; exit 2; }
  need="a GPU of $arch (capability $cc)"
fi
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -qxE "$cc" \
  || { echo "native_peak: refused: needs $need, and none is visible" >&2; exit 2; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
exec python calibration/native_peak/measure.py --out "$OUT" "$@"
