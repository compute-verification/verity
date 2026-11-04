# shellcheck shell=bash
# The ship preamble the builds here source (attn_bf16's, s8_linear's, units' and s8_dec_linear's build.sh): the toolkit,
# the cubin flags and the ship's layout. Sourced under `set -euo pipefail`; sets ROOT, the source tree's root, and each
# function below reads or sets CUDA, OUT and FLAGS.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# whether the toolkit $1 has nvcc 12.8 or later
ship_nvcc_ok() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\."; }

# CUDA, else the first of the toolkits named whose nvcc is 12.8 or later; empty when there is none
ship_find_cuda() {
  if [ -z "${CUDA:-}" ]; then
    for c in "$@"; do ship_nvcc_ok "$c" && { CUDA=$c; break; }; done
  fi
  CUDA=${CUDA:-}
}

# ship_find_cuda, refusing unless it finds one
ship_cuda() {
  ship_find_cuda "$@"
  ship_nvcc_ok "$CUDA" || { echo "no nvcc >= 12.8 at ${CUDA:-any of $*}: set CUDA" >&2; exit 1; }
}

# FLAGS: nvcc's command line for a cubin of the arch $1 (sm_90a, sm_120a): no FTZ, fast-math, FMA contraction or
# approximate division or square root, and no line info, so a cubin's bytes don't depend on the directory it's built in
ship_cubin_flags() {
  FLAGS=(-O3 -std=c++17 -gencode "arch=compute_${1#sm_},code=$1" -fmad=false -Xptxas -v -cubin)
  case " ${FLAGS[*]} " in
    *ftz*|*fast-math*|*fast_math*|*prec-div*|*prec-sqrt*|*lineinfo*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;;
  esac
}

# OUT, emptied
ship_fresh() { rm -rf "$OUT"; mkdir -p "$OUT"; }

# OUT/toolchain.txt: nvcc --version, to which a build appends its command lines
ship_toolchain() { "$CUDA/bin/nvcc" --version > "$OUT/toolchain.txt"; }

# OUT/source.txt: the commit, and what `git status` says of the paths named (from ROOT)
ship_source() {
  { git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
    git -C "$ROOT" status --porcelain -- "$@" 2>/dev/null || true; } > "$OUT/source.txt"
}

# OUT/MANIFEST.sha256 over every other file in OUT, and OUT's size
ship_manifest() {
  (cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
  du -sh "$OUT" | cut -f1 | xargs echo "ship:"
}
