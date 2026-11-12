# shellcheck shell=bash
# The ship preamble the builds here source (attn_bf16's, s8_linear's, units' and p2_stage's build.sh): the
# toolkit, the cubin flags and the ship's layout. Sourced first, under `set -euo pipefail`; sets ROOT, the source tree's
# root, and each function below reads or sets CUDA, OUT and FLAGS.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# cicc, nvcc's front end, can emit different PTX for one source from run to run: some of its decisions follow its own heap
# addresses, whose layout also moves with the lengths of the paths it reads (the source's, the toolkit's, TMPDIR's), and
# some follow elapsed time, which a busy machine stretches. So a build that sources this runs itself once more, as a child
# of this shell, with address randomization off (setarch -R), TMPDIR a fresh directory whose path has one length, and
# ship_clock.c's clock inside cicc (LD_PRELOAD), all inherited by its compilers, and compiles its sources from a copy at a
# path of one length (ship_stage) with SHIP_NVCC, nvcc with what cicc reads freed of each file's place in the tree; it
# refuses where the kernel won't turn randomization off. A rebuild at the same commit, with the same toolkit at the same
# path and the same host compiler, then reproduces every cubin; after a commit that only moves a source or a file it
# includes, cicc is handed what it was handed before, and so does that rebuild (ship_nvcc.sh's header says what that
# leaves out; ship_probe.sh holds each build's toolkit to it)
if [ -z "${SHIP_FRAME:-}" ] || [ ! -e "$SHIP_FRAME/ship_clock.so" ] \
    || [ $(( 0x$(cat /proc/self/personality) & 0x40000 )) -eq 0 ]; then
  setarch "$(uname -m)" -R true 2> /dev/null || {
    echo "setarch -R is refused here, and nvcc's PTX can depend on address layout: build where a process may turn address" \
      "randomization off" >&2; exit 1; }
  SHIP_FRAME=$(mktemp -d /tmp/ship.XXXXXXXX); trap 'rm -rf "$SHIP_FRAME"' EXIT; mkdir "$SHIP_FRAME/tmp"
  gcc -shared -fPIC -O2 -o "$SHIP_FRAME/ship_clock.so" "$ROOT/benchmarks/torch_gpu/ship_clock.c" -ldl \
    || { echo "ship_clock.c doesn't build with gcc" >&2; exit 1; }
  SHIP_FRAME=$SHIP_FRAME TMPDIR=$SHIP_FRAME/tmp LD_PRELOAD="$SHIP_FRAME/ship_clock.so${LD_PRELOAD:+:$LD_PRELOAD}" \
    setarch "$(uname -m)" -R "$BASH" "$0" "$@" || exit $?
  exit 0
fi

# the nvcc every build here compiles with (ship_nvcc.sh: its header says what it fixes); it reads CUDA, which ship_cuda
# exports
SHIP_NVCC=$ROOT/benchmarks/torch_gpu/ship_nvcc.sh

# whether the toolkit $1 has nvcc 12.8 or later
ship_nvcc_ok() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\."; }

# CUDA, else the first of the toolkits named whose nvcc is 12.8 or later; empty when there is none
ship_find_cuda() {
  if [ -z "${CUDA:-}" ]; then
    for c in "$@"; do ship_nvcc_ok "$c" && { CUDA=$c; break; }; done
  fi
  CUDA=${CUDA:-}
}

# ship_find_cuda, refusing unless it finds one, and unless SHIP_NVCC with that toolkit still gives one cubin for a source
# at two places (ship_probe.sh, once a build: a toolkit or a wrapper that no longer does builds no ship)
ship_cuda() {
  ship_find_cuda "$@"
  ship_nvcc_ok "$CUDA" || { echo "no nvcc >= 12.8 at ${CUDA:-any of $*}: set CUDA" >&2; exit 1; }
  export CUDA
  "$ROOT/benchmarks/torch_gpu/ship_probe.sh" || exit 1
}

# FLAGS: SHIP_NVCC's command line for a cubin of the arch $1 (sm_90a, sm_120a): no FTZ, fast-math, FMA contraction or
# approximate division or square root, and no line info, so a cubin's bytes don't depend on the directory it's built in
ship_cubin_flags() {
  FLAGS=(-O3 -std=c++17 -gencode "arch=compute_${1#sm_},code=$1" -fmad=false -Xptxas -v -cubin)
  case " ${FLAGS[*]} " in
    *ftz*|*fast-math*|*fast_math*|*prec-div*|*prec-sqrt*|*lineinfo*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;;
  esac
}

# SRC: the paths named (from ROOT) copied into the frame, under a path of one length, for nvcc to compile from in place of
# ROOT, whose length is the checkout's
ship_stage() {
  SRC=$SHIP_FRAME/src
  local p
  for p in "$@"; do mkdir -p "$SRC/$(dirname "$p")"; cp -a "$ROOT/$p" "$SRC/$(dirname "$p")/"; done
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

# each shared library named, its cubins (cuobjdump -xelf) beside it in <library>.cubins/: a library's own bytes differ from
# build to build (nvcc's host-side registration names), its cubins don't
ship_cubins() {
  local so d
  for so in "$@"; do
    so=$(readlink -f "$so"); d=$so.cubins
    [ -n "$("$CUDA/bin/cuobjdump" --list-elf "$so" 2> /dev/null)" ] || continue
    rm -rf "$d"; mkdir -p "$d"
    (cd "$d" && "$CUDA/bin/cuobjdump" -xelf all "$so" > /dev/null)
  done
}

# OUT/MANIFEST.sha256 over every other file in OUT, every shared library's cubins among them (ship_cubins), and OUT's size
ship_manifest() {
  local so
  while IFS= read -r so; do ship_cubins "$so"; done < <(find "$OUT" -type f -name '*.so' | LC_ALL=C sort)
  (cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
  du -sh "$OUT" | cut -f1 | xargs echo "ship:"
}
