# shellcheck shell=bash
# Sourced by each Pearl-C tree's build.sh, which states its target and adds what is its own.
#
# pearl_c_build DIR ARCH [OUT] builds DIR's NAME.cu (NAME is DIR's name) for ARCH into OUT (default /tmp/NAME-ship, with
# dashes): NAME.cubin with cubin_flags.sh's flags, refused on a GENCODE of another target, a toolkit older than 12.8 or an
# nvcc failure; then build_gate.py on DIR's sass_gate.json, ptxas's report and the SASS (sass.json), toolchain.txt, and sources.txt
# (pearl_c_sources). It leaves CUDA, OUT, FLAGS and ROOT set for the caller's own steps.
#
# pearl_c_sources SRC... appends to OUT/sources.txt, sorted and once each, the repository's files a cubin's bytes are made
# from: each SRC's include closure as nvcc -M reads it, cubin_flags.sh, this file and the calling build.sh.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../../.." && pwd)

pearl_c_build() {
  local dir=$1 arch=$2 name; name=$(basename "$dir")
  CUDA=${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit of nvcc 12.8 or later}; OUT=${3:-/tmp/${name//_/-}-ship}
  GENCODE=${GENCODE:?set GENCODE to the -gencode value of the target (benchmarks/kernels/machine.py gencode ARCH)}
  case $GENCODE in *,code=$arch) ;; *) echo "$name is $arch's: GENCODE=$GENCODE is another card's target" >&2; exit 2 ;; esac
  [ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
  "$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  . "$dir/../cubin_flags.sh"; cubin_flags "$GENCODE" || exit 1
  "$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/$name.cubin" "$dir/$name.cu" 2> "$OUT/ptxas.txt" || { cat "$OUT/ptxas.txt" >&2; exit 1; }
  PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$OUT/$name.cubin" > "$OUT/$name.sass"
  python3 "$dir/../../build_gate.py" "$dir/sass_gate.json" "$OUT/ptxas.txt" "$OUT/$name.sass" "$OUT/sass.json"
  rm "$OUT/$name.sass"
  pearl_c_sources "$dir/$name.cu"
  { "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o $name.cubin $name.cu"; } > "$OUT/toolchain.txt"
}

pearl_c_sources() {
  local src
  for src in "$@"; do "$CUDA/bin/nvcc" -std=c++17 -gencode "$GENCODE" -M "$src"; done \
    | python3 -c '
import os, sys
root, extra = os.path.realpath(sys.argv[1]), sys.argv[2:]
words = sys.stdin.read().replace("\\\n", " ").split()
print("\n".join(sorted({os.path.relpath(os.path.realpath(w), root) for w in words + extra
                        if os.path.isfile(w) and os.path.realpath(w).startswith(root + os.sep)})))' \
      "$ROOT" "$(dirname "${BASH_SOURCE[0]}")/../cubin_flags.sh" "${BASH_SOURCE[0]}" "$0" >> "$OUT/sources.txt"
  sort -u -o "$OUT/sources.txt" "$OUT/sources.txt"
}
