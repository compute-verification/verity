#!/usr/bin/env bash
# Builds the ship run.sh unpacks on the card, off the card (nvcc >= 12.8 for sm_90a, no GPU needed):
#   libu2gemm.so     every configuration of CONFIGS (one u2_gemm.cu translation unit each, its functions prefixed by its
#                    name) and hash.cu, with the CUDA runtime linked statically
#   build.json       each configuration's macros, ptxas registers and spill, its SASS MMA forms; CUTLASS's tag and commit,
#                    nvcc's version, the patch below and the source commit
#   cutlass.patch, ptxas/*.txt, MANIFEST.sha256
# CUTLASS is header-only, fetched at the pinned commit into scratch and patched there (never vendored): the cooperative and
# ping-pong kernels' setmaxnreg constants become macros, U2_LOAD_REGS and U2_MMA_REGS, defaulting to CUTLASS's values in
# u2_gemm.cu, so a configuration launched at fewer registers can hand its consumers what the launch holds.
# Usage: build.sh OUT [NAME ...] (default every configuration); env CUDA (default /usr/local/cuda-13.0, else
# /usr/local/cuda), JOBS (parallel compiles, default 4: a CUTLASS 3.x unit takes up to 7 GB in cicc), CUTLASS_DIR (an
# unpatched checkout at the pinned commit, instead of the fetch).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:?usage: build.sh OUT [NAME ...]}; shift
CUDA=${CUDA:-/usr/local/cuda-13.0}; [ -x "$CUDA/bin/nvcc" ] || CUDA=/usr/local/cuda
NVCC=$CUDA/bin/nvcc
[ -x "$NVCC" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$NVCC" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$NVCC is older than 12.8" >&2; exit 1; }
JOBS=${JOBS:-4}
CUTLASS_TAG=v4.8.0
CUTLASS_COMMIT=098de2a652cf8f00fd70b2df54051c7eccbb855a
SHA_DIR=$ROOT/benchmarks/torch_gpu/sha512_rows

# name  tile M N K  cluster M N  stages(0 auto)  pingpong  nosmem  maxnreg(0 stock)  load_regs  mma_regs
CONFIGS=(
  "c128x256_c2x1_tma     128 256 128  2 1  0  0 0    0  -   -"
  "c128x256_c2x1_ns4     128 256 128  2 1  4  0 1    0  -   -"
  "c128x256_c1x1_tma     128 256 128  1 1  0  0 0    0  -   -"
  "c256x128_c1x2_tma     256 128 128  1 2  0  0 0    0  -   -"
  "c128x256_c2x1_tma_s3  128 256 128  2 1  3  0 0    0  -   -"
  "c128x256_c2x1_ns4_r128 128 256 128 2 1  4  0 1  128 40 168"
  "c128x256_c2x1_tma_r128 128 256 128 2 1  0  0 0  128 40 168"
  "c128x256_c2x1_ns4_r120 128 256 128 2 1  4  0 1  120 24 168"
  "p128x128_c2x1_tma     128 128 128  2 1  0  1 0    0  -   -"
)
want=("$@")

S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
rm -rf "$OUT"; mkdir -p "$OUT/ptxas"
if [ -n "${CUTLASS_DIR:-}" ]; then
  cp -r "$CUTLASS_DIR" "$S/cutlass"
else
  git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass "$S/cutlass"
fi
got=$(git -C "$S/cutlass" rev-parse HEAD)
[ "$got" = "$CUTLASS_COMMIT" ] || { echo "CUTLASS at $got, want $CUTLASS_COMMIT ($CUTLASS_TAG)" >&2; exit 2; }
for f in sm90_gemm_tma_warpspecialized_cooperative sm90_gemm_tma_warpspecialized_pingpong; do
  h=$S/cutlass/include/cutlass/gemm/kernel/$f.hpp
  sed -i -e 's/LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;/LoadRegisterRequirement = U2_LOAD_REGS;/' \
    -e 's/MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;/MmaRegisterRequirement = U2_MMA_REGS;/' "$h"
  [ "$(grep -c 'U2_LOAD_REGS;\|U2_MMA_REGS;' "$h")" = 2 ] || { echo "the setmaxnreg patch didn't apply to $f" >&2; exit 2; }
done
git -C "$S/cutlass" diff > "$OUT/cutlass.patch"

FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xcompiler -fPIC --expt-relaxed-constexpr -Xptxas -v)
INC=(-I"$S/cutlass/include" -I"$S/cutlass/tools/util/include")
objs=(); names=()
compile() {  # NAME then the -D flags
  local name=$1; shift
  "$NVCC" "${FLAGS[@]}" "${INC[@]}" "$@" -c "$HERE/u2_gemm.cu" -o "$S/$name.o" > "$OUT/ptxas/$name.txt" 2>&1 \
    || { echo "compile $name failed:" >&2; tail -30 "$OUT/ptxas/$name.txt" >&2; return 1; }
}
for row in "${CONFIGS[@]}"; do
  read -r name tm tn tk cm cn st pp ns mr lr rr <<< "$row"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $name "* ]]; then continue; fi
  d=(-DU2_NAME="$name" -DU2_TM="$tm" -DU2_TN="$tn" -DU2_TK="$tk" -DU2_CM="$cm" -DU2_CN="$cn" -DU2_STAGES="$st"
     -DU2_PINGPONG="$pp" -DU2_NOSMEM="$ns" -DU2_MAXNREG="$mr")
  [ "$lr" = - ] || d+=(-DU2_LOAD_REGS="$lr" -DU2_MMA_REGS="$rr")
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
  echo "compile $name at $(date -u +%H:%M:%SZ)"
  compile "$name" "${d[@]}" &
  objs+=("$S/$name.o"); names+=("$name")
done
"$NVCC" "${FLAGS[@]}" -I"$SHA_DIR" -c "$HERE/hash.cu" -o "$S/hash.o" > "$OUT/ptxas/hash.txt" 2>&1 \
  || { cat "$OUT/ptxas/hash.txt" >&2; exit 1; }
wait
for o in "${objs[@]}"; do [ -f "$o" ] || { echo "missing $o" >&2; exit 1; }; done
"$NVCC" -gencode arch=compute_90a,code=sm_90a -shared -cudart static -o "$OUT/libu2gemm.so" "${objs[@]}" "$S/hash.o"
"$CUDA/bin/cuobjdump" -sass "$OUT/libu2gemm.so" > "$S/lib.sass"
awk '/Function :/{f=$3; next} /\/\*[0-9a-f]+\*\/ +[A-Z]*GMMA/{match($0,/[A-Z]*GMMA[.A-Z0-9x_]*/); c[f"\t"substr($0,RSTART,RLENGTH)]++}
     END{for(k in c) print k"\t"c[k]}' "$S/lib.sass" | sort > "$OUT/sass_mma.tsv"
"$NVCC" --version > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
  git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/u2_cutlass benchmarks/torch_gpu/sha512_rows 2>/dev/null || true
} > "$OUT/source.txt"
python3 - "$OUT" "$CUTLASS_TAG" "$CUTLASS_COMMIT" "${names[*]}" "${CONFIGS[@]}" <<'EOF'
import json, re, sys
out, tag, commit, names, *rows = sys.argv[1:]
cfg = {}
for row in rows:
    f = row.split()
    if f[0] not in names.split():
        continue
    ptx = open(f"{out}/ptxas/{f[0]}.txt").read()
    # the entry's line: ptxas prints one "Used N registers" per kernel; the TU has the entry alone
    regs = [int(m) for m in re.findall(r"Used (\d+) registers", ptx)]
    spill = [tuple(map(int, m)) for m in re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ptx)]
    cfg[f[0]] = {"tile": list(map(int, f[1:4])), "cluster": list(map(int, f[4:6])), "stages": int(f[6]),
                 "pingpong": int(f[7]), "nosmem": int(f[8]), "maxnreg": int(f[9]),
                 "load_regs": None if f[10] == "-" else int(f[10]), "mma_regs": None if f[11] == "-" else int(f[11]),
                 "ptxas_regs": regs, "ptxas_spill": spill, "warnings": [l for l in ptx.splitlines() if "warning" in l]}
sass = {}
for line in open(f"{out}/sass_mma.tsv"):
    fn, op, n = line.rstrip("\n").split("\t")
    sass.setdefault(fn, {})[op] = int(n)
hash_ptx = open(f"{out}/ptxas/hash.txt").read()
json.dump({"cutlass": {"tag": tag, "commit": commit}, "configs": cfg, "sass_mma": sass,
           "hash_ptxas": re.findall(r"Function properties for (\S+)|Used (\d+) registers", hash_ptx),
           "toolchain": open(f"{out}/toolchain.txt").read().strip().splitlines()[-1],
           "source": open(f"{out}/source.txt").read().split()}, open(f"{out}/build.json", "w"), indent=1)
bad = [n for n, c in cfg.items() if any(s != (0, 0) for s in c["ptxas_spill"]) or c["warnings"]]
print("spill or warning in:", bad or "none")
EOF
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "ship:"
