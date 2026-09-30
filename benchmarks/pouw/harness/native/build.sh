#!/usr/bin/env bash
# Builds libpouw_harness.so for sm_120a (no GPU needed): the harness's device side, cuBLASLt, and the CUTLASS baselines.
#
#   benchmarks/pouw/harness/native/build.sh [OUT_DIR]      (env: CUDA_HOME, CUTLASS_DIR, JOBS, PH_NO_CUTLASS=1)
#
# CUTLASS is header-only, pinned below; CUTLASS_DIR points at a checkout of that tag (the script clones it if absent).
# Writes OUT_DIR/libpouw_harness.so and OUT_DIR/build.json (the commands, versions, and each kernel's SASS MMA counts).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUTLASS_TAG="v4.8.0"
CUTLASS_COMMIT="098de2a652cf8f00fd70b2df54051c7eccbb855a"
CUTLASS_DIR="${CUTLASS_DIR:-$OUT/cutlass}"
NVCC="$CUDA_HOME/bin/nvcc"
JOBS="${JOBS:-3}"  # each CUTLASS 3.x unit takes 3-5 GB in cicc
mkdir -p "$OUT"

if [[ -z "${PH_NO_CUTLASS:-}" && ! -d "$CUTLASS_DIR/include/cutlass" ]]; then
  git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
fi
if [[ -z "${PH_NO_CUTLASS:-}" ]]; then
  got="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"
  [[ "$got" == "$CUTLASS_COMMIT" ]] || { echo "CUTLASS at $got, want $CUTLASS_COMMIT ($CUTLASS_TAG)" >&2; exit 2; }
fi

ARCH=(-gencode arch=compute_120a,code=sm_120a)
FLAGS=(-O3 -std=c++17 "${ARCH[@]}" -Xcompiler -fPIC --expt-relaxed-constexpr -lineinfo)
CUT=(-I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1)

rm -f "$OUT"/*.o "$OUT/libpouw_harness.so"
objs=("$OUT/harness.o")
"$NVCC" "${FLAGS[@]}" -c "$HERE/harness.cu" -o "$OUT/harness.o" &
if [[ -z "${PH_NO_CUTLASS:-}" ]]; then
  units=("cutlass_3x -DPH_PART=0 cutlass_3x_0" "cutlass_3x -DPH_PART=1 cutlass_3x_1" "cutlass_3x -DPH_PART=2 cutlass_3x_2"
         "cutlass_3x -DPH_PART=3 cutlass_3x_3" "cutlass_3x -DPH_PART=4 cutlass_3x_4" "cutlass_2x -DPH_PART=0 cutlass_2x"
         "cutlass_registry -DPH_PART=0 cutlass_registry")
  for u in "${units[@]}"; do
    read -r src def obj <<<"$u"
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
    "$NVCC" "${FLAGS[@]}" "${CUT[@]}" "$def" -c "$HERE/$src.cu" -o "$OUT/$obj.o" &
    objs+=("$OUT/$obj.o")
  done
else
  "$NVCC" "${FLAGS[@]}" -DPH_NO_CUTLASS -c "$HERE/cutlass_registry.cu" -o "$OUT/cutlass_registry.o" &
  objs+=("$OUT/cutlass_registry.o")
fi
wait
for o in "${objs[@]}"; do [[ -f "$o" ]] || { echo "failed to compile $o" >&2; exit 1; }; done
"$NVCC" "${ARCH[@]}" -shared -o "$OUT/libpouw_harness.so" "${objs[@]}" -L"$CUDA_HOME/lib64" -lcublasLt -lcudart

sass() { "$CUDA_HOME/bin/cuobjdump" -sass "$1" | awk '/Function :/{f=$3; next} /\/\*[0-9a-f]+\*\/ +[A-Z]*MMA/{match($0,/[A-Z]*MMA[.A-Z0-9_]*/); c[f"\t"substr($0,RSTART,RLENGTH)]++} END{for(k in c) print k"\t"c[k]}' | sort; }
sass "$OUT/libpouw_harness.so" > "$OUT/sass_mma.tsv"
python3 - "$OUT" "$NVCC" "$CUTLASS_TAG" "$CUTLASS_COMMIT" "${PH_NO_CUTLASS:-}" <<'EOF'
import hashlib, json, subprocess, sys
out, nvcc, tag, commit, no_cutlass = sys.argv[1:6]
lib = f"{out}/libpouw_harness.so"
sass: dict = {}
for line in open(f"{out}/sass_mma.tsv"):
    fn, op, n = line.rstrip("\n").split("\t")
    sass.setdefault(fn, {})[op] = int(n)
json.dump({"lib_sha256": hashlib.sha256(open(lib, "rb").read()).hexdigest(),
           "nvcc": subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1],
           "arch": "sm_120a", "cutlass": None if no_cutlass else {"tag": tag, "commit": commit},
           "sass_mma": sass}, open(f"{out}/build.json", "w"), indent=1)
EOF
echo "built $OUT/libpouw_harness.so"
