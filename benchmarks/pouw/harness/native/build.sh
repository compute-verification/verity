#!/usr/bin/env bash
# Builds libpouw_harness.so for sm_120a (no GPU needed): the harness's device side, cuBLASLt, the CUTLASS baselines, and the
# plain NVFP4 and FP8 GEMMs of ../../nvfp4_sm120/nvf4_plain.cu (no CUTLASS).
#
#   export POUW_HARNESS_LIB="$(benchmarks/pouw/harness/native/build.sh [OUT_DIR])"
#       env: POUW_HARNESS_BUILD (the build cache, default native/build), CUDA_HOME, CUTLASS_DIR, JOBS, PH_NO_CUTLASS=1,
#            LT2_HOME (a second cuBLASLt's root, include/ and lib64/: also builds libpouw_lt2.so, lt.cu alone against it)
#
# The output directory is keyed by a hash of these sources, nvcc's version and the cuBLAS headers' (OUT_DIR overrides it); a
# build that exists there is reused, so a timed run never compiles. Each object is cached under objects/ by the hash of its own
# unit, the headers, its flags and those versions, so editing one unit recompiles only that one. CUTLASS is header-only, pinned
# below (cloned into the cache if absent). Without a CUDA 12.9 toolkit, source jobs/toolkit.sh first.
# Writes libpouw_harness.so, libpouw_lt2.so with LT2_HOME, and build.json (the key, versions, and each kernel's SASS MMA counts
# and registers, stack, shared and spill bytes); prints only the library's path on stdout.
set -euo pipefail
exec 3>&1 1>&2
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
CACHE="${POUW_HARNESS_BUILD:-$HERE/build}"
CUTLASS_TAG="v4.8.0"
CUTLASS_COMMIT="098de2a652cf8f00fd70b2df54051c7eccbb855a"
CUTLASS_DIR="${CUTLASS_DIR:-$CACHE/cutlass-$CUTLASS_TAG}"
JOBS="${JOBS:-3}"  # a CUTLASS 3.x unit takes up to 7 GB in cicc
NVCC_VERSION="$("$NVCC" --version | tail -1)"
cublas_version() { awk '/^#define CUBLAS_VER_(MAJOR|MINOR|PATCH|BUILD) /{printf "%s%s", s, $3; s="."}' "$1/include/cublas_api.h"; }
CUBLAS_VERSION="$(cublas_version "$CUDA_HOME")"
LT2_VERSION="$([[ -n "${LT2_HOME:-}" ]] && cublas_version "$LT2_HOME" || true)"
PLAIN=../../nvfp4_sm120/nvf4_plain  # a unit's path from here, without .cu; mainloop_nvf4.cuh beside it
[[ -f "$HERE/$PLAIN.cu" && -f "$HERE/${PLAIN%/*}/mainloop_nvf4.cuh" ]] || { echo "no $HERE/$PLAIN.cu or its header" >&2; exit 2; }
KEY="$(cat "$HERE/build.sh" "$HERE"/*.h "$HERE"/*.cu "$HERE/$PLAIN.cu" "$HERE/${PLAIN%/*}"/*.cuh <("$NVCC" --version) \
  <(echo "${PH_NO_CUTLASS:-}" "$CUBLAS_VERSION" "${LT2_HOME:-}" "$LT2_VERSION") | sha256sum | cut -c1-16)"
OUT="${1:-$CACHE/$KEY}"
if [[ -f "$OUT/libpouw_harness.so" && -f "$OUT/build.json" ]] && grep -q "\"key\": \"$KEY\"" "$OUT/build.json"; then
  echo "$OUT/libpouw_harness.so" >&3
  exit 0
fi
mkdir -p "$OUT"

if [[ -z "${PH_NO_CUTLASS:-}" && ! -d "$CUTLASS_DIR/include/cutlass" ]]; then
  git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
fi
if [[ -z "${PH_NO_CUTLASS:-}" ]]; then
  got="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"
  [[ "$got" == "$CUTLASS_COMMIT" ]] || { echo "CUTLASS at $got, want $CUTLASS_COMMIT ($CUTLASS_TAG)" >&2; exit 2; }
fi

ARCH=(-gencode arch=compute_120a,code=sm_120a)
FLAGS=(-O3 -std=c++17 "${ARCH[@]}" -Xcompiler -fPIC --expt-relaxed-constexpr)  # ptxas gets -arch sm_120a -m 64, as pinned
CUT=(-I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1)

rm -f "$OUT"/*.o "$OUT/libpouw_harness.so" "$OUT/libpouw_lt2.so" "$OUT/build.json"
mkdir -p "$CACHE/objects"
compile() {  # SRC DEF OBJ CUTLASS(0|1): SRC is the unit's path from here without .cu; reuses objects/OBJ-<hash> when the
  #           unit, the headers (these and the .cuh beside the unit), flags and versions match
  local src="$HERE/$1.cu" def="$2" obj="$3" cut=() k o dir
  dir="$(dirname "$src")"
  [[ "$4" == 1 ]] && cut=("${CUT[@]}")
  k="$(cat "$src" "$HERE"/*.h $(find "$dir" -maxdepth 1 -name '*.cuh' | sort) \
    <(printf '%s\n' "$def" "${FLAGS[@]}" "$4" "$CUTLASS_COMMIT" "$NVCC_VERSION" "$CUBLAS_VERSION") | sha256sum | cut -c1-16)"
  o="$CACHE/objects/$obj-$k.o"
  if [[ ! -f "$o" ]]; then
    "$NVCC" "${FLAGS[@]}" -I"$HERE" -I"$dir" "${cut[@]}" ${def:+"$def"} -c "$src" -o "$o.$$.tmp" && mv "$o.$$.tmp" "$o" \
      || return 1
  fi
  cp "$o" "$OUT/$obj.o"
}
units=("harness - harness 0" "lt - lt 0" "$PLAIN - nvf4_plain 0")
if [[ -z "${PH_NO_CUTLASS:-}" ]]; then
  for p in 0 1 2 3 4 5 6; do units+=("cutlass_3x -DPH_PART=$p cutlass_3x_$p 1"); done
  units+=("cutlass_2x -DPH_PART=0 cutlass_2x 1" "cutlass_registry -DPH_PART=0 cutlass_registry 1")
else
  units+=("cutlass_registry -DPH_NO_CUTLASS cutlass_registry 0")
fi
objs=()
for u in "${units[@]}"; do
  read -r src def obj cut <<<"$u"
  [[ "$def" == - ]] && def=""
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
  compile "$src" "$def" "$obj" "$cut" &
  objs+=("$OUT/$obj.o")
done
wait
for o in "${objs[@]}"; do [[ -f "$o" ]] || { echo "failed to compile $o" >&2; exit 1; }; done
"$NVCC" "${ARCH[@]}" -shared -o "$OUT/libpouw_harness.so" "${objs[@]}" -L"$CUDA_HOME/lib64" -lcublasLt -lcudart
if [[ -n "${LT2_HOME:-}" ]]; then  # host code only, so the host compiler builds it against the second cuBLASLt's headers
  g++ -O2 -std=c++17 -fPIC -shared -DPH_LT_ONLY -x c++ "$HERE/lt.cu" -I"$LT2_HOME/include" -L"$LT2_HOME/lib64" -lcublasLt \
    -Wl,-Bsymbolic -Wl,-rpath,"$LT2_HOME/lib64" -o "$OUT/libpouw_lt2.so"
fi

sass() { "$CUDA_HOME/bin/cuobjdump" -sass "$1" | awk '/Function :/{f=$3; next} /\/\*[0-9a-f]+\*\/ +[A-Z]*MMA/{match($0,/[A-Z]*MMA[.A-Z0-9_]*/); c[f"\t"substr($0,RSTART,RLENGTH)]++} END{for(k in c) print k"\t"c[k]}' | sort; }
sass "$OUT/libpouw_harness.so" > "$OUT/sass_mma.tsv"
"$CUDA_HOME/bin/cuobjdump" -res-usage "$OUT/libpouw_harness.so" > "$OUT/res_usage.txt"
python3 - "$OUT" "$NVCC" "$CUTLASS_TAG" "$CUTLASS_COMMIT" "${PH_NO_CUTLASS:-}" "$KEY" "$CUBLAS_VERSION" "${LT2_HOME:-}" \
  "$LT2_VERSION" <<'EOF'
import hashlib, json, os, re, subprocess, sys
out, nvcc, tag, commit, no_cutlass, key, cublas, lt2_home, lt2_version = sys.argv[1:10]
lib = f"{out}/libpouw_harness.so"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
sass: dict = {}
for line in open(f"{out}/sass_mma.tsv"):
    fn, op, n = line.rstrip("\n").split("\t")
    sass.setdefault(fn, {})[op] = int(n)
res, fn = {}, None  # each kernel's registers, stack, shared and local (spill) bytes
for line in open(f"{out}/res_usage.txt"):
    if m := re.match(r"\s*Function (\S+):", line):
        fn = m.group(1)
    elif fn and (m := re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)):
        res[fn] = dict(zip(("reg", "stack", "shared", "local"), map(int, m.groups())))
lt2 = None
if lt2_home:
    dep = os.path.realpath(f"{lt2_home}/lib64/libcublasLt.so")
    lt2 = {"lib_sha256": sha(f"{out}/libpouw_lt2.so"), "cublas_headers": lt2_version, "home": lt2_home, "cublaslt": dep,
           "cublaslt_sha256": sha(dep)}
json.dump({"key": key, "lib_sha256": sha(lib),
           "nvcc": subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1],
           "arch": "sm_120a", "cutlass": None if no_cutlass else {"tag": tag, "commit": commit}, "cublas_headers": cublas,
           "lt2": lt2, "sass_mma": sass, "resources": res}, open(f"{out}/build.json", "w"), indent=1)
EOF
echo "$OUT/libpouw_harness.so" >&3
