#!/usr/bin/env bash
# The H200's write census: every SASS form a corpus of sm_90 and sm_90a code emits, with what each instruction writes
# (GPRs, predicates, uniform registers) read off `nvdisasm -plr -lrm narrow`'s register life ranges by `sass_count.py`.
# CPU only and self-contained: CUDA 13.0's nvcc and PyPI's cuobjdump and nvdisasm in a scratch venv, and the corpus
# from pinned wheels (WHEELS: the workspace's torch for CUDA 13.0 with every NVIDIA library it pulls, vLLM, sgl-kernel,
# flashinfer's cubins), Verity's own sm_90a probes and Pearl-C, and CUTLASS's Hopper examples. Writes under $OUT:
#   toolkit.txt     nvcc, cuobjdump and nvdisasm --version, and the wheels downloaded
#   corpus.tsv      a cubin a line: sha256, bytes, architecture, the file it came from
#   counts/*.json   `sass_count.py count` of each cubin's life-range listing; failed.txt the cubins nvdisasm refused
#   census.json     `sass_count.py merge` of them all; summary.txt its totals; writes.tsv `sass_count.py writes`
#   build.log       the Verity and CUTLASS builds (a source that fails is named and skipped)
#
#   research run --on auto --cwd source --mem-gb 96 -- bash benchmarks/pouw/sass_census/life.sh
# ARCHS (default "sm_90 sm_90a") the cubins kept; CUTLASS=0 skips CUTLASS (CUTLASS_JOBS builds at once, default 6).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
ARCHS=${ARCHS:-sm_90 sm_90a}
JOBS=${JOBS:-$(nproc)}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/counts" "$WORK/wheels" "$WORK/corpus" "$WORK/cubins" "$WORK/bin"
WHEELS=${WHEELS:-"torch==2.14.0 vllm sgl-kernel flashinfer-cubin"}

uv venv -q --seed --python 3.12 "$WORK/tools" || exit 2
PY=$WORK/tools/bin/python
uv pip install -q --python "$PY" nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 \
  nvidia-cuda-runtime==13.0.96 nvidia-cuda-cccl==13.0.85 || exit 2
uv pip install -q --python "$PY" --no-deps nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm || exit 2
CU=$("$PY" -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13"))')
export PATH="$CU/bin:$PATH" CUDA_HOME=$CU LIBRARY_PATH=$CU/lib NVCC=$CU/bin/nvcc POUW_CUDA_BIN=$CU/bin
{ nvcc --version; cuobjdump --version; nvdisasm --version; } >"$OUT/toolkit.txt" 2>&1

# torch's CUDA 13.0 wheel with its dependencies (every NVIDIA library it loads); the rest without theirs
read -r -a specs <<<"$WHEELS"
for s in "${specs[@]}"; do
  case "$s" in
  torch*) "$PY" -m pip download -q -d "$WORK/wheels" --only-binary :all: "$s" \
            --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple ;;
  *) "$PY" -m pip download -q -d "$WORK/wheels" --only-binary :all: --no-deps "$s" ;;
  esac >>"$OUT/build.log" 2>&1 || echo "life.sh: no wheel for $s" | tee -a "$OUT/build.log" >&2
done
ls "$WORK/wheels" | sed 's/^/wheel /' >>"$OUT/toolkit.txt"
for w in "$WORK"/wheels/*.whl; do
  unzip -q -o "$w" '*.so' '*.so.*' '*.cubin' '*.fatbin' -d "$WORK/corpus/$(basename "$w" .whl)" 2>/dev/null
done

# Verity's sm_90a code: the dc probes and Pearl-C
export ROOT WORK
ls "$ROOT"/benchmarks/pouw/dc/probes/sm90_*.cu | xargs -P "$JOBS" -I{} bash -c '
  python3 "$ROOT/benchmarks/pouw/opcode_probe/probe.py" build --arch sm_90a --out "$WORK/bin" {} >/dev/null 2>&1 \
    || echo "life.sh: build failed: {}"' >>"$OUT/build.log" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -cubin -o "$WORK/bin/pearl_c_sm90.cubin" \
  "$ROOT/benchmarks/pouw/pearl_c_sm90/pearl_c_sm90.cu" >>"$OUT/build.log" 2>&1 || echo "life.sh: build failed: pearl_c_sm90" >&2

# CUTLASS's Hopper examples at the harness's pin, device code only
if [ "${CUTLASS:-1}" != 0 ]; then
  tag=$(sed -n 's/^CUTLASS_TAG="\(.*\)"$/\1/p' "$ROOT/benchmarks/pouw/harness/native/build.sh")
  pin=$(sed -n 's/^CUTLASS_COMMIT="\(.*\)"$/\1/p' "$ROOT/benchmarks/pouw/harness/native/build.sh")
  C=$WORK/cutlass
  if git clone -q --depth 1 --branch "$tag" https://github.com/NVIDIA/cutlass "$C" >>"$OUT/build.log" 2>&1 &&
    [ "$(git -C "$C" rev-parse HEAD)" = "$pin" ]; then
    echo "cutlass $tag $pin" >>"$OUT/toolkit.txt"
    export C
    find "$C/examples" -path '*hopper*' -name '*.cu' | sort | xargs -P "${CUTLASS_JOBS:-6}" -I{} bash -c '
      f={}; n=$(basename "$(dirname "$f")")-$(basename "$f" .cu)
      timeout 1200 nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -cubin \
        -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -I"$C/include" -I"$C/tools/util/include" -I"$C/examples/common" \
        -I"$(dirname "$f")" -o "$WORK/bin/cutlass-$n.cubin" "$f" >/dev/null 2>&1 || echo "life.sh: cutlass failed: $n"
    ' >>"$OUT/build.log" 2>&1
  else
    echo "life.sh: no CUTLASS $tag at $pin" | tee -a "$OUT/build.log" >&2
  fi
fi

# every cubin of the kept architectures, one copy each: fat binaries extracted, plain cubins kept by their .target
find "$WORK/corpus" "$WORK/bin" -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.cubin' -o -name '*.fatbin' \
  -o -perm -u+x \) | sort >"$WORK/files"
n=0
while read -r f; do
  n=$((n + 1))
  d=$WORK/x/$n
  mkdir -p "$d"
  case "$f" in
  *.cubin)
    a=$(nvdisasm "$f" 2>/dev/null | grep -m1 -o '\.target\s*sm_[0-9a-z]*' | grep -o 'sm_[0-9a-z]*')
    case " $ARCHS " in *" $a "*) cp "$f" "$d/$(basename "$f" .cubin).$a.cubin" ;; esac ;;
  *) for a in $ARCHS; do (cd "$d" && cuobjdump -arch "$a" -xelf all "$f" >/dev/null 2>&1); done ;;
  esac
  for c in "$d"/*.cubin; do
    [ -f "$c" ] || continue
    sha=$(sha256sum "$c" | cut -c1-16)
    a=$(basename "$c" | grep -o 'sm_[0-9]*a\?' | tail -1)
    [ -f "$WORK/cubins/$sha.cubin" ] || { mv "$c" "$WORK/cubins/$sha.cubin"; printf '%s\t%s\t%s\t%s\n' "$sha" \
      "$(stat -c %s "$WORK/cubins/$sha.cubin")" "$a" "${f#$WORK/}"; }
  done
  rm -rf "$d"
done <"$WORK/files" >"$OUT/corpus.tsv"
wc -l <"$OUT/corpus.tsv" | xargs echo "life.sh: cubins"

export HERE OUT WORK
: >"$OUT/failed.txt"
sort -t$'\t' -k2,2nr "$OUT/corpus.tsv" | tr '\n' '\0' | xargs -0 -P "$JOBS" -I{} bash -c '
  set -o pipefail
  IFS=$'"'"'\t'"'"' read -r sha bytes arch src <<<"$1"
  lib=$(basename "$src")
  if ! nvdisasm -plr -lrm narrow "$WORK/cubins/$sha.cubin" 2>/dev/null \
      | python3 "$HERE/sass_count.py" count --source "$lib@$sha" >"$OUT/counts/$sha.json"; then
    echo "$sha $src" >>"$OUT/failed.txt"
    cuobjdump -sass "$WORK/cubins/$sha.cubin" | python3 "$HERE/sass_count.py" count --source "$lib@$sha" --arch "$arch" \
      >"$OUT/counts/$sha.json"
  fi' _ {}

python3 "$HERE/sass_count.py" merge "$OUT"/counts/*.json >"$OUT/census.json"
python3 "$HERE/sass_count.py" summary "$OUT/census.json" | tee "$OUT/summary.txt"
python3 "$HERE/sass_count.py" writes "$OUT/census.json" --family sm_90 >"$OUT/writes.tsv"
wc -l <"$OUT/failed.txt" | xargs echo "life.sh: nvdisasm refused"
