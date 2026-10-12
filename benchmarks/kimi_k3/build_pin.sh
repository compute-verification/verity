#!/usr/bin/env bash
# Kimi-K3 prep, step 1: build vLLM at the pin (d9105ea80) from source against torch 2.13.0+cu129 and CUDA 12.9.1 for
# K3_ARCHS (default "9.0a;10.0a;10.3a"), then read what the build holds for Kimi-K3 beside the pinned official cu129 wheel
# (pin_report.py: cubin and PTX archs per extension and per K3 kernel, registered ops, the CuTe DSL check). On
# vy-nebius-cpu-1, through a check slot:
#
#   research run --on vy-nebius-cpu-1 --project verity --campaign kimi-k3 --source . --cwd source --mem-gb 400 \
#     --declared-output 'out/**' -- python3 tools/verity/check/slot.py --by kimi-k3-prep --wait-max 14400 \
#     --run-max 21600 -- bash benchmarks/kimi_k3/build_pin.sh
#
# K3_CONFIGURE_ONLY=1 stops after CMake's configure (minutes, one core), which says what it would build for which archs.
# Outputs, under $RESEARCH_RUN_DIR/out (else ./out):
#   cmake-status.txt       CMake's `Building ...` / `Not building ...` / arch lines
#   build.log.gz           the whole build log
#   wheel.json             the wheel's name, sha256, size and `art:` id (preserved in the store)
#   elf.json, ops.json     the build (pin_report.py elf / ops)
#   official-elf.json, official-ops.json   the pinned official wheel, read the same way
#   cutedsl.json           nvidia-cutlass-dsl[cu13]==4.6.2 under torch cu129 (pin_report.py cutedsl)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${RESEARCH_RUN_DIR:-$PWD}/out"
CACHE="${K3_CACHE:-/workspace/research/cache/kimi-k3-prep}"
COMMIT=d9105ea8001e0a6d77a96327d17515bb5791fb36
ARCHS="${K3_ARCHS:-9.0a;10.0a;10.3a}"
VERSION="0.28.1rc1.dev472+gd9105ea80.k3sm103"
OFFICIAL="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
OFFICIAL_URL="https://wheels.vllm.ai/$COMMIT/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
OFFICIAL_SHA=7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3
mkdir -p "$OUT" "$CACHE/wheels"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/summary.txt"; }

export K3_TOOLKIT="${K3_TOOLKIT:-$CACHE/cuda-12.9.1}"
. "$here/toolkit.sh"
log "CUDA_HOME=$CUDA_HOME ($(nvcc --version | tail -1)); archs $ARCHS; $(nproc) cpus"

scratch="$(mktemp -d -p "$CACHE" build.XXXXXX)"
trap 'rm -rf "$scratch"' EXIT
export UV_HTTP_TIMEOUT=600
uv venv -q --python-preference only-managed -p 3.12 "$scratch/venv"
PY="$scratch/venv/bin/python"
export PATH="$scratch/venv/bin:$PATH"

git init -q "$scratch/vllm"
git -C "$scratch/vllm" fetch -q --depth 1 https://github.com/vllm-project/vllm.git "$COMMIT"
git -C "$scratch/vllm" checkout -q FETCH_HEAD
log "vLLM $(git -C "$scratch/vllm" rev-parse HEAD)"

uv pip install -q --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
  "torch==2.13.0+cu129" -r "$scratch/vllm/requirements/build/cuda.txt" numpy
log "$("$PY" "$here/torch_version.py")"

export TORCH_CUDA_ARCH_LIST="$ARCHS" VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release
export MAX_JOBS="${K3_JOBS:-$(nproc)}" NVCC_THREADS="${K3_NVCC_THREADS:-1}"
export SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" FETCHCONTENT_BASE_DIR="$scratch/deps"
cd "$scratch/vllm"

if [[ "${K3_CONFIGURE_ONLY:-0}" == 1 ]]; then
  mkdir -p "$scratch/cmake"
  cmake -S . -B "$scratch/cmake" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$PY" -DFETCHCONTENT_BASE_DIR="$scratch/deps" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    > "$scratch/build.log" 2>&1 || { tail -40 "$scratch/build.log"; gzip -c "$scratch/build.log" > "$OUT/build.log.gz"; exit 1; }
else
  log "building with MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"
  "$PY" -m pip wheel --no-build-isolation --no-deps -v . -w "$scratch/wheel" > "$scratch/build.log" 2>&1 \
    || { tail -60 "$scratch/build.log"; gzip -c "$scratch/build.log" > "$OUT/build.log.gz"; exit 1; }
fi
grep -E -- '-- (Building|Not building|FlashKDA|CUDA target arch|CUDA supported arch|Marlin|.*archs?:|.*will not compile)' \
  "$scratch/build.log" | sort -u > "$OUT/cmake-status.txt" || true
gzip -c "$scratch/build.log" > "$OUT/build.log.gz"
log "cmake status: $(wc -l < "$OUT/cmake-status.txt") lines"
[[ "${K3_CONFIGURE_ONLY:-0}" == 1 ]] && { log "configure only: done"; exit 0; }

whl="$(ls "$scratch"/wheel/vllm-*.whl)"
cp "$whl" "$CACHE/wheels/"
art=""
if command -v research >/dev/null; then
  art="$(research data put --kind evidence/v1 --file "$whl" --preserve 2>&1 | grep -o 'art:[0-9a-f]\{64\}' | head -1 || true)"
fi
"$PY" "$here/wheel_json.py" "$whl" "$ARCHS" "$art" > "$OUT/wheel.json"
log "wheel $(basename "$whl") ${art:-not preserved}"

uv pip install -q --python "$PY" --no-deps "$whl"
site="$("$PY" "$here/torch_version.py" --site)"
"$PY" "$here/pin_report.py" elf "$site/vllm" --out "$OUT/elf.json" | tee -a "$OUT/summary.txt"
"$PY" "$here/pin_report.py" ops "$site/vllm" --out "$OUT/ops.json" | tee -a "$OUT/summary.txt"

[[ -f "$CACHE/wheels/$OFFICIAL" ]] || curl -fsSL --retry 4 -o "$CACHE/wheels/$OFFICIAL" "$OFFICIAL_URL"
echo "$OFFICIAL_SHA  $CACHE/wheels/$OFFICIAL" | sha256sum -c --quiet
mkdir -p "$scratch/official"
(cd "$scratch/official" && "$PY" -m zipfile -e "$CACHE/wheels/$OFFICIAL" .)
log "official wheel:"
"$PY" "$here/pin_report.py" elf "$scratch/official/vllm" --out "$OUT/official-elf.json" | tee -a "$OUT/summary.txt"

uv pip install -q --python "$PY" -r "$scratch/vllm/requirements/cuda.txt" --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match "torch==2.13.0+cu129" > "$scratch/deps.log" 2>&1 \
  || { log "requirements/cuda.txt did not install: $(tail -3 "$scratch/deps.log")"; }
uv pip install -q --python "$PY" --no-deps "$whl"
log "$("$PY" "$here/torch_version.py")"
"$PY" "$here/pin_report.py" cutedsl --out "$OUT/cutedsl.json" | tee -a "$OUT/summary.txt"
log "done"
