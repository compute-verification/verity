#!/usr/bin/env bash
# attn_bf16's run on one H200, from build.sh's ship (cwd = the source tree's root):
#
#   tar -C SHIP -cf IN/ship.tar .
#   OUT=DIR IN=IN bash benchmarks/torch_gpu/attn_bf16/run.sh
#
# Needs CUDA torch and `gpu-lease` on PATH; IN defaults to a run directory's inputs/ (RESEARCH_RUN_DIR).
# Outside any lease: the ship unpacked into a scratch directory and held to its MANIFEST.sha256, and the card recorded.  Under
# one lease of at most 30 minutes: run.py, the identity check, every case of check/ and the diagnostic timing.  Every step's
# exit code is in $OUT/rc.txt; exits non-zero if any failed.  (The card's canary of sm90.mma.m16n8k16.bf16 is its own
# run, benchmarks/tc_probe's.)
# VERITY_BENCH_VENV names the Python with numpy and CUDA torch (default /workspace/cp/venv312, else /workspace/jobs/venv312);
# ATTN_EXPECT the card's name prefix (default "NVIDIA H200"); GPU_LEASE_WHO the lease's holder (default attn_bf16/ and the
# run's id).
set -uo pipefail
O=${OUT:?}; IN=${IN:-${RESEARCH_RUN_DIR:?set IN}/inputs}
export O ATTN_EXPECT=${ATTN_EXPECT:-NVIDIA H200} GPU_LEASE_WHO=${GPU_LEASE_WHO:-attn_bf16/${RESEARCH_RUN_ID:-local}}
V=${VERITY_BENCH_VENV:-}
if [ -z "$V" ]; then for c in /workspace/cp/venv312 /workspace/jobs/venv312; do [ -x "$c/bin/python" ] && V=$c && break; done; fi
export PY=${V:-/usr}/bin/python3 PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} PATH=/usr/local/cuda/bin:$PATH
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
export S
: > "$O/rc.txt"
tar -C "$S" -xf "$IN/ship.tar" && (cd "$S" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S"/sass.*.json "$S/toolchain.txt" "$S/source.txt" "$S/MANIFEST.sha256" "$S/check/fixture.json" "$O/"
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import sys, numpy; print(sys.executable, numpy.__version__); import torch; print("torch", torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  timeout 15m "$PY" "$S/run.py" "$S/check" > "$O/run.jsonl" 2> "$O/run.err"
  echo "run.py rc $?" >> "$O/rc.txt"'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
cat "$O/run.err" 2>/dev/null
! grep -qv " rc 0$" "$O/rc.txt"
