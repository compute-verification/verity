#!/usr/bin/env bash
# attn_bf16's session on one card, from build.sh's ship (cwd = the source tree's root, a --cwd source run):
#
#   tar -C BUILD_RUN/out/ship -cf /tmp/ship.tar .
#   research run --on vy-dc-h200 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE --send /tmp/ship.tar \
#     -- bash benchmarks/torch_gpu/attn_bf16/session.sh
#
# Outside any lease: the ship unpacked into a scratch directory and held to its MANIFEST.sha256, and the card recorded.  Under
# one lease of at most 30 minutes: tc_probe's canary of CANARY (default sm90.mma.m16n8k16.bf16) from the shipped library and
# its replay of the H100's BF16 specials capture, then run.py: the identity check, every case of check/ and the diagnostic
# timing.  Every step's exit code is in $OUT/rc.txt; exits non-zero if any failed.
# VERITY_BENCH_VENV names the Python with numpy and CUDA torch (default /workspace/cp/venv312, else /workspace/jobs/venv312);
# ATTN_EXPECT the card's name prefix (default "NVIDIA H200"); GPU_LEASE_WHO the lease's holder (default kernels).
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
export O CANARY=${CANARY:-sm90.mma.m16n8k16.bf16} ATTN_EXPECT=${ATTN_EXPECT:-NVIDIA H200} GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels}
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
  LIB=$(ls "$S"/tc_probe/lib*.so | head -1)
  timeout 5m "$PY" benchmarks/tc_probe/tc_probe.py --instruction "$CANARY" --library "$LIB" --canary --out "$O/canary" > "$O/canary.log" 2>&1
  echo "canary rc $?" >> "$O/rc.txt"
  timeout 8m "$PY" benchmarks/tc_probe/tc_probe.py --instruction "$CANARY" --library "$LIB" --procs 8 \
    --replay-capture "$S/tc_probe/special.reduced.json.gz" --out "$O/capture" > "$O/capture.log" 2>&1
  echo "capture rc $?" >> "$O/rc.txt"
  timeout 15m "$PY" "$S/run.py" "$S/check" > "$O/run.jsonl" 2> "$O/run.err"
  echo "run.py rc $?" >> "$O/rc.txt"'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -4 "$O/canary.log" "$O/capture.log" 2>/dev/null
cat "$O/run.err" 2>/dev/null
! grep -qv " rc 0$" "$O/rc.txt"
