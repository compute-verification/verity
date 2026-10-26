#!/usr/bin/env bash
# attn_ref's session on one H200, from build.sh's ship (cwd = the source tree's root, a --cwd source run):
#
#   tar -C BUILD_RUN/out/ship -cf /tmp/attn-ref-ship.tar .
#   research run --on vy-dc-h200 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --send /tmp/attn-ref-ship.tar -- bash benchmarks/torch_gpu/attn_ref/session.sh
#
# Outside the lease: the ship unpacked into a scratch directory and held to its MANIFEST.sha256, the gate's record and the
# fixture's summary copied to $OUT, the card and the Python recorded.  Under one lease of at most 30 minutes: check.py (the
# check, then the diagnostic timing) on the shipped cubin, its JSON lines in $OUT/check.jsonl.  Every step's exit code is in
# $OUT/rc.txt; exits non-zero if any failed.
# VERITY_BENCH_VENV names the Python with numpy and CUDA torch (default /workspace/cp/venv312, else /workspace/jobs/venv312);
# GPU_LEASE_WHO the lease's holder (default kernels-kattnref); LEASE_MIN the lease's bound (default 30).
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
export O GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels-kattnref}
V=${VERITY_BENCH_VENV:-}
if [ -z "$V" ]; then for c in /workspace/cp/venv312 /workspace/jobs/venv312; do [ -x "$c/bin/python" ] && V=$c && break; done; fi
export PY=${V:-/usr}/bin/python3
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
export S
: > "$O/rc.txt"
SHIP=$(ls "$IN"/*.tar | head -1)
tar -C "$S" -xf "$SHIP" && (cd "$S" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/sass.json" "$S/toolchain.txt" "$S/source.txt" "$S/MANIFEST.sha256" "$S/check/fixture.json" "$O/"
git rev-parse HEAD > "$O/commit.txt" 2>&1
"$PY" -c 'import sys, numpy, torch; print(sys.executable, "numpy", numpy.__version__, "torch", torch.__version__, torch.version.cuda)' \
  > "$O/python.txt" 2>&1
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  timeout 25m "$PY" "$S/check.py" "$S/check" > "$O/check.jsonl" 2> "$O/check.err"
  echo "check rc $?" >> "$O/rc.txt"'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
cat "$O/check.err" 2>/dev/null
! grep -qv " rc 0$" "$O/rc.txt"
