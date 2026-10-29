#!/usr/bin/env bash
# Pearl-C's sm_90a session on one H100 or H200, from a build's ship (cwd = the source tree's root, a --cwd clone run, since
# the pin rewrites sass_pins.json):
#
#   tar -C BUILD_RUN/out/ship -cf /tmp/ship.tar .
#   research run --on vy-dc-h200 --project verity --campaign pouw --source . --cwd clone --timeout 240m \
#     --env GPU_LEASE_WHO=kernels-pc90 --send /tmp/ship.tar -- bash benchmarks/pouw/pearl_c_sm90/h200.sh
#
# Outside any lease: the ship (build.sh's tree) unpacked into this directory and held to its MANIFEST.sha256, nvcc and the
# card recorded.  Under one lease of at most LEASE_MIN minutes (default 60): run.py's check at the fixture's shape with its
# diagnostic timing at GLM-5.3's in-domain linears at M = 64 and 8192 (`SHAPES`), then, only if the check passed,
# glm53_passes.py capture (the gate and the prefill, decode and control passes, into $OUT/passes) and glm53_timing.py: every
# in-domain linear at decode's and prefill's M, prefill's again under run.py's h3 chain (PEARLC_CHAIN=h3, into
# $OUT/timing-h3), and decode's forward captured whole, serial and deferred (--whole, into $OUT/whole).  Then
# (PIN=1, the default) harness/jobs/pin_target.sh pins this nvcc for sm_90a as TK (default 13.0.1-sm_90a) on its own lease, and
# (VERIFY=1, the default) glm53_passes.py verify replays the passes on this machine's CPUs, after every lease is released, and
# (FULL=1, the default) glm53_passes.py full checks prefill's and decode's outputs in full there (every tile's C~ and U by its
# leaf, every real row's served y): about 17 CPU-hours of vy-nebius-cpu-4's (13 min at --jobs 96), so FULL=0 leaves it to a
# run elsewhere on the published passes.
# VERITY_BENCH_VENV names the Python with torch and vLLM (default /workspace/cp/venv312), CUDA the toolkit (default
# /usr/local/cuda-13.0).  Every step's exit code is in $OUT/rc.txt; exits non-zero if any step failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/pouw/pearl_c_sm90
CUDA=${CUDA:-/usr/local/cuda-13.0}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
export PATH=$CUDA/bin:$PATH PYTHONPATH=$PWD:$PWD/tools/config:$PWD/integrations/vllm:$PWD/benchmarks/pouw${PYTHONPATH:+:$PYTHONPATH}
SHAPES=""
for nk in 2624x6144 2048x2048 6144x2048 4096x2048 3072x6144 6144x1536 512x6144 6144x256 4096x6144; do
  SHAPES="$SHAPES 64x$nk 8192x$nk"
done
export O D PY SHAPES
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
: > "$O/rc.txt"
tar -C "$S" -xf "$IN/ship.tar" && (cd "$S" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/pearl_c_sm90.cubin" "$D/" && rm -rf "$D/check" && cp -r "$S/check" "$D/check"
cp "$S/source.txt" "$S/toolchain.txt" "$S/sass.json" "$S/MANIFEST.sha256" "$O/"
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version > "$O/nvcc.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import torch, vllm; print(torch.__version__, vllm.__version__)' > "$O/python.txt" 2>&1
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-60}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  PEARLC_EXPECT_UUID=$GPU_LEASE_UUID SHAPES="$SHAPES" REPS=5 "$PY" "$D/run.py" "$D/check/check.json" > "$O/run.jsonl" 2> "$O/run.err"
  rc=$?; echo "run.py rc $rc" >> "$O/rc.txt"
  [ "$rc" = 0 ] || exit "$rc"
  "$PY" "$D/glm53_passes.py" capture --out "$O/passes" > "$O/capture.log" 2>&1
  echo "capture rc $?" >> "$O/rc.txt"
  "$PY" "$D/glm53_timing.py" --out "$O" > "$O/timing.log" 2>&1
  echo "timing rc $?" >> "$O/rc.txt"
  PEARLC_CHAIN=h3 "$PY" "$D/glm53_timing.py" --out "$O/timing-h3" --m 2048 8192 > "$O/timing-h3.log" 2>&1
  echo "timing h3 rc $?" >> "$O/rc.txt"
  "$PY" "$D/glm53_timing.py" --whole --out "$O/whole" > "$O/whole.log" 2>&1
  echo "whole rc $?" >> "$O/rc.txt"'
echo "lease rc $?" >> "$O/rc.txt"
if [ "${PIN:-1}" = 1 ]; then
  bash benchmarks/pouw/harness/jobs/pin_target.sh "${TK:-13.0.1-sm_90a}" "$CUDA" > "$O/pin.log" 2>&1
  echo "pin rc $?" >> "$O/rc.txt"
fi
if [ "${VERIFY:-1}" = 1 ] && [ -f "$O/passes/control/manifest.json" ]; then
  "$PY" "$D/glm53_passes.py" verify --out "$O/passes" --fast --jobs "$(nproc)" > "$O/verify.log" 2>&1
  echo "verify rc $?" >> "$O/rc.txt"
fi
if [ "${FULL:-1}" = 1 ] && [ -f "$O/passes/decode/manifest.json" ]; then
  "$PY" "$D/glm53_passes.py" full --out "$O/passes" --jobs "$(nproc)" > "$O/full.log" 2>&1
  echo "full rc $?" >> "$O/rc.txt"
fi
cat "$O/rc.txt"
tail -3 "$O/run.err" 2>/dev/null
tail -4 "$O/capture.log" "$O/verify.log" "$O/full.log" "$O/pin.log" 2>/dev/null
grep -E "slowdown|no stock" "$O/timing.log" "$O/timing-h3.log" "$O/whole.log" 2>/dev/null | head -80
! grep -qv " rc 0$" "$O/rc.txt"
