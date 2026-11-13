#!/usr/bin/env bash
# Pearl-C's sm_90a session on one H100 or H200, from a build's ship (cwd = the source tree's root, a --cwd clone run, since
# the pin rewrites sass_pins.json):
#
#   tar -C BUILD_RUN/out/ship -cf /tmp/ship.tar .
#   research run --on MACHINE --project verity --source . --cwd clone --timeout 240m \
#     --env GPU_LEASE_WHO=LANE --send /tmp/ship.tar -- bash benchmarks/pouw/pearl_c_sm90/h200.sh
#
# Outside any lease: the ship (build.sh's tree) unpacked into this directory and held to its MANIFEST.sha256, nvcc and the
# card recorded.  MODEL (default glm53, GLM-5.3's one TP8 rank; llama32_1b, Llama-3.2-1B's on one card) names whose
# in-domain linears every step below runs at (this directory's <MODEL>.py; glm53_passes.py and glm53_timing.py --model).
# Under one lease of at most LEASE_MIN minutes (default 60): run.py's check at the fixture's shape with its
# diagnostic timing at MODEL's in-domain linears at M = 64 and 8192 (`SHAPES`), then, only if the check passed,
# glm53_passes.py capture (the gate and the prefill, decode and control passes, into $OUT/passes) and glm53_timing.py: every
# in-domain linear at decode's and prefill's M under the serial schedule, its decode calls graphed and eager, and under
# deferred-a, prefill's again under run.py's h3 chain (PEARLC_CHAIN=h3, into $OUT/timing-h3), and decode's forward captured
# whole, serial, deferred and as served, over LAYERS layers' linears (--whole --layers, into $OUT/whole; default MODEL's
# layers for llama32_1b, 16, and one for glm53, whose in-domain linears aren't one layer's); TIMING=0 ends the lease after the
# capture, timing nothing.  Then
# (PIN=1, the default) harness/jobs/pin_target.sh pins this nvcc for sm_90a as TK (default 13.0.1-sm_90a) on its own lease,
# (VOLUNTEER=1) glm53_passes.py volunteer gives up prefill's and decode's rows over a debit cap in their records on this
# machine's CPUs, after every lease is released and before any verifier coin (VOLUNTEER=device: on a leased card,
# debit.py's check of the ship's debit/debit_sm90.cubin against fast_tile, then only if it passed the words' counts from it,
# every DEVICE_CHECK-th tile, default every one, replayed on the CPUs too and held equal), by default exactly while the
# verdict glm53_passes.py verify takes rejects a pass on a drawn row over its cap (`served_passes.py volunteer-needed
# --verdict pass`, asked before anything runs: VOLUNTEER defaults to 1 then, and any other than 1 or device is refused,
# exit 2; otherwise to 0, none), and
# (VERIFY=1, the default) glm53_passes.py verify replays DRAWS tiles of each pass there (default 459, the verdict's
# `served_passes.DRAWS`), and
# (FULL=1, the default) glm53_passes.py full checks prefill's and decode's outputs in full there (every tile's C~ and U by its
# leaf, every real row's served y): about 17 CPU-hours for glm53 (13 min on a CPU pod at --jobs 96), so FULL=0 leaves it
# to a run elsewhere on the published passes.
# With LOCK_MHZ the timing steps run with the leased card's SM clock locked there (`sudo -n nvidia-smi -lgc`, reset with
# -rgc when the lease ends), where the node allows it: the lock's outcome goes to $OUT/clock.txt, not rc.txt, so a node
# that refuses it times unlocked; either way $OUT/smi.csv samples the card's clocks every 200 ms through the timing.
# VERITY_BENCH_VENV names the Python with torch and vLLM (default /workspace/cp/venv312), CUDA the toolkit (default
# /usr/local/cuda-13.0).  Every step's exit code is in $OUT/rc.txt; exits non-zero if any step failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/pouw/pearl_c_sm90
CUDA=${CUDA:-/usr/local/cuda-13.0}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
export PATH=$CUDA/bin:$PATH PYTHONPATH=$PWD:$PWD/verity/configuration/config:$PWD/adapters/vllm:$PWD/benchmarks/pouw${PYTHONPATH:+:$PYTHONPATH}
MODEL=${MODEL:-glm53}
case $MODEL in
  glm53) NKS="2624x6144 2048x2048 6144x2048 4096x2048 3072x6144 6144x1536 512x6144 6144x256 4096x6144"; LAYERS=${LAYERS:-1} ;;
  llama32_1b) NKS="3072x2048 2048x2048 16384x2048 2048x8192"; LAYERS=${LAYERS:-16} ;;
  *) echo "MODEL is glm53 or llama32_1b, not $MODEL" >&2; exit 2 ;;
esac
SHAPES=""
for nk in $NKS; do
  SHAPES="$SHAPES 64x$nk 8192x$nk"
done
export O D PY SHAPES MODEL LAYERS LOCK_MHZ=${LOCK_MHZ:-} TIMING=${TIMING:-1}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
: > "$O/rc.txt"
NEEDED=$("$PY" benchmarks/pouw/pearl_c_vllm/served_passes.py volunteer-needed --verdict pass 2> "$O/volunteer-needed.log")
case ${NEEDED%% *} in
  yes) VOLUNTEER=${VOLUNTEER:-1} ;;
  no) VOLUNTEER=${VOLUNTEER:-0} ;;
  *) echo "volunteer-needed rc 2: '$NEEDED': $(tail -1 "$O/volunteer-needed.log")" | tee -a "$O/rc.txt"; exit 2 ;;
esac
echo "volunteer-needed: $NEEDED; VOLUNTEER=$VOLUNTEER" > "$O/volunteer-needed.txt"
if [ "${NEEDED%% *}" = yes ] && [ "$VOLUNTEER" != 1 ] && [ "$VOLUNTEER" != device ]; then
  echo "volunteer refused rc 2: the verdict rejects a pass on a drawn row over its debit cap (over-cap rule" \
    "${NEEDED#* }), so the give-up before its coin can't be skipped (VOLUNTEER=$VOLUNTEER)" | tee -a "$O/rc.txt"
  exit 2
fi
tar -C "$S" -xf "$IN/ship.tar" && (cd "$S" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/pearl_c_sm90.cubin" "$D/" && rm -rf "$D/check" && cp -r "$S/check" "$D/check"
export DEBIT=$S/debit/debit_sm90.cubin
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
  "$PY" "$D/glm53_passes.py" capture --model "$MODEL" --out "$O/passes" > "$O/capture.log" 2>&1
  echo "capture rc $?" >> "$O/rc.txt"
  [ "$TIMING" = 1 ] || exit 0
  U=$GPU_LEASE_UUID
  if [ -n "$LOCK_MHZ" ]; then
    trap "sudo -n nvidia-smi -i $U -rgc >> $O/lock.txt 2>&1" EXIT
    sudo -n nvidia-smi -i "$U" -lgc "$LOCK_MHZ,$LOCK_MHZ" > "$O/lock.txt" 2>&1
    echo "lock $LOCK_MHZ MHz: nvidia-smi -lgc exit $?" > "$O/clock.txt"
  else
    echo "unlocked" > "$O/clock.txt"
  fi
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader \
    -lms 200 -i "$U" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$PY" "$D/glm53_timing.py" --model "$MODEL" --out "$O" --graphs --schedule serial deferred-a > "$O/timing.log" 2>&1
  echo "timing rc $?" >> "$O/rc.txt"
  PEARLC_CHAIN=h3 "$PY" "$D/glm53_timing.py" --model "$MODEL" --out "$O/timing-h3" --m 2048 8192 > "$O/timing-h3.log" 2>&1
  echo "timing h3 rc $?" >> "$O/rc.txt"
  "$PY" "$D/glm53_timing.py" --model "$MODEL" --whole --layers "$LAYERS" --out "$O/whole" > "$O/whole.log" 2>&1
  echo "whole rc $?" >> "$O/rc.txt"
  kill "$smi"'
echo "lease rc $?" >> "$O/rc.txt"
if [ "${PIN:-1}" = 1 ]; then
  bash benchmarks/pouw/harness/jobs/pin_target.sh "${TK:-13.0.1-sm_90a}" "$CUDA" > "$O/pin.log" 2>&1
  echo "pin rc $?" >> "$O/rc.txt"
fi
if [ "$VOLUNTEER" = 1 ] && [ -f "$O/passes/decode/manifest.json" ]; then
  "$PY" "$D/glm53_passes.py" volunteer --out "$O/passes" --jobs "$(nproc)" > "$O/volunteer.log" 2>&1
  echo "volunteer rc $?" >> "$O/rc.txt"
elif [ "$VOLUNTEER" = device ] && [ -f "$O/passes/decode/manifest.json" ]; then
  # shellcheck disable=SC2016
  gpu-lease 1 --wait --max-min "${LEASE_MIN:-60}" -- bash -c '
    "$PY" "$D/debit.py" --cubin "$DEBIT" --out "$O/debit.json" > "$O/debit.log" 2>&1
    rc=$?; echo "debit rc $rc" >> "$O/rc.txt"
    [ "$rc" = 0 ] || exit "$rc"
    "$PY" "$D/glm53_passes.py" volunteer --out "$O/passes" --jobs "$(nproc)" --device "$DEBIT" \
      --device-check "${DEVICE_CHECK:-1}" > "$O/volunteer.log" 2>&1'
  echo "volunteer rc $?" >> "$O/rc.txt"
fi
if [ "${VERIFY:-1}" = 1 ] && [ -f "$O/passes/control/manifest.json" ]; then
  "$PY" "$D/glm53_passes.py" verify --out "$O/passes" --fast --draws "${DRAWS:-459}" --jobs "$(nproc)" > "$O/verify.log" 2>&1
  echo "verify rc $?" >> "$O/rc.txt"
fi
if [ "${FULL:-1}" = 1 ] && [ -f "$O/passes/decode/manifest.json" ]; then
  "$PY" "$D/glm53_passes.py" full --out "$O/passes" --jobs "$(nproc)" > "$O/full.log" 2>&1
  echo "full rc $?" >> "$O/rc.txt"
fi
cat "$O/rc.txt"
tail -3 "$O/run.err" 2>/dev/null
tail -4 "$O/capture.log" "$O/debit.log" "$O/volunteer.log" "$O/verify.log" "$O/full.log" "$O/pin.log" 2>/dev/null
grep -E "slowdown|no stock" "$O/timing.log" "$O/timing-h3.log" "$O/whole.log" 2>/dev/null | head -80
! grep -qv " rc 0$" "$O/rc.txt"
