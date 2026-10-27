#!/usr/bin/env bash
# HIGHUTIL on one leased GPU of vy-nebius-1 (RTX PRO 6000, sm_120), cwd = the source tree's root under `research run`:
# the build and the config (the checkpoint's weights, params) outside the lease, then under it capture.py's activations
# (a W8A8 forward of the checkpoint), period.py's diag (one step's tiles against v31.py's reference) and its interleaved
# protocol and plain periods and the negative one, and the machine model's θ beside them.
#
#   research run --on vy-nebius-1 --project verity --campaign cmp-159 --source . --cwd source \
#     --env GPU_LEASE_WHO=compute-accounting-highutil --send main.txt [--send challenger-key.json] -- \
#     env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by compute-accounting-highutil \
#     -- bash benchmarks/pouw/highutil/run.sh
#
# COINS: local (default; a development session on this machine, not of record) or gate:USER@HOST (vy-challenger-1 with
# PEER, CHALLENGER_KEY, the deployment's key record, and GATE_KEY); MAIN (default inputs/main.txt): the verifier's copy of
# main's first-parent line. CKPT: the pinned checkpoint's snapshot directory (default the node's HF cache), or `seeded`
# (seeded A and B, no capture). BLOCKS, STEPS, Q, PERIODS, NEGATIVE: period.py run's; ROW (default 32768); NSYS=1: after the
# periods, Nsight Systems over one plain and one protocol period (their kernels' share of the GPU's time); LEASE (default
# gpu-lease 1 --wait --max-min 25 --); PY (default uv run -q --no-project --with numpy python). SWEEP: `;`-separated settings
# of v31's V31_* knobs (`default`: none), timed by period.py sweep after the diag; PERIODS=0 then skips the periods. A V31_*
# variable in the run's environment sets that knob for every hu process.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/highutil
PY=${PY:-uv run -q --no-project --with numpy python}
export PYTHONPATH=.
GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a}
TORCH_PY=${TORCH_PY:-uv run -q --no-project --with torch --with numpy --with safetensors --with tokenizers --with pyarrow python}
CKPT=${CKPT:-/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7}
ROW=${ROW:-32768} BLOCKS=${BLOCKS:-4} STEPS=${STEPS:-300} Q=${Q:-2048} PERIODS=${PERIODS:-2} NEGATIVE=${NEGATIVE-skip:2@2}
COINS=${COINS:-local} MAIN=${MAIN:-${RESEARCH_RUN_DIR:-.}/inputs/main.txt}
W=$(mktemp -d /dev/shm/highutil.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
[ -n "${HYPOTHESIS:-}" ] && printf '%s\n' "$HYPOTHESIS" > "$O/hypothesis.txt"
: > "$O/rc.txt"
nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 -o "$W/hu" $D/hu.cu -lcuda -lcublasLt \
  > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
[ -x "$W/hu" ] || { cat "$O/rc.txt"; tail -30 "$O/build.log"; exit 1; }
sha256sum $D/v31.cu $D/v31.py $D/hu.cu $D/period.py > "$O/sources.sha256"
if [ "$CKPT" = seeded ]; then SRC=(--seeded); else SRC=(--checkpoint "$CKPT"); fi
uv run -q --no-project --with numpy --with safetensors python $D/period.py prep "$W/cfg" "${SRC[@]}" --row "$ROW" \
  > "$O/prep.log" 2>&1
r=$?
echo "prep rc $r" >> "$O/rc.txt"
[ $r = 0 ] || { cat "$O/rc.txt"; tail -30 "$O/prep.log"; exit 1; }
cp "$W/cfg/prep.json" "$O/prep.json"
[ "$CKPT" = seeded ] || $TORCH_PY -c "import torch, safetensors, tokenizers, pyarrow; print('torch', torch.__version__, torch.version.cuda)" \
  >> "$O/prep.log" 2>&1
(cd benchmarks/pouw/machine_model && PYTHONPATH=. uv run -q --no-project --with scipy --with numpy python -m verity_machine_model.cli \
  --device sm_120 theta) > "$O/theta.txt" 2>&1
echo "theta rc $?" >> "$O/rc.txt"
GATE=()
if [ "$COINS" != local ]; then
  GATE=(--peer "${PEER:?}" --challenger-key "${CHALLENGER_KEY:?}" --gate-key "${GATE_KEY:?}")
fi
LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
export D PY TORCH_PY CKPT ROW BLOCKS STEPS Q PERIODS NEGATIVE COINS MAIN NSYS=${NSYS:-0} SWEEP=${SWEEP:-}
export GATE_ARGS="${GATE[*]}"
$LEASE bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 500 \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi.csv" 2>&1 &
  smi=$!
  if [ "$CKPT" != seeded ]; then
    $TORCH_PY $D/capture.py "$W/cfg" --checkpoint "$CKPT" \
      > "$O/capture.log" 2>&1
    r=$?
    echo "capture rc $r" >> "$O/rc.txt"
    [ $r = 0 ] || { kill $smi 2>/dev/null; exit $r; }
    cp "$W/cfg/capture.json" "$O/capture.json"
  fi
  $PY $D/period.py diag "$W/cfg" "$W/hu" "$O" > "$O/diag.log" 2>&1
  echo "diag rc $?" >> "$O/rc.txt"
  if [ -n "$SWEEP" ]; then
    sets=()
    IFS=";" read -ra ss <<< "$SWEEP"
    for s in "${ss[@]}"; do [ "$s" = default ] && s=""; sets+=(--set "$s"); done
    $PY $D/period.py sweep "$W/cfg" "$W/hu" "$O" "${sets[@]}" > "$O/sweep.log" 2>&1
    echo "sweep rc $?" >> "$O/rc.txt"
  fi
  r=0
  [ "$PERIODS" = 0 ] || $PY $D/period.py run "$W/cfg" "$W/hu" "$O" --coins "$COINS" --main "$MAIN" $GATE_ARGS --blocks "$BLOCKS" --steps "$STEPS" \
    --q "$Q" --periods "$PERIODS" --negative "$NEGATIVE" > "$O/period.log" 2>&1
  r=$?
  echo "period rc $r" >> "$O/rc.txt"
  if [ "$NSYS" = 1 ]; then
    for arm in plain proto; do
      nsys profile -o "$O/nsys-$arm" --force-overwrite true --trace cuda $PY $D/period.py run "$W/cfg" "$W/hu" "$W/nsys-$arm" \
        --coins local --main "$MAIN" --blocks 1 --steps 64 --q 64 --periods 1 --negative "" --arms $arm >> "$O/nsys.log" 2>&1
      echo "nsys $arm rc $?" >> "$O/rc.txt"
      nsys stats --report cuda_gpu_kern_sum --format csv -o "$O/nsys-$arm" "$O/nsys-$arm.nsys-rep" >> "$O/nsys.log" 2>&1
      nsys export --type sqlite --force-overwrite true -o "$O/nsys-$arm.sqlite" "$O/nsys-$arm.nsys-rep" >> "$O/nsys.log" 2>&1
      $PY $D/timeline.py "$O/nsys-$arm.sqlite" > "$O/timeline-$arm.json" 2>> "$O/nsys.log"
    done
  fi
  kill $smi 2>/dev/null
  exit $r
'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -40 "$O/period.log"
grep -q "^period rc 0" "$O/rc.txt"
