#!/usr/bin/env bash
# HIGHUTIL on one leased GPU of vy-nebius-1 (RTX PRO 6000, sm_120), cwd = the source tree's root under `research run`:
# the build and the config (weights, params) outside the lease, then under it period.py's diag (one step's tiles
# against v31.py's reference) and its interleaved protocol and plain periods and the negative one, and the machine
# model's θ beside them.
#
#   research run --on vy-nebius-1 --project verity --campaign cmp-159 --source . --cwd source \
#     --env GPU_LEASE_WHO=compute-accounting-highutil --send main.txt [--send challenger-key.json] -- \
#     env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by compute-accounting-highutil \
#     -- bash benchmarks/pouw/highutil/run.sh
#
# COINS: local (default; a development session on this machine, not of record) or gate:USER@HOST (vy-challenger-1 with
# PEER, CHALLENGER_KEY, the deployment's key record, and GATE_KEY); MAIN (default inputs/main.txt): the verifier's copy of
# main's first-parent line. BLOCKS, STEPS, Q, PERIODS, NEGATIVE: period.py run's; ROW (default 32768); NSYS=1: after the
# periods, Nsight Systems over one plain and one protocol period (their kernels' share of the GPU's time); LEASE (default
# gpu-lease 1 --wait --max-min 25 --); PY (default uv run -q --no-project --with numpy python).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/highutil
PY=${PY:-uv run -q --no-project --with numpy python}
export PYTHONPATH=.
GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a}
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
$PY $D/period.py prep "$W/cfg" --row "$ROW" > "$O/prep.log" 2>&1
echo "prep rc $?" >> "$O/rc.txt"
(cd benchmarks/pouw/machine_model && PYTHONPATH=. uv run -q --no-project --with scipy --with numpy python -m verity_machine_model.cli \
  --device sm_120 theta) > "$O/theta.txt" 2>&1
echo "theta rc $?" >> "$O/rc.txt"
GATE=()
if [ "$COINS" != local ]; then
  GATE=(--peer "${PEER:?}" --challenger-key "${CHALLENGER_KEY:?}" --gate-key "${GATE_KEY:?}")
fi
LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
export D PY ROW BLOCKS STEPS Q PERIODS NEGATIVE COINS MAIN NSYS=${NSYS:-0}
export GATE_ARGS="${GATE[*]}"
$LEASE bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 500 \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi.csv" 2>&1 &
  smi=$!
  $PY $D/period.py diag "$W/cfg" "$W/hu" "$O" > "$O/diag.log" 2>&1
  echo "diag rc $?" >> "$O/rc.txt"
  # shellcheck disable=SC2086
  $PY $D/period.py run "$W/cfg" "$W/hu" "$O" --coins "$COINS" --main "$MAIN" $GATE_ARGS --blocks "$BLOCKS" --steps "$STEPS" \
    --q "$Q" --periods "$PERIODS" --negative "$NEGATIVE" > "$O/period.log" 2>&1
  r=$?
  echo "period rc $r" >> "$O/rc.txt"
  if [ "$NSYS" = 1 ]; then
    nsys profile -o "$O/nsys-period" --force-overwrite true --trace cuda $PY $D/period.py run "$W/cfg" "$W/hu" "$W/nsys-out" \
      --coins local --main "$MAIN" --blocks 1 --steps 100 --q 64 --periods 1 --negative "" > "$O/nsys.log" 2>&1
    echo "nsys rc $?" >> "$O/rc.txt"
    nsys stats --report cuda_gpu_kern_sum --format csv -o "$O/nsys-kern" "$O/nsys-period.nsys-rep" >> "$O/nsys.log" 2>&1
    nsys export --type sqlite -o "$O/nsys-period.sqlite" "$O/nsys-period.nsys-rep" >> "$O/nsys.log" 2>&1
  fi
  kill $smi 2>/dev/null
  exit $r
'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -40 "$O/period.log"
grep -q "^period rc 0" "$O/rc.txt"
