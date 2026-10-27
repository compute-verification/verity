#!/usr/bin/env bash
# ACCEL v3.1's prover (v31.cu) on leased GPUs, judged by v31.py (cwd = the source tree's root, under `research run`).
# Outside the lease: the build, and on the CPU the gates' operands and seeds (v31.py gpu-prep, a stub challenger's) and the
# bench shapes' CGP1 params. Under the lease: every gate (one call each into poisoned buffers, dumped with its no-write
# control), then, in bench mode, v31 bench over SHAPES (cuBLASLt beside the protocol's arms, interleaved). After the lease:
# v31.py gpu-check on each gate (every word, O and tree level against the reference, then the verifier's verdict line on
# the GPU's own commitment: honest gates accepted, cheating gates refused), gpu-sample on each bench shape's two sampled
# tiles, and the gamma table beside D6 v1.1.
#
#   research run --on vy-nebius-1 --project verity --campaign cmp-159 --source . --cwd source --env GPU_LEASE_WHO=V31-GPU \
#     --env HYPOTHESIS="..." -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py \
#     --by V31-GPU -- bash benchmarks/pouw/accel_v31/run.sh
# (a node's lane runs go through a slot of its gpu pool; the GPUs come from gpu-lease inside, for the GPU work alone)
#
# HYPOTHESIS, the attempt's, goes to $OUT/hypothesis.txt; MODE=gates|bench (default bench: the gates, then the timing; a failed gate times nothing); SHAPES: bench's NAME:M:N:K list
# (default D6's static shapes at m 8192); REPS (default 7); Q, the proof blocks a verdict opens (default 64); LP, T, ROW
# (default 256 3 1024); GENCODE (default arch=compute_120a,code=sm_120a; H200: arch=compute_90a,code=sm_90a); PY (default
# uv run -q --no-project --with numpy python: the node's python3 has no numpy); LEASE, the command that leases the GPU (default gpu-lease 1 --wait --max-min 25 --; a timed run passes
# "gpu-lease 8 --wait --timed --max-min 30 --" under `research run --no-sampler`; empty when already under a lease).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/accel_v31
PY=${PY:-uv run -q --no-project --with numpy python}
export PYTHONPATH=.
GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a}
LP=${LP:-256} T=${T:-3} ROW=${ROW:-1024} Q=${Q:-64} REPS=${REPS:-7} MODE=${MODE:-bench}
W=$(mktemp -d /dev/shm/accel-v31.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
[ -n "${HYPOTHESIS:-}" ] && printf '%s\n' "$HYPOTHESIS" > "$O/hypothesis.txt"
: > "$O/rc.txt"

nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 -o "$W/v31" $D/v31.cu -lcuda -lcublasLt \
  > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
[ -x "$W/v31" ] || { cat "$O/rc.txt"; tail -20 "$O/build.log"; exit 1; }

# gates: NAME M N K OPERANDS CHEAT [LP T ROW]. Honest ones cover a partial last segment (K 1280), wrap-heavy operands
# (uniform), zeros, a third Z digit (K 20480: 208 NT >= 2^14), a third Y digit (K 32768: 128 NT >= 2^14), L' 512 at t 2,
# and 2 KiB rows; the cheats skip a noise tile, forge one tile's T_1, or one tile's O (of 4 tiles).
GATES=${GATES:-"
g-llm 256 256 1536 llm none
g-uniform 256 384 1280 uniform none
g-zero 128 128 768 zero none
g-dz3 128 128 20480 llm none
g-dy3 128 256 32768 llm none
g-lp512 256 256 2048 llm none 512 2 1024
g-row2k 256 256 3072 llm none 256 3 2048
g-big 512 1024 4096 llm none
c-skip 256 256 1536 llm skip:2
c-forge 256 256 1536 llm forge:1:0.25
c-forgeO 256 256 1536 llm forgeO:0.25
"}
echo "$GATES" | while read -r name m n k ops cheat lp t row; do
  [ -n "$name" ] || continue
  c=(); [ "$cheat" = none ] || c=(--cheat "$cheat")
  $PY $D/v31.py gpu-prep "$W/gates/$name" "$m" "$n" "$k" --lp "${lp:-$LP}" --t "${t:-$T}" --row "${row:-$ROW}" --operands "$ops" \
    "${c[@]}" > /dev/null 2>> "$O/prep.err"
  echo "prep $name rc $?" >> "$O/rc.txt"
done
echo "$GATES" > "$W/gates.txt"

SHAPES=${SHAPES:-"8B-n6144-k4096:8192:6144:4096 8B-n4096-k4096:8192:4096:4096 8B-n28672-k4096:8192:28672:4096
8B-n4096-k14336:8192:4096:14336 8B-n128256-k4096:8192:128256:4096
70B-n10240-k8192:8192:10240:8192 70B-n8192-k8192:8192:8192:8192 70B-n57344-k8192:8192:57344:8192
70B-n8192-k28672:8192:8192:28672 70B-n128256-k8192:8192:128256:8192
V4-n43008-k7168:8192:43008:7168 V4-n50176-k3072:8192:50176:3072 V4-n1536-k7168:8192:1536:7168 V4-n65536-k1536:8192:65536:1536
V4-n512-k7168:8192:512:7168 V4-n16384-k4096:8192:16384:4096 V4-n7168-k16384:8192:7168:16384 V4-n129280-k7168:8192:129280:7168
V4-n2048-k7168:8192:2048:7168 V4-n1024-k7168:8192:1024:7168 V4-n8192-k1536:8192:8192:1536 V4-n384-k7168:8192:384:7168"}
export SHAPES
if [ "$MODE" = bench ]; then
  # shellcheck disable=SC2086
  $PY $D/v31.py params "$W/params" $SHAPES --lp "$LP" --t "$T" --row "$ROW" 2>> "$O/prep.err"
  echo "params rc $?" >> "$O/rc.txt"
fi
mkdir -p "$W/bench"
export LP T ROW REPS MODE

LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
# shellcheck disable=SC2016,SC2086
$LEASE bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
  "$W/v31" attrs > "$O/attrs.json" 2>&1
  bad=0
  while read -r name m n k ops cheat lp t row; do
    [ -n "$name" ] || continue
    "$W/v31" gate "$W/gates/$name" "$m" "$n" "$k" "${lp:-$LP}" "${t:-$T}" "${row:-$ROW}" "$cheat" >> "$O/gates.log" 2>&1
    r=$?
    echo "gate $name rc $r" >> "$O/rc.txt"
    [ $r = 0 ] || bad=1
  done < "$W/gates.txt"
  if [ "$MODE" = bench ] && [ $bad = 0 ]; then
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 \
      ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi.csv" 2>&1 &
    smi=$!
    "$W/v31" bench "$REPS" "$W/bench" "$W/params" "$LP" "$T" "$ROW" $SHAPES > "$O/bench.jsonl" 2> "$O/bench.err"
    echo "bench rc $?" >> "$O/rc.txt"
    kill $smi
  fi'
echo "lease rc $?" >> "$O/rc.txt"

# the device gates' word-for-word check and the verifier's verdicts, on the CPU
echo "$GATES" | while read -r name _; do
  [ -n "$name" ] || continue
  [ -f "$W/gates/$name/words.bin" ] || continue
  cp "$W/gates/$name/gpu.json" "$O/gate-$name.gpu.json" 2>/dev/null
  $PY $D/v31.py gpu-check "$W/gates/$name" --q "$Q" > "$O/check-$name.out" 2> "$O/check-$name.err"
  r=$?
  cp "$W/gates/$name/check.json" "$O/check-$name.json" 2>/dev/null
  echo "check $name rc $r" >> "$O/rc.txt"
  echo "$name: $(tail -1 "$O/check-$name.out")" >> "$O/verdicts.txt"
done
mkdir -p "$O/items"
cp "$W"/bench/items-*.jsonl "$O/items/" 2>/dev/null
for s in "$W"/bench/samples/*/; do
  [ -d "$s" ] || continue
  name=$(basename "$s")
  $PY $D/v31.py gpu-sample "$s" > "$O/sample-$name.out" 2> "$O/sample-$name.err"
  echo "sample $name rc $?" >> "$O/rc.txt"
done
[ -s "$O/bench.jsonl" ] && $PY $D/v31.py table "$O/bench.jsonl" > "$O/table.md" 2>> "$O/table.err"
cat "$O/rc.txt"
cat "$O/verdicts.txt" 2>/dev/null
cat "$O/table.md" 2>/dev/null
! grep -qv ' rc 0$' "$O/rc.txt"
