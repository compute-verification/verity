#!/usr/bin/env bash
# ACCEL v3.1's prover (v31.cu) on leased GPUs, judged by v31.py (cwd = the source tree's root, under `research run`).
# Outside the leases: the build, and on the CPU the gates' operands and seeds (v31.py gpu-prep, a stub challenger's) and the
# bench shapes' CGP1 params. Lease 1: every gate (one call each into poisoned buffers, dumped with its no-write control).
# Then on the CPU, v31.py gpu-check on each gate (every word, O and tree level against the reference, then the verifier's
# verdict line on the GPU's own commitment: honest gates accepted, cheating gates refused). Lease 2, in bench mode and only
# if every gate's check passed: v31 bench over SHAPES for each of VARIANTS (cuBLASLt and CUTLASS beside the protocol's arms,
# interleaved). After it: gpu-sample on each bench shape's two sampled tiles, of its checked serial call and of an
# overlapped call after the timing, and the gamma table beside D6 v1.1.
#
#   research run --on vy-nebius-1 --project verity --campaign cmp-159 --source . --cwd source --env GPU_LEASE_WHO=V31-GPU \
#     --env HYPOTHESIS="..." -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py \
#     --by V31-GPU -- bash benchmarks/pouw/accel_v31/run.sh
# (a node's lane runs go through a slot of its gpu pool; the GPUs come from gpu-lease inside, for the GPU work alone)
#
# MODE=prof: after the gates, in place of the bench, Nsight Compute (--set full) on PROF_KERNELS (an ncu -k regex, default
# the form, wrap and commit kernels; PROF_COUNT launches, default 10) of `v31 once` on each of SHAPES for each of VARIANTS:
# out/ncu-TAG-SHAPE.ncu-rep, its details page as CSV and its raw page (every metric, the stall reasons among them).
#
# HYPOTHESIS, the attempt's, goes to $OUT/hypothesis.txt; MODE=gates|bench|prof (default bench); SHAPES: bench's NAME:M:N:K list
# (default D6's static shapes at m 8192); VARIANTS: bench's variants, each a comma-separated KEY=VALUE list over row (8192),
# cprio (lo), carve (100), dyn (1), wrap (2; 1 to 4), wcw (256), ccta (1), pf (0) (v31.cu's Knobs) and nreg (the GEMM's
# register cap, a build of its own; default v31.cu's), default row=8192; REPS (default 7); Q, the proof blocks a verdict
# opens (default 64); LP, T, ROW (the gates' default 256 3 1024);
# GENCODE (default arch=compute_120a,code=sm_120a; H200: arch=compute_90a,code=sm_90a); PY (default
# uv run -q --no-project --with numpy python: the node's python3 has no numpy); LEASE, the command that leases the GPU
# (default gpu-lease 1 --wait --max-min 25 --; a timed run passes "gpu-lease 8 --wait --timed --max-min 30 --" under
# `research run --no-sampler`; empty when already under a lease); CUTLASS=0 (no CUTLASS baseline), V31_CACHE (CUTLASS's
# clone and object, default /workspace/pouw/accel_v31).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/accel_v31
PY=${PY:-uv run -q --no-project --with numpy python}
export PYTHONPATH=.
GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a}
LP=${LP:-256} T=${T:-3} ROW=${ROW:-1024} Q=${Q:-64} REPS=${REPS:-7} MODE=${MODE:-bench}
VARIANTS=${VARIANTS:-row=8192}
W=$(mktemp -d /dev/shm/accel-v31.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
[ -n "${HYPOTHESIS:-}" ] && printf '%s\n' "$HYPOTHESIS" > "$O/hypothesis.txt"
: > "$O/rc.txt"
# knob VARIANT KEY DEFAULT: the variant's KEY, else DEFAULT
knob() { local v; v=$(tr ',' '\n' <<< "$1" | sed -n "s/^$2=//p" | tail -1); echo "${v:-$3}"; }
# knob_env VARIANT: its knobs as v31.cu's environment
knob_env() {
  echo "V31_CPRIO=$(knob "$1" cprio lo) V31_CARVE=$(knob "$1" carve 100) V31_DYN=$(knob "$1" dyn 1) V31_WRAP=$(knob "$1" wrap 2)" \
    "V31_WCW=$(knob "$1" wcw 256) V31_CCTA=$(knob "$1" ccta 1) V31_PF=$(knob "$1" pf 0)"
}
export -f knob knob_env

# CUTLASS's int8 GEMMs, the second baseline: the harness's cutlass_2x.cu at the harness's CUTLASS pin, cloned once into
# V31_CACHE and its object kept there under a hash of its inputs; CUTLASS=0 builds without it (cuBLASLt's baseline alone)
CUT=()
if [ "${CUTLASS:-1}" != 0 ]; then
  H=benchmarks/pouw/harness/native
  CACHE=${V31_CACHE:-/workspace/pouw/accel_v31}
  tag=$(sed -n 's/^CUTLASS_TAG="\(.*\)"$/\1/p' $H/build.sh)
  pin=$(sed -n 's/^CUTLASS_COMMIT="\(.*\)"$/\1/p' $H/build.sh)
  cdir=$CACHE/cutlass-$tag
  mkdir -p "$CACHE"
  if [ ! -d "$cdir/include/cutlass" ]; then
    git clone -q --depth 1 --branch "$tag" https://github.com/NVIDIA/cutlass "$W/cutlass" > "$O/build-cutlass.log" 2>&1 &&
      { [ -d "$cdir" ] || mv "$W/cutlass" "$cdir"; }
  fi
  [ "$(git -C "$cdir" rev-parse HEAD 2>/dev/null)" = "$pin" ]
  echo "cutlass $tag rc $?" >> "$O/rc.txt"
  key=$(cat $H/cutlass_2x.cu $H/cutlass_api.h <(nvcc --version) <(echo "$GENCODE $pin") | sha256sum | cut -c1-16)
  obj=$CACHE/cutlass_2x-$key.o
  if [ ! -f "$obj" ]; then
    nvcc -O3 -std=c++17 -gencode "$GENCODE" --expt-relaxed-constexpr -I$H -I"$cdir/include" -I"$cdir/tools/util/include" \
      -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DPH_PART=0 -c $H/cutlass_2x.cu -o "$W/cutlass_2x.o" >> "$O/build-cutlass.log" 2>&1 &&
      cp "$W/cutlass_2x.o" "$obj.$$" && mv "$obj.$$" "$obj"
  fi
  [ -f "$obj" ]
  echo "build cutlass rc $?" >> "$O/rc.txt"
  CUT=(-DV31_CUTLASS "$obj")
fi
build() {   # BIN LOG [nvcc flags]
  nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 "${CUT[@]}" "${@:3}" -o "$1" \
    $D/v31.cu -lcuda -lcublasLt > "$2" 2>&1
}
build "$W/v31" "$O/build.log"
echo "build rc $?" >> "$O/rc.txt"
[ -x "$W/v31" ] || { cat "$O/rc.txt"; tail -20 "$O/build.log"; exit 1; }
for n in $(for v in $VARIANTS; do knob "$v" nreg ""; done | sort -u); do
  build "$W/v31-n$n" "$O/build-n$n.log" -DV31_MAXNREG="$n"
  echo "build n$n rc $?" >> "$O/rc.txt"
done

# gates: NAME M N K OPERANDS CHEAT [LP T ROW]. Honest ones cover a partial last segment (K 1280), wrap-heavy operands
# (uniform; at N 384 wrap2's 128 columns a CTA, at N 512 its 256), zeros, a third Z digit (K 20480: 208 NT >= 2^14), a
# third Y digit (K 32768: 128 NT >= 2^14), L' 512 at t 2 (and with A = B = 127, so columns past 256 wraps in a noise
# tile), rows of 2, 8 and 32 KiB; the cheats skip a noise tile, forge one tile's T_1, or one tile's O (of 4 tiles). The
# gates run the first variant's knobs (the bench's samples check the rest).
GATES=${GATES:-"
g-llm 256 256 1536 llm none
g-uniform 256 384 1280 uniform none
g-uniform512 256 512 1536 uniform none
g-zero 128 128 768 zero none
g-dz3 128 128 20480 llm none
g-dy3 128 256 32768 llm none
g-lp512 256 256 2048 llm none 512 2 1024
g-max512 256 256 2048 max none 512 2 1024
g-row2k 256 256 3072 llm none 256 3 2048
g-row8k 256 384 3072 llm none 256 3 8192
g-row32k 256 256 3072 llm none 256 3 32768
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
if [ "$MODE" != gates ]; then
  for r in $(for v in $VARIANTS; do knob "$v" row 8192; done | sort -u); do
    # shellcheck disable=SC2086
    $PY $D/v31.py params "$W/params" $SHAPES --lp "$LP" --t "$T" --row "$r" 2>> "$O/prep.err"
    echo "params r$r rc $?" >> "$O/rc.txt"
  done
fi
export LP T ROW REPS MODE VARIANTS

LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
# shellcheck disable=SC2016,SC2086
$LEASE bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
  "$W/v31" attrs > "$O/attrs.json" 2>&1
  while read -r name m n k ops cheat lp t row; do
    [ -n "$name" ] || continue
    env $(knob_env "${VARIANTS%% *}") "$W/v31" gate "$W/gates/$name" "$m" "$n" "$k" "${lp:-$LP}" "${t:-$T}" "${row:-$ROW}" \
      "$cheat" >> "$O/gates.log" 2>&1
    echo "gate $name rc $?" >> "$O/rc.txt"
  done < "$W/gates.txt"'
echo "lease gates rc $?" >> "$O/rc.txt"

# the device gates' word-for-word check and the verifier's verdicts, on the CPU
echo "$GATES" | while read -r name _; do
  [ -n "$name" ] || continue
  [ -f "$W/gates/$name/words.bin" ] || { echo "check $name rc missing" >> "$O/rc.txt"; continue; }
  cp "$W/gates/$name/gpu.json" "$O/gate-$name.gpu.json" 2>/dev/null
  $PY $D/v31.py gpu-check "$W/gates/$name" --q "$Q" > "$O/check-$name.out" 2> "$O/check-$name.err"
  r=$?
  cp "$W/gates/$name/check.json" "$O/check-$name.json" 2>/dev/null
  echo "check $name rc $r" >> "$O/rc.txt"
  echo "$name: $(tail -1 "$O/check-$name.out")" >> "$O/verdicts.txt"
done

if [ "$MODE" = bench ] && ! grep -qv ' rc 0$' "$O/rc.txt"; then
  mkdir -p "$W/bench"
  # shellcheck disable=SC2016,SC2086
  $LEASE bash -c '
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 \
      ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi.csv" 2>&1 &
    smi=$!
    for v in $VARIANTS; do
      tag=$(tr ',' '-' <<< "$v" | tr -d '=')
      n=$(knob "$v" nreg "")
      mkdir -p "$W/bench/$tag"
      env $(knob_env "$v") "$W/v31${n:+-n$n}" bench "$REPS" "$W/bench/$tag" "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" \
        $SHAPES > "$O/bench-$tag.jsonl" 2> "$O/bench-$tag.err"
      echo "bench $tag rc $?" >> "$O/rc.txt"
    done
    kill $smi'
  echo "lease bench rc $?" >> "$O/rc.txt"
elif [ "$MODE" = prof ] && ! grep -qv ' rc 0$' "$O/rc.txt"; then
  export PROF_KERNELS=${PROF_KERNELS:-form_|wrap|salt_|leaf_} PROF_COUNT=${PROF_COUNT:-10}
  NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
  export NCU
  [ -n "$NCU" ] || echo "prof rc no-ncu" >> "$O/rc.txt"
  # shellcheck disable=SC2016,SC2086
  $LEASE bash -c '
    for v in $VARIANTS; do
      tag=$(tr ',' '-' <<< "$v" | tr -d '=')
      n=$(knob "$v" nreg "")
      for s in $SHAPES; do
        f="$O/ncu-$tag-${s%%:*}"
        env $(knob_env "$v") "$NCU" --clock-control none -k "regex:$PROF_KERNELS" -c "$PROF_COUNT" --set full -f -o "$f" \
          "$W/v31${n:+-n$n}" once "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" "$s" > "$f.log" 2>&1
        echo "prof $tag ${s%%:*} rc $?" >> "$O/rc.txt"
        "$NCU" -i "$f.ncu-rep" --csv --page details > "$f.csv" 2>> "$f.log"
        "$NCU" -i "$f.ncu-rep" --csv --page raw > "$f.raw.csv" 2>> "$f.log"
      done
    done'
  echo "lease prof rc $?" >> "$O/rc.txt"
elif [ "$MODE" != gates ]; then
  echo "$MODE skipped: a gate or its check failed" >> "$O/rc.txt"
fi

for b in "$W"/bench/*/; do
  [ -d "$b" ] || continue
  tag=$(basename "$b")
  mkdir -p "$O/items/$tag"
  cp "$b"/items-*.jsonl "$O/items/$tag/" 2>/dev/null
  for s in "$b"samples/*/; do
    [ -d "$s" ] || continue
    name=$(basename "$s")
    $PY $D/v31.py gpu-sample "$s" > "$O/sample-$tag-$name.out" 2> "$O/sample-$tag-$name.err"
    echo "sample $tag $name rc $?" >> "$O/rc.txt"
  done
done
ls "$O"/bench-*.jsonl > /dev/null 2>&1 && $PY $D/v31.py table "$O"/bench-*.jsonl > "$O/table.md" 2>> "$O/table.err"
cat "$O/rc.txt"
cat "$O/verdicts.txt" 2>/dev/null
cat "$O/table.md" 2>/dev/null
! grep -qv ' rc 0$' "$O/rc.txt"
