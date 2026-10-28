#!/usr/bin/env bash
# Dither's no-fold call (v31.cu's Fill) on leased GPUs, judged by v31.py and v31lib.py (cwd = the source tree's root, under
# `research run`). Outside the leases: the builds (v31, and libv31.so: v31.cu -DV31_LIB) and the CGP1 params of every
# variant and L on the CPU. Under a lease: a canary (one call of the first variant on the first shape in at most 5
# minutes; a hang or a crash ends the GPU steps there), then v31lib.py's gate (the binding's calls: the GPU's pad, values
# and tree against v31.py's Fill, then the verifier's verdicts on the GPU's own openings, honest and cheating, filler and
# served, a filler's and a served Prover interleaved; verdicts.txt), once with the protocol GEMM and once (the filler's
# gates) with V31_LTW=1 (cuBLASLt's GEMM at S = 1; variants take it as ltw=1). Then, a lease each variant and L of
# FILL_LS, `v31 fill-bench` over FILL_SHAPES beside cuBLASLt and CUTLASS; after it, fill-sample on each shape's sampled
# tiles and fill-table.
#
#   research run --on NODE --project verity --campaign CAMPAIGN --source . --cwd source --env GPU_LEASE_WHO=WHO \
#     --env HYPOTHESIS="..." -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py \
#     --by WHO -- bash benchmarks/pouw/accel_v31/run.sh
# (a node's runs go through a slot of its gpu pool; the GPUs come from gpu-lease inside, for the GPU work alone)
#
# HYPOTHESIS, the attempt's, goes to $OUT/hypothesis.txt. VARIANTS: the variants, each a comma-separated KEY=VALUE list
# over row (the leaf row, dither.Shape.row; 16384) and v31.cu's Knobs: cprio (lo), carve (100), dyn (1), ccta (1), pf (0),
# sched (0), ltw (0), served (0; 1: served calls, O committed and T_S implied, dither.Shape(o=True), their params named
# served-*), gemm (0; 1 in a -DV31_WG build: a served call's D from wg_proto.cuh's GEMM, and the bench adds
# wgmma.linear's arm, wgl), wgbn (256), wgg (16), rp (0; 4: the O rows hashed in the GEMM's producer warpgroup; 5: and in
# a hash warpgroup of its own, r5h its registers, the prefix-sum rows too in the GEMM of the prefix sums unless r5w=0),
# r5h (96), r5w (1), tst (0; 1: RP 5's O and prefix-sum stores through shared memory with TMA), hall (0), hchunk (0),
# wgw (1), lminb (5), sahead (0), full (1), sminb (4), lfold (0) and mask (0);
# and nreg (the GEMM's register cap, V31_MAXNREG or under V31_WG V31_WG_NREG, a build of its own; default v31.cu's).
# Default "row=16384,cprio=hi,sched=1 row=8192,cprio=hi,sched=1".
# FILL_LS: the segment lengths L (default "768 2048", t 3 and 8 at LP 256; 0: L = K; each with min(L, K) at most 8 LP,
# dither/PROTOCOL.md's "The segment and the noise tile"); FILL_SHAPES: the bench's NAME:M:N:K list (default Llama 3
# 8B's QKV, O, gate-up and down at m 16384); V31_ARMS, the bench's arms to time (default all); REPS (default 7); Q, the
# proof blocks a verdict opens (default 64); LP, a noise tile's length L' (default 256).
# FILL_GATE_KNOBS=1 runs the gates (and FILL_DITHER's recheck) under the first variant's knobs (else v31.cu's defaults).
# FILL_DITHER=1 also runs fill_dither.py under the gate's lease (the GPU against dither.py itself). FILL_NSYS=1 adds,
# before the bench, Nsight Systems on `v31 fill-once` (OVL_CALLS overlapped calls, default 6, after two serial ones) for
# each variant, L and shape: out/nsys-TAG-L-SHAPE.nsys-rep, its kernel trace and `v31.py timeline`'s .timeline.md.
# FILL_PROF=1, after the bench, runs Nsight Compute's pipe utilization on `v31 fill-once` (OVL_CALLS overlapped calls,
# default 4) for the first variant on each shape and L, and h200_pipes.py's duty of each pipe over the overlapped call
# (pipes.md).
# GENCODE (default arch=compute_120a,code=sm_120a; H200: arch=compute_90a,code=sm_90a); PY (default
# uv run -q --no-project --with numpy python: the node's python3 has no numpy); LEASE, the command that leases the GPU
# (default gpu-lease 1 --wait --max-min 25 --; a timed run passes "gpu-lease 8 --wait --timed --max-min 30 --" under
# `research run --no-sampler`; empty when already under a lease); LEASE_ONCE=1: every GPU step under one LEASE, after
# the builds and the CPU's prep, in place of a lease each; CUTLASS=0 (no CUTLASS baseline), V31_CACHE (CUTLASS's clone
# and object, default /workspace/pouw/accel_v31); NVFLAGS, more nvcc flags for every build (H200: -DV31_WG=1, the GEMMs
# on warpgroup MMAs); SHA_FLAGS, sha512.cuh's issue choices (default -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1, tuned on
# sm_120).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/accel_v31
PY=${PY:-uv run -q --no-project --with numpy python}
export PYTHONPATH=.
GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a}
LP=${LP:-256} Q=${Q:-64} REPS=${REPS:-7}
VARIANTS=${VARIANTS:-"row=16384,cprio=hi,sched=1 row=8192,cprio=hi,sched=1"}
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
  echo -n "V31_FULL=$(knob "$1" full 1) V31_SMINB=$(knob "$1" sminb 4) V31_LFOLD=$(knob "$1" lfold 0) V31_MASK=$(knob "$1" mask 0) "
  echo "V31_CPRIO=$(knob "$1" cprio lo) V31_CARVE=$(knob "$1" carve 100) V31_DYN=$(knob "$1" dyn 1) V31_CCTA=$(knob "$1" ccta 1)" \
    "V31_PF=$(knob "$1" pf 0) V31_SCHED=$(knob "$1" sched 0) V31_LTW=$(knob "$1" ltw 0) V31_SERVED=$(knob "$1" served 0)" \
    "V31_GEMM=$(knob "$1" gemm 0) V31_WGBN=$(knob "$1" wgbn 256) V31_WGG=$(knob "$1" wgg 16)" \
    "V31_RP=$(knob "$1" rp 0) V31_R5H=$(knob "$1" r5h 96) V31_HALL=$(knob "$1" hall 0) V31_HCHUNK=$(knob "$1" hchunk 0)" \
    "V31_WGW=$(knob "$1" wgw 1) V31_R5W=$(knob "$1" r5w 1) V31_TST=$(knob "$1" tst 0)" \
    "V31_LMINB=$(knob "$1" lminb 5) V31_SAHEAD=$(knob "$1" sahead 0)"
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
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v ${SHA_FLAGS:--DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1} "${CUT[@]}" ${NVFLAGS:-} "${@:3}" \
    -o "$1" $D/v31.cu -lcuda -lcublasLt > "$2" 2>&1
}
build "$W/v31" "$O/build.log"
echo "build rc $?" >> "$O/rc.txt"
[ -x "$W/v31" ] || { cat "$O/rc.txt"; tail -20 "$O/build.log"; exit 1; }
for n in $(for v in $VARIANTS; do knob "$v" nreg ""; done | sort -u); do
  build "$W/v31-n$n" "$O/build-n$n.log" -DV31_MAXNREG="$n" -DV31_WG_NREG="$n"
  echo "build n$n rc $?" >> "$O/rc.txt"
done

LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
FILL_LS=${FILL_LS:-"768 2048"}
FILL_SHAPES=${FILL_SHAPES:-"8B-qkv:16384:6144:4096 8B-o:16384:4096:4096 8B-up:16384:28672:4096 8B-down:16384:4096:14336"}
export VARIANTS FILL_SHAPES FILL_LS LP REPS Q D PY
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v ${SHA_FLAGS:--DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1} -DV31_LIB -DCOMMIT_GPU_DOMS=8 ${NVFLAGS:-} \
  -Xcompiler -fPIC -shared -o "$W/libv31.so" $D/v31.cu -lcuda -lcublasLt > "$O/build-lib.log" 2>&1
echo "build lib rc $?" >> "$O/rc.txt"
for v in $VARIANTS; do
  for L in $FILL_LS; do
    sv=(); [ "$(knob "$v" served 0)" = 1 ] && sv=(--served)
    # shellcheck disable=SC2086
    $PY $D/v31.py fill-params "$W/fparams" $FILL_SHAPES --lp "$LP" --L "$L" --row "$(knob "$v" row 16384)" "${sv[@]}" \
      2>> "$O/prep.err"
    echo "fill-params $(knob "$v" row 16384)${sv[*]} L$L rc $?" >> "$O/rc.txt"
  done
done
# the GPU steps, a lease each (LEASE_ONCE=1: all of them under one lease, LEASE's). First a canary, one call of the
# first variant on the first shape in at most 5 minutes: a hang or a crash ends them there (exit 3), not at the lease's cap
fill_gpu() {
  # shellcheck disable=SC2016
  $LEASE bash -c '
    nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv \
      ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
    v=${VARIANTS%% *} L=${FILL_LS%% *} s=${FILL_SHAPES%% *}
    # shellcheck disable=SC2046
    env $(knob_env "$v") timeout 300 "$W/v31" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 16384)" "$s" > "$O/canary.log" 2>&1
    rc=$?
    echo "canary rc $rc" >> "$O/rc.txt"
    [ $rc = 0 ] || exit 3
    gk=(); [ "${FILL_GATE_KNOBS:-0}" = 1 ] && gk=($(knob_env "${VARIANTS%% *}"))
    for x in 0 1; do
      only=(); [ $x = 1 ] && only=(--only fill)
      env "${gk[@]}" V31_LTW=$x $PY $D/v31lib.py gate --lib "$W/libv31.so" --out "$O/fill-gates-ltw$x" --q "$Q" "${only[@]}" \
        > "$O/fill-gates-ltw$x.log" 2> "$O/fill-gates-ltw$x.err"
      echo "fill gate ltw$x rc $?" >> "$O/rc.txt"
      if [ "${FILL_DITHER:-0}" = 1 ]; then
        env "${gk[@]}" V31_LTW=$x $PY $D/fill_dither.py --lib "$W/libv31.so" --out "$O/fill-dither-ltw$x" > "$O/fill-dither-ltw$x.log" \
          2> "$O/fill-dither-ltw$x.err"
        echo "fill dither ltw$x rc $?" >> "$O/rc.txt"
      fi
    done'
  rc=$?
  echo "lease fill gate rc $rc" >> "$O/rc.txt"
  [ $rc = 3 ] && return 3
  for x in 0 1; do
    grep -hE '^[a-z].*(ACCEPT|REFUSE)' "$O/fill-gates-ltw$x.log" "$O/fill-dither-ltw$x.log" 2> /dev/null | sed "s/^/ltw$x /" \
      >> "$O/verdicts.txt"
  done
  if [ "${FILL_NSYS:-0}" = 1 ]; then
    NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
    export NSYS OVL_CALLS=${OVL_CALLS:-6}
    [ -n "$NSYS" ] || echo "nsys rc no-nsys" >> "$O/rc.txt"
    # shellcheck disable=SC2016
    $LEASE bash -c '
      for v in $VARIANTS; do
        for L in $FILL_LS; do
          tag=$(tr "," "-" <<< "$v" | tr -d "=")-L$L
          for s in $FILL_SHAPES; do
            f="$O/nsys-$tag-${s%%:*}"
            # shellcheck disable=SC2046
            env $(knob_env "$v") V31_ONCE_OVL="$OVL_CALLS" "$NSYS" profile -t cuda -s none --cpuctxsw none -f true -o "$f" \
              "$W/v31" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 16384)" "$s" > "$f.log" 2>&1
            echo "nsys $tag ${s%%:*} rc $?" >> "$O/rc.txt"
          done
        done
      done'
    echo "lease nsys rc $?" >> "$O/rc.txt"
    for f in "$O"/nsys-*.nsys-rep; do
      [ -f "$f" ] || continue
      "$NSYS" stats -r cuda_gpu_trace -f csv --force-export=true -o "${f%.nsys-rep}" "$f" >> "${f%.nsys-rep}.log" 2>&1
      $PY $D/v31.py timeline "${f%.nsys-rep}_cuda_gpu_trace.csv" --gemm gemm --from bits_kernel > "${f%.nsys-rep}.timeline.md" \
        2>> "${f%.nsys-rep}.log"
      echo "timeline $(basename "${f%.nsys-rep}") rc $?" >> "$O/rc.txt"
    done
  fi
  # the bench runs whatever the gate's verdicts (its own checked call is sampled below); cite it only beside a passed gate
  for v in $VARIANTS; do
    for L in $FILL_LS; do
      export v L
      # shellcheck disable=SC2016
      $LEASE bash -c '
        tag=$(tr "," "-" <<< "$v" | tr -d "=")-L$L
        n=$(knob "$v" nreg "")
        nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 \
          ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi-$tag.csv" 2>&1 &
        smi=$!
        mkdir -p "$W/fbench/$tag"
        # shellcheck disable=SC2046,SC2086
        env $(knob_env "$v") "$W/v31${n:+-n$n}" fill-bench "$REPS" "$W/fbench/$tag" "$W/fparams" "$LP" "$L" \
          "$(knob "$v" row 16384)" $FILL_SHAPES > "$O/fill-bench-$tag.jsonl" 2> "$O/fill-bench-$tag.err"
        echo "fill-bench $tag rc $?" >> "$O/rc.txt"
        kill $smi'
      echo "lease fill-bench $v L$L rc $?" >> "$O/rc.txt"
    done
  done
  if [ "${FILL_PROF:-0}" = 1 ]; then
    NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
    export NCU
    [ -n "$NCU" ] || echo "fill prof rc no-ncu" >> "$O/rc.txt"
    # shellcheck disable=SC2016,SC2086
    $LEASE bash -c '
      v=${VARIANTS%% *}
      tag=$(tr "," "-" <<< "$v" | tr -d "=")
      n=$(knob "$v" nreg "")
      # the tensor pipes this chip counts, beside the section pipes (ncu refuses a metric it lacks)
      "$NCU" --query-metrics --chip "${NCU_CHIP:-gh100}" > "$O/ncu-metrics.txt" 2>&1
      tens=$(awk "{print \$1}" "$O/ncu-metrics.txt" | grep -E "^sm__(pipe_tensor[a-z0-9_]*_cycles_active|inst_executed_pipe_tensor[a-z0-9_]*)$" |
        sed "s/$/.avg.pct_of_peak_sustained_active/" | paste -sd, -)
      for L in $FILL_LS; do
        for s in $FILL_SHAPES; do
          f="$O/ncu-$tag-L$L-${s%%:*}"
          for extra in "${tens:+,$tens}" ""; do
            env $(knob_env "$v") V31_ONCE_OVL="${OVL_CALLS:-4}" V31_LT_TUNE=0 "$NCU" --clock-control none --section ComputeWorkloadAnalysis \
              --section SpeedOfLight --metrics "sm__cycles_active.avg,sm__cycles_elapsed.avg$extra" -f -o "$f" \
              "$W/v31${n:+-n$n}" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 16384)" "$s" > "$f.log" 2>&1
            rc=$?
            [ $rc = 0 ] && break
          done
          echo "fill prof $tag L$L ${s%%:*} rc $rc" >> "$O/rc.txt"
          "$NCU" -i "$f.ncu-rep" --csv --page raw > "$f.raw.csv" 2>> "$f.log"
        done
      done'
    echo "lease fill prof rc $?" >> "$O/rc.txt"
    $PY $D/h200_pipes.py "$O" > "$O/pipes.md" 2> "$O/pipes.err"
    echo "pipes rc $?" >> "$O/rc.txt"
  fi
}
if [ "${LEASE_ONCE:-0}" = 1 ] && [ -n "$LEASE" ]; then
  export -f fill_gpu
  # shellcheck disable=SC2086
  $LEASE env LEASE= bash -c 'set -uo pipefail; fill_gpu'
  echo "lease once rc $?" >> "$O/rc.txt"
else
  fill_gpu
fi
for s in "$W"/fbench/*/samples/*/; do
  [ -d "$s" ] || continue
  name=$(basename "$(dirname "$(dirname "$s")")")-$(basename "$s")
  $PY $D/v31.py fill-sample "$s" > "$O/fill-sample-$name.out" 2> "$O/fill-sample-$name.err"
  echo "fill-sample $name rc $?" >> "$O/rc.txt"
done
ls "$O"/fill-bench-*.jsonl > /dev/null 2>&1 && $PY $D/v31.py fill-table "$O"/fill-bench-*.jsonl > "$O/table.md" 2>> "$O/table.err"
cat "$O/rc.txt"
cat "$O/verdicts.txt" 2>/dev/null
cat "$O/table.md" 2>/dev/null
! grep -qv ' rc 0$' "$O/rc.txt"
