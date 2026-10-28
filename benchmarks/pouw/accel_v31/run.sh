#!/usr/bin/env bash
# Dither v3.1's prover (v31.cu) on leased GPUs, judged by v31.py (cwd = the source tree's root, under `research run`).
# Outside the leases: the build, and on the CPU the gates' operands and seeds (v31.py gpu-prep, a stub challenger's) and the
# bench shapes' CGP1 params. Lease 1: every gate (one call each into poisoned buffers, dumped with its no-write control).
# Then on the CPU, v31.py gpu-check on each gate (every prefix sum, O and tree level against the reference, then the verifier's
# verdict line on the GPU's own commitment: honest gates accepted, cheating gates refused). Lease 2, in bench mode and only
# if every gate's check passed: v31 bench over SHAPES for each of VARIANTS (cuBLASLt and CUTLASS beside the protocol's arms,
# interleaved). After it: gpu-sample on each bench shape's two sampled tiles, of its checked serial call and of an
# overlapped call after the timing, and the gamma table beside D6 v1.1.
#
#   research run --on vy-nebius-1 --project verity --campaign CAMPAIGN --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --env HYPOTHESIS="..." -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py \
#     --by LANE -- bash benchmarks/pouw/accel_v31/run.sh
# (a node's lane runs go through a slot of its gpu pool; the GPUs come from gpu-lease inside, for the GPU work alone)
#
# MODE=prof: after the gates, in place of the bench, Nsight Compute (--set full) on PROF_KERNELS (an ncu -k regex, default
# the form, wrap and commit kernels; PROF_COUNT launches, default 10) of `v31 once` on each of SHAPES for each of VARIANTS:
# out/ncu-TAG-SHAPE.ncu-rep, its details page as CSV and its raw page (every metric, the stall reasons among them).
# MODE=nsys: after the gates, Nsight Systems on `v31 once` with OVL_CALLS (default 6) overlapped calls after its two
# serial ones, on each of SHAPES for each of VARIANTS: out/nsys-TAG-SHAPE.nsys-rep, its kernel trace as CSV, and
# `v31.py timeline`'s summary of it (.timeline.md: each kernel serial and overlapped, the steady period, the streams' overlap).
# MODE=duty: after the gates, Nsight Compute's application-range replay over OVL_CALLS (default 6) overlapped calls of
# `v31 once`, and over as many wgmma.linear calls (gemm=1), on each of SHAPES for each of VARIANTS: every pipe's duty
# (duty.metrics) with the kernels concurrent as they run, out/duty-TAG-SHAPE-{ovl,wgl}.csv and duty.py's table, duty.md.
#
# MODE=fill: Dither's no-fold call in place of all the above. libv31.so (v31.cu -DV31_LIB) and, under a lease,
# v31lib.py's gate (the binding's calls: the GPU's pad, values and tree against v31.py's Fill, then the verifier's verdicts
# on the GPU's own openings, honest and cheating, filler and served, a filler's and a served Prover interleaved;
# verdicts.txt), once with the protocol GEMM and once (the filler's gates) with V31_LTW=1 (cuBLASLt's GEMM at S = 1;
# variants take it as ltw=1); then, a lease each variant and L of FILL_LS (default "768 2048", t 3 and 8 at LP 256; 0: L =
# K; each with min(L, K) at most 8 LP, dither/PROTOCOL.md's "The segment and the noise period"),
# `v31 fill-bench` over FILL_SHAPES (default Llama 3 8B's QKV, O, gate-up and down at m 16384) beside cuBLASLt and
# CUTLASS; fill-sample on each shape's sampled tiles; fill-table. VARIANTS here default to "row=1024,cprio=hi,sched=1
# row=8192,cprio=hi,sched=1" (row: the leaf row, dither.Shape.row); served=1 benches the served call (O committed, T_S
# implied: dither.Shape(o=True); its params named served-*); with gemm=1 (a -DV31_WG build) its D comes from wg_proto.cuh's
# GEMM, rp=4 hashing its O rows in the GEMM's producer warpgroup (rp=5: and a hash warpgroup of its own, r5h its
# registers, and the prefix-sum rows in the GEMM of the prefix sums unless r5w=0; tst=1 its stores through shared memory
# with TMA), and the bench adds wgmma.linear's arm (wgl).
# FILL_GATE_KNOBS=1 runs the gates (and FILL_DITHER's recheck) under the first variant's knobs (else v31.cu's defaults). FILL_DITHER=1 also runs fill_dither.py under the gate's lease (the GPU against dither.py itself; the tree must have
# circuit/dither.py). FILL_NSYS=1 adds, before the bench, Nsight Systems on `v31 fill-once` (OVL_CALLS overlapped calls
# after two serial ones) for each variant, L and shape: out/nsys-TAG-L-SHAPE.nsys-rep, its kernel trace and `v31.py
# timeline`'s .timeline.md. FILL_PROF=1, after the bench, runs Nsight Compute's pipe utilization on `v31 fill-once`
# (OVL_CALLS overlapped calls, default 4) for the first variant on each shape and L, and h200_pipes.py's duty of each pipe
# over the overlapped call (pipes.md).
#
# HYPOTHESIS, the attempt's, goes to $OUT/hypothesis.txt; MODE=gates|bench|prof|nsys|fill (default bench); SHAPES: bench's NAME:M:N:K list
# (default D6's static shapes at m 8192); VARIANTS: bench's variants, each a comma-separated KEY=VALUE list over row (8192),
# cprio (lo), carve (100), dyn (1), wrap (2; 1 to 7), wcw (256), ccta (1), pf (0), diag (0), sched (0), ltw (0), served (0),
# gemm (0), wgbn (256), cm (0), wgg (16), rp (0), r5h (96), r5w (1), tst (0), hall (0), hchunk (0), wgw (1), lminb (5), sahead (0) (v31.cu's
# Knobs) and nreg (the GEMM's
# register cap, V31_MAXNREG or under V31_WG V31_WG_NREG, a build of its own; default v31.cu's), and the commit's full (1),
# sminb (4), lfold (0) and mask (0) (v31.cu's Knobs too), default row=8192; V31_ARMS, the fill bench's arms to time
# (default all); REPS
# (default 7); Q, the proof blocks a
# verdict opens (default 64); LP, T, ROW (the gates' default 256 3 1024);
# GENCODE (default arch=compute_120a,code=sm_120a; H200: arch=compute_90a,code=sm_90a); PY (default
# uv run -q --no-project --with numpy python: the node's python3 has no numpy); LEASE, the command that leases the GPU
# (default gpu-lease 1 --wait --max-min 25 --; a timed run passes "gpu-lease 8 --wait --timed --max-min 30 --" under
# `research run --no-sampler`; empty when already under a lease); LEASE_ONCE=1 (MODE=fill): its GPU steps all under one
# LEASE, after the builds and the CPU's prep, in place of a lease each; CUTLASS=0 (no CUTLASS baseline), V31_CACHE (CUTLASS's
# clone and object, default /workspace/pouw/accel_v31); NVFLAGS, more nvcc flags for every build (H200: -DV31_WG=1, the
# GEMM on warpgroup MMAs); SHA_FLAGS, sha512.cuh's issue choices (default -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1, tuned
# on sm_120).
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
# knob_env VARIANT: its knobs as v31.cu's environment, past those above rp 6 and 7 (v31.cu's ping-pong GEMM, each consumer
# warpgroup hashing O rows while the other runs its mainloop; 7: warps 9 to 11 hashing too), ppm (0), pph (1) and lord (0)
knob_env() {
  echo -n "V31_FULL=$(knob "$1" full 1) V31_SMINB=$(knob "$1" sminb 4) V31_LFOLD=$(knob "$1" lfold 0) V31_MASK=$(knob "$1" mask 0) "
  echo "V31_CPRIO=$(knob "$1" cprio lo) V31_CARVE=$(knob "$1" carve 100) V31_DYN=$(knob "$1" dyn 1) V31_WRAP=$(knob "$1" wrap 2)" \
    "V31_WCW=$(knob "$1" wcw 256) V31_CCTA=$(knob "$1" ccta 1) V31_PF=$(knob "$1" pf 0) V31_DIAG=$(knob "$1" diag 0)" \
    "V31_SCHED=$(knob "$1" sched 0) V31_LTW=$(knob "$1" ltw 0) V31_SERVED=$(knob "$1" served 0)" \
    "V31_GEMM=$(knob "$1" gemm 0) V31_WGBN=$(knob "$1" wgbn 256) V31_CM=$(knob "$1" cm 0) V31_WGG=$(knob "$1" wgg 16)" \
    "V31_RP=$(knob "$1" rp 0) V31_R5H=$(knob "$1" r5h 96) V31_HALL=$(knob "$1" hall 0) V31_HCHUNK=$(knob "$1" hchunk 0)" \
    "V31_WGW=$(knob "$1" wgw 1) V31_R5W=$(knob "$1" r5w 1) V31_TST=$(knob "$1" tst 0)" \
    "V31_LMINB=$(knob "$1" lminb 5) V31_SAHEAD=$(knob "$1" sahead 0) V31_PPM=$(knob "$1" ppm 0) V31_PPH=$(knob "$1" pph 1)" \
    "V31_LORD=$(knob "$1" lord 0)"
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
if [ "$MODE" = fill ]; then
  [ "$VARIANTS" = row=8192 ] && VARIANTS="row=1024,cprio=hi,sched=1 row=8192,cprio=hi,sched=1"
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
      $PY $D/v31.py fill-params "$W/fparams" $FILL_SHAPES --lp "$LP" --L "$L" --row "$(knob "$v" row 1024)" "${sv[@]}" \
        2>> "$O/prep.err"
      echo "fill-params $(knob "$v" row 1024)${sv[*]} L$L rc $?" >> "$O/rc.txt"
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
    env $(knob_env "$v") timeout 300 "$W/v31" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 1024)" "$s" > "$O/canary.log" 2>&1
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
              "$W/v31" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 1024)" "$s" > "$f.log" 2>&1
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
          "$(knob "$v" row 1024)" $FILL_SHAPES > "$O/fill-bench-$tag.jsonl" 2> "$O/fill-bench-$tag.err"
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
              "$W/v31${n:+-n$n}" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 1024)" "$s" > "$f.log" 2>&1
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
  exit
fi

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

# the device gates' exact check and the verifier's verdicts, on the CPU
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
elif [ "$MODE" = nsys ] && ! grep -qv ' rc 0$' "$O/rc.txt"; then
  NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
  export NSYS OVL_CALLS=${OVL_CALLS:-6}
  [ -n "$NSYS" ] || echo "nsys rc no-nsys" >> "$O/rc.txt"
  # shellcheck disable=SC2016,SC2086
  $LEASE bash -c '
    for v in $VARIANTS; do
      tag=$(tr ',' '-' <<< "$v" | tr -d '=')
      n=$(knob "$v" nreg "")
      for s in $SHAPES; do
        f="$O/nsys-$tag-${s%%:*}"
        env $(knob_env "$v") V31_ONCE_OVL="$OVL_CALLS" "$NSYS" profile -t cuda -s none --cpuctxsw none -f true -o "$f" \
          "$W/v31${n:+-n$n}" once "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" "$s" > "$f.log" 2>&1
        echo "nsys $tag ${s%%:*} rc $?" >> "$O/rc.txt"
      done
    done'
  echo "lease nsys rc $?" >> "$O/rc.txt"
  for f in "$O"/nsys-*.nsys-rep; do
    [ -f "$f" ] || continue
    "$NSYS" stats -r cuda_gpu_trace -f csv --force-export=true -o "${f%.nsys-rep}" "$f" >> "${f%.nsys-rep}.log" 2>&1
    $PY $D/v31.py timeline "${f%.nsys-rep}_cuda_gpu_trace.csv" > "${f%.nsys-rep}.timeline.md" 2>> "${f%.nsys-rep}.log"
    echo "timeline $(basename "${f%.nsys-rep}") rc $?" >> "$O/rc.txt"
  done
elif [ "$MODE" = duty ] && ! grep -qv ' rc 0$' "$O/rc.txt"; then
  NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
  export NCU OVL_CALLS=${OVL_CALLS:-6}
  [ -n "$NCU" ] || echo "duty rc no-ncu" >> "$O/rc.txt"
  # shellcheck disable=SC2016,SC2086
  $LEASE bash -c '
    for v in $VARIANTS; do
      tag=$(tr "," "-" <<< "$v" | tr -d "=")
      n=$(knob "$v" nreg "")
      for s in $SHAPES; do
        for r in ovl wgl; do
          f="$O/duty-$tag-${s%%:*}-$r"
          env $(knob_env "$v") V31_ONCE_OVL="$OVL_CALLS" V31_ONCE_RANGE=$r "$NCU" --clock-control none --replay-mode app-range \
            --metrics "$(tr -d " \n" < '"$D"'/duty.metrics)" -f -o "$f" \
            "$W/v31${n:+-n$n}" once "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" "$s" > "$f.log" 2>&1
          echo "duty $tag ${s%%:*} $r rc $?" >> "$O/rc.txt"
          "$NCU" -i "$f.ncu-rep" --csv --page raw --print-units base > "$f.csv" 2>> "$f.log"
        done
      done
    done'
  echo "lease duty rc $?" >> "$O/rc.txt"
  $PY $D/duty.py "$O"/duty-*.csv > "$O/duty.md" 2>> "$O/duty.err"
  echo "duty table rc $?" >> "$O/rc.txt"
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
