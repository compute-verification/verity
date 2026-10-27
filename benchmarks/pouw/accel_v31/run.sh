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
# MODE=nsys: after the gates, Nsight Systems on `v31 once` with OVL_CALLS (default 6) overlapped calls after its two
# serial ones, on each of SHAPES for each of VARIANTS: out/nsys-TAG-SHAPE.nsys-rep, its kernel trace as CSV, and
# `v31.py timeline`'s summary of it (.timeline.md: each kernel serial and overlapped, the steady period, the streams' overlap).
#
# MODE=fill: Dither's no-fold call (CMP-172) in place of all the above. libv31.so (v31.cu -DV31_LIB) and, under a lease,
# v31lib.py's gate (the binding's calls: the GPU's pad, values and tree against v31.py's Fill, then the verifier's verdicts
# on the GPU's own openings, honest and cheating, filler and served, a filler's and a served Prover interleaved;
# verdicts.txt), once with the protocol GEMM (ltw0), once (the filler's gates) with V31_LTW=1 (cuBLASLt's GEMM at S = 1;
# variants take it as ltw=1) and once (the served gates and the pipeline, c3) at three copies with the leaf phases on
# their own streams (V31_COPIES=3 V31_LSTR=1, sched 1, pf 2, ccta 3); then, a lease each variant and L of FILL_LS (default "768 4096 8192 0", 0: L = K),
# `v31 fill-bench` over FILL_SHAPES (default Llama 3 8B's QKV, O, gate-up and down at m 16384) beside cuBLASLt and
# CUTLASS; fill-sample on each shape's sampled tiles; fill-table. VARIANTS here default to "row=1024,cprio=hi,sched=1
# row=8192,cprio=hi,sched=1" (row: the leaf row, dither.Shape.row); served=1 benches the served call (O committed, T_S
# implied: dither.Shape(o=True); its params named served-*).
# FILL_NSYS=1 adds, before the bench, Nsight Systems on `v31 fill-once` (OVL_CALLS overlapped calls after two serial ones)
# for each variant, L and shape: out/nsys-TAG-L-SHAPE.nsys-rep, its kernel trace and `v31.py timeline`'s .timeline.md.
# MODE=contend: no gates; `v31 contend` (cuBLASLt's GEMM beside a filler kernel issuing SHA-512 on registers, integer ALU,
# FP32 FMA or nothing, at 1 to 3 CTAs an SM: what each costs the other) over CONTEND_SHAPES (default 8B O, 8B gate-up and
# 70B O at m 16384), REPS each, once per CONTEND_CARVE (the fillers' V31_CARVE; default "-1 100": the driver's carveout,
# then the GEMM's); CONTEND_GEMMS (default lt: cuBLASLt's pick; else CUTLASS entry names too) and CONTEND_MODES (default
# "sha alu fma sleep") choose the GEMMs and fillers; out/contend.jsonl and `v31.py contend-table`'s table.md.
# FILL_PROF=1 adds, before the bench, Nsight Compute (--set full) on `v31 fill-once`'s serial call for each variant, L and
# shape: the first PROF_COUNT (default 3) launches matching PROF_KERNELS (default cuBLASLt's GEMM, salt_persist and
# leaf_persist), each alone: out/ncu-TAG-L-SHAPE.ncu-rep and its details and raw pages as CSV (pipes, issue, stalls).
#
# HYPOTHESIS, the attempt's, goes to $OUT/hypothesis.txt; MODE=gates|bench|prof|nsys|fill|contend (default bench); SHAPES: bench's NAME:M:N:K list
# (default D6's static shapes at m 8192); VARIANTS: bench's variants, each a comma-separated KEY=VALUE list over row (8192),
# cprio (lo), carve (100), dyn (1), wrap (2; 1 to 5), wcw (256), ccta (1), pf (0), diag (0), sched (0), copies (2), lstr (0)
# (v31.cu's Knobs), nreg (the GEMM's register cap) and rot (sha512.cuh's SHA_ROT, the sigmas rotating on the FMA pipe),
# each a build of its own (default v31.cu's, 0), default row=8192; REPS (default
# 7); Q, the proof blocks a verdict opens (default 64); LP, T, ROW (the gates' default 256 3 1024);
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
    "V31_WCW=$(knob "$1" wcw 256) V31_CCTA=$(knob "$1" ccta 1) V31_PF=$(knob "$1" pf 0) V31_DIAG=$(knob "$1" diag 0)" \
    "V31_SCHED=$(knob "$1" sched 0) V31_LTW=$(knob "$1" ltw 0) V31_SERVED=$(knob "$1" served 0)" \
    "V31_COPIES=$(knob "$1" copies 2) V31_LSTR=$(knob "$1" lstr 0)"
}
# bin VARIANT: the v31 build it runs
bin() { local n r; n=$(knob "$1" nreg ""); r=$(knob "$1" rot ""); echo "$W/v31${n:+-n$n}${r:+-r$r}"; }
export -f knob knob_env bin

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
build() {   # BIN LOG SHA_ROT [nvcc flags]
  nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT="$3" -DSHA_ROLL=1 "${CUT[@]}" "${@:4}" -o "$1" \
    $D/v31.cu -lcuda -lcublasLt > "$2" 2>&1
}
build "$W/v31" "$O/build.log" 0
echo "build rc $?" >> "$O/rc.txt"
[ -x "$W/v31" ] || { cat "$O/rc.txt"; tail -20 "$O/build.log"; exit 1; }
for v in $VARIANTS; do
  b=$(bin "$v") n=$(knob "$v" nreg "")
  [ -e "$b" ] && continue
  build "$b" "$O/build${b#"$W/v31"}.log" "$(knob "$v" rot 0)" ${n:+-DV31_MAXNREG="$n"}
  echo "build ${b#"$W/v31-"} rc $?" >> "$O/rc.txt"
done

LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
if [ "$MODE" = fill ]; then
  [ "$VARIANTS" = row=8192 ] && VARIANTS="row=1024,cprio=hi,sched=1 row=8192,cprio=hi,sched=1"
  FILL_LS=${FILL_LS:-"768 4096 8192 0"}
  FILL_SHAPES=${FILL_SHAPES:-"8B-qkv:16384:6144:4096 8B-o:16384:4096:4096 8B-up:16384:28672:4096 8B-down:16384:4096:14336"}
  export VARIANTS FILL_SHAPES FILL_LS LP REPS Q D PY
  nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 -DV31_LIB -DCOMMIT_GPU_DOMS=8 \
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
  # shellcheck disable=SC2016
  $LEASE bash -c '
    nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv \
      ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
    for x in ltw0 ltw1 c3; do
      case $x in
        ltw0) only=(); ev=(V31_LTW=0) ;;
        ltw1) only=(--only fill); ev=(V31_LTW=1) ;;
        c3) only=(--only served); ev=(V31_COPIES=3 V31_LSTR=1 V31_SCHED=1 V31_PF=2 V31_CCTA=3) ;;
      esac
      env "${ev[@]}" $PY $D/v31lib.py gate --lib "$W/libv31.so" --out "$O/fill-gates-$x" --q "$Q" "${only[@]}" \
        > "$O/fill-gates-$x.log" 2> "$O/fill-gates-$x.err"
      echo "fill gate $x rc $?" >> "$O/rc.txt"
    done'
  echo "lease fill gate rc $?" >> "$O/rc.txt"
  for x in ltw0 ltw1 c3; do grep -E '^[a-z].*(ACCEPT|REFUSE)' "$O/fill-gates-$x.log" | sed "s/^/$x /" >> "$O/verdicts.txt"; done
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
              "$(bin "$v")" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 1024)" "$s" > "$f.log" 2>&1
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
  if [ "${FILL_PROF:-0}" = 1 ]; then
    NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
    export NCU PROF_KERNELS=${PROF_KERNELS:-cutlass|salt_persist|leaf_persist} PROF_COUNT=${PROF_COUNT:-3}
    [ -n "$NCU" ] || echo "prof rc no-ncu" >> "$O/rc.txt"
    # shellcheck disable=SC2016
    $LEASE bash -c '
      for v in $VARIANTS; do
        for L in $FILL_LS; do
          tag=$(tr "," "-" <<< "$v" | tr -d "=")-L$L
          for s in $FILL_SHAPES; do
            f="$O/ncu-$tag-${s%%:*}"
            # shellcheck disable=SC2046
            env $(knob_env "$v") V31_ONCE_OVL=0 "$NCU" --clock-control none --kernel-name-base demangled \
              -k "regex:$PROF_KERNELS" -c "$PROF_COUNT" --set full -f -o "$f" \
              "$(bin "$v")" fill-once "$W/fparams" "$LP" "$L" "$(knob "$v" row 1024)" "$s" > "$f.log" 2>&1
            echo "prof $tag ${s%%:*} rc $?" >> "$O/rc.txt"
            "$NCU" -i "$f.ncu-rep" --csv --page details > "$f.csv" 2>> "$f.log"
            "$NCU" -i "$f.ncu-rep" --csv --page raw > "$f.raw.csv" 2>> "$f.log"
          done
        done
      done'
    echo "lease prof rc $?" >> "$O/rc.txt"
  fi
  # the bench runs whatever the gate's verdicts (its own checked call is sampled below); cite it only beside a passed gate
  for v in $VARIANTS; do
    for L in $FILL_LS; do
      export v L
      # shellcheck disable=SC2016
      $LEASE bash -c '
        tag=$(tr "," "-" <<< "$v" | tr -d "=")-L$L
        nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 \
          ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi-$tag.csv" 2>&1 &
        smi=$!
        mkdir -p "$W/fbench/$tag"
        # shellcheck disable=SC2046,SC2086
        env $(knob_env "$v") "$(bin "$v")" fill-bench "$REPS" "$W/fbench/$tag" "$W/fparams" "$LP" "$L" \
          "$(knob "$v" row 1024)" $FILL_SHAPES > "$O/fill-bench-$tag.jsonl" 2> "$O/fill-bench-$tag.err"
        echo "fill-bench $tag rc $?" >> "$O/rc.txt"
        kill $smi'
      echo "lease fill-bench $v L$L rc $?" >> "$O/rc.txt"
    done
  done
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
if [ "$MODE" = contend ]; then
  export SHAPES=${CONTEND_SHAPES:-"8B-o:16384:4096:4096 8B-up:16384:28672:4096 70B-o:16384:8192:8192"} REPS
  export CONTEND_CARVE=${CONTEND_CARVE:-"-1 100"} V31_CGEMMS=${CONTEND_GEMMS:-lt} V31_CMODES=${CONTEND_MODES:-"sha alu fma sleep"}
  # shellcheck disable=SC2016,SC2086
  $LEASE bash -c '
    for c in $CONTEND_CARVE; do
      V31_CARVE=$c "$W/v31" contend "$REPS" $SHAPES >> "$O/contend.jsonl" 2>> "$O/contend.err"
      echo "contend carve $c rc $?" >> "$O/rc.txt"
    done'
  echo "lease contend rc $?" >> "$O/rc.txt"
  $PY $D/v31.py contend-table "$O/contend.jsonl" > "$O/table.md" 2>> "$O/table.err"
  echo "contend-table rc $?" >> "$O/rc.txt"
  cat "$O/rc.txt" "$O/table.md"
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
      mkdir -p "$W/bench/$tag"
      env $(knob_env "$v") "$(bin "$v")" bench "$REPS" "$W/bench/$tag" "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" \
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
      for s in $SHAPES; do
        f="$O/ncu-$tag-${s%%:*}"
        env $(knob_env "$v") "$NCU" --clock-control none -k "regex:$PROF_KERNELS" -c "$PROF_COUNT" --set full -f -o "$f" \
          "$(bin "$v")" once "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" "$s" > "$f.log" 2>&1
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
      for s in $SHAPES; do
        f="$O/nsys-$tag-${s%%:*}"
        env $(knob_env "$v") V31_ONCE_OVL="$OVL_CALLS" "$NSYS" profile -t cuda -s none --cpuctxsw none -f true -o "$f" \
          "$(bin "$v")" once "$W/params" "$LP" "$T" "$(knob "$v" row 8192)" "$s" > "$f.log" 2>&1
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
