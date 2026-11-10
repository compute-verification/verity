#!/usr/bin/env bash
# A real-operand GEMM's tensor rate on one RTX PRO 6000 (sm_120) against the card's bound, and the tensor rate a co-resident
# undeclared kernel takes beside it (gemm_idle.cu). cwd: the source tree's root; writes $OUT (default the run dir's out/).
#
#   bash benchmarks/gemm_idle/run.sh build      # no GPU: gemm_idle into $GEMM_IDLE_CACHE/<key>/, with the PoUW harness's
#                                               # CUTLASS 2.x int8 and 3.x sm_120 FP8 tables (CUTLASS v4.8.0); a unit
#                                               # that fails to build is left out and said so in build.txt
#   research run --on vy-nebius-2 --project verity --source . --cwd source -- env \
#     CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by LANE -- \
#     gpu-lease 1 --wait --max-min 30 -- bash benchmarks/gemm_idle/run.sh measure  # builds first if the cache lacks the key
#
# measure, on the leased GPU at the node's clock lock (vy-clocks: -lgc 2100,2100): the register-resident mma.sync peak
# (zero and uniform operands, s8 and e4m3), the co-runner alone, then every LIBS GEMM at Llama-3.2-1B's and Llama-3.1-8B's
# linear shapes at each TS (tokens; 1 and 32 are decode), alone and beside the co-runner at each CO entry (W[:SLEEP]: W
# warps an SM, SLEEP ns after each batch), each also timed alone. Then, if HI_MHZ, the same GPU locked at HI_MHZ for the
# peaks and the 8B linears at T 8192 (uniform and gauss weights, beside HI_CO), and the node's lock put back
# (-lgc NODE_MHZ,NODE_MHZ; never -rgc, which would leave the GPU unlocked).
# Env: KINDS (s8 e4m3), TS (1 32 512 2048 8192), SECS (1), PEAK_SECS (6), CO (1:0 2:0 4:0 8:0), HI_CO (4:0),
# WCLASS (uniform), HI_MHZ (2430; 0 skips), NODE_MHZ (2100), MODELS (1b 8b).
# Writes passes.jsonl (gemm_idle's lines, each with its clock label), info.json, gpu.txt, smi.csv, rc.txt, build.txt,
# wgmma_sm120.txt (whether a wgmma compiles for sm_120a: the #6630 lead kernel's mainloop is wgmma) and summary.txt.
set -uo pipefail
D=benchmarks/gemm_idle
H=benchmarks/pouw/harness/native
O=${OUT:-${RESEARCH_RUN_DIR:?set OUT}/out}
mkdir -p "$O"
CUDA=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA/bin:$PATH CUDA_MODULE_LOADING=EAGER
CACHE=${GEMM_IDLE_CACHE:-/workspace/research/cache/gemm-idle}
CUTLASS_TAG=v4.8.0 CUTLASS_COMMIT=098de2a652cf8f00fd70b2df54051c7eccbb855a
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/steps.txt"; }

build() {
  mkdir -p "$CACHE"
  local cut=
  for c in /workspace/pouw/harness/build/cutlass-$CUTLASS_TAG "$CACHE/cutlass-$CUTLASS_TAG"; do
    [ "$(git -C "$c" rev-parse HEAD 2>/dev/null)" = "$CUTLASS_COMMIT" ] && { cut=$c; break; }
  done
  if [ -z "$cut" ]; then
    cut=$CACHE/cutlass-$CUTLASS_TAG
    (flock 9; [ -d "$cut/include/cutlass" ] || git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass \
      "$cut") 9> "$CACHE/.clone.lock"
    [ "$(git -C "$cut" rev-parse HEAD 2>/dev/null)" = "$CUTLASS_COMMIT" ] || { say "CUTLASS not at $CUTLASS_COMMIT"; cut=; }
  fi
  local key
  key=$(cat "$D/gemm_idle.cu" "$H/cutlass_2x.cu" "$H/cutlass_3x.cu" "$H/cutlass_api.h" <(nvcc --version) \
    <(echo "$CUTLASS_COMMIT $cut") | sha256sum | cut -c1-16)
  B=$CACHE/$key
  if [ -x "$B/gemm_idle" ]; then say "build $key cached"; cp "$B/build.txt" "$O/build.txt"; return 0; fi
  local S
  S=$(mktemp -d "$CACHE/tmp-XXXXXX")
  local F=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a --expt-relaxed-constexpr)
  local C=(-I"$cut/include" -I"$cut/tools/util/include" -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -I"$H")
  say "build $key (CUTLASS ${cut:-none})"
  if [ -n "$cut" ]; then
    nvcc "${F[@]}" "${C[@]}" -DPH_PART=0 -c "$H/cutlass_2x.cu" -o "$S/c2x.o" > "$S/c2x.log" 2>&1 &
    nvcc "${F[@]}" "${C[@]}" -DPH_PART=0 -c "$H/cutlass_3x.cu" -o "$S/c3x.o" > "$S/c3x.log" 2>&1 &
    wait
  fi
  local defs=() objs=()
  for u in c2x c3x; do
    if [ -s "$S/$u.o" ]; then defs+=("-DWITH_${u^^}"); objs+=("$S/$u.o"); echo "$u built" >> "$S/build.txt"
    else echo "$u not built: $(tail -c 2000 "$S/$u.log" 2>/dev/null)" >> "$S/build.txt"; fi
  done
  nvcc "${F[@]}" -Xptxas -v "${defs[@]}" -I"$H" "$D/gemm_idle.cu" "${objs[@]}" -o "$S/gemm_idle" -lcublasLt \
    -L"$CUDA/lib64/stubs" -lnvidia-ml > "$S/main.log" 2>&1 || { say "gemm_idle failed to build"; cat "$S/main.log" >> "$O/build.txt"; return 1; }
  grep -A2 "tensor_k" "$S/main.log" | grep -E "registers|Compiling" >> "$S/build.txt"
  { nvcc --version | tail -1; echo "cutlass $CUTLASS_TAG $CUTLASS_COMMIT $cut"; echo "key $key"; } >> "$S/build.txt"
  cp "$S/build.txt" "$O/build.txt"
  rm -rf "$B" && mv "$S" "$B"
  say "build $key done"
}

wgmma_probe() {
  local S
  S=$(mktemp -d)
  printf '__global__ void k() { asm volatile("wgmma.fence.sync.aligned;\\n"); }\n' > "$S/w.cu"
  for a in sm_90a sm_120a; do
    nvcc -gencode "arch=compute_${a#sm_},code=$a" -c "$S/w.cu" -o "$S/w.o" > "$S/log" 2>&1
    echo "$a rc $? $(tr '\n' ' ' < "$S/log")" >> "$O/wgmma_sm120.txt"
  done
  rm -rf "$S"
}

# pass LABEL ARGS...: one gemm_idle process, its lines tagged with the clock label
pass() {
  local l=$1
  shift
  timeout 300 "$BIN" "$@" 2>> "$O/err.txt" | sed "s/^{/{\"clock\": \"$l\", /" >> "$O/passes.jsonl"
  echo "$l $* rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
}

peaks() {
  for k in $KINDS; do
    for c in zero uniform; do for w in 8 16; do pass "$1" peak "$k" "$c" "$w" 8 "$PEAK_SECS"; done; done
    for e in $2; do
      local s=0
      [[ $e == *:* ]] && s=${e#*:}
      pass "$1" peak "$k" uniform "${e%%:*}" 8 "$SECS" "$s"
    done
  done
}

# Llama-3.2-1B (hidden 2048, MLP 8192, 32 q and 8 kv heads of 64) and Llama-3.1-8B (4096, 14336, 32 and 8 of 128):
# qkv, o, gate_up and down as N x K
SHAPES_1b="3072x2048 2048x2048 16384x2048 2048x8192"
SHAPES_8b="6144x4096 4096x4096 28672x4096 4096x14336"

measure() {
  U=${GPU_LEASE_UUID:-$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null)}
  [ -n "$U" ] || { say "no leased GPU"; return 1; }
  BIN=$B/gemm_idle
  nvidia-smi -i "$U" --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,temperature.gpu \
    --format=csv > "$O/gpu.txt" 2>&1
  nvidia-smi -i "$U" -q -d CLOCK,PERFORMANCE,POWER > "$O/clock-before.txt" 2>&1
  nvidia-smi -i "$U" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu \
    --format=csv,noheader -lms 200 > "$O/smi.csv" 2>&1 &
  local smi=$!
  "$BIN" info > "$O/info.json" 2>> "$O/err.txt"
  local node
  node=$(nvidia-smi -i "$U" --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')
  local L=node-$node
  say "measure on $U at the node's lock ($node MHz idle)"
  peaks "$L" "$CO"
  for k in $KINDS; do
    local libs="lt c2x"
    [ "$k" = e4m3 ] && libs="lt c3x"
    for lib in $libs; do
      for m in $MODELS; do
        local v=SHAPES_$m
        for s in ${!v}; do
          for t in $TS; do pass "$L" gemm "$lib" "$k" "$t" "${s%x*}" "${s#*x}" "$SECS" "$WCLASS" $CO; done
        done
      done
      say "$k $lib done"
    done
  done
  if [ "${HI_MHZ:-0}" != 0 ]; then
    trap 'sudo -n nvidia-smi -i "$U" -lgc "$NODE_MHZ,$NODE_MHZ" >> "$O/lock.txt" 2>&1' EXIT INT TERM
    sudo -n nvidia-smi -i "$U" -lgc "$HI_MHZ,$HI_MHZ" > "$O/lock.txt" 2>&1
    echo "lock $HI_MHZ rc $?" >> "$O/rc.txt"
    L=lock-$HI_MHZ
    say "measure at $L"
    peaks "$L" "$HI_CO"
    for k in $KINDS; do
      for wc in uniform gauss; do
        for s in $SHAPES_8b; do pass "$L" gemm lt "$k" 8192 "${s%x*}" "${s#*x}" "$SECS" "$wc" $HI_CO; done
      done
    done
    sudo -n nvidia-smi -i "$U" -lgc "$NODE_MHZ,$NODE_MHZ" >> "$O/lock.txt" 2>&1
    echo "lock back $NODE_MHZ rc $?" >> "$O/rc.txt"
    trap - EXIT INT TERM
  fi
  kill "$smi" 2>/dev/null
  nvidia-smi -i "$U" -q -d CLOCK,PERFORMANCE > "$O/clock-after.txt" 2>&1
  nvidia-smi -i "$U" --query-gpu=clocks.sm --format=csv,noheader >> "$O/rc.txt" 2>&1
  python3 "$D/summarize.py" "$O" > "$O/summary.txt" 2>&1
  say "measure done"
}

export KINDS=${KINDS:-s8 e4m3} TS=${TS:-1 32 512 2048 8192} SECS=${SECS:-1} PEAK_SECS=${PEAK_SECS:-6}
export CO=${CO:-1:0 2:0 4:0 8:0} HI_CO=${HI_CO:-4:0}
export WCLASS=${WCLASS:-uniform} HI_MHZ=${HI_MHZ:-2430} NODE_MHZ=${NODE_MHZ:-2100} MODELS=${MODELS:-1b 8b}
case "${1:-measure}" in
  build) build && wgmma_probe ;;
  measure) build && { wgmma_probe; measure; } ;;
  *) echo "usage: run.sh build|measure" >&2; exit 2 ;;
esac
