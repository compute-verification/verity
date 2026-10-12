#!/usr/bin/env bash
# Llama-3.1-8B or 70B decode in vLLM on one GPU with its linear weights stored only encoded, against the same engine with plain FP8 weights
# (cwd = the source tree).  Builds ve.cu outside the lease, runs the layout tests, then under one gpu-lease 1 --wait
# --max-min MAXMIN (default 25) one serve.py process per arm, and compare.py's gates and points.  Results go to
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-vllm-encoded \
#     -- taskset -c 48-95 bash benchmarks/vllm_encoded/run.sh
#
# WORK is the e2e's vLLM environment on node 2 (benchmarks/pouw/pearl_c_vllm/setup.sh).  ARMS (default
# "stock plain enc:2 enc:3 stock") are serve.py's arms in order, an encoded arm as enc:K or enc:K:LIB (LIB pb56 or pb60),
# ra-codec-wide's fused GEMM as wide:BITS:K or wide:BITS:K:tag (BITS 127, 255 or 521; K = 0 its plain payloads; tag: the
# ro-blake3/v1 tag of every block read, as decode runs), ra-codec-tensor's as tensor:FIELD:K or tensor:FIELD:K:tag (FIELD
# 255, 511, 512, 521, 767, 768, 1023 or 1279, codec_tensor/wt.cu built for each the arms name; K >= 1, 512 and 768 being
# full-slot fields at K = 1; tag: its tag warps tag the
# store's 4,088-byte blocks), any arm with +c compiled as vLLM serves by default (serve.py
# --compiled) or +d compiled without any choice made by timing (--no-autotune, the compiled gate); wide.cu is built for
# each BITS the arms name (WW_KMAX=${WW_KMAX:-2}, the PTX-carry squaring); a last /g makes an arm its gates and one
# point (batch 1, context 512, 16 steps); SERVE_ARGS passes through to every arm; ARM_MIN bounds one arm (default 7 minutes).
# MODEL (default the e2e's Llama-3.1-8B-Instruct; for 70B the hub snapshot of unsloth/Meta-Llama-3.1-70B with SERVE_ARGS
# "--mem 0.93") is read into the page cache before the lease, so each arm loads it from memory if the run's --mem-gb holds
# it (192 for 70B's 132 GB of BF16).  MODE=selftest runs selftest.py
# instead (no vLLM engine): its --check one launch at a time (under compute-sanitizer when SANITIZE=1), then its --time,
# with SELFTEST_ARGS; SELFTEST_PY (default the pous benchmark's torch 2.14 cu130 venv) runs the check a second time.
# MODE=profile runs the arms (default "stock enc:2") under Nsight Compute, each point's profiler window over two decode
# steps, and traffic.py's measured HBM bytes a step instead of compare.py (its tokens/s are not timings).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || { echo "no $WORK/ready.json: run benchmarks/pouw/pearl_c_vllm/setup.sh"; exit 3; }
export PATH=/usr/local/cuda-13.0/bin:$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf}
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export PYTHONPATH=$PWD:$PWD/integrations/vllm CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
VE=benchmarks/vllm_encoded
B=$(mktemp -d /tmp/vllm-encoded.XXXXXX); trap 'rm -rf "$B"' EXIT
CU=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
FLAGS=("${CU[@]}" -I benchmarks/word_codec -DWC_KMAX="${WC_KMAX:-3}")
declare -A DEFS=([pb56]="-DWC_PB=56" [pb60]="-DWC_PB=60")
for lib in ${LIBSET:-pb56 pb60}; do
  # shellcheck disable=SC2086
  nvcc "${FLAGS[@]}" ${DEFS[$lib]} -o "$B/ve-$lib.so" $VE/ve.cu > "$O/ve-$lib.ptxas" 2>&1 &
done
[ "${MODE:-}" = profile ] && ARMS=${ARMS:-stock enc:2}
ARMS=${ARMS:-stock plain enc:2 enc:3 stock}
WIDE=$(for a in $ARMS; do case $a in wide:*) a=${a#wide:}; echo "${a%%:*}" ;; esac; done | sort -u)
for bits in $WIDE; do
  nvcc "${FLAGS[@]}" -I benchmarks/bandwidth -DWW_BITS="$bits" -DWW_SQR=1 -DWW_KMAX="${WW_KMAX:-2}" -o "$B/ww-p$bits.so" \
    benchmarks/word_codec/wide.cu > "$O/ww-p$bits.ptxas" 2>&1 &
done
TENSOR=$(for a in $ARMS; do case $a in tensor:*) a=${a#tensor:}; echo "${a%%:*}" ;; esac; done | sort -u)
for n in $TENSOR; do
  nvcc "${CU[@]}" -DWT_FIELD="$n" -o "$B/wt-$n.so" benchmarks/codec_tensor/wt.cu > "$O/wt-$n.ptxas" 2>&1 &
done
cat "$MODEL"/*.safetensors > /dev/null &
wait || true
for lib in ${LIBSET:-pb56 pb60}; do [ -e "$B/ve-$lib.so" ] || { tail -30 "$O/ve-$lib.ptxas"; exit 4; }; done
for bits in $WIDE; do [ -e "$B/ww-p$bits.so" ] || { tail -30 "$O/ww-p$bits.ptxas"; exit 4; }; done
for n in $TENSOR; do [ -e "$B/wt-$n.so" ] || { tail -30 "$O/wt-$n.ptxas"; exit 4; }; done
sha256sum "$B"/*.so > "$O/libs.sha256"
(cd $VE && "$PY" -m pytest -q -p no:cacheprovider tests) > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ("model", "revision", "vllm", "torch")}))' \
  "$WORK/ready.json" > "$O/ready.json"
export NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu} NCU_ARGS=
if [ "${MODE:-}" = profile ]; then
  SERVE_ARGS=${SERVE_ARGS:---profile-steps 2 --steps 16} ARM_MIN=${ARM_MIN:-10}
  NCU_ARGS="--profile-from-start off --graph-profiling graph --cache-control none --clock-control none"
  NCU_ARGS+=" --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum"
fi
export B PY MODEL VE ARMS SERVE_ARGS=${SERVE_ARGS:-} ARM_MIN=${ARM_MIN:-7}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
if [ "${MODE:-}" = selftest ]; then
  export SELFTEST_ARGS=${SELFTEST_ARGS:-} SANITIZE=${SANITIZE:-0}
  export SELFTEST_PY=${SELFTEST_PY:-/workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python}
  # shellcheck disable=SC2016
  "${LEASE[@]}" bash -c '
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
    lib=$B/ve-${LIB:-pb56}.so
    S=(); [ "$SANITIZE" = 1 ] && S=(/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --print-limit 20)
    # shellcheck disable=SC2086
    CUDA_LAUNCH_BLOCKING=1 timeout 600 "${S[@]}" "$PY" $VE/selftest.py --lib "$lib" --check --out "$O/check.json" $SELFTEST_ARGS \
      > "$O/check.log" 2>&1; rc=$?; echo "check $rc" >> "$O/arms.rc"
    if [ $rc != 0 ] && [ "$SANITIZE" != 1 ]; then
      timeout 600 /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --print-limit 20 "$PY" $VE/selftest.py --lib "$lib" \
        --check --shapes o --batches 1 --ks 0 --out "$O/check-sanitized.json" > "$O/check-sanitized.log" 2>&1
      echo "check-sanitized $?" >> "$O/arms.rc"
    fi
    if [ -x "$SELFTEST_PY" ]; then
      # shellcheck disable=SC2086
      CUDA_LAUNCH_BLOCKING=1 timeout 600 "$SELFTEST_PY" $VE/selftest.py --lib "$lib" --check --ks 0 --batches 1,32 \
        --out "$O/check-cu130.json" > "$O/check-cu130.log" 2>&1; echo "check-cu130 $?" >> "$O/arms.rc"
    fi
    # shellcheck disable=SC2086
    timeout 600 "$PY" $VE/selftest.py --lib "$lib" --time --out "$O/time.json" $SELFTEST_ARGS > "$O/time.log" 2>&1
    echo "time $?" >> "$O/arms.rc"'
  cat "$O/arms.rc"; tail -5 "$O/check.log"
  exit 0
fi
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  UUID=${GPU_LEASE_UUID%%,*}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,memory.used \
          --format=csv,noheader -i "$UUID" | sed "s/^/$1, /" >> "$O/gpu.txt" 2>&1; }
  i=0
  for arm in $ARMS; do
    i=$((i + 1)); C=(); c=; G=(); g=
    case $arm in */g) G=(--batches 1 --contexts 512 --steps 16); g=-g; arm=${arm%/g} ;; esac
    case $arm in
      *+c) C=(--compiled); c=-c; arm=${arm%+c} ;;
      *+d) C=(--compiled --no-autotune); c=-d; arm=${arm%+d} ;;
    esac
    IFS=: read -r name k lib <<< "$arm"
    case $name in
      enc) A=(--arm enc --k "$k" --lib "$B/ve-${lib:-pb56}.so"); tag=enc$k${lib:+-$lib} ;;
      wide) A=(--arm wide --k "${lib%%:*}" --lib "$B/ve-pb56.so" --wide-lib "$B/ww-p$k.so"); tag=w$k-${lib%%:*}
            case $lib in *:tag) A+=(--tag); tag+=t ;; esac ;;
      tensor) A=(--arm tensor --k "${lib%%:*}" --lib "$B/ve-pb56.so" --tensor-lib "$B/wt-$k.so" --gate-reps 3)
              tag=t$k-${lib%%:*}
              case $lib in *:tag) A+=(--tag); tag+=t ;; esac ;;
      plain) A=(--arm plain --lib "$B/ve-${k:-pb56}.so"); tag=plain${k:+-$k} ;;
      *) A=(--arm "$name"); tag=$name ;;
    esac
    A+=("${C[@]}" "${G[@]}"); tag=$i-$tag$c$g; q "$tag start"
    # shellcheck disable=SC2206
    P=(); [ -n "$NCU_ARGS" ] && P=("$NCU_BIN" $NCU_ARGS -f -o "$O/ncu-$tag")
    # shellcheck disable=SC2086
    timeout "$((ARM_MIN * 60))" "${P[@]}" "$PY" $VE/serve.py "${A[@]}" --model "$MODEL" --out "$O/arm-$tag.json" $SERVE_ARGS \
      > "$O/arm-$tag.log" 2>&1
    echo "$tag $?" >> "$O/arms.rc"; q "$tag end"
  done'
if [ "${MODE:-}" = profile ]; then
  for r in "$O"/ncu-*.ncu-rep; do
    "$NCU_BIN" --import "$r" --csv --page raw --print-units base > "${r%.ncu-rep}.csv" 2>&1 && rm -f "$r"
  done
  (cd $VE && "$PY" traffic.py "$O") > "$O/traffic.log" 2>&1; rc=$?
  cat "$O/arms.rc" "$O/traffic.log"
  exit $rc
fi
"$PY" $VE/compare.py "$O" > "$O/compare.log" 2>&1; rc=$?
cat "$O/arms.rc" "$O/compare.log"
exit $rc
