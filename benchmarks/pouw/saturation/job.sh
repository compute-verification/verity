#!/usr/bin/env bash
# CMP-8 (Daniel's #7 and #8): the saturation budget's measurements on one RTX PRO 6000 of vy-nebius-2, from the clone cwd.
#   STAGES (default "split micro ncu"):
#   split  lease 1: vllm_split.py on Llama-3.1-8B, {bf16, fp8} x {eager, graphs, compiled}: per-step kernel time by category
#          at decode batch 32 (context 1,024) and prefills of 2,048 and 8,192 tokens
#   micro  lease 2: fresh_operand.py gemm (torch._scaled_mm E4M3 n^3) and commit (hm96-sha512 rows of a fresh n x n E4M3
#          operand, gated against Python), n = 4,096 / 8,192 / 16,384
#   ncu    lease 2: one eager decode step (batch 32) per precision under ncu (pipe counts), summarized by ncu_summary.py
#   ncu_prefill (not default): the same for one eager 2,048-token prefill step
# Untimed diagnostic (tier 2a): one leased GPU, the fill queue keeps running beside it.
set -euo pipefail
: "${OUT:?}"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ROOT=$PWD
WORK=/workspace/jobs
PY=$WORK/venv312/bin/python
CUDA=/usr/local/cuda-13.0
NCU=$CUDA/bin/ncu
MODEL=${MODEL:-unsloth/Llama-3.1-8B}
REV=${REV:-3f0d51f8e5640f98f1a96ea9044a0e55c0a83814}
STAGES=${STAGES:-split micro ncu}
export HF_HOME=${HF_HOME:-/workspace/hf} CUDA_HOME=$CUDA PATH=$CUDA/bin:$WORK/venv312/bin:$PATH
export PYTHONPATH=$ROOT/integrations/vllm:$ROOT:$ROOT/tools/research/src${PYTHONPATH:+:$PYTHONPATH}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 TOKENIZERS_PARALLELISM=false
. "$ROOT/benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh"
S=$ROOT/benchmarks/pouw/saturation
stamp() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/stages.log"; }
git -C "$ROOT" rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || true

if [[ " $STAGES " == *" micro "* || " $STAGES " == *" ncu "* ]]; then
  stamp "build hm96_rows.cubin"
  nice -n 19 "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin -o "$d/hm96_rows.cubin" \
    "$ROOT/benchmarks/pouw/served_zk/hm96_rows.cu" 2> "$OUT/nvcc-rows.txt"
fi

split() {
  for cfg in "$@"; do
    set -- $cfg
    stamp "split $1 $2"
    if timeout 600 "$PY" "$S/vllm_split.py" --model "$MODEL" --revision "$REV" --precision "$1" --mode "$2" \
        --scratch "$d" --out "$OUT/split-$1-$2.json" > "$d/split-$1-$2.log" 2>&1; then
      stamp "split $1 $2 ok"
    else
      stamp "split $1 $2 FAILED rc=$?"
    fi
    grep -v "^\s*$" "$d/split-$1-$2.log" | tail -60 > "$OUT/split-$1-$2.log.tail"
  done
}

micro() {
  stamp "micro gemm"
  "$PY" "$S/fresh_operand.py" gemm --out "$OUT/fresh-gemm.json" > "$OUT/fresh-gemm.log" 2>&1 || stamp "micro gemm FAILED"
  stamp "micro commit"
  PEARLC_EXPECT_UUID=$GPU_LEASE_UUID "$PY" "$S/fresh_operand.py" commit --rows-cubin "$d/hm96_rows.cubin" \
    --out "$OUT/fresh-commit.json" > "$OUT/fresh-commit.log" 2>&1 || stamp "micro commit FAILED rc=$?"
}

METRICS=gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__thread_inst_executed.sum,sm__inst_executed.sum
METRICS+=,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_xu.sum
METRICS+=,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_uniform.sum
for p in sm__inst_executed_pipe_fma sm__inst_executed_pipe_alu sm__inst_executed_pipe_xu sm__pipe_fma_cycles_active \
         sm__pipe_alu_cycles_active sm__inst_issued sm__inst_executed_pipe_tensor dram__throughput sm__throughput; do
  METRICS+=,$p.avg.pct_of_peak_sustained_elapsed
done

ncu_stage() {
  for cfg in "$@"; do
    set -- $cfg
    local prec=$1 phase=$2 name=ncu-$1-$2 step=--ncu-step
    [[ $phase == prefill* ]] && step="--ncu-prefill ${phase#prefill}"
    stamp "ncu $prec $phase"
    if timeout 720 "$NCU" --target-processes all --nvtx --nvtx-include "ncu_step/" --clock-control none \
        --metrics "$METRICS" --export "$d/$name" --force-overwrite \
        "$PY" "$S/vllm_split.py" --model "$MODEL" --revision "$REV" --precision "$prec" --mode eager $step \
        --out "$OUT/$name-run.json" > "$d/$name.log" 2>&1; then
      stamp "ncu $prec $phase ok"
    else
      stamp "ncu $prec $phase FAILED rc=$?"
    fi
    grep -v "^\s*$" "$d/$name.log" | tail -40 > "$OUT/$name.log.tail"
  done
}

leased() {
  stamp "lease for: $*"
  gpu-lease 1 --wait --max-min 30 -- bash -c "$(declare -f stamp split micro ncu_stage); $(declare -p OUT d S PY MODEL REV NCU METRICS); $*" \
    || stamp "lease for '$*' FAILED rc=$?"
}
[[ " $STAGES " == *" split "* ]] && leased split "'bf16 eager'" "'fp8 eager'" "'bf16 graphs'" "'fp8 graphs'"
[[ " $STAGES " == *" split "* ]] && leased split "'fp8 compiled'" "'bf16 compiled'"
[[ " $STAGES " == *" micro "* ]] && leased micro
[[ " $STAGES " == *" ncu "* ]] && leased ncu_stage "'bf16 decode'" "'fp8 decode'"
[[ " $STAGES " == *" ncu_prefill "* ]] && leased ncu_stage "'fp8 prefill2048'" "'bf16 prefill2048'"

for rep in "$d"/ncu-*.ncu-rep; do
  [ -f "$rep" ] || continue
  name=$(basename "$rep" .ncu-rep)
  "$NCU" --import "$rep" --csv --page raw > "$d/$name-raw.csv"
  gzip -c "$d/$name-raw.csv" > "$OUT/$name-raw.csv.gz"
  "$PY" "$S/ncu_summary.py" "$d/$name-raw.csv" --out "$OUT/$name-summary.json" > "$OUT/$name-summary.log" 2>&1 \
    || stamp "ncu_summary $name FAILED"
done
stamp "done"
