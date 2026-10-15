#!/usr/bin/env bash
# CMP-8 (Daniel's #7 and #8): the saturation budget's measurements on one RTX PRO 6000 of vy-nebius-2, from the clone cwd.
#   STAGES (default "split micro ncu"):
#   split  lease 1: vllm_split.py on Llama-3.1-8B, {bf16, fp8} x {eager, graphs, compiled}: per-step kernel time by category
#          at decode batch 32 (context 1,024) and prefills of 2,048 and 8,192 tokens
#   micro  lease 2: fresh_operand.py gemm (torch._scaled_mm E4M3 n^3) and commit (hm96-sha512 rows of a fresh n x n E4M3
#          operand, gated against Python), n = 4,096 / 8,192 / 16,384
#   ncu    lease 2: one eager decode step (batch 32) per precision under ncu (pipe counts), summarized by ncu_summary.py;
#          its first 200 kernels (every kernel kind, ~13 layers), with a 6 GiB KV cache so kernel replay's memory backup
#          stays on the device (at 0.85 it goes to host memory, ~40 s a kernel: r20261015-042637-c9be)
#   ncu_prefill (not default): the same for one eager 2,048-token prefill step; ncu_prefill8k: FP8 at 8,192 tokens
#   phi    (not default) lease 1: phi_probe.cu, a keyed ARX receipt step over register-resident E4M3 mma.sync accumulators
#          at commit strides L = 32 ... 256 (gated), at 2 and 4 blocks an SM; its SASS and an opcode count per kernel; ncu's
#          warp-state, scheduler and pipe sections over seven of its kernels
#   isa    (not default) lease 1: isa_rates.cu, the throughput of single CUDA-core instructions (FFMA, IADD3, LOP3, IMAD,
#          IDP4A, HFMA2, PRMT, the E4M3 decode, shared-memory lookups) alone and in pairs (gated), at 2 and 4 blocks an SM;
#          its SASS and an opcode count per kernel
#   rgemm  (not default) lease 1: receipt_gemm.cu, Pearl-C's 128 x 128 TMA mainloop with the phi receipt over every total
#          word at L = 128, 256, 512 (receipts after the window's FADDs, or spread across its MMAs) against its receipt-free
#          twin, n = 8,192 and 16,384 (gated); its ptxas report, SASS and opcode counts; ncu's pipe, scheduler and warp-state
#          sections over eight of its kernels at 8,192
# Timeouts run in the foreground so `research cancel` reaches the GPU process.
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
probe_build() {  # NAME SOURCE [NVCC FLAGS]: the binary in $d, its SASS and the opcode count of each kernel in $OUT
  stamp "build $1"
  nice -n 19 "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a "${@:3}" -o "$d/$1" "$S/$2" \
    2> "$OUT/nvcc-$1.txt"
  "$CUDA/bin/cuobjdump" -sass "$d/$1" > "$d/$1-sass.txt"
  gzip -c "$d/$1-sass.txt" > "$OUT/$1-sass.txt.gz"
  awk '/Function :/ { f = $3; next }
       match($0, /\/\*[0-9a-f][0-9a-f][0-9a-f][0-9a-f]+\*\/[ \t]+(@!?U?P[0-9T][ \t]+)?[A-Z][A-Z0-9_]*/) {
         n = split(substr($0, RSTART, RLENGTH), a, /[ \t]+/); c[f "\t" a[n]]++ }
       END { for (k in c) print k "\t" c[k] }' "$d/$1-sass.txt" | sort > "$OUT/$1-sass-opcodes.tsv"
}
[[ " $STAGES " == *" phi "* ]] && probe_build phi_probe phi_probe.cu
[[ " $STAGES " == *" isa "* ]] && probe_build isa_rates isa_rates.cu
[[ " $STAGES " == *" rgemm "* ]] && probe_build receipt_gemm receipt_gemm.cu -Xptxas -v

split() {
  for cfg in "$@"; do
    set -- $cfg
    stamp "split $1 $2"
    if timeout --foreground 600 "$PY" "$S/vllm_split.py" --model "$MODEL" --revision "$REV" --precision "$1" --mode "$2" \
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

METRICS=gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__thread_inst_executed.sum
for p in sm__inst_executed_pipe_fma sm__inst_executed_pipe_alu sm__pipe_fma_cycles_active sm__pipe_alu_cycles_active \
         sm__inst_issued sm__inst_executed_pipe_tensor dram__throughput; do
  METRICS+=,$p.avg.pct_of_peak_sustained_elapsed
done

ncu_stage() {
  for cfg in "$@"; do
    set -- $cfg
    local prec=$1 phase=$2 name=ncu-$1-$2 step=--ncu-step
    [[ $phase == prefill* ]] && step="--ncu-prefill ${phase#prefill}"
    stamp "ncu $prec $phase"
    if timeout --foreground 840 "$NCU" --target-processes all --nvtx --nvtx-include "ncu_step/" --clock-control none \
        --launch-count 200 --metrics "$METRICS" --export "$d/$name" --force-overwrite \
        "$PY" "$S/vllm_split.py" --model "$MODEL" --revision "$REV" --precision "$prec" --mode eager $step \
        --kv-cache-gib 6 --out "$OUT/$name-run.json" > "$d/$name.log" 2>&1; then
      stamp "ncu $prec $phase ok"
    else
      stamp "ncu $prec $phase FAILED rc=$?"
    fi
    grep -v "^\s*$" "$d/$name.log" | tail -40 > "$OUT/$name.log.tail"
  done
}

phi() {
  for blocks in 2 4; do
    stamp "phi probe, $blocks blocks an SM"
    if timeout --foreground 300 "$d/phi_probe" --ms 60 --reps 5 --blocks $blocks > "$OUT/phi-probe-b$blocks.json" \
        2> "$OUT/phi-probe-b$blocks.err"; then
      stamp "phi probe b$blocks ok"
    else
      stamp "phi probe b$blocks FAILED rc=$?"
    fi
  done
  local pick=mode0_per1_r1_alu,mode1_per1_r1_alu,mode2_per1_r1_alu,mode2_per2_r1_alu,mode2_per3_r2_alu,mode4_per2_r1_alu
  pick+=,mode4_per3_r2_alu
  stamp "phi ncu"
  if timeout --foreground 600 "$NCU" --clock-control none --section LaunchStats --section WarpStateStats \
      --section SchedulerStats --section ComputeWorkloadAnalysis --export "$d/phi-ncu" --force-overwrite \
      "$d/phi_probe" --ms 5 --reps 1 --only $pick > "$d/phi-ncu.log" 2>&1; then
    "$NCU" --import "$d/phi-ncu.ncu-rep" --csv --page details > "$OUT/phi-ncu-details.csv"
    "$NCU" --import "$d/phi-ncu.ncu-rep" --csv --page raw | gzip -c > "$OUT/phi-ncu-raw.csv.gz"
    stamp "phi ncu ok"
  else
    stamp "phi ncu FAILED rc=$?"
  fi
  grep -v "^\s*$" "$d/phi-ncu.log" | tail -40 > "$OUT/phi-ncu.log.tail"
}

isa() {
  for blocks in 2 4; do
    stamp "isa rates, $blocks blocks an SM"
    if timeout --foreground 300 "$d/isa_rates" --ms 60 --reps 5 --blocks $blocks > "$OUT/isa-rates-b$blocks.json" \
        2> "$OUT/isa-rates-b$blocks.err"; then
      stamp "isa rates b$blocks ok"
    else
      stamp "isa rates b$blocks FAILED rc=$?"
    fi
  done
}

rgemm() {
  stamp "receipt gemm"
  if timeout --foreground 300 "$d/receipt_gemm" --sizes 8192,16384 --reps 15 > "$OUT/receipt-gemm.json" \
      2> "$OUT/receipt-gemm.err"; then
    stamp "receipt gemm ok"
  else
    stamp "receipt gemm FAILED rc=$?"
  fi
  stamp "receipt gemm ncu"
  if timeout --foreground 600 "$NCU" --clock-control none --kernel-name regex:rgemm --section LaunchStats \
      --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section SpeedOfLight \
      --metrics "$METRICS" --export "$d/rgemm-ncu" --force-overwrite \
      "$d/receipt_gemm" --sizes 8192 --profile --only alu_p1,alu_p1s,alu_p4,imad_p1,alu2_p1,alu2_p1s,alu4_p1 > "$d/rgemm-ncu.log" 2>&1; then
    "$NCU" --import "$d/rgemm-ncu.ncu-rep" --csv --page details > "$OUT/rgemm-ncu-details.csv"
    "$NCU" --import "$d/rgemm-ncu.ncu-rep" --csv --page raw | gzip -c > "$OUT/rgemm-ncu-raw.csv.gz"
    stamp "receipt gemm ncu ok"
  else
    stamp "receipt gemm ncu FAILED rc=$?"
  fi
  grep -v "^\s*$" "$d/rgemm-ncu.log" | tail -40 > "$OUT/rgemm-ncu.log.tail"
}

leased() {
  stamp "lease for: $*"
  gpu-lease 1 --wait --max-min 30 -- bash -c "$(declare -f stamp split micro ncu_stage phi isa rgemm); $(declare -p OUT d S PY MODEL REV NCU METRICS); $*" \
    || stamp "lease for '$*' FAILED rc=$?"
}
[[ " $STAGES " == *" split "* ]] && leased split "'bf16 eager'" "'fp8 eager'" "'bf16 graphs'" "'fp8 graphs'"
[[ " $STAGES " == *" split "* ]] && leased split "'fp8 compiled'" "'bf16 compiled'"
[[ " $STAGES " == *" micro "* ]] && leased micro
[[ " $STAGES " == *" ncu "* ]] && leased ncu_stage "'bf16 decode'" "'fp8 decode'"
[[ " $STAGES " == *" ncu_prefill "* ]] && leased ncu_stage "'fp8 prefill2048'" "'bf16 prefill2048'"
[[ " $STAGES " == *" ncu_prefill8k "* ]] && leased ncu_stage "'fp8 prefill8192'"
[[ " $STAGES " == *" phi "* ]] && leased phi
[[ " $STAGES " == *" isa "* ]] && leased isa
[[ " $STAGES " == *" rgemm "* ]] && leased rgemm

for rep in "$d"/ncu-*.ncu-rep; do
  [ -f "$rep" ] || continue
  name=$(basename "$rep" .ncu-rep)
  "$NCU" --import "$rep" --csv --page raw > "$d/$name-raw.csv"
  gzip -c "$d/$name-raw.csv" > "$OUT/$name-raw.csv.gz"
  "$PY" "$S/ncu_summary.py" "$d/$name-raw.csv" --out "$OUT/$name-summary.json" > "$OUT/$name-summary.log" 2>&1 \
    || stamp "ncu_summary $name FAILED"
done
stamp "done"
