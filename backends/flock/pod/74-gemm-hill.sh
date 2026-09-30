#!/usr/bin/env bash
# BF16 GemmCoordinate hillclimb, one step at one K (gemm_hill.py): the gate (byte identity at B = 16: the GPU's proofs and
# transcripts equal the CPU prover's, and the device paths agree), then one m = 35 statement of
# GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1} proved warm once and timed three times, nvidia-smi dmon sampling the GPU every
# second from the start; out/metrics.json and result.json carry the overhead against the census's spec peak, the throughput, the
# GPU utilization and the verifier's seconds, out/results.jsonl the statement's record.
# Run from the verity tree (research run --cwd source), in a Kueue job on vy-nebius-1 (48 vCPU, 1 GPU). Staged circuits, instance
# files and proofs go under FLOCK_WORK/scratch/<run id>, removed at exit: the run dir keeps records and logs only.
# env: K= (required: 2048, 4096, 8192 or 16384)  N= (instances; default gemm_hill.INSTANCES[K], m = 35)  CPUS=48 (the job pins
#      itself to the first CPUS of its cores)  GATE=1  WARM=1  RUNS=3  PEAK_ID=rtx-pro-6000-blackwell-server/bf16
#      FLOCK_WORK=/workspace/jobs/proofs-bf16-hill (builds, venvs and cached binaries)  DISK_MAX=80 (don't start with /workspace
#      this full, in %)  GPU=1 (0: a CPU dry run, with BINARY= and PYBIN= as for 70-class-sweep.sh)
#      any other KEY=VALUE reaches the prover's environment (FC_PIPELINE_DEPTH=4 and FC_HOST_PREPIN=1 by default, as #20)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
: "${K:?K= is required}"
if [ -z "${HILL_PINNED:-}" ]; then
  use=$(df --output=pcent /workspace 2>/dev/null | tail -1 | tr -dc 0-9)
  if [ -n "$use" ] && [ "$use" -ge "${DISK_MAX:-80}" ]; then echo "/workspace is at $use%: not starting"; exit 1; fi
  cpus=$(python3 -c "import os; c = sorted(os.sched_getaffinity(0)); print(','.join(map(str, c[:int(os.environ.get('CPUS', '48'))])))")
  export HILL_PINNED=$cpus RAYON_NUM_THREADS=$(echo $cpus | tr ',' '\n' | wc -l) HILL_T0=$(date +%s.%N)
  export OMP_NUM_THREADS=$RAYON_NUM_THREADS
  exec taskset -c $cpus bash "$0" "$@"
fi
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/gemm-hill}; mkdir -p $R/out
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-bf16-hill}; mkdir -p $FLOCK_WORK/home
export HOME=$FLOCK_WORK/home
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S
INST=${N:-}; unset N
DEFS=$R/out/defs.json; INST=$(python3 backends/flock/pod/gemm_hill.py defs --K $K ${INST:+--n $INST} --out $DEFS) || exit 1
# the job template's cargo target dir is shared; the builds here keep their own
unset CARGO_TARGET_DIR
export FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
if [ "${GPU:-1}" = 1 ] && [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ]; then echo "no CUDA 13.3 (run 71-gemm-slowdown.sh once to unpack it)"; exit 1; fi
ph() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date +%s.%N)" >> $R/out/phases.tsv; }
printf 'job\tstart\t%s\n' "$HILL_T0" >> $R/out/phases.tsv
Q=name,uuid,driver_version,memory.total,clocks.sm,clocks.max.sm,power.limit,temperature.gpu
{ echo $Q; nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits 2>/dev/null | head -1; } > $R/out/gpu.csv
{ nvidia-smi -L; lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$HILL_PINNED rayon=$RAYON_NUM_THREADS"; free -g | head -2
  df -h /workspace | tail -1; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-} K=$K instances=$INST"; } > $R/out/host.txt 2>&1
DG=; case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*|*-*) ;; *) DG="--gpu $CUDA_VISIBLE_DEVICES" ;; esac
python3 backends/flock/pod/gemm_hill.py dmon --out $R/out/dmon.tsv $DG &
DMON=$!
python3 backends/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $R/out/others_cores.txt --every 5 &
SAMPLER=$!
# the phase's records and logs into the run dir: not its circuits, instance files or proofs (listed with their sizes instead)
keep() {
  [ -d $S/$1 ] || return 0
  mkdir -p $R/out/$1
  (cd $S/$1 && find . -type f ! -name '*.bin' ! -name circuit.txt -size -8M -print0 | xargs -0 -r cp --parents -t $R/out/$1/)
  (cd $S/$1 && find . -type f \( -name '*.bin' -o -name circuit.txt -o -size +8M \) -printf '%p\t%s\n') > $R/out/$1/not-kept.tsv
}
done_() { kill $DMON $SAMPLER 2>/dev/null; wait $DMON $SAMPLER 2>/dev/null; rm -rf $S; }
trap done_ EXIT
GS=; [ "${GPU:-1}" = 1 ] && GS="SELFTEST_GPU=1 SELFTEST_CASES=gpu_paths_agree,gpu_proofs_match_cpu"
ok=1
if [ "${GATE:-1}" = 1 ]; then
  ph gate start
  OUT=$S/gate RESEARCH_RUN_DIR=$S/gate bash backends/flock/pod/70-class-sweep.sh DEFS=$DEFS BATCH=16 WARM=0 RUNS=1 SELFTEST=1 $GS PORT=7720 \
    > $R/out/gate.log 2>&1 || ok=0
  ph gate end
  keep gate
  cp $R/out/gate/binary.sha256 $R/out/ 2>/dev/null
fi
if [ $ok = 1 ]; then
  ph sweep start
  OUT=$S/m35 RESEARCH_RUN_DIR=$S/m35 WRAP="python3 $REPO/backends/flock/pod/gemm_hill.py stamp --out $R/out/stamps.tsv --" \
    bash backends/flock/pod/70-class-sweep.sh DEFS=$DEFS BATCH=fixed N=$INST MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} \
    WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 PORT=7721 > $R/out/sweep.log 2>&1 || ok=0
  ph sweep end
  keep m35
  cp $S/m35/classes/results.jsonl $R/out/results.jsonl 2>/dev/null
fi
kill $DMON $SAMPLER 2>/dev/null; wait $DMON $SAMPLER 2>/dev/null
python3 backends/flock/pod/gemm_hill.py metrics --K $K --gate $R/out/gate/classes/results.jsonl --sweep $R/out/results.jsonl \
  --stamps $R/out/stamps.tsv --dmon $R/out/dmon.tsv --phases $R/out/phases.tsv --others $R/out/others_cores.txt --gpu-csv $R/out/gpu.csv \
  --census census/hardware.json --peak-id ${PEAK_ID:-rtx-pro-6000-blackwell-server/bf16} --out $R/out/metrics.json --result $R/result.json || ok=0
[ $ok = 1 ] && grep -q '"status": "passed"' $R/result.json
