#!/usr/bin/env bash
# GemmCoordinate hillclimb, one step at one K and dtype (gemm_hill.py): the gate (byte identity at B = 16: the GPU's proofs and
# transcripts equal the CPU prover's, the device paths agree, and sessions verified beside the next ones' proving equal those
# verified one at a time), then one m = 35 statement of the dtype's coordinate (BF16's
# GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}, or gemm_fp.py's sm_120 FP8 / FP4 row, its instances filling m = 35 from the
# gate's staging) proved warm once and timed three times, nvidia-smi dmon sampling the GPU's utilization and memory every second
# from the start, and the job's own cores sampled every 5 s (out/slice_cores.tsv: another job pinned to them flags the point
# cpu-slice-shared); result.json (the campaign contract's) and out/metrics.json carry the overhead against the census's spec
# peak, the throughput, the GPU utilization and the verifier's seconds, out/results.jsonl the statement's record, and
# hillclimb.json the step's verity/hillclimb-point/v0 record (declared in outputs.json, so the store keeps it as its own artifact).
# Run from the verity tree (research run --cwd source), in a Kueue job on vy-nebius-1 (1 GPU). Staged circuits, instance
# files and proofs go under FLOCK_WORK/scratch/<run id>, removed at exit: the run dir keeps records and logs only.
# env: K= (required: 2048, 4096, 8192 or 16384)  DTYPE=bf16 (or e4m3, nvf4, mxf4)  N= (instances; default BF16's
#      gemm_hill.INSTANCES[K], an FP8 / FP4 row's `gemm_hill.py fill` of the gate, m = 35; required with GATE=0)  CPUSET= (the
#      cores to pin to, e.g. 96-111)  CPUS=48 (without CPUSET: the first CPUS cores of free 16-core slices). Node 1's dispatched
#      jobs all share 96-159, so a job holds a lock on every slice it pins (cpu-slices.sh, SLICE_LOCKS=/workspace/jobs/slices) and
#      waits for slices another job holds  GATE=1  WARM=1  RUNS=3  PEAK_ID= (default the dtype's)
#      STEP=0 LABEL=baseline (the hillclimb step and what changed)  QUESTION= (the question the job answers; default the step's)
#      HILL_NODE=vy-nebius-1
#      FLOCK_WORK=/workspace/jobs/proofs-bf16-hill (builds, venvs and cached binaries)  DISK_MAX=80 (don't start with /workspace
#      this full, in %)  GPU=1 (0: a CPU dry run, with BINARY= and PYBIN= as for 70-class-sweep.sh)
#      STAGE_ONLY=1 (a job with no GPU: build the prover and stage the gate's and the step's statements into FLOCK_STAGE_CACHE,
#      default FLOCK_WORK/stage-cache, then stop; the GPU job with the same settings takes both from the cache, so it holds its
#      GPU from the gate's selftests on, and its point is flagged staged-on-gpu if it staged the step's statement itself)  SM=120
#      STATEMENT_DIGEST=1 (with STAGE_ONLY: each staged statement's digest in out/staged.jsonl, as the verifier builds it)
#      FLOCK_PACK_WORDS=1 (an FP row's elements packed four FP4 or two FP8 a u16 leaf: a different statement, its point flagged)
#      any other KEY=VALUE reaches the prover's environment (FC_PIPELINE_DEPTH=4 and FC_HOST_PREPIN=1 by default, as #20;
#      FC_VERIFY_AHEAD=A FC_VERIFY_SERVERS=S: A sessions verified beside the next one's proving, by S verifier processes in turn)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
: "${K:?K= is required}"
if [ -z "${HILL_PINNED:-}" ]; then
  use=$(df --output=pcent /workspace 2>/dev/null | tail -1 | tr -dc 0-9)
  if [ -n "$use" ] && [ "$use" -ge "${DISK_MAX:-80}" ]; then echo "/workspace is at $use%: not starting"; exit 1; fi
  . backends/flock/pod/cpu-slices.sh
  hold_slices "${CPUSET:-}" "${CPUS:-48}"
  cpus=$(taskset -c "$SLICE_CPUS" python3 -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0)))))") || exit 1
  export HILL_PINNED=$cpus RAYON_NUM_THREADS=$(echo $cpus | tr ',' '\n' | wc -l) HILL_T0=$(date +%s.%N)
  export OMP_NUM_THREADS=$RAYON_NUM_THREADS
  exec taskset -c $cpus bash "$0" "$@"
fi
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/gemm-hill}; mkdir -p $R/out
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-bf16-hill}; mkdir -p $FLOCK_WORK/home
export HOME=$FLOCK_WORK/home FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-$FLOCK_WORK/stage-cache}
[ "${STAGE_ONLY:-0}" = 1 ] && export SM=${SM:-120}
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S
DTYPE=${DTYPE:-bf16}; INST=${N:-}; unset N
# "n SOURCE": n instances and the class sweep's source for them (DEFS= BF16's Gemm_v2 row, PROGRAM= gemm_fp.py's FP8 / FP4 row)
src() { python3 backends/flock/pod/gemm_hill.py defs --dtype $DTYPE --K $K ${1:+--n $1} --out $R/out/defs.json; }
SRC=
if [ $DTYPE = bf16 ] || [ -n "$INST" ]; then S0=$(src $INST) || exit 1; INST=${S0%% *}; SRC=${S0#* }; GSRC=$SRC
elif [ "${GATE:-1}" = 1 ]; then S0=$(src 16) || exit 1; GSRC=${S0#* }
else echo "DTYPE=$DTYPE with GATE=0: N= is required"; exit 1; fi
# the job template's cargo target dir is shared; the builds here keep their own
unset CARGO_TARGET_DIR
export FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
if [ "${GPU:-1}" = 1 ] && [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ]; then echo "no CUDA 13.3 (run 71-gemm-slowdown.sh once to unpack it)"; exit 1; fi
ph() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date +%s.%N)" >> $R/out/phases.tsv; }
printf 'job\tstart\t%s\n' "$HILL_T0" >> $R/out/phases.tsv
Q=name,uuid,driver_version,memory.total,compute_cap,clocks.sm,clocks.max.sm,power.limit,temperature.gpu
{ echo $Q; nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits 2>/dev/null | head -1; } > $R/out/gpu.csv
{ nvidia-smi -L; lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$HILL_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-} slice_wait_s=${SLICE_WAIT_S:-}"; free -g | head -2
  df -h /workspace | tail -1; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-} dtype=$DTYPE K=$K instances=${INST:-from the gate}"; } > $R/out/host.txt 2>&1
DG=; case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*|*-*) ;; *) DG="--gpu $CUDA_VISIBLE_DEVICES" ;; esac
python3 backends/flock/pod/gemm_hill.py dmon --out $R/out/dmon.tsv $DG &
DMON=$!
python3 backends/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $R/out/others_cores.txt --every 5 &
SAMPLER=$!
python3 backends/flock/pod/gemm_hill.py cores --cpus $HILL_PINNED --out $R/out/slice_cores.tsv --every 5 &
SLICE=$!
# the phase's records and logs into the run dir: not its circuits, instance files or proofs (listed with their sizes instead)
keep() {
  [ -d $S/$1 ] || return 0
  mkdir -p $R/out/$1
  (cd $S/$1 && find . -type f ! -name '*.bin' ! -name circuit.txt -size -8M -print0 | xargs -0 -r cp --parents -t $R/out/$1/)
  (cd $S/$1 && find . -type f \( -name '*.bin' -o -name circuit.txt -o -size +8M \) -printf '%p\t%s\n') > $R/out/$1/not-kept.tsv
}
done_() { kill $DMON $SAMPLER $SLICE 2>/dev/null; wait $DMON $SAMPLER $SLICE 2>/dev/null; rm -rf $S; }
trap done_ EXIT
# the gate's three GPU cases at once, each its own process (B = 16: about 7-10 GB each of host memory, under 5 GB of the GPU's)
GS=; [ "${GPU:-1}" = 1 ] && GS="SELFTEST_GPU=1 SELFTEST_CASES=gpu_paths_agree,gpu_proofs_match_cpu,verify_ahead_matches_serial SELFTEST_JOBS=3"
ok=1
if [ "${GATE:-1}" = 1 ]; then
  ph gate start
  OUT=$S/gate RESEARCH_RUN_DIR=$S/gate bash backends/flock/pod/70-class-sweep.sh $GSRC BATCH=16 WARM=0 RUNS=1 SELFTEST=1 $GS PORT=7720 \
    > $R/out/gate.log 2>&1 || ok=0
  ph gate end
  keep gate
  cp $R/out/gate/binary.sha256 $R/out/ 2>/dev/null
  if [ $ok = 1 ] && [ -z "$SRC" ]; then
    GREC=results.jsonl; [ "${STAGE_ONLY:-0}" = 1 ] && GREC=staged.jsonl
    INST=$(python3 backends/flock/pod/gemm_hill.py fill --dtype $DTYPE --K $K --gate $R/out/gate/classes/$GREC) && S0=$(src $INST) \
      && SRC=${S0#* } && echo "instances=$INST (m = 35 from the gate's staging)" >> $R/out/host.txt || ok=0
  fi
fi
if [ $ok = 1 ]; then
  ph sweep start
  OUT=$S/m35 RESEARCH_RUN_DIR=$S/m35 WRAP="python3 $REPO/backends/flock/pod/gemm_hill.py stamp --out $R/out/stamps.tsv --" \
    bash backends/flock/pod/70-class-sweep.sh $SRC BATCH=fixed N=$INST MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} \
    WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 PORT=7721 > $R/out/sweep.log 2>&1 || ok=0
  ph sweep end
  keep m35
  cp $S/m35/classes/results.jsonl $R/out/results.jsonl 2>/dev/null
fi
kill $DMON $SAMPLER $SLICE 2>/dev/null; wait $DMON $SAMPLER $SLICE 2>/dev/null
if [ "${STAGE_ONLY:-0}" = 1 ]; then
  cp $S/gate/classes/staged.jsonl $R/out/gate-staged.jsonl; cp $S/m35/classes/staged.jsonl $R/out/staged.jsonl
  [ $ok = 1 ] && grep -q '"stage"' $R/out/staged.jsonl; exit
fi
# the contract's own reader (verity_numerical.bench.contract) needs the tree's packages: the prover's venv, as 70-class-sweep.sh runs
PYBIN=${PYBIN:-$FLOCK_WORK/flock-circuit/py/bin/python3}; [ -x $PYBIN ] || PYBIN=python3
PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python \
  $PYBIN backends/flock/pod/gemm_hill.py metrics --dtype $DTYPE --K $K --gate $R/out/gate/classes/results.jsonl --sweep $R/out/results.jsonl \
  --stamps $R/out/stamps.tsv --warm ${WARM:-1} --runs ${RUNS:-3} --dmon $R/out/dmon.tsv --phases $R/out/phases.tsv --others $R/out/others_cores.txt --slice $R/out/slice_cores.tsv \
  --gpu-csv $R/out/gpu.csv --census census/hardware.json ${PEAK_ID:+--peak-id $PEAK_ID} --binary-sha $R/out/binary.sha256 --step ${STEP:-0} \
  --label "${LABEL:-baseline}" ${QUESTION:+--question "$QUESTION"} --node ${HILL_NODE:-vy-nebius-1} --out $R/out/metrics.json --result $R/result.json --hillclimb $R/hillclimb.json || ok=0
[ -f $R/hillclimb.json ] && printf '%s\n' '{"schema": "research/outputs/v0.1", "outputs": [{"name": "hillclimb", "kind": "hillclimb-point/v0", "file": "hillclimb.json", "meta": {"campaign": "proofs-hillclimb"}}]}' > $R/outputs.json
[ $ok = 1 ] && grep -q '"status": "passed"' $R/result.json
