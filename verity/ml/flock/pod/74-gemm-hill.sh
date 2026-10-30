#!/usr/bin/env bash
# GemmCoordinate hillclimb, one step at one K and dtype (gemm_hill.py): the preflight check (byte identity at B = 16: the GPU's
# proofs and transcripts equal the CPU prover's, the device paths agree, and sessions verified beside the next ones' proving
# equal those verified one at a time), then one m = 35 statement of the dtype's coordinate (BF16's
# GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}, or gemm_fp.py's sm_120 FP8 / FP4 row, its instances filling m = 35 from the
# preflight check's staging) proved warm once and timed three times, nvidia-smi dmon sampling the GPU's
# utilization and memory every second from the start, and the job's own cores sampled every 5 s (out/slice_cores.tsv: another
# job pinned to them flags the point cpu-slice-shared); result.json (the campaign contract's) and out/metrics.json carry the
# overhead against the census's spec peak, the throughput, the GPU utilization and the verifier's seconds, out/results.jsonl
# the statement's record, and hillclimb.json the step's verity/hillclimb-point/v0 record (declared in outputs.json, so the
# store keeps it as its own artifact).
# Run from the verity tree (research run --cwd source), in a Kueue job on vy-nebius-1 (1 GPU). Staged circuits, instance
# files and proofs go under FLOCK_WORK/scratch/<run id>, removed at exit: the run dir keeps records and logs only.
# env: K= (required: 2048, 4096, 8192, 16384 or 32768)  DTYPE=bf16 (or e4m3, nvf4, mxf4)  N= (instances; default BF16's
#      gemm_hill.INSTANCES[K], an FP8 / FP4 row's `gemm_hill.py fill` of the preflight check, m = 35; required with PREFLIGHT=0
#      or SERVE_ONLY=1)  CPUSET= (the cores to pin to, inside the job's affinity)  CPUS=48 (without CPUSET: the first CPUS cores
#      of free 16-core slices; pass CPUS=16).
#      Node 1's dispatcher starts every `provers` task on the same cores (128-175), so a job holds a lock on every slice it pins
#      (cpu-slices.sh, SLICE_LOCKS=/workspace/jobs/slices) and waits for slices another job holds
#      PREFLIGHT=1 (0: skip the preflight check; GATE= is its old name)  WARM=1  RUNS=3  PEAK_ID= (default the dtype's)
#      POINTS=1 (N: one GPU session of N points, one preflight check and then the step's statement swept N times, each point
#      under out/p<i>/ with its own hillclimb-<i>.json citing the check, and result.json the session's; gemm_hill.py;
#      0: the preflight check alone, its record the run's one output, and with STAGE_ONLY=1 only its statement staged)
#      STEP=0 LABEL=baseline (the hillclimb step and what changed)  QUESTION= (the job's research question, else the
#      campaign's)  HILL_NODE=vy-nebius-1
#      FLOCK_WORK=/workspace/jobs/proofs-bf16-hill (builds, venvs and cached binaries)  DISK_MAX=80 (don't start with /workspace
#      this full, in %)  GPU=1 (0: a CPU dry run, with BINARY= and PYBIN= as for 70-class-sweep.sh)
#      STAGE_ONLY=1 (a job with no GPU: build the prover and stage the preflight check's and the step's statements into
#      FLOCK_STAGE_CACHE, default FLOCK_WORK/stage-cache, then stop; the GPU job with the same settings takes both from the
#      cache, so it holds its GPU from the preflight check's selftests on, and its point is flagged staged-on-gpu if it staged
#      the step's statement itself)  SM=120
#      STATEMENT_DIGEST=1 (with STAGE_ONLY: each staged statement's digest in out/staged.jsonl, as the verifier builds it)
#      FLOCK_PACK_WORDS=1 (an FP row's elements packed four FP4 or two FP8 a u16 leaf: a different statement, its point flagged)
#      FLOCK_CORE_ROWS=1 (an FP row is core's bit row, sha512/row/v2, its codes then its scales: a different statement,
#      its point flagged; any row length, the verifiers deriving its prefix, fill bits and padding from it)
#      VERIFIERS=DIR with SERVE_ONLY=1 (a job with no GPU: stage the preflight check's and the step's statements and serve their
#      sessions on the pod's address, published under DIR, until they are served or FC_SERVE_WAIT_S pass), and VERIFIERS=DIR
#      alone (the GPU job, submitted once DIR holds both statements' addresses: its sessions' verifier is that pod's, over the
#      pod network, so this pod holds only the prover); give both jobs the same K, N, POINTS, WARM, RUNS and
#      FC_VERIFY_AHEAD, and FC_VERIFY_SERVERS to the serving one (its verifier processes serve every point's sessions)
#      any other KEY=VALUE reaches the prover's environment (FC_PIPELINE_DEPTH=4 and FC_HOST_PREPIN=1 by default, as #20;
#      FC_VERIFY_AHEAD=A FC_VERIFY_SERVERS=S: A sessions verified beside the next one's proving, by S verifier processes in turn)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
: "${K:?K= is required}"
if [ -z "${HILL_PINNED:-}" ]; then
  use=$(df --output=pcent /workspace 2>/dev/null | tail -1 | tr -dc 0-9)
  if [ -n "$use" ] && [ "$use" -ge "${DISK_MAX:-80}" ]; then echo "/workspace is at $use%: not starting"; exit 1; fi
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "${CPUSET:-}" "${CPUS:-48}" || exit 1
  cpus=$(taskset -c "$SLICE_CPUS" python3 -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0)))))") || exit 1
  export HILL_PINNED=$cpus RAYON_NUM_THREADS=$(echo $cpus | tr ',' '\n' | wc -l) HILL_T0=$(date +%s.%N)
  export OMP_NUM_THREADS=$RAYON_NUM_THREADS
  exec taskset -c $cpus bash "$0" "$@"
fi
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/gemm-hill}; mkdir -p $R/out
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-bf16-hill}; mkdir -p $FLOCK_WORK/home
export HOME=$FLOCK_WORK/home FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-$FLOCK_WORK/stage-cache}
[ "${STAGE_ONLY:-0}" = 1 ] || [ "${SERVE_ONLY:-0}" = 1 ] && export SM=${SM:-120}
[ -n "${VERIFIERS:-}" ] && [ "${SERVE_ONLY:-0}" != 1 ] && export FC_VERIFIERS=$VERIFIERS
POINTS=${POINTS:-1}; [ "${STAGE_ONLY:-0}" = 1 ] && [ "$POINTS" != 0 ] && POINTS=1
case $POINTS in ''|*[!0-9]*|0?*) echo "POINTS=$POINTS: a count of points (0: the preflight check alone)"; exit 1 ;; esac
[ "$POINTS" = 0 ] && [ "${PREFLIGHT:-${GATE:-1}}" != 1 ] && { echo "POINTS=0 runs the preflight check alone: not with PREFLIGHT=0"; exit 1; }
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S
DTYPE=${DTYPE:-bf16}; INST=${N:-}; unset N
# "n SOURCE": n instances and the class sweep's source for them (DEFS= BF16's Gemm_v2 row, PROGRAM= gemm_fp.py's FP8 / FP4 row)
src() { python3 verity/ml/flock/pod/gemm_hill.py defs --dtype $DTYPE --K $K ${1:+--n $1} --out $R/out/defs.json; }
SRC=
if [ $DTYPE = bf16 ] || [ -n "$INST" ]; then S0=$(src $INST) || exit 1; INST=${S0%% *}; SRC=${S0#* }; GSRC=$SRC
elif [ "${PREFLIGHT:-${GATE:-1}}" = 1 ]; then S0=$(src 16) || exit 1; GSRC=${S0#* }
else echo "DTYPE=$DTYPE with PREFLIGHT=0: N= is required"; exit 1; fi
# the job template's cargo target dir is shared; the builds here keep their own
unset CARGO_TARGET_DIR
export FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
# K = 32768's coordinate (16,860,955 ANDs, k_log 27) is past the class sweep's default cap of 2^24 ANDs a shape
export MAX_ANDS=${MAX_ANDS:-33554432}
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
DMON=
if command -v nvidia-smi >/dev/null; then python3 verity/ml/flock/pod/gemm_hill.py dmon --out $R/out/dmon.tsv $DG & DMON=$!; fi
python3 verity/ml/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $R/out/others_cores.txt --every 5 &
SAMPLER=$!
python3 verity/ml/flock/pod/gemm_hill.py cores --cpus $HILL_PINNED --out $R/out/slice_cores.tsv --every 5 &
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
GS=; [ "${GPU:-1}" = 1 ] && GS="SELFTEST_GPU=1 SELFTEST_CASES=gpu_paths_agree,gpu_proofs_match_cpu,verify_ahead_matches_serial"
# the preflight check's three GPU cases at once, each its own process, while 3 x one case's device memory fits the GPU's free
# memory (class_statement's guard), else one at a time. One case's bound per K (B = 16) is its measured peak (selftest.per_case)
# with some room: at most 4794, 5074, 9272 and 17678 MiB at K = 2048, 4096, 8192 and 16384, with the pod's host memory at most 12, 16, 27 and 53 GB with the three at once. K = 32768 has none, so
# its cases run one at a time: at BF16's k_log 27 one case holds about 30 GiB, and three at once overfilled the 97,887 MiB card.
# Each case's stderr is kept as out/gate/classes/*/selftest-<case>.err, beside the class's prove.out
case $K in 2048|4096) CASE_MIB=6144 ;; 8192) CASE_MIB=12288 ;; 16384) CASE_MIB=20480 ;; *) CASE_MIB= ;; esac
[ -n "$GS" ] && [ -n "$CASE_MIB" ] && GS+=" SELFTEST_JOBS=3 SELFTEST_CASE_MIB=$CASE_MIB"
ok=1
if [ "${SERVE_ONLY:-0}" = 1 ]; then
  : "${VERIFIERS:?SERVE_ONLY=1 needs VERIFIERS=DIR, where the GPU job finds the addresses}"
  [ -n "$SRC" ] || { echo "SERVE_ONLY=1 with DTYPE=$DTYPE: N= is required"; exit 1; }
  ph serve start
  OUT=$S/preflight RESEARCH_RUN_DIR=$S/preflight SERVE_ONLY=$VERIFIERS bash verity/ml/flock/pod/70-class-sweep.sh $GSRC BATCH=16 WARM=0 RUNS=1 \
    SELFTEST=0 > $R/out/preflight.log 2>&1 &
  GP=$!
  OUT=$S/m35 RESEARCH_RUN_DIR=$S/m35 SERVE_ONLY=$VERIFIERS SERVE_POINTS=$POINTS bash verity/ml/flock/pod/70-class-sweep.sh $SRC \
    BATCH=fixed N=$INST MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 > $R/out/sweep.log 2>&1 &
  SP=$!
  wait $GP || ok=0
  wait $SP || ok=0
  ph serve end
  keep preflight; keep m35
  cp $S/preflight/classes/served.jsonl $R/out/preflight-served.jsonl; cp $S/m35/classes/served.jsonl $R/out/served.jsonl
  [ $ok = 1 ] && python3 -c 'import json, sys; r = [json.loads(l)["serve"] for f in sys.argv[1:] for l in open(f) if l.strip()]
sys.exit(0 if len(r) == 2 and all(x["accepted"] and x["served"] == x["sessions"] * x.get("points", 1) and not x["timed_out"] for x in r) else 1)' \
    $R/out/preflight-served.jsonl $R/out/served.jsonl
  exit
fi
if [ "${PREFLIGHT:-${GATE:-1}}" = 1 ]; then
  ph preflight start
  OUT=$S/preflight RESEARCH_RUN_DIR=$S/preflight bash verity/ml/flock/pod/70-class-sweep.sh $GSRC BATCH=16 WARM=0 RUNS=1 SELFTEST=1 $GS PORT=7720 \
    > $R/out/preflight.log 2>&1 || ok=0
  ph preflight end
  keep preflight
  python3 verity/ml/flock/pod/gemm_hill.py key --binary-sha $R/out/preflight/binary.sha256 --out $R/out/preflight/key.json
  if [ $ok = 1 ] && [ -z "$SRC" ]; then
    PREC=results.jsonl; [ "${STAGE_ONLY:-0}" = 1 ] && PREC=staged.jsonl
    INST=$(python3 verity/ml/flock/pod/gemm_hill.py fill --dtype $DTYPE --K $K --preflight $R/out/preflight/classes/$PREC) && S0=$(src $INST) \
      && SRC=${S0#* } && echo "instances=$INST (m = 35 from the preflight check's staging)" >> $R/out/host.txt || ok=0
  fi
fi
# the session's points, one after another (P: each one's directory under out/ and the scratch, none with one point); a point
# that fails ends the session
pdir() { [ "$POINTS" = 1 ] || echo p$1/; }
ran=0
for i in $(seq 0 $((POINTS - 1))); do
  [ $ok = 1 ] || break
  P=$(pdir $i); mkdir -p $R/out/$P
  ph sweep${P:+-p$i} start
  OUT=$S/${P}m35 RESEARCH_RUN_DIR=$S/${P}m35 WRAP="python3 $REPO/verity/ml/flock/pod/gemm_hill.py stamp --out $R/out/${P}stamps.tsv --" \
    bash verity/ml/flock/pod/70-class-sweep.sh $SRC BATCH=fixed N=$INST MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} \
    WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 PORT=7721 > $R/out/${P}sweep.log 2>&1 || ok=0
  ph sweep${P:+-p$i} end
  keep ${P}m35
  cp $S/${P}m35/classes/results.jsonl $R/out/${P}results.jsonl 2>/dev/null
  python3 verity/ml/flock/pod/gemm_hill.py key --binary-sha $R/out/${P}m35/binary.sha256 --out $R/out/${P}key.json
  ran=$((ran + 1))
done
kill $DMON $SAMPLER $SLICE 2>/dev/null; wait $DMON $SAMPLER $SLICE 2>/dev/null
if [ "${STAGE_ONLY:-0}" = 1 ]; then
  cp $S/preflight/classes/staged.jsonl $R/out/preflight-staged.jsonl
  [ "$POINTS" = 0 ] && { [ $ok = 1 ] && grep -q '"stage"' $R/out/preflight-staged.jsonl; exit; }
  cp $S/m35/classes/staged.jsonl $R/out/staged.jsonl
  [ $ok = 1 ] && grep -q '"stage"' $R/out/staged.jsonl; exit
fi
if [ "$POINTS" = 0 ]; then
  python3 verity/ml/flock/pod/gemm_hill.py session --dir $R --points 0 || ok=0
  [ $ok = 1 ]; exit
fi
# the contract's own reader (verity_numerical.bench.contract) needs the tree's packages: the prover's venv, as 70-class-sweep.sh runs
PYBIN=${PYBIN:-$FLOCK_WORK/flock-circuit/py/bin/python3}; [ -x $PYBIN ] || PYBIN=python3
# every point that ran, or the one the session refused before it ran (its record says why)
ph metrics start
for i in $(seq 0 $(( (ran > 0 ? ran : 1) - 1 ))); do
  P=$(pdir $i); mkdir -p $R/out/$P
  BS=$R/out/${P}m35/binary.sha256; [ -f $BS ] || BS=$R/out/preflight/binary.sha256
  if [ "$POINTS" = 1 ]; then RES=$R/result.json HC=$R/hillclimb.json; else RES=$R/out/${P}result.json HC=$R/hillclimb-$i.json; fi
  PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python \
    $PYBIN verity/ml/flock/pod/gemm_hill.py metrics --dtype $DTYPE --K $K --preflight $R/out/preflight/classes/results.jsonl --sweep $R/out/${P}results.jsonl \
    --stamps $R/out/${P}stamps.tsv --warm ${WARM:-1} --runs ${RUNS:-3} --dmon $R/out/dmon.tsv --phases $R/out/phases.tsv --others $R/out/others_cores.txt --slice $R/out/slice_cores.tsv \
    --gpu-csv $R/out/gpu.csv --census benchmarks/census/hardware.json ${PEAK_ID:+--peak-id $PEAK_ID} --binary-sha $BS --step ${STEP:-0} \
    --label "${LABEL:-baseline}" ${QUESTION:+--question "$QUESTION"} --node ${HILL_NODE:-vy-nebius-1} --point $i --points $POINTS --preflight-key $R/out/preflight/key.json \
    --point-key $R/out/${P}key.json --out $R/out/${P}metrics.json --result $RES --hillclimb $HC || ok=0
done
ph metrics end
python3 verity/ml/flock/pod/gemm_hill.py session --dir $R --points $POINTS || ok=0
[ $ok = 1 ] && grep -q '"status": "passed"' $R/result.json
