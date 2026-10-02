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
# env: K= (required: 2048, 4096, 8192, 16384 or 32768)  DTYPE=bf16 (or e4m3, nvf4, mxf4)  N= (instances; default BF16's
#      gemm_hill.INSTANCES[K], an FP8 / FP4 row's `gemm_hill.py fill` of the gate, m = 35; required with GATE=0 or SERVE_ONLY=1;
#      an FP row's gate is always its N = 16 row)  CPUSET= (the cores to
#      pin to, inside the job's affinity)  CPUS=48 (without CPUSET: the first CPUS cores of free 16-core slices; pass CPUS=16).
#      Node 1's dispatcher starts every `provers` task on the same cores (128-175), so a job holds a lock on every slice it pins
#      (cpu-slices.sh, SLICE_LOCKS=/workspace/jobs/slices) and waits for slices another job holds  GATE=1  WARM=1  RUNS=3
#      PEAK_ID= (default the dtype's)
#      STEP=0 LABEL=baseline (the hillclimb step and what changed)  QUESTION= (the job's research question, else the
#      campaign's)  HILL_NODE=vy-nebius-1
#      FLOCK_WORK=/workspace/jobs/proofs-bf16-hill (builds, venvs and cached binaries)  DISK_MAX=80 (don't start with /workspace
#      this full, in %)  GPU=1 (0: a CPU dry run, with BINARY= and PYBIN= as for 70-class-sweep.sh)
#      STAGE_ONLY=1 (a job with no GPU: build the prover and stage the gate's and the step's statements into FLOCK_STAGE_CACHE,
#      default FLOCK_WORK/stage-cache, then stop; the GPU job with the same settings takes both from the cache, so it holds its
#      GPU from the gate's selftests on, and its point is flagged staged-on-gpu if it staged the step's statement itself)  SM=120
#      VERIFIERS=DIR with SERVE_ONLY=1 (a job with no GPU: stage the gate's and the step's statements and serve their sessions on
#      the pod's address, published under DIR, until they are served or FC_SERVE_WAIT_S pass), and VERIFIERS=DIR alone (the GPU
#      job, submitted once DIR holds both statements' addresses: its sessions' verifier is that pod's, over the pod network, so
#      this pod holds only the prover); give both jobs the same K, N, WARM, RUNS and FC_VERIFY_AHEAD, and FC_VERIFY_SERVERS to
#      the serving one
#      ZK=1 (the gate's selftests, the serve and the prove with --zk; then zkaudit, the selftest build's, on the gate's and the
#      step's statements: out/zkaudit-{gate,m35}.json, and the job fails unless every byte the verifier kept is attributed and
#      none is in the clear)  ZKAUDIT_RUNS=RUN,... (only that zkaudit, on earlier runs' statements: see zkaudit below)
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
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S
DTYPE=${DTYPE:-bf16}; INST=${N:-}; unset N
# "n SOURCE": n instances and the class sweep's source for them (DEFS= BF16's Gemm_v2 row, PROGRAM= gemm_fp.py's FP8 / FP4 row)
src() { python3 backends/flock/pod/gemm_hill.py defs --dtype $DTYPE --K $K ${1:+--n $1} --out $R/out/defs.json; }
SRC=
if [ $DTYPE = bf16 ]; then S0=$(src $INST) || exit 1; INST=${S0%% *}; SRC=${S0#* }; GSRC=$SRC
else
  S0=$(src 16) || exit 1; GSRC=${S0#* }
  if [ -n "$INST" ]; then S0=$(src $INST) || exit 1; SRC=${S0#* }
  elif [ "${GATE:-1}" != 1 ] || [ "${SERVE_ONLY:-0}" = 1 ]; then echo "DTYPE=$DTYPE with GATE=0 or SERVE_ONLY=1: N= is required"; exit 1; fi
fi
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
if command -v nvidia-smi >/dev/null; then python3 backends/flock/pod/gemm_hill.py dmon --out $R/out/dmon.tsv $DG & DMON=$!; fi
python3 backends/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $R/out/others_cores.txt --every 5 &
SAMPLER=$!
python3 backends/flock/pod/gemm_hill.py cores --cpus $HILL_PINNED --out $R/out/slice_cores.tsv --every 5 &
SLICE=$!
python3 backends/flock/pod/gemm_hill.py vmstat --out $R/out/vmstat.tsv &
VMSTAT=$!
# the phase's records and logs into the run dir: not its circuits, instance files or proofs (listed with their sizes instead)
keep() {
  [ -d $S/$1 ] || return 0
  mkdir -p $R/out/$1
  (cd $S/$1 && find . -type f ! -name '*.bin' ! -name circuit.txt -size -8M -print0 | xargs -0 -r cp --parents -t $R/out/$1/)
  (cd $S/$1 && find . -type f \( -name '*.bin' -o -name circuit.txt -o -size +8M \) -printf '%p\t%s\n') > $R/out/$1/not-kept.tsv
}
done_() { kill $DMON $SAMPLER $SLICE $VMSTAT 2>/dev/null; wait $DMON $SAMPLER $SLICE $VMSTAT 2>/dev/null; rm -rf $S; }
trap done_ EXIT
zok=1 ZKA=; [ "${ZK:-0}" = 1 ] && [ "${GPU:-1}" = 1 ] && [ "${STAGE_ONLY:-0}" != 1 ] && [ "${NSYS:-0}" != 1 ] && [ "${NCU:-0}" != 1 ] && ZKA=1
# zkaudit NAME RESULTS BINARY_SHA: the statement a phase's classes/results.jsonl names, as FLOCK_STAGE_CACHE keeps it (the
# phase's own copy is gone once proved), under the phase's selftest build (binary.sha256's second line), on CUDA 13.3's
# libraries as 70-class-sweep.sh runs it: out/zkaudit-NAME.json
zkaudit() {
  local f sb C13=${CUDA_TK:-/usr/local/cuda-13.3}
  f=$(python3 - $2 $FLOCK_STAGE_CACHE <<'EOF'
import json, sys
from pathlib import Path
recs = [json.loads(ln) for ln in open(sys.argv[1]) if ln.strip()]
rec = recs[-1] if recs else {}
n, c = (rec.get("stage") or {}).get("n"), rec.get("stage_cached")
d = Path(sys.argv[2]) / c if c else None
print(f"{d}/circuit.txt {d}/inst-{n}.bin" if d and (d / "circuit.txt").exists() and (d / f"inst-{n}.bin").exists() else "")
EOF
  )
  sb=$(awk 'NR == 2 {print $2}' $3 2>/dev/null)
  if [ -z "$f" ] || [ ! -x "$sb" ]; then
    echo '{"skipped": "the statement is not in FLOCK_STAGE_CACHE, or there is no selftest build"}' > $R/out/zkaudit-$1.json; zok=0; return
  fi
  ph zkaudit-$1 start
  LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-} $sb zkaudit --zk --gpu --circuit ${f% *} --instances ${f#* } \
    > $R/out/zkaudit-$1.out 2> $R/out/zkaudit-$1.err
  ph zkaudit-$1 end
  grep '^ZKAUDIT' $R/out/zkaudit-$1.out | cut -f2- > $R/out/zkaudit-$1.json
  [ -s $R/out/zkaudit-$1.json ] || echo "{\"failed\": \"no ZKAUDIT record: zkaudit-$1.err\"}" > $R/out/zkaudit-$1.json
  python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("checks_pass") and d.get("attributed") else 1)' \
    $R/out/zkaudit-$1.json || zok=0
}
# ZKAUDIT_RUNS=RUN,... (GATE=0): only zkaudit, on each earlier run's gate and step statements (its out/{gate,m35} records name
# them, and FLOCK_STAGE_CACHE still holds them): out/zkaudit-RUN-{gate,m35}.json; the job fails unless every one passes
if [ -n "${ZKAUDIT_RUNS:-}" ]; then
  for r in ${ZKAUDIT_RUNS//,/ }; do
    for p in gate m35; do zkaudit $r-$p $(dirname $R)/$r/out/$p/classes/results.jsonl $(dirname $R)/$r/out/$p/binary.sha256; done
  done
  [ $zok = 1 ]; exit
fi
# the gate's three GPU cases at once, each its own process (B = 16: about 7-10 GB each of host memory, and of the GPU's about 9,
# 18 and 31 GiB at k_log 25, 26 and 27, so class_statement runs a k_log 27 gate's one at a time); each case's stderr is kept as
# out/gate/classes/*/selftest-<case>.err, beside the class's prove.out
GS=; [ "${GPU:-1}" = 1 ] && GS="SELFTEST_GPU=1 SELFTEST_CASES=gpu_paths_agree,gpu_proofs_match_cpu,verify_ahead_matches_serial SELFTEST_JOBS=3"
ok=1
if [ "${SERVE_ONLY:-0}" = 1 ]; then
  : "${VERIFIERS:?SERVE_ONLY=1 needs VERIFIERS=DIR, where the GPU job finds the addresses}"
  ph serve start
  OUT=$S/gate RESEARCH_RUN_DIR=$S/gate SERVE_ONLY=$VERIFIERS bash backends/flock/pod/70-class-sweep.sh $GSRC BATCH=16 WARM=0 RUNS=1 \
    SELFTEST=0 > $R/out/gate.log 2>&1 &
  GP=$!
  OUT=$S/m35 RESEARCH_RUN_DIR=$S/m35 SERVE_ONLY=$VERIFIERS bash backends/flock/pod/70-class-sweep.sh $SRC BATCH=fixed N=$INST \
    MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 > $R/out/sweep.log 2>&1 &
  SP=$!
  wait $GP || ok=0
  wait $SP || ok=0
  ph serve end
  keep gate; keep m35
  cp $S/gate/classes/served.jsonl $R/out/gate-served.jsonl; cp $S/m35/classes/served.jsonl $R/out/served.jsonl
  [ $ok = 1 ] && python3 -c 'import json, sys; r = [json.loads(l)["serve"] for f in sys.argv[1:] for l in open(f) if l.strip()]
sys.exit(0 if len(r) == 2 and all(x["accepted"] and x["served"] == x["sessions"] and not x["timed_out"] for x in r) else 1)' \
    $R/out/gate-served.jsonl $R/out/served.jsonl
  exit
fi
if [ "${GATE:-1}" = 1 ]; then
  ph gate start
  OUT=$S/gate RESEARCH_RUN_DIR=$S/gate bash backends/flock/pod/70-class-sweep.sh $GSRC BATCH=16 WARM=0 RUNS=1 SELFTEST=1 $GS PORT=7720 \
    > $R/out/gate.log 2>&1 || ok=0
  ph gate end
  keep gate
  cp $R/out/gate/binary.sha256 $R/out/ 2>/dev/null
  [ -n "$ZKA" ] && [ $ok = 1 ] && zkaudit gate $S/gate/classes/results.jsonl $S/gate/binary.sha256
  if [ $ok = 1 ] && [ -z "$SRC" ]; then
    GREC=results.jsonl; [ "${STAGE_ONLY:-0}" = 1 ] && GREC=staged.jsonl
    INST=$(python3 backends/flock/pod/gemm_hill.py fill --dtype $DTYPE --K $K --gate $R/out/gate/classes/$GREC) && S0=$(src $INST) \
      && SRC=${S0#* } && echo "instances=$INST (m = 35 from the gate's staging)" >> $R/out/host.txt || ok=0
  fi
fi
# NSYS=1: a profile, not a point: the step's prove process under Nsight Systems' CUDA trace (out/nsys/prove-<pid>.nsys-rep, and
# its kernel, memcpy and API summaries beside it), unpacked into NSYS_ROOT (default FLOCK_WORK/nsys-root) as 71-gemm-slowdown.sh
# does; no metrics, result.json or hillclimb.json, since the trace slows the prove. Pass GATE=0: the gate proves nothing it needs.
# NSYS_ARGS replaces the trace's options (default: CUDA and NVTX, no CPU sampling); `nsys status -e` in the pod goes to
# out/nsys/status.txt, so a CPU-sampled profile records whether the pod could sample
NSW=
if [ "${NSYS:-0}" = 1 ]; then
  NR=${NSYS_ROOT:-$FLOCK_WORK/nsys-root}; NS=$NR/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
  if [ ! -x $NS ]; then
    mkdir -p $NR/debs && (cd $NR/debs && { apt-get download nsight-systems-2026.1.3 2>/dev/null || curl -fsSLO \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb; }) || exit 1
    for d in $NR/debs/*.deb; do dpkg-deb -x $d $NR; done
  fi
  mkdir -p $R/out/nsys; $NS status -e > $R/out/nsys/status.txt 2>&1
  NSW="$NS profile ${NSYS_ARGS:--t cuda,nvtx --sample=none --cpuctxsw=none} -o $R/out/nsys/prove-%p"
fi
# NCU=1: a profile, not a point: one kernel (NCU_KERNEL, default zerocheck's round 1) under Nsight Compute, from NCU_ROOT (an ncu
# CLI tree, default FLOCK_WORK/ncu-root). A kernel replayed over several passes has every allocation it can reach saved first, to
# host memory once the GPU's is full, so the full section set runs on the gate's B = 16 statement (NCU_B16_COUNT launches,
# default 2: one per rep) and the step's statement gets only NCU_M35_METRICS, single-pass counters (no save: its 70+ GiB would
# not fit the job's memory), on NCU_COUNT launches (default 2) after the first NCU_SKIP (default 2: the warm session's). The B = 16
# statement takes NCU_M35_METRICS on one launch first: unless ncu collects them in one pass there, the step's statement gets
# NCU_M35_FALLBACK (out/ncu/m35-metrics.txt says which). out/ncu/{b16,m35}-<pid>.ncu-rep and their CSV pages. Pass GATE=0; not
# with NSYS=1
NCW=
if [ "${NCU:-0}" = 1 ]; then
  [ -n "$NSW" ] && { echo "NCU=1 and NSYS=1: one profiler at a time"; exit 1; }
  NC=${NCU_ROOT:-$FLOCK_WORK/ncu-root}/ncu; [ -x $NC ] || { echo "no ncu at $NC"; exit 1; }
  mkdir -p $R/out/ncu; NCW="$NC --kernel-name regex:${NCU_KERNEL:-zerocheck_first_round_cpu_structured} --clock-control none"
  M35=${NCU_M35_METRICS:-gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,smsp__inst_executed.sum,sm__cycles_elapsed.avg}
  b16() {   # b16 NAME NCU-OPTION...: the B = 16 statement under ncu, out/stamps-NAME.tsv and out/ncu/NAME-<pid>.ncu-rep
    local n=$1; shift
    ph ncu-$n start
    OUT=$S/$n RESEARCH_RUN_DIR=$S/$n \
      WRAP="python3 $REPO/backends/flock/pod/gemm_hill.py stamp --out $R/out/stamps-$n.tsv -- $NCW $* -o $R/out/ncu/$n-%p" \
      bash backends/flock/pod/70-class-sweep.sh $GSRC BATCH=16 WARM=0 RUNS=1 SELFTEST=0 PORT=7720 > $R/out/$n.log 2>&1 || ok=0
    ph ncu-$n end
    keep $n
  }
  if [ $ok = 1 ]; then
    b16 b16 --set full --launch-count ${NCU_B16_COUNT:-2}
    for f in $R/out/ncu/b16-*.ncu-rep; do
      [ -f "$f" ] && { $NC --import $f --page details --csv > ${f%.ncu-rep}.details.csv; $NC --import $f --page raw --csv > ${f%.ncu-rep}.raw.csv; }
    done
  fi
  if [ $ok = 1 ]; then
    b16 passes --metrics $M35 --launch-count 1
    grep -qE -- '- 1 pass\s*$' $R/out/stamps-passes.tsv && ! grep -qE -- ' passes\s*$' $R/out/stamps-passes.tsv \
      || M35=${NCU_M35_FALLBACK:-gpu__time_duration.sum,smsp__inst_executed.sum,sm__cycles_elapsed.avg}
    echo "$M35" > $R/out/ncu/m35-metrics.txt
  fi
  NSW="$NCW --metrics $M35 --launch-skip ${NCU_SKIP:-2} --launch-count ${NCU_COUNT:-2} -o $R/out/ncu/m35-%p"
fi
if [ $ok = 1 ]; then
  ph sweep start
  OUT=$S/m35 RESEARCH_RUN_DIR=$S/m35 WRAP="python3 $REPO/backends/flock/pod/gemm_hill.py stamp --out $R/out/stamps.tsv -- $NSW" \
    bash backends/flock/pod/70-class-sweep.sh $SRC BATCH=fixed N=$INST MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} \
    WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 PORT=7721 > $R/out/sweep.log 2>&1 || ok=0
  ph sweep end
  keep m35
  cp $S/m35/classes/results.jsonl $R/out/results.jsonl 2>/dev/null
  [ -n "$ZKA" ] && [ $ok = 1 ] && zkaudit m35 $S/m35/classes/results.jsonl $S/m35/binary.sha256
fi
kill $DMON $SAMPLER $SLICE $VMSTAT 2>/dev/null; wait $DMON $SAMPLER $SLICE $VMSTAT 2>/dev/null
if [ -n "$NCW" ]; then
  for f in $R/out/ncu/passes-*.ncu-rep $R/out/ncu/m35-*.ncu-rep; do [ -f "$f" ] && $NC --import $f --page raw --csv > ${f%.ncu-rep}.raw.csv; done
  [ $ok = 1 ] && ls $R/out/ncu/b16-*.ncu-rep $R/out/ncu/m35-*.ncu-rep > /dev/null; exit
fi
if [ -n "$NSW" ]; then
  for f in $R/out/nsys/*.nsys-rep; do
    [ -f "$f" ] && $NS stats -q -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum -f csv -o ${f%.nsys-rep} $f \
      >> $R/out/nsys/stats.log 2>&1
  done
  [ $ok = 1 ] && ls $R/out/nsys/*.nsys-rep > /dev/null; exit
fi
if [ "${STAGE_ONLY:-0}" = 1 ]; then
  cp $S/gate/classes/staged.jsonl $R/out/gate-staged.jsonl; cp $S/m35/classes/staged.jsonl $R/out/staged.jsonl
  [ $ok = 1 ] && grep -q '"stage"' $R/out/staged.jsonl; exit
fi
# the contract's own reader (verity_numerical.bench.contract) needs the tree's packages: the prover's venv, as 70-class-sweep.sh runs
PYBIN=${PYBIN:-$FLOCK_WORK/flock-circuit/py/bin/python3}; [ -x $PYBIN ] || PYBIN=python3
PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python \
  $PYBIN backends/flock/pod/gemm_hill.py metrics --dtype $DTYPE --K $K --gate $R/out/gate/classes/results.jsonl --sweep $R/out/results.jsonl \
  --stamps $R/out/stamps.tsv --warm ${WARM:-1} --runs ${RUNS:-3} --dmon $R/out/dmon.tsv --phases $R/out/phases.tsv --others $R/out/others_cores.txt --slice $R/out/slice_cores.tsv \
  --vmstat $R/out/vmstat.tsv \
  --gpu-csv $R/out/gpu.csv --census census/hardware.json ${PEAK_ID:+--peak-id $PEAK_ID} --binary-sha $R/out/binary.sha256 --step ${STEP:-0} \
  --label "${LABEL:-baseline}" ${QUESTION:+--question "$QUESTION"} --node ${HILL_NODE:-vy-nebius-1} --out $R/out/metrics.json --result $R/result.json --hillclimb $R/hillclimb.json || ok=0
[ -f $R/hillclimb.json ] && printf '%s\n' '{"schema": "research/outputs/v0.1", "outputs": [{"name": "hillclimb", "kind": "hillclimb-point/v0", "file": "hillclimb.json", "meta": {"campaign": "proofs-hillclimb"}}]}' > $R/outputs.json
[ $ok = 1 ] && [ $zok = 1 ] && grep -q '"status": "passed"' $R/result.json
