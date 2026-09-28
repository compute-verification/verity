#!/usr/bin/env bash
# GEMM proving slowdown, one attempt (gemm_slowdown.py): the preflight check (byte identity at B = 16: the GPU's proofs and
# transcripts equal the CPU prover's, and the device paths agree), then both GEMM coordinates timed at their largest statement,
# then the same GPU's native GEMMs; result.json carries the prefill and decode slowdowns. Run from the verity tree (research run
# --cwd source).
# env: PREFLIGHT=1 (0: skip the preflight check; GATE= is its old name)  WARM=1 RUNS=3  BATCH_ANDS=2^32
#      MAX_STATEMENT_BITS=2^34  ITERS=100
#      DEFS=verity/ml/flock/pod/gemm-coordinates.defs.json (the Call counts: gemm-coordinates-4k fills 256 tiles of K = 2048)
#      GPUS="4 5 6 7" (on a machine with /run/gpu-lease: take the first free of these, as gpu-lease does)
#      any other KEY=VALUE reaches the prover's environment (FC_NATIVE_SHA=0, FC_HOST_PINNED=0, ...)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/ovg}; mkdir -p $R/out
DEFS=$REPO/${DEFS:-backends/flock/pod/gemm-coordinates.defs.json}
if [ -d /run/gpu-lease ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  for i in ${GPUS:-4 5 6 7}; do
    exec 9>>/run/gpu-lease/$i.lock
    if flock -n -x 9; then
      export CUDA_VISIBLE_DEVICES=$i CUDA_DEVICE_ORDER=PCI_BUS_ID
      echo "$$ ${RESEARCH_RUN_ID:-} $(date -u +%FT%TZ)" > /run/gpu-lease/$i.owner
      break
    fi
    exec 9>&-
  done
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no free GPU among ${GPUS:-4 5 6 7}"; exit 75; }
fi
# a machine whose /workspace the run's user can't write (vy-nebius-1): the builds and venvs go under its cache, or under
# /workspace/jobs in a Kueue job (another uid); staged statements are kept there for the next attempt (FLOCK_STAGE_CACHE)
if [ ! -w /workspace ] && [ -z "${FLOCK_WORK:-}" ]; then
  if mkdir -p /workspace/cache/flock 2>/dev/null && [ -w /workspace/cache/flock ]; then export FLOCK_WORK=/workspace/cache/flock
  else export FLOCK_WORK=/workspace/jobs/flock-m0; mkdir -p $FLOCK_WORK/home; export HOME=$FLOCK_WORK/home; fi
fi
export FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-${FLOCK_WORK:-/workspace}/stage-cache}
# a Kueue job's template sets its own cargo target dir; the builds here keep theirs. Its pod has no CPU limit: the prover's
# threads are the job's vCPUs
unset CARGO_TARGET_DIR
[ -n "${RAYON_NUM_THREADS:-}" ] || export RAYON_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
# CUDA 13.3 (upstream Flock's clmad needs its ptxas) where it isn't installed and can't be, without root: its packages unpacked
# under FLOCK_WORK (apt-get download needs no root); no compat library: an R580 driver runs 13.3 by minor-version compatibility
for c in ${FLOCK_WORK:-/workspace}/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
if [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ]; then
  CR=${FLOCK_WORK:-/workspace}/cuda-root; export CUDA_TK=$CR/usr/local/cuda-13.3
  if [ ! -x $CUDA_TK/bin/nvcc ]; then
    # the 13-3 packages' dependency closure, from apt-cache show (a pinned-out package has no candidate for apt-cache depends)
    todo="cuda-nvcc-13-3 cuda-cudart-dev-13-3"; seen=
    while [ -n "$todo" ]; do
      set -- $todo; q=$1; shift; todo="$*"
      case " $seen " in *" $q "*) continue ;; esac
      seen="$seen $q"
      todo="$todo $(apt-cache show $q 2>/dev/null | awk '/^Depends:/ {sub(/^Depends: /, ""); print; exit}' | tr ',' '\n' \
        | sed 's/(.*//; s/|.*//; s/ //g' | grep -E -- '-13-3$' | tr '\n' ' ')"
    done
    mkdir -p $CR/debs && (cd $CR/debs && apt-get download $(for q in $seen; do echo "$q=$(apt-cache show $q | awk '/^Version:/ {print $2; exit}')"; done)) \
      && for d in $CR/debs/*.deb; do dpkg-deb -x $d $CR; done
    [ -x $CUDA_TK/bin/nvcc ] || { echo "CUDA 13.3 unpack failed"; exit 1; }
  fi
fi
{ nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; nproc; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-}"; } | tee $R/out/host.txt
# cores busy outside this attempt every 30 s (Builds overlapping it: ov.noisy)
python3 verity/ml/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $R/out/others_cores.txt &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
PREFLIGHT=${PREFLIGHT:-${GATE:-1}}
if [ "$PREFLIGHT" = 1 ]; then
  OUT=$R/out/preflight bash verity/ml/flock/pod/70-class-sweep.sh DEFS=$DEFS BATCH=16 WARM=0 RUNS=1 SELFTEST=1 SELFTEST_GPU=1 \
    SELFTEST_CASES=gpu_paths_agree,gpu_proofs_match_cpu,verify_ahead_matches_serial || exit 1
fi
# NSYS=1: a profile, not an attempt: the m = 34 prove sessions under Nsight Systems' CUDA trace (unpacked like CUDA 13.3), each
# process's kernel, memcpy and API summaries in out/nsys; no native GEMMs, no result.json
if [ "${NSYS:-0}" = 1 ]; then
  NR=${FLOCK_WORK:-/workspace}/nsys-root; NS=$NR/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
  if [ ! -x $NS ]; then
    # a job's image has no CUDA apt source: the package from NVIDIA's repository directly
    mkdir -p $NR/debs && (cd $NR/debs && { apt-get download nsight-systems-2026.1.3 2>/dev/null || curl -fsSLO \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb; }) || exit 1
    for d in $NR/debs/*.deb; do dpkg-deb -x $d $NR; done
  fi
  mkdir -p $R/out/nsys; export WRAP="$NS profile -t cuda --sample=none --cpuctxsw=none -o $R/out/nsys/prove-%p"
fi
OUT=$R/out/m34 bash verity/ml/flock/pod/70-class-sweep.sh DEFS=$DEFS BATCH=${BATCH:-auto} ${N:+N=$N} BATCH_ANDS=${BATCH_ANDS:-4294967296} \
  MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-17179869184} WARM=${WARM:-1} RUNS=${RUNS:-3} SELFTEST=0 || exit 1
if [ "${NSYS:-0}" = 1 ]; then
  for f in $R/out/nsys/*.nsys-rep; do
    $NS stats -q -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum -f csv -o ${f%.nsys-rep} $f > /dev/null
  done
  exit 0
fi
W=${FLOCK_WORK:-/workspace}/flock-circuit; UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $W/torch/bin/python ] || { $UV venv -q --python 3.12 $W/torch && $UV pip install -q --python $W/torch/bin/python 'torch>=2.7' \
  --index-url https://download.pytorch.org/whl/cu128; } || exit 1
$W/torch/bin/python verity/ml/flock/pod/gemm_slowdown.py native --out $R/out/native.json --iters ${ITERS:-100} || exit 1
G=; [ "$PREFLIGHT" = 1 ] && G="--preflight $R/out/preflight/classes/results.jsonl"
$W/py/bin/python3 verity/ml/flock/pod/gemm_slowdown.py compute --sweep $R/out/m34/classes/results.jsonl $G --native $R/out/native.json \
  --result $R/result.json
