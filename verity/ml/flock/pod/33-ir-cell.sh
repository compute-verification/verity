#!/usr/bin/env bash
# flock-ir-lowering: a C-Flock cell of an IR template under frame-v3 (verity/flock-ir-frame/v3) through the bench spine
# (bench.drivers.c_interactive): the sweep verity_flock.ir_bench, prover on the line's GPU, verifier on a separate pod. The
# cell's input set arrives as inputs/<content digest>.tar (research run --send); each pod stages its own instance files from it.
#   ROLE=prover   TEMPLATE POINTS PER RUNS VERIFIER=host:port RTT_TARGET=host:22 LOOPBACK=1 OPERATOR [ART]   (GPU build)
#   ROLE=verifier TEMPLATE POINTS PER RUNS OPERATOR                                                        (CPU build; serves on :7400)
# The prover builds Flock-CUDA for its own GPU's compute capability (SM, from nvidia-smi: 90 H100, 89 L40S / RTX 4090).
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=${RESEARCH_RUN_DIR:-/tmp/flock-ir-cell}/out; W=/workspace/flock-ir; mkdir -p $O $W
ROLE=${ROLE:-prover}; TEMPLATE=${TEMPLATE:?}; POINTS=${POINTS:?}; PER=${PER:-1073741824}; RUNS=${RUNS:-5}
G=1; [ $ROLE = verifier ] && G=0
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-90}
[ -x /usr/bin/time ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq time >/dev/null; }
MODE=build GPU=$G SM=$SM bash verity/ml/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
# the cgroup's CPU quota (v2 cpu.max, else v1 cfs_quota_us / cfs_period_us, as timing_guard.cpu_quota), rounded up: more
# threads than the quota only get throttled
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER_=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); }
TH=$( { [ "$Q" = max ] || [ "$Q" = -1 ]; } && nproc || echo $(( (Q + PER_ - 1) / PER_ )) ); export RAYON_NUM_THREADS=$TH
F=/workspace/flock-gpu-link/flock; FEAT=; [ $G = 1 ] && FEAT="--features gpu"
BIN=flock-ir-frame; [ "$TEMPLATE" = gumbel-top-p-token-select ] && BIN=flock-ir-sampling   # verity/flock-ir-sampling/v1's own binary
(cd $F && cargo build --release -p flock-live $FEAT --bin $BIN -j $TH) > $O/build-ir.txt 2>&1 || { tail -40 $O/build-ir.txt; exit 1; }
B=$F/target/release/$BIN; sha256sum $B | tee $O/binary.sha256
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
rm -rf $W/set && mkdir -p $W/set && tar -xf $(ls $I/*.tar | head -1) -C $W/set || exit 1
SET=$(dirname $(find $W/set -name manifest.json | head -1))
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
if [ $ROLE = verifier ]; then
  python3 -m verity_flock.ir_bench --template $TEMPLATE --input-set $SET --bin $B --out-dir $O/verifier --points "$POINTS" --per-proof $PER \
    --runs $RUNS --serve-plan --listen 0.0.0.0:7400 --operator ${OPERATOR:-flock-ir-lowering-verifier-pod} 2>&1 | tee $O/serve-plan.log
else
  python3 -m verity_flock.ir_bench --template $TEMPLATE --input-set $SET ${ART:+--art $ART} --bin $B --out-dir $O/cell --points "$POINTS" \
    --per-proof $PER --runs $RUNS --verifier ${VERIFIER:?} --operator ${OPERATOR:-flock-ir-lowering-verifier-pod} --prove-arg=--gpu \
    ${RTT_TARGET:+--rtt-target $RTT_TARGET} ${LOOPBACK:+--loopback} 2>&1 | tee $O/sweep.log | cut -c1-600
fi
true
