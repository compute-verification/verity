#!/usr/bin/env bash
# flock-vllm-v1: the Flock cell over vllm-v1 commitments (verity/flock-vllm-block/v1, flock-vllm-v1). Builds with
# 20-gpu-link.sh's checkout, then
#   ROLE=prover   (GPU) CPU and GPU selftests at SELF="8 64", then the TABLES sweep (verity_flock.bench --scheme vllm-v1) at
#                 POINTS against VERIFIER=host:port (RTT_TARGET=host:sshport; LOOPBACK=1: the same-run loopback probe)
#   ROLE=verifier (CPU) its own instance files for POINTS, serving the prover's order on :7400
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-vllm-v1}/out; W=/workspace/flock-vllm-v1; mkdir -p $O $W
ROLE=${ROLE:-prover}; REL=${REL:-fp8-hopper}; POINTS=${POINTS:-1024 2048 4096 8192 16384}; PER=${PER:-8192}; RUNS=${RUNS:-5}
G=1; [ $ROLE = verifier ] && G=0
[ -x /usr/bin/time ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq time >/dev/null; }
MODE=build GPU=$G bash verity/ml/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER_=100000; }
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER_ ))); export RAYON_NUM_THREADS=$TH
F=/workspace/flock-gpu-link/flock; FEAT=; [ $G = 1 ] && FEAT="--features gpu"
(cd $F && cargo build --release -p flock-live $FEAT --bin flock-vllm-v1 -j $TH) >> $O/build.log 2>&1 || { tail -40 $O/build.log; exit 1; }
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python
B=$F/target/release/flock-vllm-v1; sha256sum $B | tee $O/binary.sha256
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
if [ $ROLE = verifier ]; then
  python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/verifier --points "$POINTS" --per-proof $PER --runs $RUNS \
    --procs $TH --scheme vllm-v1 --serve-plan --listen 0.0.0.0:7400 --operator ${OPERATOR:-flock-vllm-v1-verifier-pod} 2>&1 | tee $O/serve-plan.log
  exit 0
fi
python3 - <<EOF
from verity_flock import instances, lowering
open("$O/net-$REL.txt", "w").write(lowering.netlist("$REL"))
for n in "${SELF:-8 64}".split():
    instances.write(f"$O/self-{n}.bin", "$REL", int(n), procs=$TH, scheme="vllm-v1")
EOF
for n in ${SELF:-8 64}; do
  $B selftest --instances $O/self-$n.bin --netlist $O/net-$REL.txt > $O/selftest-cpu-$n.txt 2>&1; grep -h SELFTEST $O/selftest-cpu-$n.txt
  [ $G = 1 ] && { $B selftest --gpu --instances $O/self-$n.bin --netlist $O/net-$REL.txt > $O/selftest-gpu-$n.txt 2>&1
                  grep -h 'SELFTEST\|"pass":false' $O/selftest-gpu-$n.txt | cut -c1-400; }
done
[ "${SELFTEST_ONLY:-0}" = 1 ] && exit 0
python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/cell --points "$POINTS" --per-proof $PER --runs $RUNS \
  --procs $TH --scheme vllm-v1 --verifier ${VERIFIER:?} --operator ${OPERATOR:-flock-vllm-v1-verifier-pod} --prove-arg=--gpu \
  ${RTT_TARGET:+--rtt-target $RTT_TARGET} ${LOOPBACK:+--loopback} 2>&1 | tee $O/sweep.log | cut -c1-600
true
