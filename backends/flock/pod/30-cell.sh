#!/usr/bin/env bash
# flock-backend: a Flock Table 2 cell with flock-gpu-link's flock-pure-gpu (verity/flock-pure-block): the TABLES sweep through
# verity_flock.bench, prover on the line's GPU, verifier on a separate same-datacenter pod.
#   ROLE=prover   REL POINTS PER RUNS VERIFIER=host:port  (GPU build; sweep against the verifier)
#   ROLE=verifier REL POINTS PER RUNS OPERATOR=lane       (CPU build; its own instance files; serves the prover's order on :7400)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-cell}/out; W=/workspace/flock-backend; mkdir -p $O $W
ROLE=${ROLE:-prover}; REL=${REL:-bf16-hopper}; POINTS=${POINTS:-1024 2048 4096 8192 16384 32768}; PER=${PER:-8192}; RUNS=${RUNS:-5}
G=1; [ $ROLE = verifier ] && G=0
[ -x /usr/bin/time ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq time >/dev/null; }
MODE=build GPU=$G bash backends/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER_=100000; }
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER_ ))); export RAYON_NUM_THREADS=$TH
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python
B=/workspace/flock-gpu-link/flock/target/release/flock-pure-gpu; sha256sum $B | tee $O/binary.sha256
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
if [ $ROLE = verifier ]; then
  python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/verifier --points "$POINTS" --per-proof $PER --runs $RUNS \
    --procs $TH --serve-plan --listen 0.0.0.0:7400 --operator ${OPERATOR:-flock-backend-verifier-pod} 2>&1 | tee $O/serve-plan.log
else
  python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/cell --points "$POINTS" --per-proof $PER --runs $RUNS \
    --procs $TH --verifier ${VERIFIER:?} --operator ${OPERATOR:-flock-backend-verifier-pod} --prove-arg=--gpu 2>&1 | tee $O/sweep.log | cut -c1-600
fi
true
