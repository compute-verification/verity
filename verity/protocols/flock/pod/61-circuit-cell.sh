#!/usr/bin/env bash
# flock-circuit: a Flock cell of an IR template under verity/flock-circuit through the bench spine
# (bench.drivers.c_interactive, relation circuit:<template>): the sweep verity_flock.circuit_bench, the prover on the line's GPU, the
# verifier on a separate pod serving its own public files (no rows). The cell's input set arrives as inputs/<digest>.tar.
#   ROLE=prover   TEMPLATE POINTS PER RUNS VERIFIER=host:port RTT_TARGET=host:22 LOOPBACK=1 OPERATOR [ART]   (GPU build)
#   ROLE=verifier TEMPLATE POINTS PER RUNS OPERATOR                                                        (CPU build; serves on :7501)
#   FC_COINS=tree (both roles: the challenger's keyed coin tree, every server the verifier pod or LOOPBACK=1 runs on its own
#   challenger, pod/challenger.sh, its records audited against the challenger's release log; empty: the binary's default,
#   seed coins)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
export FC_COINS=${FC_COINS:-tree}
if [ "$FC_COINS" = tree ]; then export FLOCK_CHALLENGER=${FLOCK_CHALLENGER:-1}; else export FLOCK_CHALLENGER=0; fi
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=${RESEARCH_RUN_DIR:-/tmp/flock-circuit-cell}/out; W=/workspace/flock-circuit; mkdir -p $O $W
ROLE=${ROLE:-prover}; TEMPLATE=${TEMPLATE:?}; POINTS=${POINTS:?}; PER=${PER:-1073741824}; RUNS=${RUNS:-5}
G=1; [ $ROLE = verifier ] && G=0
[ -x /usr/bin/time ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq time >/dev/null; }
MODE=build GPU=$G bash verity/protocols/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER_=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); }
# one core under the quota's floor: the host witness (row inputs, hm96 slots) runs beside the CUDA driver's threads, and a
# pool at the quota throttles the cgroup, which the timing guard reads as contention
TH=$( { [ "$Q" = max ] || [ "$Q" = -1 ]; } && nproc || echo $(( Q / PER_ > 2 ? Q / PER_ - 1 : 1 )) ); export RAYON_NUM_THREADS=$TH
B=/workspace/flock-gpu-link/flock/target/release/flock-circuit; sha256sum $B | tee $O/binary.sha256
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
# the challengers' Lean, where this pod serves sessions: the verifier role's servers and LOOPBACK=1's
if [ "$FLOCK_CHALLENGER" = 1 ] && { [ $ROLE = verifier ] || [ -n "${LOOPBACK:-}" ]; }; then
  export CHALLENGER_REPO=$REPO CHALLENGER_PY=python3 CHALLENGER_WORK=$O/challenger-build
  . verity/protocols/flock/prover/pod/challenger.sh
  challenger_build; challenger_built || exit 1
fi
rm -rf $W/set && mkdir -p $W/set && tar -xf $(ls $I/*.tar | head -1) -C $W/set || exit 1
SET=$(dirname $(find $W/set -name manifest.json | head -1))
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
if [ $ROLE = verifier ]; then
  python3 -m verity_flock.circuit_bench --template $TEMPLATE --input-set $SET --bin $B --out-dir $O/verifier --points "$POINTS" --per-proof $PER \
    --runs $RUNS --serve-plan --listen 0.0.0.0:7501 --operator ${OPERATOR:-flock-circuit-verifier-pod} 2>&1 | tee $O/serve-plan.log
else
  python3 -m verity_flock.circuit_bench --template $TEMPLATE --input-set $SET ${ART:+--art $ART} --bin $B --out-dir $O/cell --points "$POINTS" \
    --per-proof $PER --runs $RUNS --verifier ${VERIFIER:?} --operator ${OPERATOR:-flock-circuit-verifier-pod} --prove-arg=--gpu \
    ${RTT_TARGET:+--rtt-target $RTT_TARGET} ${LOOPBACK:+--loopback} 2>&1 | tee $O/sweep.log | cut -c1-600
fi
true
