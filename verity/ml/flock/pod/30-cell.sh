#!/usr/bin/env bash
# flock-backend: a Flock Table 2 cell with flock-gpu-link's flock-pure-gpu (verity/flock-pure-block): the TABLES sweep through
# verity_flock.bench, prover on the line's GPU, verifier on a separate same-datacenter pod.
#   ROLE=prover   REL POINTS PER RUNS VERIFIER=host:port  (GPU build; sweep against the verifier)
#   ROLE=verifier REL POINTS PER RUNS OPERATOR=lane       (CPU build; its own instance files; serves the prover's order on :7400)
#   SET=name.tar  a spine input set sent with the run (inputs/name.tar or .tgz: its manifest.json and port files, at the top
#                 or in one directory, as bench.cell's driver packs it), both roles
#   YMODEL=1      with SET: y is REL's chain output (a set captured on other silicon)
#   K=2048        an 8-bit relation's synthetic E4M3 stream at K words
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-cell}/out; W=/workspace/flock-backend; mkdir -p $O $W
ROLE=${ROLE:-prover}; REL=${REL:-bf16-hopper}; POINTS=${POINTS:-1024 2048 4096 8192 16384 32768}; PER=${PER:-8192}; RUNS=${RUNS:-5}
G=1; [ $ROLE = verifier ] && G=0
[ -x /usr/bin/time ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq time >/dev/null; }
MODE=build GPU=$G bash verity/ml/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { read -r Q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us && read -r PER_ < /sys/fs/cgroup/cpu/cpu.cfs_period_us; } 2>/dev/null || { Q=max; PER_=100000; }; [ "$Q" = -1 ] && Q=max
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER_ ))); TH=${THREADS:-$TH}; export RAYON_NUM_THREADS=$TH
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python
if [ "$REL" = bf16-ampere ] && [ -z "${SET:-}" ] && [ -z "${K:-}" ]; then
  # the A100 row's frozen bench-instances/v1 set: rebuilt here from the committed seeds, refused unless every array matches the
  # committed manifest's sha256, then linked into the source tree (as pod_bootstrap.sh BENCH_INSTANCES=1 does)
  BI=$W/bench-instances/v1; FZ=$REPO/fixtures/bench-instances/v1
  [ -f $BI/vu-k1536.x.u16 ] || python3 -m verity_numerical.bench.instances build --out $BI --seeds $FZ/seeds --procs $TH 2>&1 | tail -3
  python3 - $BI $FZ <<'PYEOF' | tee $O/bench-instances.txt || exit 4
import json, sys
from pathlib import Path
from verity_numerical.bench.instances import sha256_file
bi, fz = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((fz / "manifest.json").read_text())
built = {n: f for t in man["tiers"].values() for n, f in t.get("files", {}).items() if not f.get("committed")}
bad = [n for n, f in built.items() if not (bi / n).is_file() or sha256_file(bi / n) != f["sha256"]]
assert not bad, f"rebuilt arrays differ from the committed manifest: {bad}"
for n in built:
    dst = fz / n
    dst.unlink(missing_ok=True)
    dst.symlink_to(bi / n)
print(f"{len(built)} rebuilt arrays match the committed manifest; linked into {fz}")
PYEOF
fi
B=/workspace/flock-gpu-link/flock/target/release/flock-pure-gpu; sha256sum $B | tee $O/binary.sha256
SRC=""
if [ -n "${SET:-}" ]; then
  S=$W/set-${SET%%.*}; rm -rf $S; mkdir -p $S; tar xf ${RESEARCH_RUN_DIR:?}/inputs/$SET -C $S || exit 5
  S=$(dirname "$(find $S -name manifest.json | head -1)")
  (cd $S && sha256sum manifest.json $(ls | grep -E '^[xwy]\.u(8|16|32)$')) | tee $O/set.sha256; SRC="--set $S"
fi
[ -n "${K:-}" ] && SRC="$SRC --k $K"
[ -n "${YMODEL:-}" ] && SRC="$SRC --y-model"
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
if [ $ROLE = verifier ]; then
  python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/verifier --points "$POINTS" --per-proof $PER --runs $RUNS \
    --procs $TH --leaf ${LEAF:-${SCHEME:-blake3}} $SRC --serve-plan --listen 0.0.0.0:7400 --operator ${OPERATOR:-flock-backend-verifier-pod} 2>&1 | tee $O/serve-plan.log
else
  python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/cell --points "$POINTS" --per-proof $PER --runs $RUNS \
    --procs $TH --leaf ${LEAF:-${SCHEME:-blake3}} $SRC --verifier ${VERIFIER:?} --operator ${OPERATOR:-flock-backend-verifier-pod} --prove-arg=--gpu ${RTT_TARGET:+--rtt-target $RTT_TARGET} ${LOOPBACK:+--loopback} 2>&1 | tee $O/sweep.log | cut -c1-600
fi
true
