#!/usr/bin/env bash
# flock-backend: build flock-pure into a flock b684b12 checkout (flock-link's patch, verity/ml/flock/live as crates/flock-live,
# verity_unit.rs into flock-prover), then (MODE):
#   selftest  flock-pure's negatives at SELF_VUS for REL (in-process verifier)
#   sweep     the TABLES.md sweep (verity_flock.bench) from START, RUNS timed sessions per point, loopback verifier
#   serve     only a verifier for REL at SERVE_VUS on 0.0.0.0:7400 (a verifier pod; VERIFIER=host:port on the prover)
#   replay    re-verify recorded sessions: REPLAY_DIR holds pN-*/ point dirs (instances regenerated here, not read)
# env: REL=bf16-hopper MODE="selftest sweep" SELF_VUS="8 64" START=1024 MAX=65536 RUNS=5 VERIFIER= SERVE_VUS=4096 CUDA=
# Settings also come as K=V arguments (the research Tool's key params: verity/ml/flock/tool.py).
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
W=/workspace/flock-backend; F=$W/flock; O=${RESEARCH_RUN_DIR:-/tmp/flock-backend}/out; mkdir -p $W $O
REPO=$(pwd); SRC=$REPO/verity/ml/flock
REL=${REL:-bf16-hopper}; MODE=${MODE:-selftest sweep}
command -v cc >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential pkg-config >/dev/null; }
[ -x /usr/bin/time ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq time >/dev/null; }
[ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
source $HOME/.cargo/env
# Python 3.12 (the core's syntax) in a venv of its own, with numpy and blake3
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH; python3 --version
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python
cd $W; [ -d flock ] || git clone -q https://github.com/succinctlabs/flock.git
cd $F && git checkout -q b684b12 && git checkout -q -- . && git clean -qfd crates && git apply $SRC/flock-link-b684b12.patch && git apply $SRC/flock-gpu-link-b684b12.patch || exit 1
rm -rf crates/flock-live && cp -r $SRC/live crates/flock-live
grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
cp $SRC/verity_unit.rs crates/flock-prover/src/r1cs_hashes/verity_unit.rs
grep -q 'pub mod verity_unit;' crates/flock-prover/src/r1cs_hashes.rs || printf '\npub mod verity_unit;\n' >> crates/flock-prover/src/r1cs_hashes.rs
QUOTA=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); TH=${QUOTA:-$(nproc)}
export RAYON_NUM_THREADS=$TH
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo "threads $TH" | tee -a $O/host.txt; free -g | tee -a $O/host.txt
git -C $F diff --stat | tail -1 | tee $O/patch-stat.txt
( time cargo build --release -p flock-live --features verity-unit --bin flock-pure -j $TH ) 2>&1 | grep -E '^error|-->|Finished|real' | head -40
B=$F/target/release/flock-pure; [ -x $B ] || exit 1
rustc --version | tee -a $O/host.txt
cd $REPO
python3 - <<EOF | tee $O/lowering.txt
from verity_flock import lowering
import hashlib
for r in ("$REL",) + (("fp8-hopper",) if "$REL" != "fp8-hopper" else ("bf16-hopper",)):
    t = lowering.netlist(r); open("$O/net-%s.txt" % r, "w").write(t)
    d = hashlib.sha256(t.encode()).hexdigest(); print(r, d, "PINNED" if d == lowering.PINS[r] else "NOT-PINNED")
EOF
grep -q "^$REL .* PINNED" $O/lowering.txt || exit 3
OTHER=$(awk '$1 != "'$REL'" {print $1}' $O/lowering.txt | head -1)
PIN=$(awk '$1 == "'$REL'" {print $2}' $O/lowering.txt)
for mode in $MODE; do case $mode in
selftest)
  for v in ${SELF_VUS:-8 64}; do
    python3 -m verity_flock.instances $REL $v $O/inst-$REL-$v.bin --procs $TH | tee -a $O/instances.txt
    $B selftest --instances $O/inst-$REL-$v.bin --netlist $O/net-$REL.txt --netlist-other $O/net-$OTHER.txt 2>$O/selftest-$v.err \
      | tee $O/selftest-$REL-$v.txt | grep -E '^(NEG|SELFTEST)' | cut -c1-240
  done
  grep -h '^NEG' $O/selftest-$REL-*.txt > $O/negatives-$REL.tsv ;;
sweep)
  python3 -m verity_flock.bench --relation $REL --bin $B --out-dir $O/sweep-$REL --start ${START:-1024} --max ${MAX:-65536} \
    --runs ${RUNS:-5} --procs $TH ${VERIFIER:+--verifier $VERIFIER} 2>&1 | tee $O/sweep-$REL.log | cut -c1-600 ;;
serve)
  python3 -m verity_flock.instances $REL ${SERVE_VUS:-4096} $O/inst-$REL-serve.bin --procs $TH
  $B serve --listen 0.0.0.0:7400 --out $O/sessions-$REL --pin $PIN --instances $O/inst-$REL-serve.bin --netlist $O/net-$REL.txt ;;
replay)
  for d in ${REPLAY_DIR:?}/p*-*/; do
    n=$(basename $d | cut -d- -f2)
    python3 -m verity_flock.instances $REL $n $O/inst-$REL-$n.bin --procs $TH >/dev/null
    for s in $d/sessions/p*/; do
      i=$(basename $s | cut -c2-5); r=$(printf "%02d" $((10#$i)))
      $B replay --instances $O/inst-$REL-$n.bin --netlist $O/net-$REL.txt --pin $PIN --session $s/session.json \
        --proofs $d/proofs/run$r-rep0.proof,$d/proofs/run$r-rep1.proof 2>/dev/null | sed "s|^|$(basename $d)/$(basename $s)\t|" | tee -a $O/replay-$REL.tsv | cut -c1-240
    done
  done ;;
esac; done
true
