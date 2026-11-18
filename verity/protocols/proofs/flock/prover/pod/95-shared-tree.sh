#!/usr/bin/env bash
# The shared-tree toy (tests/shared_tree.py): one output tree for (T, out) that writers M1 = units [0, n/2) and M2 =
# [n/2, n) open by path, and a reader R = units [n, 2n) whose `x` is a link row reading writer n-1-i's row by path in it
# (a tree port, its reads the program's) and whose own outputs are its leaves [n, 2n). Each statement is proved in one loopback --zk session (every unit sampled) and verified by Lean's
# `verity-verify-proof verify --zk --registered` with its own file; ra, rb, mc and rd are the four tamperings, each
# refused. Run from the verity tree (research run --cwd source); CPU only. Scratch is a `mktemp -d` under FLOCK_WORK's
# scratch/, deleted at exit. Outputs under out/: host.txt, stage-8.json and stage-64.json (each statement's public file
# size and the shared rows' strings it holds), the public files and own files at n = 8, and per statement serve.log,
# prove.txt, prove.err, challenger-NAME/ and lean.log (its VERDICT line); verdicts.txt sums them.
# env: CPUS=16  N=8 (the proved size)  SIZES="8 64" (the staged sizes)  CASES="m1 m2 r ra rb mc rd"
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${ST_PINNED:-}" ]; then
  if mkdir -p "${SLICE_LOCKS:-/workspace/jobs/slices}" 2>/dev/null; then
    . verity/protocols/proofs/flock/prover/pod/cpu-slices.sh
    hold_slices "" "${CPUS:-16}" || exit 1
    export ST_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
    exec taskset -c $SLICE_CPUS bash "$0" "$@"
  fi
  export ST_PINNED=slot RAYON_NUM_THREADS=$(nproc)
fi
export FC_COINS=tree
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/shared-tree}/out; mkdir -p $O
. verity/protocols/proofs/spot/verifier/one_stage/verifier_env.sh
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/served-registered; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W $FW/scratch || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$(mktemp -d -p $FW/scratch shared-tree.XXXXXX) || exit 1
L=$FW/cache/served-registered; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$ST_PINNED rayon=$RAYON_NUM_THREADS"; } > $O/host.txt 2>&1

LSHA=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py key verity/protocols/proofs/flock/verifier)
LEAN_BIN=$L/lean-$LSHA/verity-verify-proof
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ]; then
    rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py copy verity/protocols/proofs/flock/verifier $L/lean-src) \
      && (cd $LEAN_SRC && lake build verity-verify-proof) && mkdir -p $L/lean-$LSHA && cp $LEAN_SRC/.lake/build/bin/verity-verify-proof $LEAN_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!
CHALLENGER_REPO=$REPO CHALLENGER_WORK=$S/challenger-work
. verity/protocols/proofs/flock/prover/pod/challenger.sh
challenger_build

KEY=$(cat verity/protocols/proofs/flock/prover/pod/20-gpu-link.sh verity/protocols/proofs/flock/prover/pod/60-circuit.sh verity/protocols/proofs/flock/prover/*.patch verity/protocols/proofs/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/proofs/flock/prover/cuda verity/protocols/proofs/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/proofs/flock/prover/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch

N=${N:-8}
for n in ${SIZES:-8 64}; do
  python3 tests/verity/protocols/proofs/flock/prover/shared_tree.py --out $S/n$n --n $n > $O/stage-$n.json 2> $O/stage-$n.err \
    || { tail -20 $O/stage-$n.err; exit 1; }
done
D=$S/n$N
mkdir -p $O/n$N && cp $D/*.pub $D/*.reads.json $D/*.links.json $D/program.json $D/partition.json $O/n$N/ || exit 1
wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
challenger_built || exit 1
CHALLENGER_PY=python3
: > $O/verdicts.txt
for c in ${CASES:-m1 m2 r ra rb mc rd}; do
  C=$O/$c; mkdir -p $C
  K=$(python3 -c 'import json, sys; u = json.load(open(sys.argv[1]))["statements"][sys.argv[2]]["units"]; print(u[1] - u[0])' $O/stage-$N.json $c)
  CIR=$D/$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["statements"][sys.argv[2]]["circuit"])' $O/stage-$N.json $c)
  PIN=$(sha512sum $CIR | cut -d' ' -f1)
  LINKS=(); [ -f $D/$c.links.json ] && LINKS=(--links $D/$c.links.json)
  $B statement --zk --circuit $CIR --instances $D/$c.pub --tables $D/tables "${LINKS[@]}" > $C/statement.txt 2>&1
  challenger_start $C/challenger-$c || { echo "$c challenger refused" >> $O/verdicts.txt; continue; }
  PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  $B serve --zk --listen 127.0.0.1:$PORT --out $S/sessions-$c --instances $D/$c.pub --circuit $CIR --tables $D/tables \
    --pin $PIN "${LINKS[@]}" --operator shared-tree --draw subset:$K --sessions 1 "${TREE_FROM[@]}" > $C/serve.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $C/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  $B prove --zk --verifier 127.0.0.1:$PORT --instances $D/$c.inst --circuit $CIR --tables $D/tables --runs 1 --warm 0 \
    > $C/prove.txt 2> $C/prove.err
  echo "$c prove rc $?" >> $O/verdicts.txt
  for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill $SP 2>/dev/null; wait $SP
  challenger_end $C/challenger-$c
  SESS=$(dirname "$(ls -1 $S/sessions-$c/*/session.json 2>/dev/null | LC_ALL=C sort | tail -1)")
  [ -f "$SESS/session.json" ] || { echo "$c no session" >> $O/verdicts.txt; continue; }
  challenger_audit $C/challenger-$c $SESS > $C/audit.txt 2>&1; echo "$c audit rc $?" >> $O/verdicts.txt
  $(verifier_env) $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $CIR --public $D/$c.pub \
    --partition $D/partition.json --program $D/program.json --registered $D/$c.reads.json --session $SESS > $C/lean.log 2>&1
  echo "$c lean rc $?" >> $O/verdicts.txt
  grep -h '^VERDICT' $C/lean.log | head -1 | cut -c1-2000 >> $O/verdicts.txt
  tail -3 $C/lean.log | cut -c1-600 >> $O/verdicts.txt
done
cat $O/verdicts.txt
