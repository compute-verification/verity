#!/usr/bin/env bash
# The sessions-of-several-tables fixture (tests/test_lean_session_tables.py): RoPE d64, 8 instances of the committed
# synthetic set (seed 20260928) staged by verity_flock.circuit (circuit.txt, the verifier's pub-8.bin), proved by a
# selftest build (seed-injection, on seed coins) in loopback sessions of several tables: sessions/j2 (J = 2, no draw),
# sessions/j2-draw (J = 2, `--draw subset:4`, two sessions) and sessions/j4 (J = 4, no draw), each with its live verdicts
# (index.jsonl; each record's verdict refuses its form, several tables without --zk, and keeps the verdict without that
# refusal, `without_refusal`). The staged verifier's files and the sessions go into the evidence store as a fixture (`research data put
# --kind fixture/v1 --preserve`), whose id `out/fixture.txt` names (else they are out/fixture). Run from the verity tree
# (research run --cwd source); CPU only. It exits nonzero unless every session passed every check of its server.
# env: FLOCK_WORK=/workspace, or where /workspace isn't writable /workspace/cache/session-tables (the prover build and its venv)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/session-tables}/out; mkdir -p $O
[ -n "${FLOCK_WORK:-}" ] || [ -w /workspace ] || export FLOCK_WORK=/workspace/cache/session-tables
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$(nproc)}

# the CPU selftest prover, cached per pod under its sources' key
KEY=$(cat verity/protocols/proof/flock/prover/pod/20-gpu-link.sh verity/protocols/proof/flock/prover/pod/60-circuit.sh verity/protocols/proof/flock/prover/*.patch verity/protocols/proof/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/proof/flock/prover/cuda verity/protocols/proof/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-seed; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build SEED_INJECTION=1 GPU=0 bash verity/protocols/proof/flock/prover/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
RESEARCH=$(command -v research || true)
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proof/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch:$REPO/tests/verity/protocols/proof/flock/prover

D=$S/stage; N=8
python3 - $D <<'EOF' > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
import json, sys
from pathlib import Path
from synthetic_sets import synthetic_set
from verity_flock import circuit as FC, subcircuit_templates as TM
from verity_flock.input_sets import InputSet
out = Path(sys.argv[1])
s = InputSet.open(synthetic_set(TM.rope_head(64), 8, 20260928, out.parent / "set")["path"])
FC.stage(FC.lowering_for_set(s), s, 0, 8, out)
m = json.loads((out / "circuit.txt").read_text().split("\n", 2)[1][5:])
print(json.dumps({"statement": m["statement"], "program_sha512": m["program_sha512"], "files": sorted(p.name for p in out.iterdir())}))
EOF
cat $O/stage.json
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
OK=1
run() {  # name J sessions [serve args...]
  local name=$1 j=$2 n=$3; shift 3
  local port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  $B serve --listen 127.0.0.1:$port --out $S/sessions/$name --instances $D/pub-$N.bin --circuit $D/circuit.txt --pin $PIN \
    --session-tables $j --sessions $n --operator loopback "$@" > $O/$name.serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1800); do grep -q SERVING $O/$name.serve.log && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  $B prove --verifier 127.0.0.1:$port --instances $D/inst-$N.bin --circuit $D/circuit.txt --session-tables $j --runs $n \
    > $O/$name.prove.txt 2> $O/$name.prove.err
  wait $sp
  # the server refuses the form (several tables without --zk) once every check has passed, as Lean does: its verdict without
  # the refusal is the sessions' own
  local got=$(python3 -c 'import json, sys
vs = [json.load(open(p))["verdict"] for p in sys.argv[1:]]
print(sum(bool(v["accepted"] or (v.get("without_refusal") or {}).get("accepted")) for v in vs))' $S/sessions/$name/l*/session.json 2>/dev/null)
  echo "$name: $got of $n pass every check of the server"
  [ "$got" = "$n" ] || OK=0
}
run j2 2 1
run j2-draw 2 2 --draw subset:4
run j4 4 1

# the fixture: the verifier's files and the sessions (no prover file: it holds the values)
X=$S/fixture; mkdir -p $X/stage
cp $D/circuit.txt $D/pub-$N.bin $X/stage/ && cp -r $S/sessions $X/sessions && cp $O/stage.json $X/stage.json || exit 1
[ $OK = 1 ] || exit 1
if [ -n "$RESEARCH" ]; then
  $RESEARCH data put --kind fixture/v1 --tree $X --preserve > $O/fixture.txt 2> $O/fixture.err || tail -5 $O/fixture.err
fi
[ -s $O/fixture.txt ] || { mkdir -p $O/fixture && cp -r $X/. $O/fixture/; echo "not in the store (no research CLI, or fixture.err): the fixture is out/fixture" > $O/fixture.txt; }
cat $O/fixture.txt
