#!/usr/bin/env bash
# The served-registered fixture (tests/served_registered.py, tests/test_lean_served_registered.py): a RoPE d64 statement
# whose `cs` rows are a served tree registered as a port (`one_stage.served.register_served`), proved in one loopback --zk
# session under a subset draw, then Lean's `flock-verify verify --zk --registered` on it with the verifier's own file
# (accepted) and with the tree's root changed (refused). The staged files and the session go into the evidence store as a
# fixture (`research data put --kind fixture/v1 --preserve`), whose id `out/fixture.txt` names. Run from the verity tree
# (research run --cwd source); CPU only. Outputs under out/: host.txt, stage.json, statement.txt, serve.log, prove.txt and
# prove.err, live.json (the LIVE record), lean-honest.log and lean-root.log (each its VERDICT line), fixture.txt.
# ZK=0 is the plain session instead (no --zk anywhere): a stratified draw of K units, Lean's own (`flock-verify draw
# --stratified K` on the stage's program and partition, the OS's coins), served from its file and verified with
# `--stratified K --coins os` (FC_COINS=os: every round's coins the OS's); Lean also verifies it without --registered
# (lean-none.log, refused). Its fixture adds vectors/: honest, root and none, each the session with its case.json (the live
# server's verdict, from serve's index.jsonl) and a registered.json for the two negatives (agree.py's per-session reads;
# vectors.json's live set).
# env: CPUS=16  K=16 (the draw's units, of the stage's 32)  ZK=1  SCHEMA=v1 (the tree's row format: v1, v2 or seg,
#      served_registered.py's --schema)  FLOCK_WORK=/workspace, or /workspace/jobs/pc-read-gadget
#      where /workspace isn't writable (node 1) (the prover build and its venv, shared as 76-read-gadget.sh's)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${SR_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export SR_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=${FC_COINS:-os}
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/served-registered}/out; mkdir -p $O
[ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/jobs/pc-read-gadget
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/served-registered; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$SR_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-}"; } > $O/host.txt 2>&1

# Lean's flock-verify, built once per source, while the prover builds
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ]; then
    rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
      && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!

# the CPU prover, cached per pod under its sources' key (76-read-gadget.sh's)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
RESEARCH=$(command -v research || true)
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm

D=$S/stage; N=32; K=${K:-16}
if [ "${ZK:-1}" = 0 ]; then ZF=(); else ZF=(--zk); fi
python3 verity/ml/flock/tests/served_registered.py --out $D --schema ${SCHEMA:-v1} > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
$B statement "${ZF[@]}" --circuit $D/circuit.txt --instances $D/pub-$N.bin --tables $D/tables > $O/statement.txt 2>&1
if [ "${ZK:-1}" = 0 ]; then
  # the plain claims draw by the stratified law: the verifier's own draw, served from its file
  wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
  $LEAN_BIN draw --program $D/program.json --partition $D/partition.json --stratified $K > $O/draw.log 2>&1 || { cat $O/draw.log; exit 1; }
  grep -h '^DRAW ' $O/draw.log | head -1 | cut -d' ' -f2- > $D/draw.json
  DRAW=(--draw-file $D/draw.json); VFLAGS=(--stratified $K --coins os)
else
  DRAW=(--draw subset:$K); VFLAGS=(--zk)
fi
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
$B serve "${ZF[@]}" --listen 127.0.0.1:$PORT --out $S/sessions --instances $D/pub-$N.bin --circuit $D/circuit.txt --tables $D/tables \
  --pin $PIN --operator served-registered "${DRAW[@]}" --sessions 1 > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $O/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
$B prove "${ZF[@]}" --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin --circuit $D/circuit.txt --tables $D/tables --runs 1 --warm 0 \
  > $O/prove.txt 2> $O/prove.err
grep -h '^LIVE' $O/prove.txt | head -1 | cut -f2- > $O/live.json
for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill $SP 2>/dev/null; wait $SP
SESS=$(dirname $(ls -1 $S/sessions/*/session.json | LC_ALL=C sort | tail -1)) || exit 1

wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
python3 - $D/registered.json $D/registered-root.json <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
for p in doc["ports"].values():
    p["root"] = "00" * 64
json.dump(doc, open(sys.argv[2], "w"))
EOF
CASES="honest:registered.json root:registered-root.json"
[ "${ZK:-1}" = 0 ] && CASES="$CASES none:"
for c in $CASES; do
  f=${c#*:}; REG=(); [ -n "$f" ] && REG=(--registered $D/$f)
  $LEAN_BIN verify "${VFLAGS[@]}" --statement verity/flock-circuit --circuit $D/circuit.txt --public $D/pub-$N.bin \
    --partition $D/partition.json --program $D/program.json "${REG[@]}" --session $SESS > $O/lean-${c%%:*}.log 2>&1
  echo "${c%%:*} rc $?"
  grep -h '^VERDICT' $O/lean-${c%%:*}.log | head -1 | cut -c1-600
done

# the fixture: the verifier's files, the served tree and the session (no prover file: it holds the values)
X=$S/fixture; mkdir -p $X/stage $X/session
cp $D/circuit.txt $D/pub-$N.bin $D/program.json $D/partition.json $D/served.json $D/registered.json $X/stage/ && cp -r $D/tables $X/stage/
cp -r $SESS/. $X/session/
cp $O/live.json $X/live.json
if [ "${ZK:-1}" = 0 ]; then
  cp $D/draw.json $X/stage/
  python3 - $SESS $D/registered-root.json $X/vectors <<'EOF'
import json, shutil, sys
from pathlib import Path
sess, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
v = json.loads((sess / "session.json").read_text())["verdict"]
live = {"accepted": v["accepted"], "why": v.get("why") or ""}
for name, expect, reads in (("honest", "accept", None), ("root", "reject", root.read_text()), ("none", "reject", "null")):
    d = out / name
    shutil.copytree(sess, d)
    (d / "case.json").write_text(json.dumps({"case": name, "expect": expect, "live": live}, indent=1))
    if reads is not None:
        (d / "registered.json").write_text(reads)
EOF
fi
if [ -n "$RESEARCH" ]; then
  $RESEARCH data put --kind fixture/v1 --tree $X --preserve > $O/fixture.txt 2> $O/fixture.err || tail -5 $O/fixture.err
fi
[ -s $O/fixture.txt ] || { mkdir -p $O/fixture && cp -r $X/. $O/fixture/; echo "no research CLI: the fixture is out/fixture" > $O/fixture.txt; }
cat $O/fixture.txt
