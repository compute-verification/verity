#!/usr/bin/env bash
# The plain hidden-output multi-table fixture (tests/test_lean_plain_hidden_tables.py): RoPE d64 (8 instances) on the
# default hidden-output statement (`verity/flock-circuit`, hm96-sha512 leaves), proved without --zk in loopback sessions of
# two tables under a subset draw of 4 (`serve/prove --session-tables 2 --draw subset:4`), then Lean's `flock-verify verify
# --plain-tables --session-tables 2` on them with the verifier's own program (the RoPE d64 Definition's descriptor, META's
# `program_sha512`), each coin form (seed, os) and without the flag. The staged verifier files and the sessions go to
# out/fixture, which the launcher puts into the evidence store (`research data put --kind fixture/v1 --tree out/fixture
# --preserve`). Run from the verity tree (research run --cwd source); CPU only. Outputs under out/: host.txt, stage.json,
# serve.log, prove.txt, prove.err, lean-*.log (each its VERDICT lines) and fixture/.
# env: CPUS=16  SESSIONS=2  K=4  FLOCK_WORK=/workspace (the prover build and its venv, shared as 89-served-registered.sh's)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${PH_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export PH_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/plain-hidden-tables}/out; mkdir -p $O
[ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/jobs/pc-read-gadget
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/plain-hidden-tables; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$PH_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-}"; } > $O/host.txt 2>&1

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

# the CPU prover, cached per pod under its sources' key (89-served-registered.sh's)
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
(cd $REPO && $UV sync -q --all-packages) > $O/uv-sync.log 2>&1 || { tail -20 $O/uv-sync.log; exit 1; }
PY=$REPO/.venv/bin/python

D=$S/stage; N=8; J=2
$PY - $N $S <<'EOF' > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
import json, sys
from pathlib import Path
from verity_flock import circuit as FC
from verity_numerical.bench import templates as TM
from verity_numerical.bench.generate import generate
from verity_numerical.bench.input_sets import InputSet
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.program import Program
n = int(sys.argv[1]); out = Path(sys.argv[2])
s = InputSet.open(generate(TM.rope_head(64), n, 20260928, out / "set")["path"])
FC.stage(FC.lowering_for_set(s), s, 0, n, out / "stage")
prog = Program(TM.rope_head(64).definition())
(out / "stage" / "program.json").write_bytes(canonical_json(encode_program(prog)))
m = json.loads((out / "stage" / "circuit.txt").read_text().split("\n", 2)[1][5:])
print(json.dumps({"statement": m["statement"], "k_log": m["k_log"], "program_sha512": m["program_sha512"],
                  "program_digest": program_digest(prog), "leaf_scheme": m["leaf_scheme"].get("id"),
                  "stage": sorted(p.name for p in (out / "stage").iterdir())}))
EOF
cat $O/stage.json
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
TABLES=(); [ -d $D/tables ] && TABLES=(--tables $D/tables)
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
$B serve --listen 127.0.0.1:$PORT --out $S/sessions --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TABLES[@]}" \
  --pin $PIN --session-tables $J --sessions ${SESSIONS:-2} --operator loopback --draw subset:${K:-4} > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $O/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
$B prove --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin --circuit $D/circuit.txt "${TABLES[@]}" --session-tables $J \
  --runs ${SESSIONS:-2} > $O/prove.txt 2> $O/prove.err
for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill $SP 2>/dev/null; wait $SP
mapfile -t SESS < <(dirname $(ls -1 $S/sessions/*/session.json) | LC_ALL=C sort)
[ ${#SESS[@]} -gt 0 ] || { tail -20 $O/serve.log $O/prove.err; exit 1; }

wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
for c in seed os; do
  $LEAN_BIN verify --circuit $D/circuit.txt --public $D/pub-$N.bin "${TABLES[@]}" --coins $c --session-tables $J --plain-tables \
    --program $D/program.json --session "${SESS[@]}" > $O/lean-plain-$c.log 2>&1
  echo "plain $c rc $?"; grep -h '^VERDICT' $O/lean-plain-$c.log | cut -c1-400
done
$LEAN_BIN verify --circuit $D/circuit.txt --public $D/pub-$N.bin "${TABLES[@]}" --session-tables $J --program $D/program.json \
  --session "${SESS[@]}" > $O/lean-unflagged.log 2>&1
echo "unflagged rc $?"; grep -h '^VERDICT' $O/lean-unflagged.log | cut -c1-400

# the fixture: the verifier's files and the sessions (no prover file: it holds the values)
X=$O/fixture; mkdir -p $X/stage $X/sessions
cp $D/circuit.txt $D/pub-$N.bin $D/program.json $X/stage/ && { [ -d $D/tables ] && cp -r $D/tables $X/stage/ || true; }
cp -r $S/sessions/. $X/sessions/
cp $O/stage.json $X/stage.json
du -sh $X
