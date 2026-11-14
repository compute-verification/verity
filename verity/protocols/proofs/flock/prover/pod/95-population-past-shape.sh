#!/usr/bin/env bash
# A part past the spec's shape, proved at a sample in it: 91-plain-hidden.sh's statement (tests/plain_hidden.py: RoPE d64
# with hidden outputs) at N units, one VU a block, so its population's dense size m = k_log + nbl is past 35 while a subset
# sample of K units is inside it. Stage the population; `statement` (which proves the population itself) is refused in the
# shape's words; one loopback session without --zk under `--draw subset:K` on the keyed coin tree (FC_COINS=tree), its
# coins from the auditor's challenger (challenger.sh), proved by the CPU prover; Lean's `flock-verify verify --coins tree`
# on it with the verifier's own program and partition. Then a session at `--draw subset:N`, every unit, which the server
# refuses at Register in the shape's words. Run from the verity tree (research run --cwd source); CPU only.
# Outputs under out/: host.txt, stage.json, statement.txt, serve.log, prove.txt, prove.err, live.json, challenger/,
# lean-honest.log, past-serve.log, past-prove.txt, past-session.json, result.json (each step's verdict).
# env: CPUS=16  N=16385 (k_log 21 at one VU a block: nbl 15, m 36)  K=16 (nbl 4, m 25)  FLOCK_WORK as 91-plain-hidden.sh's
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${PH_PINNED:-}" ]; then
  if mkdir -p "${SLICE_LOCKS:-/workspace/jobs/slices}" 2>/dev/null; then
    . verity/protocols/proofs/flock/prover/pod/cpu-slices.sh
    hold_slices "" "${CPUS:-16}" || exit 1
    export PH_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
    exec taskset -c $SLICE_CPUS bash "$0" "$@"
  fi
  export PH_PINNED=slot RAYON_NUM_THREADS=$(nproc)
fi
export FC_COINS=tree
N=${N:-16385}; K=${K:-16}
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/population-past-shape}/out; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/plain-hidden; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/plain-hidden; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$PH_PINNED rayon=$RAYON_NUM_THREADS N=$N K=$K"; } > $O/host.txt 2>&1

# Lean's flock-verify, built once per source (91-plain-hidden.sh's cache), while the prover builds
LSHA=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py key verity/protocols/proofs/flock/verifier)
LEAN_BIN=$L/lean-$LSHA/flock-verify
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ]; then
    rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py copy verity/protocols/proofs/flock/verifier $L/lean-src) \
      && (cd $LEAN_SRC && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $LEAN_SRC/.lake/build/bin/flock-verify $LEAN_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!
CHALLENGER_REPO=$REPO CHALLENGER_WORK=$S/challenger-work
. verity/protocols/proofs/flock/prover/pod/challenger.sh
challenger_build

# the CPU prover, cached per pod under its sources' key (91-plain-hidden.sh's)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/protocols/proofs/flock/prover/pod/20-gpu-link.sh verity/protocols/proofs/flock/prover/pod/60-circuit.sh verity/protocols/proofs/flock/prover/*.patch verity/protocols/proofs/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/proofs/flock/prover/cuda verity/protocols/proofs/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/protocols/proofs/flock/prover/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm

D=$S/stage
python3 verity/protocols/proofs/flock/prover/tests/plain_hidden.py --out $D --n $N --vus-per-block 1 > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
# `statement` proves the population itself, so it still tests the shape at the population's count
$B statement --circuit $D/circuit.txt --instances $D/pub-$N.bin --tables $D/tables > $O/statement.txt 2>&1
port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
# one session at `--draw subset:$1`, its server's log, the prover's LIVE line and the server's session directory under $2
session() {
  local k=$1 tag=$2 p sp; p=$(port)
  $B serve --listen 127.0.0.1:$p --out $S/$tag-sessions --instances $D/pub-$N.bin --circuit $D/circuit.txt --tables $D/tables \
    --pin $PIN --operator population-past-shape --draw subset:$k --sessions 1 "${@:3}" > $O/$tag-serve.log 2>&1 &
  sp=$!
  for i in $(seq 1800); do grep -q SERVING $O/$tag-serve.log && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  $B prove --verifier 127.0.0.1:$p --instances $D/inst-$N.bin --circuit $D/circuit.txt --tables $D/tables --runs 1 --warm 0 \
    > $O/$tag-prove.txt 2> $O/$tag-prove.err
  grep -h '^LIVE' $O/$tag-prove.txt | head -1 | cut -f2- > $O/$tag-live.json
  for i in $(seq 600); do kill -0 $sp 2>/dev/null || break; sleep 1; done
  kill $sp 2>/dev/null; wait $sp
}
CHALLENGER_PY=python3
challenger_built && challenger_start $O/challenger || exit 1
session $K honest "${TREE_FROM[@]}"
SESS=$(dirname $(ls -1 $S/honest-sessions/*/session.json | LC_ALL=C sort | tail -1)) || exit 1
challenger_end $O/challenger; echo "challenger exit $?"
challenger_audit $O/challenger $SESS; echo "challenger audit $?"

wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
. verity/protocols/proofs/spot/verifier/one_stage/verifier_env.sh
$(verifier_env) $LEAN_BIN verify --statement verity/flock-circuit --coins tree --circuit $D/circuit.txt --public $D/pub-$N.bin \
  --partition $D/partition.json --program $D/program.json --session $SESS > $O/lean-honest.log 2>&1
echo "honest rc $?"
grep -h '^VERDICT' $O/lean-honest.log | head -1 | cut -c1-600

# every unit: the sample's own statement is the population's, past the shape
session $N past
PAST=$(ls -1 $S/past-sessions/*/session.json 2>/dev/null | LC_ALL=C sort | tail -1)
[ -n "$PAST" ] && cp $PAST $O/past-session.json

python3 - "$O" "$N" "$K" <<'EOF'
import json, sys
from pathlib import Path
o, n, k = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
def first(path, tag):
    for line in (o / path).read_text(errors="replace").splitlines() if (o / path).exists() else []:
        if line.startswith(tag):
            return json.loads(line.split(None, 1)[1])
def live(tag):
    t = (o / f"{tag}-live.json").read_text().strip() if (o / f"{tag}-live.json").exists() else ""
    return json.loads(t) if t else None
statement, honest, verdict, past = first("statement.txt", "REFUSED"), live("honest"), first("lean-honest.log", "VERDICT"), live("past")
words = (statement or {}).get("why") or ""
past_rec = json.loads((o / "past-session.json").read_text()) if (o / "past-session.json").exists() else None
got = {"n": n, "k": k,
       "statement_refused": statement and statement.get("why"),
       "honest": honest and {x: honest.get(x) for x in ("accepted", "why", "instances", "dense_m", "e2e_s")},
       "lean": verdict and {x: verdict.get(x) for x in ("accepted", "why")},
       "past": past and {x: past.get(x) for x in ("accepted", "why", "prover_stopped")},
       "past_session_why": past_rec and past_rec.get("why")}
got["ok"] = {"population refused by statement in the shape's words": words.endswith(": outside the spec's shape (m <= 35)"),
             "sample proved": bool(honest and honest.get("accepted")),
             "Lean accepts the sampled session": bool(verdict and verdict.get("accepted")),
             "every unit refused at Register in the same words": bool(words and past and not past.get("accepted") and words in str(past.get("why")))}
(o / "result.json").write_text(json.dumps(got, indent=1))
print(json.dumps(got, indent=1))
sys.exit(0 if all(got["ok"].values()) else 1)
EOF
