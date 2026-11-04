#!/usr/bin/env bash
# One-stage audit, run A5, on a CPU machine: a deferred Commit's window, a stand-in, through registration, the verifier's Lean
# draw, M0's prover and the Lean verifier of record (a5.py). The Commit is the CPU RoPE d64 fixture (integrations/vllm
# tests/pipeline/served_rope: the production committer, sealed into a deferred replay bundle), and the window is
# `verity-vllm serving-rows` on it; with WINDOW=served it is the in-Commit hook's over the same members (served_rope --served),
# which A5 takes only over a SHA-512 run root of record. The fixture's members are committed under rows-sha512, which the window
# names as its record's; COMMIT_SCHEME, if set, must say so (vllm-v1 is kept only to read recorded runs, and A5 over one is a5.py on
# its bundle). M0's flock-circuit and Lean's flock-verify are built from this tree, in scratch (A5_FLOCK_WORK=DIR keeps M0's build).
#   python3 tools/verity/check/slot.py --by LANE -- bash benchmarks/one_stage/a5.sh
#   (a job run from a clone of the tree, 16 GB)
# A5_ROW_DIR=D: a single-GPU deferred Commit's row dir in place of the fixture (its sealed D/commit/replay_bundle_p0 and the Build's
# Programs of record in D, as a5_llama1b_h200.sh leaves them), with A5_PARTITION the partition its --serving-rows named and, with
# WINDOW=served, A5_WINDOW the window its C2 site committed (`c2_replay.serving_rows`); A5_SUBCIRCUIT is the template's input set
# for the lowering (default rope-head/d64/neox-bf16, the fixture's RoPEHead_v1{D=64}).
# Kept in $OUT: a5's records and logs, the public files, the session (its proofs and record), the binaries' sha256 and the
# circuit's pin; never a prover's file (inst-*, m*-private.bin, v*-private.bin: the rows and their salts) nor the 56 MB
# circuit (its pin is).
set -uxo pipefail
if [ "${COMMIT_SCHEME:-rows-sha512}" != rows-sha512 ]; then
  echo "a5.sh: COMMIT_SCHEME=$COMMIT_SCHEME: the fixture commits under rows-sha512 only (vllm-v1 is kept to read recorded runs)" >&2; exit 2
fi
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock HEAD:benchmarks/numerical; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }
PY=$REPO/.venv/bin/python

# Lean's flock-verify, in the background while M0 builds (its own .lake in this clone)
(cd verity/core/service/flock && lake build flock-verify flock-firewall) > $O/build-lean.log 2>&1 &
LP=$!
mkdir -p $O/m0-build
FW=${A5_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > $O/m0-build/build.log 2>&1 \
  || { tail -40 $O/m0-build/build.log; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 $O/build-lean.log; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify; FWL=${LEAN%/*}/flock-firewall
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > $O/build-sampled-proofs.log 2>&1 || { tail -40 $O/build-sampled-proofs.log; exit 7; }
sha256sum $BIN $LEAN $FWL | tee $O/binaries.sha256

# serving: the CPU Commit and its window, from the sealed bundle, or the recorded Commit of A5_ROW_DIR and its window
S=$W/served
WINDOW=${WINDOW:-stand-in}
if [ -n "${A5_ROW_DIR:-}" ]; then
  BUNDLE=$A5_ROW_DIR/commit/replay_bundle_p0 ROWD=$A5_ROW_DIR PART=${A5_PARTITION:?A5_ROW_DIR needs A5_PARTITION}
  [ -s $BUNDLE/MANIFEST.json ] || { echo "a5.sh: no sealed bundle at $BUNDLE" >&2; exit 2; }
else
  BUNDLE=$S/commit/replay_bundle_p1 ROWD=$S/row PART=$S/partition.json
fi
if [ "$WINDOW" = served ] && [ -n "${A5_ROW_DIR:-}" ]; then
  WIN=${A5_WINDOW:?A5_ROW_DIR with WINDOW=served needs A5_WINDOW, the window the Commit committed}
  [ ! -e $WIN/error.txt ] || { echo "a5.sh: the Commit's window failed:"; tail -20 $WIN/error.txt; exit 8; }
elif [ "$WINDOW" = served ]; then
  (cd integrations/vllm && $PY -m tests.pipeline.served_rope $S --served) > $O/commit.json 2> $O/commit.err || { tail -20 $O/commit.err; exit 8; }
  WIN=$S/hook
else
  [ -n "${A5_ROW_DIR:-}" ] || { (cd integrations/vllm && $PY -m tests.pipeline.served_rope $S) > $O/commit.json 2> $O/commit.err || { tail -20 $O/commit.err; exit 8; }; }
  t0=$(date +%s%N)
  $REPO/.venv/bin/verity-vllm serving-rows $BUNDLE --row-dir $ROWD --partition $PART --out $S/window \
    > $O/serving-rows.log 2>&1 || { tail -20 $O/serving-rows.log; exit 8; }
  echo "serving_rows_s=$(( ($(date +%s%N) - t0) / 1000000 ))e-3" | tee -a $O/serving-rows.log
  WIN=$S/window
fi
mkdir -p $O/window && cp $WIN/{registration,window,partition,index}.json $WIN/m*-public.bin $O/window/

# the verifier's lowering: a synthetic input set of the template (its values are never read)
$PY -m verity_numerical.bench.generate --subcircuit ${A5_SUBCIRCUIT:-rope-head/d64/neox-bf16} --n 4 --seed 1 --out $W/sets > $O/sets.log 2>&1 || exit 9

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
$PY benchmarks/one_stage/a5.py --bundle $BUNDLE --row-dir $ROWD --partition $PART --window $WIN \
  --template-sets $W/sets --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --out $W/a5 2>&1 | tee $O/a5.log | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p $O/a5
(cd $W/a5 && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -path './*/archive/*' -print0 \
  | xargs -0 -I{} cp --parents {} $O/a5/)
exit $rc
