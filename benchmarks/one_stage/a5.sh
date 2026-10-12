#!/usr/bin/env bash
# One-stage audit, run A5, on a CPU machine: a deferred Commit's window, a stand-in, through registration, the verifier's Lean
# draw, M0's prover and the Lean verifier of record (a5.py). The Commit is the CPU RoPE d64 fixture (integrations/vllm
# tests/pipeline/served_rope: the production committer, sealed into a deferred replay bundle), and the window is
# `verity-vllm serving-rows` on it. M0's flock-circuit and Lean's flock-verify are built from this tree, in scratch.
#   research run --on M --project verity --cwd clone --mem-gb 16 -- python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/one_stage/a5.sh
# Kept in $OUT: a5's records and logs, the public files, the session (its proofs and record), the binaries' sha256 and the
# circuit's pin; never a prover's file (inst-*, m*-private.bin: the rows and their salts) nor the 56 MB circuit (its pin is).
set -uxo pipefail
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock HEAD:benchmarks/numerical; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }
PY=$REPO/.venv/bin/python

# Lean's flock-verify, in the background while M0 builds (its own .lake in this clone)
(cd verity/core/service/flock && lake build flock-verify) > $O/build-lean.log 2>&1 &
LP=$!
mkdir -p $O/m0-build
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > $O/m0-build/build.log 2>&1 \
  || { tail -40 $O/m0-build/build.log; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 $O/build-lean.log; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > $O/build-sampled-proofs.log 2>&1 || { tail -40 $O/build-sampled-proofs.log; exit 7; }
sha256sum $BIN $LEAN | tee $O/binaries.sha256

# serving: the CPU Commit and its window, from the sealed bundle
S=$W/served
(cd integrations/vllm && $PY -m tests.pipeline.served_rope $S) > $O/commit.json 2> $O/commit.err || { tail -20 $O/commit.err; exit 8; }
t0=$(date +%s%N)
$REPO/.venv/bin/verity-vllm serving-rows $S/commit/replay_bundle_p1 --row-dir $S/row --partition $S/partition.json --out $S/window \
  > $O/serving-rows.log 2>&1 || { tail -20 $O/serving-rows.log; exit 8; }
echo "serving_rows_s=$(( ($(date +%s%N) - t0) / 1000000 ))e-3" | tee -a $O/serving-rows.log
mkdir -p $O/window && cp $S/window/{registration,window,partition,index}.json $S/window/m*-public.bin $O/window/

# the verifier's lowering: a synthetic input set of the template (its values are never read)
$PY -m verity_numerical.bench.generate --subcircuit rope-head/d64/neox-bf16 --n 4 --seed 1 --out $W/sets > $O/sets.log 2>&1 || exit 9

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
$PY benchmarks/one_stage/a5.py --bundle $S/commit/replay_bundle_p1 --row-dir $S/row --partition $S/partition.json --window $S/window \
  --template-sets $W/sets --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --out $W/a5 2>&1 | tee $O/a5.log | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p $O/a5
(cd $W/a5 && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -path './*/archive/*' -print0 \
  | xargs -0 -I{} cp --parents {} $O/a5/)
exit $rc
