#!/usr/bin/env bash
# One-stage audit, run A5, over a TP Commit's windows, one per rank (a5.py --tp-commit), on a CPU machine: each rank's served
# window through registration, the verifier's Lean draw, M0's prover and the Lean verifier of record. M0's flock-circuit and
# Lean's flock-verify are built from this tree, in scratch.
#   GLM-5.3 TP8: A5_TP=DIR holds the TP Commit's record (commit/runs.jsonl, commit/versions.json) and its windows (serving-rows/
#     rank<r>/), A5_BUILD=DIR each rank's Programs of record (rank<r>/); the workload (integrations/vllm/workloads/$ROW.json) and
#     the partition file (glm53_silu_mul_v2_tp8.json) are this tree's.
#   The stand-in (no A5_TP): the CPU TP Commit tests/pipeline/served_tp, world 2, once per STANDIN entry I:LAW (its SiluMul_v2{I}
#     rows, its partition's law).
#   research run --on M --project verity --cwd clone --mem-gb 64 -- bash benchmarks/one_stage/a5_tp.sh
# Kept in $OUT: a5's records and logs, the public files, the sessions (their proofs and records), the binaries' sha256 and the
# circuits' pins; never a prover's file (inst-*, m*-inst-*, m*-private.bin: the rows and their salts) nor a circuit (its pin is).
set -uxo pipefail
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock HEAD:benchmarks/numerical; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }
PY=$REPO/.venv/bin/python

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

# the verifier's lowering: a synthetic input set per template (its values are never read)
for I in 256 1536; do
  $PY -m verity_numerical.bench.generate --subcircuit silu-mul-v2/i$I/bf16 --n 4 --seed 1 --out $W/sets/i$I > $O/sets-i$I.log 2>&1 || exit 9
done
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)

TIME=$([ -x /usr/bin/time ] && echo "/usr/bin/time -v")
audit() {  # NAME TP_COMMIT BUILD WORKLOAD WINDOWS PARTITION
  mkdir -p $O/$1
  $TIME $PY benchmarks/one_stage/a5.py --tp-commit $2 --build-root $3 --workload $4 --windows $5 --partition $6 \
    ${A5_RANKS:+--ranks $A5_RANKS} --template-sets $W/sets --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --out $W/$1 \
    > $O/$1/a5.log 2> $O/$1/a5.err
  local rc=$?
  grep '^A5' $O/$1/a5.log | cut -c1-4000; tail -25 $O/$1/a5.err
  (cd $W/$1 && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name 'm*-private.bin' ! -path './*/archive/*' -print0 \
    | xargs -0 -r -I{} cp --parents {} $O/$1/)
  return $rc
}

public() {  # WINDOWS DEST: serving's public files of every rank
  for d in $1/rank*; do
    mkdir -p $2/$(basename $d) && cp $d/{registration,window,partition,index}.json $d/m*-public.bin $2/$(basename $d)/
  done
  cp $1/tp.json $2/ 2>/dev/null
}

rc=0
if [ -n "${A5_TP:-}" ]; then
  public $A5_TP/serving-rows $O/glm/window
  audit glm $A5_TP/commit ${A5_BUILD:-$A5_TP/build} integrations/vllm/workloads/$ROW.json $A5_TP/serving-rows \
    benchmarks/one_stage/glm53_silu_mul_v2_tp8.json || rc=1
else
  for e in ${STANDIN:-256:subset:8 1536:subset:8}; do
    I=${e%%:*}; LAW=${e#*:}; S=$W/served-i$I
    (cd integrations/vllm && $PY -m tests.pipeline.served_tp $S --world 2 --I $I --law $LAW) > $O/standin-i$I.json 2> $O/standin-i$I.err \
      || { tail -20 $O/standin-i$I.err; exit 8; }
    public $S/serving-rows $O/standin-i$I/window && cp -r $S/commit $S/workload.json $S/partition.json $O/standin-i$I/
    audit standin-i$I $S/commit $S/build $S/workload.json $S/serving-rows $S/partition.json || rc=1
  done
fi
exit $rc
