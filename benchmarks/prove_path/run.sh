#!/usr/bin/env bash
# Prove path (prove_path.py) on a CPU machine: a TP Commit's ranks committed over the stand-in compiler's Programs for the run's
# advice, each rank's window registered with the compiler's digest (its cap ℓ in it) and the root of the advice's hiding
# commitment, drawn through a5's per-rank service.Stream on the verifier's own coins, proved by M0 --zk behind the Lean firewall
# and verified by Lean, the advice's root then held to the registration the draw followed; then one forged run, a unit whose
# committed value is changed (committed consistently, the prover's self-check skipped), which Lean must refuse. M0's
# flock-circuit and Lean's flock-verify are built from this tree, in scratch.
#   PP_TP=DIR: a deferred TP Commit's row dir (DIR/commit with its sealed bundles, DIR/build/rank<r>), as GLM-5.3 B1's Commit
#     r20261021-141533-38b2 on vy-nebius-cpu-2 (/workspace/cp/circuits/glm-v2/rows/$ROW). Its advice is the declared run's,
#     from integrations/vllm/workloads/$ROW.json. The honest partition is PP_PARTITION (default a5's step-16 SiluMul_v2,
#     stratified:4); the forged one SiluMul_v2{I=1536} at step 16, all 3 units drawn, on ranks PP_FORGED_RANKS (default 0).
#   The stand-in (no PP_TP): the CPU TP Commit tests/pipeline/served_tp --deferred, world 2, SiluMul_v2{I=256}: honest subset:4,
#     forged subset:8 (every unit drawn).
#   PP_JOBS=N ranks at once (a GLM-5.3 rank: its bundle window ~2.5 min and ~11 GB, its audit ~10 GB); PP_RANKS=0,3 those ranks
#   only; PP_FORGE=MEMBER:UNIT the forged unit (default 0:1); PP_FLOCK_WORK=DIR keeps M0's build across runs.
#   research run --on vy-nebius-1 --project verity --source . --cwd clone --mem-gb 16 --mem-wait 3600 -- bash benchmarks/prove_path/run.sh
# Kept in $OUT: the records and logs, the public files, the sessions (their proofs and records), the binaries' sha256; never the
# advice or its opening (the prover's, private: the verifier is handed the opening, the named gap), a prover's file (inst-*,
# *-private.bin: the rows and their salts) nor a circuit (its pin is).
set -uxo pipefail
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }
PY=$REPO/.venv/bin/python

(cd verity/core/service/flock && lake build flock-verify flock-firewall) > $O/build-lean.log 2>&1 &
LP=$!
mkdir -p $O/m0-build
FW=${PP_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > $O/m0-build/build.log 2>&1 \
  || { tail -40 $O/m0-build/build.log; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 $O/build-lean.log; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify; FWL=${LEAN%/*}/flock-firewall
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > $O/build-sampled-proofs.log 2>&1 || { tail -40 $O/build-sampled-proofs.log; exit 7; }
sha256sum $BIN $LEAN $FWL | tee $O/binaries.sha256

for I in 256 1536; do   # the verifier's lowering: a synthetic input set per template (its values are never read)
  $PY -m verity_numerical.bench.generate --subcircuit silu-mul-v2/i$I/bf16 --n 4 --seed 1 --out $W/sets/i$I > $O/sets-i$I.log 2>&1 || exit 9
done
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
TIME=$([ -x /usr/bin/time ] && echo "/usr/bin/time -v")
PP="$PY benchmarks/prove_path/prove_path.py"

keep() {  # NAME: $W/NAME's records into $O/NAME
  mkdir -p $O/$1
  (cd $W/$1 && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' ! -path './*/archive/*' -print0 \
    | xargs -0 -r -I{} cp --parents {} $O/$1/)
}
run() {  # NAME COMMIT BUILD ADVICE PARTITION RANKS [--forge M:U]: the prover's windows, then the verifier's audit
  local name=$1 commit=$2 build=$3 advice=$4 part=$5 ranks=$6; shift 6
  local forged=; [ $# -gt 0 ] && forged=--forged
  mkdir -p $O/$name
  $TIME $PP commit --tp-commit $commit --build-root $build --advice $advice --opening $W/$name-opening.json --partition $part \
    ${ranks:+--ranks $ranks} \
    --jobs ${PP_JOBS:-1} --out $W/$name-windows "$@" > $O/$name/commit.log 2>&1 || { tail -30 $O/$name/commit.log; return 1; }
  for d in $W/$name-windows/rank*; do   # serving's public files, never a private body
    mkdir -p $O/$name/windows/$(basename $d) && cp $d/{registration,window,partition,index}.json $d/m*-public.bin $O/$name/windows/$(basename $d)/
  done
  cp $W/$name-windows/commit.json $O/$name/windows/
  $TIME $PP audit --tp-commit $commit --build-root $build --opening $W/$name-opening.json --windows $W/$name-windows --partition $part \
    ${ranks:+--ranks $ranks} --jobs ${PP_JOBS:-1} ${PP_SESSION_TIMEOUT:+--session-timeout $PP_SESSION_TIMEOUT} $forged \
    --template-sets $W/sets --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --out $W/$name > $O/$name/audit.log 2> $O/$name/audit.err
  local rc=$?
  grep '^PROVE-PATH' $O/$name/audit.log | cut -c1-6000; tail -25 $O/$name/audit.err
  keep $name
  return $rc
}

rc=0
FORGE=${PP_FORGE:-0:1}
if [ -n "${PP_TP:-}" ]; then
  $PP advice --workload integrations/vllm/workloads/$ROW.json --out $W/advice.bin | tee $O/advice.log || exit 8
  run glm $PP_TP/commit $PP_TP/build $W/advice.bin ${PP_PARTITION:-benchmarks/one_stage/glm53_silu_mul_v2_tp8_step16_stratified.json} \
    "${PP_RANKS:-}" || rc=1
  run glm-forged $PP_TP/commit $PP_TP/build $W/advice.bin benchmarks/prove_path/glm53_silu_mul_v2_i1536_step16_all.json \
    "${PP_FORGED_RANKS:-0}" --forge $FORGE || rc=1
else
  S=$W/served
  (cd integrations/vllm && $PY -m tests.pipeline.served_tp $S --world 2 --I 256 --law subset:4 --deferred) \
    > $O/standin.json 2> $O/standin.err || { tail -20 $O/standin.err; exit 8; }
  mkdir -p $O/standin && cp $S/workload.json $O/standin/ && find $S/commit -maxdepth 1 -type f -exec cp {} $O/standin/ \;
  $PP advice --workload $S/workload.json --out $W/advice.bin | tee $O/advice.log || exit 8
  echo '{"templates": ["SiluMul_v2{I=256}"], "law": "subset:8"}' > $W/all.json
  run standin $S/commit $S/build $W/advice.bin $S/partition.json "${PP_RANKS:-}" || rc=1
  run standin-forged $S/commit $S/build $W/advice.bin $W/all.json "${PP_FORGED_RANKS:-0}" --forge $FORGE || rc=1
fi
exit $rc
