#!/usr/bin/env bash
# Prove path (prove_path.py) on a CPU machine: a TP Commit's ranks committed over the stand-in compiler's Programs for the run's
# advice, each rank's window registered with the compiler's digest (its cap ℓ in it) and the root of the advice's hiding
# commitment, drawn through a5's per-rank service.Stream on the verifier's own coins, proved by M0 --zk behind the Lean firewall
# and verified by Lean, the advice's root then held to the registration the draw followed; then one forged run, a unit whose
# committed value is changed (committed consistently, the prover's self-check skipped), which Lean must refuse. M0's
# flock-circuit and Lean's flock-verify are built from this tree, in scratch.
#   PP_TP=DIR: a deferred TP Commit's row dir (DIR/commit with its sealed bundles, DIR/build/rank<r>), as GLM-5.3 B1's Commit
#     (/workspace/cp/circuits/glm-v2/rows/$ROW). Its advice is the declared run's,
#     from integrations/vllm/workloads/$ROW.json. The honest partition is PP_PARTITION (default a5's step-16 SiluMul_v2,
#     stratified:4); the forged one SiluMul_v2{I=1536} at step 16, all 3 units drawn, on ranks PP_FORGED_RANKS (default 0).
#   PP_LLAMA=DIR: a single-GPU (TP1) Commit's row dir on this node: Llama-3.2-1B's Commit ($LROW, the sealed bundle
#     commit/replay_bundle_p0 and the Build's build_request_LP*). Its compiler is I2's compile_steps; its advice the declared
#     run's. The bundle's context records the Commit's absolute node paths, and none is read: the prover's Programs are read from
#     --row-dir, the verifier's from its own compile of each, and the Build manifest falls back to
#     <row dir>/manifest.json (sha256-checked against the record). The honest partition is
#     PP_PARTITION (default layer 0's SiluMul_v2{I=8192} at step 1, 4 units, subset:2); the forged one layer 0's at step 24, the
#     one unit only r0 decodes there, drawn (subset:1), forged at PP_FORGE (default 0:0). With PP_SERVE_STANDIN=1, the same
#     Commit laid out as a serve-openai run (prove_path.py serve-standin) and audited as a served window.
#   PP_SERVE_RUN=DIR PP_WINDOW=K PP_SERVE_ROWS=DIR: a serve-openai run's closed Commit window K (default 0), its
#     advice, requests and engine from the run dir, its Commit from the window's sealed row dir; PP_TARGET=F the
#     serving device's target profile where serve.json records none. The honest and forged partitions are PP_PARTITION and
#     PP_FORGED_PARTITION (default the Llama ones above), forged at PP_FORGE (default 0:0). The window's compiler lists its
#     model's transfer gates of record (prove_path.GATES_OF_RECORD, which PP_TRANSFER_GATES may only restate), and its advice is
#     registered under that compiler (`verity-vllm advice --compiler-name/--compiler-digest`, the inner name and digest
#     `prove_path.py compiler` prints). With PP_RECORDS=1 (a served window, PP_SERVE_RUN or PP_SERVE_STANDIN), each window is
#     audited from its records alone (records_audit.py: circuit.json, the registration and the values, the advice's opening),
#     its registered compiler run as a command (prove_path.py circuit), instead of by prove_path.py audit.
#   PP_LLAMA, PP_SERVE_RUN and PP_SERVE_ROWS name directories on this node, the serve directory's; an art: id is refused,
#     since a published row dir's sealed bundle and a published run's requests.jsonl hold request values.
#   PP_VLLM_PYTHON=PY: ignored (prove_path.py audit's and circuit's --vllm-python): a TP1 or served window's verifier writes its
#     own Build of each Program I2 compiles from its compile, with no vLLM, since the rows' names, the parameters' names and
#     types, supplied, the correspondence and prefix_cache.json are no Program digest's and never read from the row dir.
#   PP_WEIGHTS_TIE=1: the checkpoint tie of vLLM's weights registration at TINY (verity_vllm.program.predict.weights_tie;
#     its inputs tests/program/predict/weights_tie_fixture): the committer registers checkpoint X's weights and records
#     its tie window, one SHA-512 chain a shard over the registered rows (the embedding table's row-seg rows a chain of
#     their own), ending at the verifier's own digest; the verifier makes its tie of X from X's public bytes (the circuit's
#     SHA-512, the run root, the registered entries and its supplied values, every unit as the output class), audits the
#     window against it under PP_LAW (default work:1) in
#     members of PP_TIE_CAP units (default 16), and pins the roots only on an accepting audit. Two refused arms: a
#     committer serving checkpoint Y's words as X's (its last units' digest reads open no root), and the same with those
#     digests forged to X's (those units aren't their Definition's); each must be refused, and pin nothing.
#   The stand-in (no PP_TP, no PP_LLAMA): the CPU TP Commit tests/pipeline/served_tp --deferred, world 2, SiluMul_v2{I=256}:
#     honest subset:4, forged subset:8 (every unit drawn); with PP_PLAIN=1, stratified:4 and stratified:8.
#   PP_PLAIN=1: plain C-Flock sessions, not --zk (prove_path.py audit --plain, a5's --plain): one table under a stratified
#     draw, until multi-table coverage lands, so each partition's law must be stratified:K over a window of one member.
#   PP_KEEP_WINDOWS=DIR: each run's committed windows copied whole to DIR/<name>-windows (the prover's files with them, so
#   DIR stays on the node, under `research keep`), for a later job such as recursion_e2e/stage.sh WINDOW=DIR/served-windows/rank0.
#   PP_JOBS=N ranks at once (a GLM-5.3 rank: its bundle window ~2.5 min and ~11 GB, its audit ~10 GB); PP_RANKS=0,3 those ranks
#   only; PP_FORGE=MEMBER:UNIT the forged unit (default 0:1); PP_ONLY=honest|forged that run alone; PP_FLOCK_WORK=DIR keeps M0's
#   build across runs. The Llama honest audit peaks at ~20 GB (one process), its forged one well under.
#   PP_TRANSFER_GATES="T ..." the templates the registered compiler lists as holding transfer gates (--transfer-gate, both
#   sides), every unit of theirs proved beside the partition's work:K of the rest; PP_SETS="SPEC ..." more input sets, a
#   subcircuit spec each (rope-head/d64/bf16); PP_FORGED_PARTITION the forged run's partition. E.g. Llama TP1 with layer 0's
#   160 RoPE heads at step 1 as the stand-in transfer gates: PP_PARTITION=benchmarks/prove_path/llama32_1b_tg_l0_step1.json
#   PP_TRANSFER_GATES='RoPEHead_v1{D=64}' PP_SETS=rope-head/d64/bf16, its forged run PP_FORGED_PARTITION=benchmarks/prove_path/
#   llama32_1b_tg_l0_step24.json PP_FORGE=0:0 (a RoPE head, drawn because every one is).
#   PP_CHALLENGER=local: each window drawn on a local challenger session's coins (--challenger local, a5_coins), not the
#   verifier's own, pinned on main's first-parent history fetched from the clone's origin. PP_CHALLENGER=gate:USER@HOST:
#   the auditor host's, through its SSH gate with the lane's gate key PP_GATE_KEY, its challenger on PP_PEER's 127.0.0.1 (this
#   node, as the gate's /etc/vy/peers names it), under the key record PP_CHALLENGER_KEY (challenger-key/v1). The gate opens a
#   root once, and both arms register under the window's one run root: with the gate, run one arm (PP_ONLY).
# From a clone of the tree, with 64 GB of memory, the outputs in $OUT (default $RESEARCH_RUN_DIR/out):
#   PP_LLAMA=DIR bash benchmarks/prove_path/run.sh
# Kept in $OUT: the records and logs, the public files, the sessions (their proofs and records), the binaries' sha256; never the
# advice or its opening (the prover's, private: the verifier is handed the opening, the named gap), a prover's file (inst-*,
# *-private.bin: the rows and their salts), a circuit (its pin is) nor a request value (the stand-in's workload.json, its prompts).
set -uxo pipefail
for v in PP_LLAMA PP_SERVE_RUN PP_SERVE_ROWS; do
  case "${!v:-}" in art:*) echo "$v=${!v}: a published row dir or serve run holds request values; give $v the serve" \
    "directory on this node (a DIR)"; exit 2;; esac
done
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
LROW=llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager
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

for I in 256 1536 8192 ${PP_SETS:-}; do   # the verifier's lowering: a synthetic input set per template (its values are never read)
  case $I in */*) SUB=$I SET=${I//\//_};; *) SUB=silu-mul-v2/i$I/bf16 SET=i$I;; esac
  $PY -m verity_numerical.bench.generate --subcircuit $SUB --n 4 --seed 1 --out $W/sets/$SET > $O/sets-$SET.log 2>&1 || exit 9
done
TG=(); for t in ${PP_TRANSFER_GATES:-}; do TG+=(--transfer-gate "$t"); done
CH=()
if [ -n "${PP_CHALLENGER:-}" ]; then   # the verifier's own copy of main's first-parent history, which the session's pin must be on
  UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
  git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > $O/fetch-main.log 2>&1 \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> $O/fetch-main.log 2>&1 \
    && git rev-list --first-parent refs/remotes/origin/main > $O/main.txt || { tail -5 $O/fetch-main.log; exit 4; }
  CH=(--challenger $PP_CHALLENGER --main $O/main.txt)
  case $PP_CHALLENGER in gate:*)
    [ -n "${PP_PEER:-}" ] && [ -n "${PP_CHALLENGER_KEY:-}" ] || { echo "PP_CHALLENGER=gate:… needs PP_PEER and PP_CHALLENGER_KEY"; exit 4; }
    [ -n "${PP_ONLY:-}" ] || { echo "PP_CHALLENGER=gate:… opens the window's root once: run one arm, PP_ONLY=honest|forged"; exit 4; }
    CH+=(--peer $PP_PEER --challenger-key $PP_CHALLENGER_KEY ${PP_GATE_KEY:+--gate-key $PP_GATE_KEY});;
  esac
fi
PLAIN=(); LAW=subset; [ -n "${PP_PLAIN:-}" ] && PLAIN=(--plain) LAW=stratified
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover at m=27 can overflow on a many-core
# node; its own worker and verifier threads take 64 MiB
export RUST_MIN_STACK=$((64 << 20))
TIME=$([ -x /usr/bin/time ] && echo "/usr/bin/time -v")
PP="$PY benchmarks/prove_path/prove_path.py"

keep() {  # NAME: $W/NAME's records into $O/NAME, among them the circuit each drawn member was proved on (window/m<k>/circuit.txt)
  mkdir -p $O/$1
  (cd $W/$1 && find . -type f \( ! -name circuit.txt -o -path './window/m*/circuit.txt' \) ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' ! -path './*/archive/*' \
    ! -path './builds/*' ! -path './proof-edits/*/sessions/*' ! -path './proof-edits/mutants/*/*' ! -path './proof-edits/*/member/*' -print0 \
    | xargs -0 -r -I{} cp --parents {} $O/$1/)
}
run() {  # NAME RUN ADVICE PARTITION RANKS [--forge M:U]: the prover's windows, then the verifier's audit; RUN is the committed
         # run's arguments (--tp-commit DIR --build-root DIR, --bundle DIR --row-dir DIR --workload F, or --serve-run DIR
         # --window K --row-dir DIR, whose advice and opening are the server's: ADVICE -)
  local name=$1 src=$2 advice=$3 part=$4 ranks=$5; shift 5
  local forged=; [ $# -gt 0 ] && forged=--forged
  case "${PP_ONLY:-}:${forged:+forged}" in honest:forged|forged:) return 0;; esac
  local adv="--advice $advice" op="--opening $W/$name-opening.json"
  case "$src" in *--serve-run*) adv=; op=;; esac
  mkdir -p $O/$name
  $TIME $PP commit $src $adv $op --partition $part \
    ${ranks:+--ranks $ranks} "${TG[@]}" \
    --jobs ${PP_JOBS:-1} --out $W/$name-windows "$@" > $O/$name/commit.log 2>&1 || { tail -30 $O/$name/commit.log; return 1; }
  for d in $W/$name-windows/rank*; do   # serving's public files, never a private body
    mkdir -p $O/$name/windows/$(basename $d) && cp $d/{registration,window,partition,index}.json $d/m*-public.bin $O/$name/windows/$(basename $d)/
    [ ! -e $d/circuit.json ] || cp $d/circuit.json $O/$name/windows/$(basename $d)/
  done
  cp $W/$name-windows/commit.json $O/$name/windows/
  if [ -n "${PP_KEEP_WINDOWS:-}" ]; then   # the windows whole, the prover's files included, for a later job on this node
    mkdir -p "$PP_KEEP_WINDOWS" && rm -rf "$PP_KEEP_WINDOWS/$name-windows" && cp -a $W/$name-windows "$PP_KEEP_WINDOWS/" || return 1
    R keep "$PP_KEEP_WINDOWS" --owner @proofs --expires 7d --priority low \
      --why "prove_path/run.sh PP_KEEP_WINDOWS: committed windows (rows and salts) a later job stages from" >> $O/keep.log 2>&1
  fi
  if [ -n "${PP_RECORDS:-}" ] && [ -n "${REC_RUN:-}" ]; then   # the window's audit from its records alone
    records $name $part $forged
    local rc=$?
    keep $name
    return $rc
  fi
  $TIME $PP audit $src $op --windows $W/$name-windows --partition $part \
    ${ranks:+--ranks $ranks} --jobs ${PP_JOBS:-1} ${PP_SESSION_TIMEOUT:+--session-timeout $PP_SESSION_TIMEOUT} $forged "${TG[@]}" "${CH[@]}" "${PLAIN[@]}" \
    ${PP_VLLM_PYTHON:+--vllm-python $PP_VLLM_PYTHON} --template-sets $W/sets --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --out $W/$name > $O/$name/audit.log 2> $O/$name/audit.err
  local rc=$?
  grep '^PROVE-PATH' $O/$name/audit.log | cut -c1-6000; tail -25 $O/$name/audit.err
  keep $name
  return $rc
}
records() {  # NAME PARTITION [--forged]: records_audit.py on the served window REC_RUN's window REC_K, the registered compiler
             # the served one as a command (prove_path.py circuit), the run root the one windows.jsonl publishes
  local name=$1 part=$2 forged=${3:-}
  $PY -c 'import json, sys
c = json.load(open(sys.argv[1]))
print(json.dumps({c["name"]: {"digest": c["digest"], "argv": sys.argv[2:]}}))' $O/$REC_NAME-records/compiler.json \
    $PY $REPO/benchmarks/prove_path/prove_path.py circuit --serve-run $REC_RUN --window $REC_K --vllm-python ${PP_VLLM_PYTHON:-$PY} \
    ${PP_TARGET:+--target $PP_TARGET} --partition $(realpath $part) "${TG[@]}" > $O/$name/compilers.json || return 1
  local root law
  root=$($PY -c 'import json, sys
print(next(w["run_root"] for w in map(json.loads, open(sys.argv[1])) if int(w["window"]) == int(sys.argv[2])))' $REC_RUN/windows.jsonl $REC_K) || return 1
  law=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["law"])' $part) || return 1
  $TIME $PY benchmarks/prove_path/records_audit.py audit --window $W/$name-windows/rank0 --law $law --run $root \
    --opening $REC_RUN/advice/w$REC_K/private/opening.json --compilers $O/$name/compilers.json --template-sets $W/sets \
    ${PP_SESSION_TIMEOUT:+--session-timeout $PP_SESSION_TIMEOUT} $forged "${CH[@]}" \
    --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --out $W/$name > $O/$name/audit.log 2> $O/$name/audit.err
  local rc=$?
  grep '^RECORDS' $O/$name/audit.log | cut -c1-6000; tail -25 $O/$name/audit.err
  return $rc
}

rc=0
FORGE=${PP_FORGE:-0:1}
if [ -n "${PP_TP:-}" ]; then
  $PP advice --workload integrations/vllm/workloads/$ROW.json --out $W/advice.bin | tee $O/advice.log || exit 8
  GLM="--tp-commit $PP_TP/commit --build-root $PP_TP/build"
  run glm "$GLM" $W/advice.bin ${PP_PARTITION:-benchmarks/one_stage/glm53_silu_mul_v2_tp8_step16_stratified.json} "${PP_RANKS:-}" || rc=1
  run glm-forged "$GLM" $W/advice.bin benchmarks/prove_path/glm53_silu_mul_v2_i1536_step16_all.json "${PP_FORGED_RANKS:-0}" \
    --forge $FORGE || rc=1
elif [ -n "${PP_LLAMA:-}${PP_SERVE_RUN:-}" ]; then
  D=${PP_SERVE_ROWS:-${PP_LLAMA:-}}
  [ -d "$D" ] || { echo "no row dir ${D:-(PP_SERVE_ROWS or PP_LLAMA)}: give the Commit's row dir on this node"; exit 8; }
  STEP1=benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step1.json STEP24=benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step24.json
  served() {  # NAME SERVE_RUN K: the served window's honest and forged audits, its public records and its compiler kept
    mkdir -p $O/$1-records
    cp $2/serve.json $2/windows.jsonl $2/advice/w$3/registration.json $O/$1-records/
    $PP compiler --serve-run $2 ${PP_TARGET:+--target $PP_TARGET} "${TG[@]}" | tee $O/$1-records/compiler.json || return 1
    local src="--serve-run $2 --window $3 --row-dir $D ${PP_TARGET:+--target $PP_TARGET}"
    REC_NAME=$1 REC_RUN=$2 REC_K=$3
    run $1 "$src" - ${PP_PARTITION:-$STEP1} "" || rc=1
    run $1-forged "$src" - ${PP_FORGED_PARTITION:-$STEP24} "" --forge ${PP_FORGE:-0:0} || rc=1
    REC_RUN=
  }
  if [ -n "${PP_SERVE_RUN:-}" ]; then
    served served $PP_SERVE_RUN ${PP_WINDOW:-0} || rc=1
  else
    LLAMA="--bundle $D/commit/replay_bundle_p0 --row-dir $D --workload examples/vllm/workloads/$LROW.json"
    if [ -n "${PP_SERVE_STANDIN:-}" ]; then
      $PP serve-standin $LLAMA --out $W/serve-run | tee $O/serve-standin.json || exit 8
      served llama-served $W/serve-run 0 || rc=1
    else
      $PP advice --workload examples/vllm/workloads/$LROW.json --out $W/advice.bin | tee $O/advice.log || exit 8
      run llama "$LLAMA" $W/advice.bin ${PP_PARTITION:-$STEP1} "" || rc=1
      run llama-forged "$LLAMA" $W/advice.bin ${PP_FORGED_PARTITION:-$STEP24} "" --forge ${PP_FORGE:-0:0} || rc=1
    fi
  fi
elif [ -n "${PP_WEIGHTS_TIE:-}" ]; then
  TLAW=${PP_LAW:-work:1} TCAP=${PP_TIE_CAP:-16} F=$W/tie-fixture
  (cd integrations/vllm && $PY -m tests.program.predict.weights_tie_fixture $F) > $O/tie-fixture.json 2> $O/tie-fixture.err \
    || { tail -20 $O/tie-fixture.err; exit 8; }
  WT="$PY -m verity_vllm.pipeline.cli weights-tie"
  tie() {  # NAME [--served] [--forge-digest]: the committer's registration of X and its tie window (--served: Y's words served
           # as X's; --forge-digest: its ends' digests forged to X's), the verifier's tie of X, the records audit of the window
           # against it, and the pins, which only an accepting audit writes. Each arm is a run (and a root) of its own
    local name=$1 forged= id=${RESEARCH_RUN_ID:-local}/$1 commit=() c run rc pinned; shift
    for x in "$@"; do
      case $x in
        --served) forged=--forged; commit+=(--served-dir $F/y/snap --served-manifest $F/y/checkpoints.json);;
        --forge-digest) commit+=(--forge-digest);;
      esac
    done
    local common=(--phi $F/phi.json --checkpoint-dir $F/x/snap --manifest $F/x/checkpoints.json
                  --grain $($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["grain"])' $O/tie-fixture.json)
                  --law $TLAW --cap $TCAP --run-id $id)
    mkdir -p $O/$name
    $WT commit "${common[@]}" --key-file $W/$name-key --registration-out $O/$name/registration.json --out $W/$name-window \
      "${commit[@]}" | tee $O/$name/commit.json || return 1
    $WT verify "${common[@]}" --registration $O/$name/registration.json --out $O/$name/verifier | tee $O/$name/verify.json || return 1
    c=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["circuit"])' $O/$name/verifier/tie.json) || return 1
    run=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' $O/$name/verifier/tie.json) || return 1
    mkdir -p $O/$name/window && cp $W/$name-window/{registration,window,index,circuit}.json $W/$name-window/m*-public.bin $O/$name/window/
    $TIME $PY benchmarks/prove_path/records_audit.py audit --window $W/$name-window --law $TLAW --circuit $c --run $run \
      --registered $O/$name/verifier/registered.json --outputs $O/$name/verifier/outputs.json \
      ${PP_SESSION_TIMEOUT:+--session-timeout $PP_SESSION_TIMEOUT} $forged "${CH[@]}" \
      --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --out $W/$name > $O/$name/audit.log 2> $O/$name/audit.err
    rc=$?
    grep '^RECORDS' $O/$name/audit.log | cut -c1-6000; tail -25 $O/$name/audit.err
    keep $name
    $WT pins "${common[@]}" --registration $O/$name/registration.json --summary $W/$name/summary.json --out $O/$name/pins.json \
      | tee $O/$name/pins-verdict.json
    pinned=$?
    if [ -n "$forged" ]; then [ $pinned != 0 ] && [ ! -e $O/$name/pins.json ] || rc=1; else [ $pinned = 0 ] || rc=1; fi
    return $rc
  }
  tie weights-tie || rc=1
  [ "${PP_ONLY:-}" = honest ] || tie weights-tie-other --served || rc=1
  [ "${PP_ONLY:-}" = honest ] || tie weights-tie-forged --served --forge-digest || rc=1
else
  S=$W/served
  (cd integrations/vllm && $PY -m tests.pipeline.served_tp $S --world 2 --I 256 --law $LAW:4 --deferred) \
    > $O/standin.json 2> $O/standin.err || { tail -20 $O/standin.err; exit 8; }
  mkdir -p $O/standin && find $S/commit -maxdepth 1 -type f -exec cp {} $O/standin/ \;
  $PP advice --workload $S/workload.json --out $W/advice.bin | tee $O/advice.log || exit 8
  echo "{\"templates\": [\"SiluMul_v2{I=256}\"], \"law\": \"$LAW:8\"}" > $W/all.json
  SI="--tp-commit $S/commit --build-root $S/build"
  run standin "$SI" $W/advice.bin $S/partition.json "${PP_RANKS:-}" || rc=1
  run standin-forged "$SI" $W/advice.bin $W/all.json "${PP_FORGED_RANKS:-0}" --forge $FORGE || rc=1
fi
exit $rc
