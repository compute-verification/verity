#!/usr/bin/env bash
# Prove path on a CPU machine: the checkpoint tie of vLLM's weights registration, proved by M0 --zk behind the Lean ZK
# server and verified by Lean. M0's flock-circuit and Lean's verity-verify-proof are built from this tree, in scratch.
#   PP_WEIGHTS_TIE=1: the checkpoint tie of vLLM's weights registration at TINY (verity_vllm.program.predict.weights_tie;
#     its inputs tests/program/predict/weights_tie_fixture): the committer registers checkpoint X's weights and records
#     its tie window, one SHA-512 chain a shard over the registered rows (the embedding table's row-seg rows a chain of
#     their own), ending at the verifier's own digest; the verifier makes its tie of X from X's public bytes (the circuit's
#     SHA-512, the run root, the registered entries and its supplied values), audits the window against it under the
#     tie's own design, work:N over its N units in members of PP_TIE_CAP units (default 16), which the audit's rule
#     samples at the rate N/N, so every unit is proved and no output class is passed, and pins the roots only on an
#     accepting audit whose sample holds every unit.
#   PP_FLOCK_WORK=DIR keeps M0's build across runs.
#   PP_CHALLENGER=local (the default): each window sampled on a local challenger session's coins (--challenger local, a5_coins),
#   pinned on main's first-parent history fetched from the clone's origin; the verifier flips no coins of its own. PP_CHALLENGER=gate:USER@HOST:
#   the auditor device's, through its SSH gate with the lane's gate key PP_GATE_KEY, its challenger on PP_PEER's 127.0.0.1 (this
#   node, as the gate's /etc/vy/peers names it), under the key record PP_CHALLENGER_KEY (challenger-key/v1). PP_MAIN (a
#   file, `git rev-list --first-parent` of the gate's pin) is that history instead, for a gate pinned on `next`.
# From a clone of the tree, with 64 GB of memory, the outputs in $OUT (default $RESEARCH_RUN_DIR/out, outside a research run
# out/prove_path under the clone); $OUT/ran_with.txt is the line naming the run's coins and parameters (prove_path.py ran-with):
#   PP_WEIGHTS_TIE=1 bash benchmarks/prove_path/run.sh
# Kept in $OUT: the records and logs, the public files, the sessions (their proofs and records) of a records audit, an
# audit's summaries in their public form (public_log.py summaries), the binaries' sha256; never the
# advice or its opening (the prover's, private: the verifier is handed the opening, the named gap), a prover's file (inst-*,
# *-private.bin: the rows and their salts), a circuit (its pin is), a request value (the stand-in's workload.json, its prompts)
# nor a serve's own records (serve.json, windows.jsonl: only vLLM's mirror publishes them, through serve_mirror.PUBLIC).
# The rest goes to PP_KEEP=DIR (0700, refused under $OUT or the run dir and named in its private-paths.json; without it,
# scratch removed at exit): a window's
# registration, commit.json, partition and the files beside them, an audit's own records whole (its summaries with the
# advice's length and what is counted from it, its copy of the registration, its sessions), the tools' logs and error output,
# the machine's shape, and the commit's and audits' logs whole: $OUT and stdout get their lines as public_log.py lets
# them go.
set -uxo pipefail
PUBF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../one_stage/public_files.py
PL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/public_log.py  # a log as it may leave: the fields its reader reads, errors by their class
[ -z "${PP_TP:-}" ] || { echo "PP_TP=$PP_TP: refused: the deferred TP Commit is deleted; give PP_WEIGHTS_TIE"; exit 2; }
[ -n "${PP_WEIGHTS_TIE:-}" ] || { echo "refused: the served arm is deleted; give PP_WEIGHTS_TIE"; exit 2; }
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:+$RESEARCH_RUN_DIR/out}}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
[ -n "$O" ] || { O=$REPO/out/prove_path; echo "OUT unset outside a research run: the outputs go to $O"; }
mkdir -p $O
K=${PP_KEEP:-$W/keep}
for p in "$O" "${RESEARCH_RUN_DIR:-}"; do
  [ -z "$p" ] || case $(realpath -m "$K")/ in "$(realpath -m "$p")"/*)
    echo "PP_KEEP=$K is under $p, which custody publishes: give a directory outside it"; exit 2;; esac
done
(umask 077 && mkdir -p "$K") && chmod 700 "$K" || exit 2
if [ -n "${PP_KEEP:-}" ] && [ -d "${RESEARCH_RUN_DIR:-}" ]; then   # the keep named for a reader on this node (`prove layers`)
  python3 - "$RESEARCH_RUN_DIR/private-paths.json" "$(realpath -m "$K")" <<'EOF' || exit 2
import json, pathlib, sys
f, k = pathlib.Path(sys.argv[1]), sys.argv[2]
doc = json.loads(f.read_text()) if f.is_file() else {}
f.write_text(json.dumps({**doc, "schema": "private-paths/v1", "private": sorted({*doc.get("private", []), k}),
                         "names": list(doc.get("names", []))}, indent=1) + "\n")
EOF
fi
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; } > $K/machine.txt
git rev-parse HEAD HEAD:verity/protocols/proof/flock | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $K/sync.log 2>&1 || { tail -20 $K/sync.log; exit 5; }
PY=$REPO/.venv/bin/python
(cd verity/protocols/proof/flock/verifier && lake build verity-verify-proof flock-zk-server flock-read-tmpl) > $K/build-lean.log 2>&1 &
LP=$!
mkdir -p $K/m0-build
FW=${PP_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$K/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/proof/flock/prover/pod/60-circuit.sh > $K/m0-build/build.log 2>&1 \
  || { tail -40 $K/m0-build/build.log; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 $K/build-lean.log; exit 7; }
LEAN=$REPO/verity/protocols/proof/flock/verifier/.lake/build/bin/verity-verify-proof; FWL=${LEAN%/*}/flock-zk-server
(cd verity/protocols/proof/spot/verifier/lean && lake build spot-verifier) > $K/build-sampled-proofs.log 2>&1 || { tail -40 $K/build-sampled-proofs.log; exit 7; }
sha256sum $BIN $LEAN $FWL | tee $O/binaries.sha256
PP_CHALLENGER=${PP_CHALLENGER:-local}   # the sample's coins are a challenger's: a local development session unless named
$PY benchmarks/prove_path/prove_path.py ran-with --challenger "$PP_CHALLENGER" ${PP_CHALLENGER_KEY:+--challenger-key "$PP_CHALLENGER_KEY"} \
  | tee $O/ran_with.txt || exit 4

CH=()
if [ -n "${PP_CHALLENGER:-}" ] && [ -n "${PP_MAIN:-}" ]; then   # the run's first-parent list, which holds the gate's pin
  [ -s "$PP_MAIN" ] || { echo "PP_MAIN $PP_MAIN is missing or empty"; exit 4; }
  [ "$PP_MAIN" -ef $O/main.txt ] || cp "$PP_MAIN" $O/main.txt
elif [ -n "${PP_CHALLENGER:-}" ]; then   # the verifier's own copy of main's first-parent history, which the session's pin must be on
  UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
  git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > $K/fetch-main.log 2>&1 \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> $K/fetch-main.log 2>&1 \
    && git rev-list --first-parent refs/remotes/origin/main > $O/main.txt || { tail -5 $K/fetch-main.log; exit 4; }
fi
if [ -n "${PP_CHALLENGER:-}" ]; then
  CH=(--challenger $PP_CHALLENGER --main $O/main.txt)
  case $PP_CHALLENGER in gate:*)
    [ -n "${PP_PEER:-}" ] && [ -n "${PP_CHALLENGER_KEY:-}" ] || { echo "PP_CHALLENGER=gate:… needs PP_PEER and PP_CHALLENGER_KEY"; exit 4; }
    CH+=(--peer $PP_PEER --challenger-key $PP_CHALLENGER_KEY ${PP_GATE_KEY:+--gate-key $PP_GATE_KEY});;
  esac
fi
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proof/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover at m=27 can overflow on a many-core
# node; its own worker and verifier threads take 64 MiB
export RUST_MIN_STACK=$((64 << 20))
TIME=$([ -x /usr/bin/time ] && echo "/usr/bin/time -v")

keep() {  # NAME [public]: $W/NAME's records into $O/NAME, among them the circuit each drawn member was proved on
          # (window/m<k>/circuit.txt); with public, into $K/NAME, and $O/NAME gets only their summaries' public form
  local to=$O/$1; [ "${2:-}" = public ] && to=$K/$1
  mkdir -p $to
  (cd $W/$1 && find . -type f \( ! -name circuit.txt -o -path './window/m*/circuit.txt' \) ! -name 'inst-*' ! -name 'm*-inst-*' ! -path './*/archive/*' \
    ! -path './builds/*' ! -path './proof-edits/*/sessions/*' ! -path './proof-edits/mutants/*/*' ! -path './proof-edits/*/member/*' -print0 \
    | python3 "$PUBF" filter | xargs -0 -r -I{} cp --parents {} $to/)
  # a case's blobs name its files in $W, which the EXIT trap removes: name the kept copies instead
  (cd $to && find . -name '*.json' -print0 | xargs -0 -r grep -lZ '"blobs"' \
    | xargs -0 -r $PY -c 'import json,sys
w, o = sys.argv[1:3]
for f in sys.argv[3:]:
    c = json.load(open(f))
    if isinstance(c, dict) and isinstance(c.get("blobs"), list):
        c["blobs"] = [o + b[len(w):] if isinstance(b, str) and b.startswith(w) else b for b in c["blobs"]]
        open(f, "w").write(json.dumps(c, sort_keys=True))' "$W/$1/" "$(cd $to && pwd)/")
  [ "${2:-}" != public ] || python3 "$PL" summaries $to $O/$1
}
rc=0
if [ -n "${PP_WEIGHTS_TIE:-}" ]; then
  TCAP=${PP_TIE_CAP:-16} F=$W/tie-fixture
  (cd tests/adapters/vllm && $PY -m tests.program.predict.weights_tie_fixture $F) > $O/tie-fixture.json 2> $K/tie-fixture.err \
    || { tail -20 $K/tie-fixture.err; exit 8; }
  WT="$PY -m verity_vllm.pipeline.cli weights-tie"
  pins_answered() {  # STATUS VERDICT ERR PINS: `weights-tie pins` gave its own answer, with no traceback: 0 and
                     # `"pinned": true` with PINS written
    ! grep -q '^Traceback (most recent call last):' "$3" || return 1
    [ "$1" = 0 ] && grep -q '^{"pinned": true, ' "$2" && [ -e "$4" ]
  }
  tie() {  # NAME: the committer's registration of X and its tie window, the verifier's tie of X, the records audit of the
           # window against it, and the pins, which only an accepting audit writes
    local name=$1 id=${RESEARCH_RUN_ID:-local}/$1 c run units rc pinned
    local common=(--phi $F/phi.json --checkpoint-dir $F/x/snap --manifest $F/x/checkpoints.json
                  --grain $($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["grain"])' $O/tie-fixture.json)
                  --cap $TCAP --run-id $id)
    mkdir -p $O/$name $K/$name/window
    $WT commit "${common[@]}" --registration-out $W/$name-registration.json --out $W/$name-window \
      | tee $O/$name/commit.json || return 1
    cp $W/$name-registration.json $O/$name/registration.json
    $WT verify "${common[@]}" --registration $O/$name/registration.json --out $O/$name/verifier | tee $O/$name/verify.json || return 1
    c=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["circuit"])' $O/$name/verifier/tie.json) || return 1
    run=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' $O/$name/verifier/tie.json) || return 1
    units=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["units"])' $O/$name/verifier/tie.json) || return 1
    mkdir -p $O/$name/window && cp $W/$name-window/{registration,circuit}.json $O/$name/window/ \
      && cp $W/$name-window/{window,index}.json $W/$name-window/m*-public.bin $K/$name/window/
    $TIME $PY benchmarks/prove_path/records_audit.py audit --window $W/$name-window --sampling-design work-rate:$units/$units --circuit $c --run $run \
      --registered $O/$name/verifier/registered.json \
      ${PP_SESSION_TIMEOUT:+--session-timeout $PP_SESSION_TIMEOUT} "${CH[@]}" \
      --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --zk-server $FWL --out $W/$name > $K/$name/audit.log 2> $K/$name/audit.err
    rc=$?
    python3 "$PL" $K/$name/audit.log > $O/$name/audit.log
    grep '^RECORDS' $O/$name/audit.log | cut -c1-6000; python3 "$PL" $K/$name/audit.err | tail -25
    keep $name
    $WT pins "${common[@]}" --registration $O/$name/registration.json --summary $W/$name/summary.json --out $O/$name/pins.json \
      2> $K/$name/pins.err | tee $O/$name/pins-verdict.json
    pinned=$?
    pins_answered $pinned $O/$name/pins-verdict.json $K/$name/pins.err $O/$name/pins.json \
      || { echo "PINS ERROR $name: weights-tie pins ended with status $pinned and not its own pins"; tail -5 $K/$name/pins.err; rc=1; }
    return $rc
  }
  tie weights-tie || rc=1
fi
cat $O/ran_with.txt
exit $rc
