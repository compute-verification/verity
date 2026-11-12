#!/usr/bin/env bash
# stop_audit.py end to end on a CPU node, on a stop class at small statics (stop_audit_small.py: no engine):
# flock-circuit, flock-verify and SPOT's Lean verifier built from this tree as two_stage_small.sh builds them, then
# `two_stage.py first --stop --words` (the boundary registration, the stop registration under the class's certified
# words table and both samples on a local challenger session) and seven audits of the stop population. The words
# tables are a stand-in network certifier's (stop_words.py), read under the verifier's copy of its key:
#   accept      the class as served: the audit record accepts on Lean's verdict of record, and is not complete
#   length      a class registered with stop words that are not the rule's: Lean refuses its session
#   unlogged    a stop record on a boundary record the boundary stream never logged: refused before any session
#   unaccepted  a boundary stream's outcome that did not accept: the boundary-audit entry refuses the record
#   incomplete  a boundary stream's outcome that accepted with an entry that did not run: the record accepts, the
#               boundary-audit entry has no verdict, and it is not complete
#   table       the class's own hand-off under a certifier's table whose max_tokens differs for one request: the stop
#               record registers the table its blocks were proved at, and is refused before any session (R7-certified)
#   words       a class registered under such a table and proved at the engine's max_tokens: Lean refuses its session
#               at the registered read, and the certified-words entry names the row (stop-read-root)
# Exit 0 when the seven are what stop_audit_small.py expect holds them to. Records under $OUT (never a circuit, a
# prover file, a private body or salts, the certifier's opening or signer state, or a challenger's key or secret).
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/stop_audit_small.sh
#   SMALL_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); SMALL_ARGS="--serve NONCE"
#   passes stop_audit_small.py serve more; SMALL_AUDIT="--firewall PATH" passes stop_audit.py more
set -euo pipefail
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; } | tee "$O/machine.txt"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }

{ (cd verity/protocols/flock/verifier && lake build flock-verify flock-firewall) \
    && (cd verity/protocols/spot/verifier/lean && lake build sampled-proofs-verifier); } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
FW=${SMALL_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/protocols/flock/verifier/.lake/build/bin/flock-verify
sha256sum "$BIN" "$LEAN" | tee "$O/binaries.sha256"

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
export RAYON_NUM_THREADS=$(nproc) PYTHONPATH=$M0PATH
# a local challenger session's pin is held to main's first-parent line: a run's clone has the shipped tree as its origin,
# which holds main as its own main or origin/main
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$O/fetch-main.log" 2>&1 \
  && git rev-list --first-parent refs/remotes/origin/main > "$O/main.txt" || { tail -5 "$O/fetch-main.log"; exit 4; }

PY=(uv run --no-sync python)
DESIGN=rate:2/6
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/stop_audit_small.py serve --out "$W/serve" ${SMALL_ARGS:-} | tee "$O/serve.json"
field() { "${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$O/serve.json" "$1"; }
CHECK=$(field check); STOP=$(field stop); CLASS=$(field class); WRONG=$(field wrong); WRONG_FILES=$(field wrong_files)
WORDS_CLASS=$(field words); CERTIFIER=$(field certifier)
# each class's certifier's files: the signed head (public), and the table's opening, which only a prover's half reads
CERTIFIED=$(field certified); CHANGED=$(field changed); WRONG_CERTIFIED=$(field wrong_certified)
WORDS_CERTIFIED=$(field words_certified)
NAME=$(basename "$CLASS")

first() {  # NAME CLASSDIR CERTIFIED: the class's boundary and stop registrations and their samples
  "${PY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$2" --check "$CHECK" --replay "$DESIGN" --k 1 --stop "$STOP" \
    --words "$3/words.json" --certifier "$CERTIFIER" \
    --challenger local --main "$O/main.txt" --out "$W/$1" > "$O/$1.log" 2>&1 || { tail -20 "$O/$1.log"; exit 3; }
  tail -2 "$O/$1.log"
}
audit() {  # NAME STAGE1 CLASSDIR HEAD OPENING [more]: one audit of the stop population; its exit code beside its records
  local name=$1 stage1=$2 class=$3 head=$4 opening=$5 rc=0; shift 5
  # shellcheck disable=SC2086
  "${PY[@]}" benchmarks/one_stage/stop_audit.py --stage1 "$stage1" --boundary "$class" --stop "$STOP" --design "$DESIGN" \
    --words "$head/words.json" --certifier "$CERTIFIER" --words-opening "$opening/words-opening.json" \
    --bin "$BIN" --lean "$LEAN" --out "$W/$name" "$@" ${SMALL_AUDIT:-} > "$O/$name.log" 2>&1 || rc=$?
  grep '^STOP ' "$O/$name.log" || tail -30 "$O/$name.log"
  echo "$name rc=$rc" | tee -a "$O/machine.txt"
}

first first "$CLASS" "$CERTIFIED"
first first-wrong "$WRONG" "$WRONG_CERTIFIED"
first first-words "$WORDS_CLASS" "$WORDS_CERTIFIED"
S1=$W/first/$NAME/stage1.json
"${PY[@]}" benchmarks/one_stage/stop_audit_small.py unlogged "$S1" --words "$CERTIFIED/words.json" --certifier "$CERTIFIER" \
  > "$W/unlogged.json"
"${PY[@]}" benchmarks/one_stage/stop_audit_small.py unaccepted "$S1" > "$W/unaccepted.json"
"${PY[@]}" benchmarks/one_stage/stop_audit_small.py incomplete "$S1" > "$W/incomplete.json"
audit accept "$S1" "$CLASS" "$CERTIFIED" "$CERTIFIED"
audit words "$W/first-words/$NAME/stage1.json" "$WORDS_CLASS" "$WORDS_CERTIFIED" "$WORDS_CERTIFIED" --engine-words
audit table "$S1" "$CLASS" "$CHANGED" "$CERTIFIED"
audit length "$W/first-wrong/$NAME/stage1.json" "$WRONG_FILES" "$WRONG_CERTIFIED" "$WRONG_CERTIFIED" --forge "$(seq -s, 0 5)"
audit unlogged "$W/unlogged.json" "$CLASS" "$CERTIFIED" "$CERTIFIED"
audit unaccepted "$S1" "$CLASS" "$CERTIFIED" "$CERTIFIED" --boundary-outcome "$W/unaccepted.json"
audit incomplete "$S1" "$CLASS" "$CERTIFIED" "$CERTIFIED" --boundary-outcome "$W/incomplete.json"

RECORDS=(-type f ! -name circuit.txt ! -name 'inst-*' ! -name '*-private.bin' ! -name index.json ! -name index-salt.bin
         ! -name salts.bin ! -path '*/archive/*' ! -path './prover/*' ! -path '*/challenger/*' -size -64M)
for d in first first-wrong first-words accept length unlogged unaccepted incomplete table words; do
  [ -d "$W/$d" ] || continue
  mkdir -p "$O/$d"
  (cd "$W/$d" && find . "${RECORDS[@]}" -print0 | xargs -0 -r -I{} cp --parents {} "$O/$d/")
done
cp "$W/unlogged.json" "$W/unaccepted.json" "$W/incomplete.json" "$O/"
for d in "$CERTIFIED" "$CHANGED" "$WRONG_CERTIFIED" "$WORDS_CERTIFIED"; do
  mkdir -p "$O/certifier/${d#"$W/serve/"}"
  cp "$d/words.json" "$d/certifier.json" "$O/certifier/${d#"$W/serve/"}/"
done
"${PY[@]}" benchmarks/one_stage/stop_audit_small.py expect "$W" | tee "$O/expect.txt"
