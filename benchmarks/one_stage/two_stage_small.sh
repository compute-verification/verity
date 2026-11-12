#!/usr/bin/env bash
# two_stage.py's two commands end to end on a CPU node, on a token select class at small statics (two_stage_small.py: no
# engine): flock-circuit, flock-verify and SPOT's Lean verifier built from this tree as the toy's audit.sh builds them,
# then `first` (the boundary registration and the first stage on a local challenger session) and `second` (the interior,
# the second stage on a session of its own, a --zk session per member, Lean's verdicts, round 2 and the stream's outcome).
# Exit 0 when the second stage is what today's record gives (two_stage_small.py expect): round 1 commits no port for the
# block's return, so everything holds but the return's read. Records under $OUT (never a circuit, a prover file, a
# private body or salts, or a challenger's key or secret); a record over 63 MiB (a served vocabulary's public bodies and
# written window, a challenger host's ledger) is not kept, and large.txt lists its SHA-512, size and path.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/two_stage_small.sh
#   SMALL_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); SMALL_ARGS="--v 256" passes
#   two_stage_small.py serve more; SMALL_FIRST and SMALL_SECOND="--firewall PATH" pass `first` and `second` more;
#   SMALL_INLINE=1 runs the class as a served vocabulary's is, its block the check's body inlined, with `second
#   --unlinked`: exit 0 when everything holds but round 2's wires (second-wires)
# On a challenger host's gate, from one of its peers: G="--challenger gate:USER@HOST --peer NODE --challenger-key
# RECORD --ssh-key KEY" as both SMALL_FIRST and SMALL_SECOND (the last --challenger given is the one taken), and
# SMALL_ARGS="--serve NONCE" with a nonce no run has used: the gate opens a run's root once, ever.
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
INLINE=(); UNLINKED=(); [ -n "${SMALL_INLINE:-}" ] && { INLINE=(--inline); UNLINKED=(--unlinked); }
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage_small.py serve --out "$W/serve" "${INLINE[@]}" ${SMALL_ARGS:-} | tee "$O/serve.json"
CHECK=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["check"])' "$O/serve.json")
CLASS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["class"])' "$O/serve.json")
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$CLASS" --check "$CHECK" --replay rate:2/6 --k 1 \
  --challenger local --main "$O/main.txt" --out "$W/first" ${SMALL_FIRST:-} > "$O/first.log" 2>&1 || { tail -20 "$O/first.log"; exit 3; }
tail -3 "$O/first.log"
rc=0
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py second --stage1 "$W/first/$(basename "$CLASS")/stage1.json" --boundary "$CLASS" \
  --capture "$W/serve/capture" --challenger local --main "$O/main.txt" --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
  --out "$W/second" "${UNLINKED[@]}" ${SMALL_SECOND:-} > "$O/second.log" 2>&1 || rc=$?
grep '^TWO ' "$O/second.log" || tail -30 "$O/second.log"
echo "second rc=$rc" | tee -a "$O/machine.txt"
RECORDS=(-type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' ! -name index.json
         ! -name index-salt.bin ! -name salts.bin ! -path './*/archive/*' ! -path '*-work/*')
for d in first second; do
  mkdir -p "$O/$d"
  (cd "$W/$d" && find . "${RECORDS[@]}" -size -64M -print0 | xargs -0 -r -I{} cp --parents {} "$O/$d/")
  (cd "$W/$d" && find . "${RECORDS[@]}" -size +63M -print0 | sort -z \
     | xargs -0 -r -I{} sh -c 'printf "%s  %s  %s\n" "$(sha512sum < "$1" | cut -d" " -f1)" "$(stat -c %s "$1")" "$1"' _ {}) > "$O/$d/large.txt"
done
[ -f "$W/second/summary.json" ] || exit 2
"${PY[@]}" benchmarks/one_stage/two_stage_small.py expect "$W/second/summary.json" "${UNLINKED[@]}" | tee "$O/expect.txt"
