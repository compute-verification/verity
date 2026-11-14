#!/usr/bin/env bash
# two_stage.py's two commands end to end on a CPU node, on a token select class at small statics (two_stage_small.py: no
# engine): flock-circuit, verity-verify-proof and SPOT's Lean verifier built from this tree as the toy's audit.sh builds them,
# then `first` (the boundary registration and the first stage on a local challenger session) and `second` (the interior,
# the second stage on a session of its own, a --zk session per member, Lean's verdicts, round 2 and the stream's outcome).
# Exit 0 when the second stage is what the play expects (two_stage_small.py expect). With no arguments the play is the
# honest one round 2 accepts: the class inlined as the greedy select in one unit, round 1 committing the ids the block
# returns as a port and the select's logits beside them (two_stage_small.py serve --inline --units 1 --logits), so the
# interior registers no value, the sessions read both at round 1's rows and round 2 has no code. A play round 2 refuses
# is a negative case and runs only by its name. SMALL_NEGATIVE=second-inputs: round 1 commits the ids alone, so the
# logits are a value the prover registers after the first stage's sample, and exit 0 is when everything holds but
# round 2's second-inputs, by which it refuses the statements. Records under $OUT (never a circuit, a prover file, a
# private body or salts, or a challenger's key or secret); a record over 63 MiB (a served vocabulary's public bodies and
# written window, a challenger host's ledger) is not kept, and large.txt lists its SHA-512, size and path.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/two_stage_small.sh
#   SMALL_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); SMALL_ARGS, if given, is the
#   serve's arguments, not the play's; SMALL_FIRST and SMALL_SECOND="--firewall PATH" pass `first` and `second` more;
#   SMALL_INLINE=1 runs the class as a served vocabulary's is, its block the check's body inlined, with `second
#   --unlinked`: exit 0 when everything holds but round 2's wires (second-wires)
#   SMALL_NEGATIVE=second-inputs runs that negative case on the class of whole instances (serve with no arguments), and
#   with SMALL_ARGS="--inline --units 20 --v 2560" on the class inlined as the greedy select in 20 units
#   (GreedySelect_v1{V, N=V / 20}: 20 templates, one unit of each a token), which round 2 derives its wires and its
#   return link from: `second` runs it linked, and exit 0 is that case's, as for the class of whole instances;
#   adding --stop serves it under a stop rule, a stop class, which `first` and `second` take as --stop; adding --tables
#   commits each request's history table beside it instead, as a serve with no stop rule does; adding --logits commits
#   the select's inputs beside the ids, kept in the scratch serve's private/ (boundary_logits.load), which `second`
#   takes as --kept: the honest play in 20 units, which takes no negative case's name
#   SMALL_GPU=1 builds flock-circuit with its device prover (on CUDA 13.3's toolkit: CUDA_TK, the node's own, or the one
#   verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh finds or unpacks beside flock-circuit's build) and runs `second` with
#   --devices, so its sessions prove on a card: the cards the run already holds (the indices in CUDA_VISIBLE_DEVICES, as
#   research run --gpus N sets them), else one that gpu-lease hands `second` alone, for at most SMALL_LEASE_MIN minutes
#   (30), so the builds and the first stage hold none
# On a challenger host's gate, from one of its peers: G="--challenger gate:USER@HOST --peer NODE --challenger-key
# RECORD --ssh-key KEY" as both SMALL_FIRST and SMALL_SECOND (the last --challenger given is the one taken), and
# SMALL_ARGS="--inline --units 1 --logits --serve NONCE" with a nonce no run has used: the gate opens a run's root once,
# ever.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
# the play: the honest one, unless a negative case is named, which `expect` holds to round 2's refusal by its one code
PLAY="--inline --units 1 --logits"; INLINE=(); UNLINKED=(); INPUTS=()
[ -n "${SMALL_INLINE:-}" ] && { PLAY=; INLINE=(--inline); UNLINKED=(--unlinked); }
case "${SMALL_NEGATIVE:-}" in
  "") ;;
  second-inputs) PLAY=; INPUTS=(--inputs) ;;
  *) echo "SMALL_NEGATIVE=$SMALL_NEGATIVE names no negative case: second-inputs is the one (second-wires' is SMALL_INLINE=1)" >&2; exit 9 ;;
esac
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; } | tee "$O/machine.txt"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }

{ (cd verity/protocols/proofs/flock/verifier && lake build verity-verify-proof flock-firewall) \
    && (cd verity/protocols/proofs/spot/verifier/lean && lake build sampled-proofs-verifier); } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
FW=${SMALL_FLOCK_WORK:-$W}
GPU=0; ON_CARDS=()
if [ -n "${SMALL_GPU:-}" ]; then
  # the unpacking runs in a subshell, under its own shell options; sourced again, the script only names the toolkit it
  # finds. With the toolkit named, the build installs nothing on the node, and the prover finds its runtime beside it
  ( set +euo pipefail; FLOCK_WORK=$FW . verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh ) > "$O/cuda.log" 2>&1 \
    || { tail -20 "$O/cuda.log"; echo "SMALL_GPU=1: the node has no CUDA 13.3 toolkit and none could be unpacked" >&2; exit 8; }
  set +euo pipefail; FLOCK_WORK=$FW . verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh >> "$O/cuda.log" 2>&1; set -euo pipefail
  TK=${CUDA_TK:-/usr/local/cuda-13.3}
  [ -x "$TK/bin/nvcc" ] || { echo "SMALL_GPU=1: no nvcc under $TK" >&2; exit 8; }
  export CUDA_TK=$TK LD_LIBRARY_PATH=$TK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  # `second` is told its cards as whoever holds them names them: this run's, else a lease's taken for it alone
  GPU=1; ON_CARDS=(bash -c 'echo "cards $CUDA_VISIBLE_DEVICES ${GPU_LEASE_UUID:-}" >&2; exec "$@" --devices "$CUDA_VISIBLE_DEVICES"' _)
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    command -v gpu-lease >/dev/null || { echo "SMALL_GPU=1: the run holds no card (CUDA_VISIBLE_DEVICES) and the node has no gpu-lease" >&2; exit 8; }
    export GPU_LEASE_WHO=${GPU_LEASE_WHO:-two-stage-small/${RESEARCH_RUN_ID:-local}}
    ON_CARDS=(gpu-lease 1 --wait --max-min "${SMALL_LEASE_MIN:-30}" -- "${ON_CARDS[@]}")
  fi
  echo "toolkit $TK" | tee -a "$O/machine.txt"
fi
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=$GPU bash verity/protocols/proofs/flock/prover/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/protocols/proofs/flock/verifier/.lake/build/bin/verity-verify-proof
sha256sum "$BIN" "$LEAN" | tee "$O/binaries.sha256"

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm
export RAYON_NUM_THREADS=$(nproc) PYTHONPATH=$M0PATH
# a local challenger session's pin is held to main's first-parent line: a run's clone has the shipped tree as its origin,
# which holds main as its own main or origin/main
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$O/fetch-main.log" 2>&1 \
  && git rev-list --first-parent refs/remotes/origin/main > "$O/main.txt" || { tail -5 "$O/fetch-main.log"; exit 4; }

PY=(uv run --no-sync python)
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage_small.py serve --out "$W/serve" "${INLINE[@]}" ${SMALL_ARGS:-$PLAY} | tee "$O/serve.json"
CHECK=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["check"])' "$O/serve.json")
CLASS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["class"])' "$O/serve.json")
RULE=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stop", ""))' "$O/serve.json")
STOP=(); [ -z "$RULE" ] || STOP=(--stop "$RULE")
# a run whose record of round 1 commits the select's logits opens the kept rows; a linked one whose record doesn't is
# refused by round 2's second-inputs, and goes on only as the negative case of that name
LOGITS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("logits", ""))' "$O/serve.json")
KEPT=(); [ -z "$LOGITS" ] || KEPT=(--kept "$LOGITS")
[ -n "$LOGITS${SMALL_INLINE:-}${SMALL_NEGATIVE:-}" ] || { echo "the serve's record commits no logits, a play round 2 refuses (second-inputs):" \
  "give SMALL_ARGS --inline --units K --logits, or name the negative case, SMALL_NEGATIVE=second-inputs" >&2; exit 9; }
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$CLASS" --check "$CHECK" --replay rate:2/6 --k 1 \
  --challenger local --main "$O/main.txt" --out "$W/first" "${STOP[@]}" ${SMALL_FIRST:-} > "$O/first.log" 2>&1 || { tail -20 "$O/first.log"; exit 3; }
tail -3 "$O/first.log"
rc=0
# shellcheck disable=SC2086
"${ON_CARDS[@]}" "${PY[@]}" benchmarks/one_stage/two_stage.py second --stage1 "$W/first/$(basename "$CLASS")/stage1.json" --boundary "$CLASS" \
  --capture "$W/serve/capture" --challenger local --challenger-host "$W/first/challenger-host" --main "$O/main.txt" --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
  --out "$W/second" "${UNLINKED[@]}" "${STOP[@]}" "${KEPT[@]}" ${SMALL_SECOND:-} > "$O/second.log" 2>&1 || rc=$?
grep '^TWO ' "$O/second.log" || tail -30 "$O/second.log"
echo "second rc=$rc" | tee -a "$O/machine.txt"
RECORDS=(-type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' ! -name index.json
         ! -name index-salt.bin ! -name salts.bin ! -path './*/archive/*' ! -path '*-work/*' ! -path './challenger-host/*')
for d in first second; do
  mkdir -p "$O/$d"
  (cd "$W/$d" && find . "${RECORDS[@]}" -size -64M -print0 | xargs -0 -r -I{} cp --parents {} "$O/$d/")
  (cd "$W/$d" && find . "${RECORDS[@]}" -size +63M -print0 | sort -z \
     | xargs -0 -r -I{} sh -c 'printf "%s  %s  %s\n" "$(sha512sum < "$1" | cut -d" " -f1)" "$(stat -c %s "$1")" "$1"' _ {}) > "$O/$d/large.txt"
done
[ -f "$W/second/summary.json" ] || exit 2
"${PY[@]}" benchmarks/one_stage/two_stage_small.py expect "$W/second/summary.json" "${UNLINKED[@]}" "${INPUTS[@]}" | tee "$O/expect.txt"
