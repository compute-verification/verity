#!/usr/bin/env bash
# two_stage.py's two commands end to end on a CPU node, on a token select class at small statics (two_stage_small.py: no
# engine): flock-circuit, flock-verify and SPOT's Lean verifier built from this tree as the toy's audit.sh builds them,
# then `first` (the boundary registration and the first stage on a local challenger session) and `second` (the interior,
# the second stage on a session of its own, a --zk session per member, Lean's verdicts, round 2 and the stream's outcome).
# Exit 0 when the second stage held (two_stage_small.py expect): round 1 commits the ids the block returns as a port, the
# return's sessions read it at round 1's rows, round 2 accepts the statements, and the stream's outcome is accepted or
# refused by the two-stage form alone. Records under $OUT (never a circuit, a prover file, a
# private body or salts, or a challenger's key or secret); a record over 63 MiB (a served vocabulary's public bodies and
# written window, a challenger host's ledger) is not kept, and large.txt lists its SHA-512, size and path.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/two_stage_small.sh
#   SMALL_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); SMALL_ARGS="--v 256" passes
#   two_stage_small.py serve more; SMALL_FIRST and SMALL_SECOND="--firewall PATH" pass `first` and `second` more;
#   SMALL_INLINE=1 runs the class as a served vocabulary's is, its block the check's body inlined, with `second
#   --unlinked`: exit 0 when everything holds but round 2's wires (second-wires)
#   SMALL_ARGS="--inline --units 20 --v 2560" runs the class inlined as the greedy select in 20 units
#   (GreedySelect_v1{V, N=V / 20}: 20 templates, one unit of each a token), which round 2 derives its wires and its
#   return link from: `second` runs it linked, and exit 0 is the first line's, as for the class of whole instances
#   SMALL_GPU=1 builds flock-circuit with its device prover (on CUDA 13.3's toolkit: CUDA_TK, the node's own, or the one
#   verity/protocols/flock/pod/cuda-13.3.sh finds or unpacks beside flock-circuit's build) and runs `second` with
#   --devices, so its sessions prove on a card: the cards the run already holds (the indices in CUDA_VISIBLE_DEVICES, as
#   research run --gpus N sets them), else one that gpu-lease hands `second` alone, for at most SMALL_LEASE_MIN minutes
#   (30), so the builds and the first stage hold none
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
GPU=0; ON_CARDS=()
if [ -n "${SMALL_GPU:-}" ]; then
  # the unpacking runs in a subshell, under its own shell options; sourced again, the script only names the toolkit it
  # finds. With the toolkit named, the build installs nothing on the node, and the prover finds its runtime beside it
  ( set +euo pipefail; FLOCK_WORK=$FW . verity/protocols/flock/pod/cuda-13.3.sh ) > "$O/cuda.log" 2>&1 \
    || { tail -20 "$O/cuda.log"; echo "SMALL_GPU=1: the node has no CUDA 13.3 toolkit and none could be unpacked" >&2; exit 8; }
  set +euo pipefail; FLOCK_WORK=$FW . verity/protocols/flock/pod/cuda-13.3.sh >> "$O/cuda.log" 2>&1; set -euo pipefail
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
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=$GPU bash verity/protocols/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
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
"${ON_CARDS[@]}" "${PY[@]}" benchmarks/one_stage/two_stage.py second --stage1 "$W/first/$(basename "$CLASS")/stage1.json" --boundary "$CLASS" \
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
