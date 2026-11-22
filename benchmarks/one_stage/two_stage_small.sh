#!/usr/bin/env bash
# two_stage.py's two commands end to end on a CPU node, on a token select class at small statics (two_stage_small.py: no
# engine): flock-circuit, verity-verify-proof and SPOT's Lean verifier built from this tree,
# then `first` (the boundary docket and the first stage on a local challenger session) and `second` (the interior,
# the second stage on a session of its own, a --zk session per member, Lean's verdicts, round 2 and the stream's outcome).
# Exit 0 when the second stage is what the play expects (two_stage_small.py expect). With no arguments the play is the
# honest one round 2 accepts: the class inlined as the greedy select in one unit, round 1 committing the ids the block
# returns as a port and the select's logits beside them (two_stage_small.py serve --inline --units 1 --logits), so the
# interior registers no value, the sessions read both at round 1's rows and round 2 has no code. Records under $OUT (never a circuit, a prover file, a
# private body or salts, or a challenger's key or secret); a record over 63 MiB (a served vocabulary's public bodies and
# written window, a challenger host's ledger) is not kept, and large.txt lists its SHA-512, size and path.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/two_stage_small.sh
#   SMALL_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); SMALL_ARGS, if given, is the
#   serve's arguments, not the play's; SMALL_FIRST and SMALL_SECOND="--zk-server PATH" pass `first` and `second` more;
#   SMALL_ARGS="--inline --units 20 --v 2560 --logits" runs the class inlined as the greedy select in 20 units
#   (GreedySelect_v1{V, N=V / 20}: 20 templates, one unit of each a token); adding --stop serves it under a stop rule,
#   which `first` and `second` take as --stop; adding --tables commits each request's history table beside it
#   SMALL_GPU=1 builds flock-circuit with its device prover (on CUDA 13.3's toolkit: CUDA_TK, the node's own, or the one
#   verity/protocols/proof/flock/prover/pod/cuda-13.3.sh finds or unpacks beside flock-circuit's build) and runs `second` with
#   --gpus, so its sessions prove on a card: the cards the run already holds (the indices in CUDA_VISIBLE_DEVICES, as
#   research run --gpus N sets them), else one that gpu-lease hands `second` alone, for at most SMALL_LEASE_MIN minutes
#   (30), so the builds and the first stage hold none
# On a challenger host's gate, from one of its peers: G="--challenger gate:USER@HOST --peer NODE --challenger-key
# RECORD --ssh-key KEY" as both SMALL_FIRST and SMALL_SECOND (the last --challenger given is the one taken), and
# SMALL_ARGS="--inline --units 1 --logits --serve NONCE" with a nonce no run has used: the gate opens a run's root once,
# ever.
set -euo pipefail
PUBF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/public_files.py
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
PLAY="--inline --units 1 --logits"
DEV="DEVELOPMENT RUN, not an accepted run of record: the second stage's circuit is taken as the staging side wrote it after the first sample, not derived by the verifier"
FW=${SMALL_FLOCK_WORK:-$W}
GPU=0; ON_CARDS=()
if [ -n "${SMALL_GPU:-}" ]; then
  # with the toolkit named, the build installs nothing on the node, and the prover finds its runtime beside it
  . benchmarks/one_stage/toolkit.sh
  FLOCK_WORK=$FW cuda_13_3 "$O/cuda.log" \
    || { tail -20 "$O/cuda.log"; echo "SMALL_GPU=1: the node has no CUDA 13.3 toolkit and none could be unpacked" >&2; exit 8; }
  TK=$CUDA_TK
  [ -x "$TK/bin/nvcc" ] || { echo "SMALL_GPU=1: no nvcc under $TK" >&2; exit 8; }
  export CUDA_TK=$TK LD_LIBRARY_PATH=$TK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  # `second` is told its cards as whoever holds them names them: this run's, else a lease's taken for it alone
  GPU=1; ON_CARDS=(bash -c 'echo "cards $CUDA_VISIBLE_DEVICES ${GPU_LEASE_UUID:-}" >&2; exec "$@" --gpus "$CUDA_VISIBLE_DEVICES"' _)
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    command -v gpu-lease >/dev/null || { echo "SMALL_GPU=1: the run holds no card (CUDA_VISIBLE_DEVICES) and the node has no gpu-lease" >&2; exit 8; }
    export GPU_LEASE_WHO=${GPU_LEASE_WHO:-two-stage-small/${RESEARCH_RUN_ID:-local}}
    ON_CARDS=(gpu-lease 1 --wait --max-min "${SMALL_LEASE_MIN:-30}" -- "${ON_CARDS[@]}")
  fi
  echo "toolkit $TK" | tee -a "$O/machine.txt"
fi
. benchmarks/one_stage/two_stage_build.sh
export PYTHONPATH=$M0PATH

PY=(uv run --no-sync python)
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage_small.py serve --out "$W/serve" ${SMALL_ARGS:-$PLAY} | tee "$O/serve.json"
CHECK=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["check"])' "$O/serve.json")
CLASS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["class"])' "$O/serve.json")
RULE=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stop", ""))' "$O/serve.json")
STOP=(); [ -z "$RULE" ] || STOP=(--stop "$RULE")
# a run whose record of round 1 commits the select's logits opens the kept rows; one whose record doesn't is refused by
# round 2's second-inputs, so it isn't run
LOGITS=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("logits", ""))' "$O/serve.json")
KEPT=(); [ -z "$LOGITS" ] || KEPT=(--kept "$LOGITS")
[ -n "$LOGITS" ] || { echo "the serve's record commits no logits, a play round 2 refuses (second-inputs): give SMALL_ARGS" \
  "--inline --units K --logits" >&2; exit 9; }
# shellcheck disable=SC2086
"${PY[@]}" benchmarks/one_stage/two_stage.py first --boundary "$CLASS" --check "$CHECK" --replay rate:2/6 --k 1 \
  --challenger local --main "$O/main.txt" --out "$W/first" "${STOP[@]}" ${SMALL_FIRST:-} > "$O/first.log" 2>&1 || { tail -20 "$O/first.log"; exit 3; }
tail -3 "$O/first.log"
rc=0
# shellcheck disable=SC2086
"${ON_CARDS[@]}" "${PY[@]}" benchmarks/one_stage/two_stage.py second --first-stage "$W/first/$(basename "$CLASS")/stage1.json" --boundary "$CLASS" \
  --capture "$W/serve/capture" --challenger local --challenger-host "$W/first/challenger-host" --main "$O/main.txt" --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
  --out "$W/second" "${STOP[@]}" "${KEPT[@]}" ${SMALL_SECOND:-} > "$O/second.log" 2>&1 || rc=$?
echo "$DEV" | tee "$O/summary.txt"
grep '^TWO ' "$O/second.log" | tee -a "$O/summary.txt" || tail -30 "$O/second.log"
echo "second rc=$rc" | tee -a "$O/machine.txt"
RECORDS=(-type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name index.json
         ! -path './*/archive/*' ! -path '*-work/*' ! -path './challenger-host/*')
for d in first second; do
  mkdir -p "$O/$d"
  (cd "$W/$d" && find . "${RECORDS[@]}" -size -64M -print0 | python3 "$PUBF" filter | xargs -0 -r -I{} cp --parents {} "$O/$d/")
  (cd "$W/$d" && find . "${RECORDS[@]}" -size +63M -print0 | sort -z \
     | xargs -0 -r -I{} sh -c 'printf "%s  %s  %s\n" "$(sha512sum < "$1" | cut -d" " -f1)" "$(stat -c %s "$1")" "$1"' _ {}) > "$O/$d/large.txt"
done
[ -f "$W/second/summary.json" ] || exit 2
"${PY[@]}" benchmarks/one_stage/two_stage_small.py expect "$W/second/summary.json" | tee "$O/expect.txt"
