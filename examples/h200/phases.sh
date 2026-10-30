#!/usr/bin/env bash
# `bash examples/h200/phases.sh PHASE [--dry-run] [FLAG...]`: one phase of Demo A's M3 proper on the demo H200
# (its interfaces in examples/h200/interfaces; run.py is the step driver).
# Demo A is not one command: it is eight runs, each phase's output the next one's input, the coin flipped only after the
# registration is in the challenger's ledger. Each phase below starts its job by the run recorder (`research run`) on its
# machine (or, for `register`, here on the CPU) with the variables it names; --dry-run prints the command and starts
# nothing, and further FLAGs go to the recorder. Each phase's run of record is in example.toml's `runs`:
#   commit    the step on the H200 under a fresh salt secret kept on the node: 32 steps at batch 512, every op's output
#             committed, roots.json out (m3proper_job.sh PHASE=commit)
#   weights   the registration of the 733 seeded INT8 weights under the served Program, its 0600 witness kept on the
#             node (e2e_h200.weights_commit commit). PROGRAM: the Program digest (hex)
#   register  the step registration, fixed before any coin: every step root, step 31's slot roots, the weights and the law
#             subset:8 over the step's 3,124,276,224 proof blocks (m3stepreg.py). ROOTS: commit's roots.json, WEIGHTS_REG:
#             weights' registration.json, PROGRAM, COMMIT_RUN: commit's run id; then REG_OUT/step-registration.json
#             goes to the evidence store (kind m3-step-registration/v1)
#   coin      one live coin from the challenger host's gate, its session opened at the registration's digest, no resample; the
#             receipt and draw.json out (m3coin.py). REG: step-registration.json
#   open      every registered weight row a sampled proof block reads, opened from weights' witness (m3bundle.py weights). REG, DRAW:
#             coin's draw.json, WEIGHTS_STATE: weights' state directory on WEIGHTS_NODE
#   replay    the step again on the H200 under the salt secret, every root held to roots.json, the drawn slots' rows kept
#             beside the secret (m3proper_job.sh PHASE=replay). ROOTS, REG, DRAW
#   bundle    each sampled proof block's one-member window and replay bundle, checked against both registrations (m3proper_job.sh
#             PHASE=build); then the tree goes to the evidence store (kind m3-drawn-blocks/v1). REG, DRAW,
#             RECEIPT: coin's receipt.json, WEIGHTS_OPEN: open's weights-open.json
#   prove     the sample re-derived from the coin, every block held to the step registration, then each proved by Flock --zk
#             behind the ZK server one at a time and decided by Lean, one flipped bit refused (prove_bundle.sh). TREE_ART:
#             bundle's art id, COIN_DIR: coin's out/coin with its session's ledger, on CPU_NODE
# Machines: H200_NODE (an H200 node), CPU_NODE and WEIGHTS_NODE (CPU nodes); each defaults to the node its run of record
# used. A phase whose job isn't in this checkout (prove_bundle.sh's PB_IN_ART and PB_COIN aren't on main) says so and starts
# nothing. The memory audits (PoUS and the bandwidth audit) are not a phase here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PHASES="commit weights register coin open replay bundle prove"
PHASE=${1:-}
[[ " $PHASES " == *" $PHASE "* ]] || { echo "examples/h200/phases.sh PHASE [--dry-run] [FLAG...]: PHASE is one of $PHASES" >&2; exit 2; }
shift
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
H200=${H200_NODE:-vy-dc-h200-2} CPU=${CPU_NODE:-vy-nebius-cpu-4} WNODE=${WEIGHTS_NODE:-vy-nebius-cpu-2}
need() { for v; do [ -n "${!v:-}" ] || { echo "h200 phases $PHASE: set $v (examples/h200/phases.sh's header says what it is)" >&2; exit 2; }; done; }
have() {
  grep -qs -- "${3:-}" "$1" || { echo "h200 phases $PHASE: $1${3:+ with $3} isn't in this checkout: it is on $2 until it lands" >&2; exit 3; }
}
named() { [ "$(basename "$1")" = "$2" ] || { echo "h200 phases $PHASE: $1 must be named $2 (the job reads inputs/$2)" >&2; exit 2; }; }
main_txt() { T=$(mktemp -d); git rev-list --first-parent origin/main > "$T/main.txt"; echo "$T/main.txt"; }
RR=(research run --project verity --source .)
JOB=examples/h200/m3proper_job.sh BR=cursor/m3-proper-1de3
case $PHASE in
  commit)
    have $JOB $BR
    CMD=("${RR[@]}" --on "$H200" --cwd source --timeout 5400 --env PHASE=commit "$@" -- bash $JOB) ;;
  weights)
    need PROGRAM
    CMD=("${RR[@]}" --on "$WNODE" --env PROGRAM="$PROGRAM" "$@" -- bash -c '
set -euo pipefail
uv sync -q --all-packages --extra torch-cpu
ST=/workspace/research/lanes/compute-accounting/h200-weights/$(basename "$RESEARCH_RUN_DIR")
mkdir -p "${ST%/*}"
PYTHONPATH=examples/h200 uv run --no-sync python -m e2e_h200.weights_commit commit --shape h200 \
  --seed demo --program "$PROGRAM" --state "$ST" --workers "$(nproc)" --out "$OUT/registration.json" | tee "$OUT/summary.json"
uv run --no-sync research keep "$ST" --owner @compute-accounting --expires 30d --priority high \
  --why "the H200 weights registration state and its 0600 OS-salt witness (prover-private), for the opens" || echo "keep failed" >&2') ;;
  register)
    have examples/h200/m3stepreg.py $BR
    need ROOTS WEIGHTS_REG PROGRAM COMMIT_RUN
    CMD=(uv run python examples/h200/m3stepreg.py --roots "$ROOTS" --weights "$WEIGHTS_REG" --program "$PROGRAM"
         --run "$COMMIT_RUN" --out "${REG_OUT:-out/h200/registration}") ;;
  coin)
    have examples/h200/m3coin.py $BR
    need REG; named "$REG" step-registration.json
    CMD=("${RR[@]}" --on "$CPU" --send "$REG" --send "$(main_txt)" "$@" -- bash -c 'uv sync -q && uv run --no-sync python \
examples/h200/m3coin.py --registration "$RESEARCH_RUN_DIR/inputs/step-registration.json" --gate-key \
"$HOME/.ssh/vy-gate-compute-accounting" --peer '"$CPU"' --minutes 20 --main "$RESEARCH_RUN_DIR/inputs/main.txt" --out "$OUT/coin"') ;;
  open)
    have examples/h200/m3bundle.py $BR
    need REG DRAW WEIGHTS_STATE; named "$REG" step-registration.json; named "$DRAW" draw.json
    CMD=("${RR[@]}" --on "$WNODE" --send "$REG" --send "$DRAW" --env WEIGHTS_STATE="$WEIGHTS_STATE" "$@" -- bash -c 'uv sync -q \
--all-packages --extra torch-cpu && uv run --no-sync python examples/h200/m3bundle.py weights --state "$WEIGHTS_STATE" \
--seed demo --registration "$RESEARCH_RUN_DIR/inputs/step-registration.json" --draw "$RESEARCH_RUN_DIR/inputs/draw.json" \
--out "$OUT/weights-open.json"') ;;
  replay)
    have $JOB $BR
    need ROOTS REG DRAW; named "$ROOTS" roots.json; named "$REG" step-registration.json; named "$DRAW" draw.json
    CMD=("${RR[@]}" --on "$H200" --cwd source --timeout 5400 --env PHASE=replay --send "$ROOTS" --send "$REG" --send "$DRAW" "$@"
         -- bash $JOB) ;;
  bundle)
    have $JOB $BR
    need REG DRAW RECEIPT WEIGHTS_OPEN
    named "$REG" step-registration.json; named "$DRAW" draw.json; named "$RECEIPT" receipt.json; named "$WEIGHTS_OPEN" weights-open.json
    CMD=("${RR[@]}" --on "$H200" --cwd source --timeout 5400 --env PHASE=build --send "$REG" --send "$DRAW" --send "$RECEIPT"
         --send "$WEIGHTS_OPEN" "$@" -- bash $JOB) ;;
  prove)
    have benchmarks/one_stage/prove_bundle.sh cursor/proofs-m3-proper-92aa PB_COIN
    need TREE_ART COIN_DIR
    CMD=("${RR[@]}" --on "$CPU" --pool proofs --as proofs --cwd clone --send "$(main_txt)" "$@" -- bash
         benchmarks/one_stage/prove_bundle.sh PB_IN_ART="$TREE_ART" PB_COIN="$COIN_DIR" PB_FLIPS=1 PB_LEAN_JOBS=1) ;;
esac
if [ $DRY = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
