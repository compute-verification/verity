#!/usr/bin/env bash
# A gate run of the remote memory audit from an agent's VM, one command (cwd = this repository): the prover's whole node
# serving MODEL on PROVER_NODE, the verifier on vy-auditor-1 through its gate, the coins from vy-challenger-1 under
# period PERIOD, and the line of record and the what-if beside it. Each step is a `research run`; their ids go to
# ~/.research/gate-runs/PERIOD.env.
#
#   benchmarks/remote_session/gate_run.sh [--dry-run] PERIOD    # check, probe, launch the prover, fire
#   benchmarks/remote_session/gate_run.sh whatif PERIOD         # once fire has ended: gate_whatif.sh over its records
#
#   1. checkpoint  checkpoint.sh with CHECK_ONLY=1 on PROVER_NODE: the pinned checkpoint's files against core's pin.
#   2. probe       gate.sh probe on GATE_ON (the machine with GATE_KEY): vy-challenger-1's key is PIN, PERIOD unused and
#                  no period of the lane open, vy-auditor-1 pinned at COMMIT and idle. Nothing is written to a gate.
#   3. prover      run.sh on PROVER_NODE inside `gpu-lease ENGINES --wait` (32 GB of host memory an engine; not
#                  preemptible: a gate run stopped midway spends its period), ROLE=prover SERVE=node with PLAN at the
#                  snapshot core's pin names, gate run 7's engine and node settings, VERIFIER_SSH to vy-auditor-1's
#                  tunnel account; once leased it waits up to TUNNEL_WAIT_S for the host's verifier to listen.
#   4. fire        gate.sh fire on GATE_ON: once the prover holds its GPUs, probe again, period-open (circuit of
#                  COMMIT's remote_bandwidth tree, MODEL), the host's run, period-close, auditor.sh (the line of record,
#                  its auditor_line.json and credit.md in the fire run's out/).
#   whatif         gate_whatif.sh at origin/main's head on WHATIF_ON over the fire run's artifact (the host run in it,
#                  its ledger and pins) and the prover's; RUN_ART, SUB, AUDITOR_ART and PROVER_ART name others.
#
# COMMIT is the commit vy-auditor-1's verifier is pinned at (default PERIOD.env's, else origin/main's head; the probe
# refuses another and names the host's, and a COMMIT given here replaces the recorded one); the steps run from a
# worktree of it, with this file's gate.sh and checkpoint.sh sent beside. GATE_KEY is the key's path on GATE_ON (default
# gate.sh's, under that machine's user's ~/.ssh). A dry run's ids go to PERIOD.dry.env. --dry-run runs steps 1 and 2
# (with DRY=1: no request reaches a gate, and GATE_KEY need not exist), prints step 3, and runs step 4 with DRY=1
# against PROVER_RUN, any run id (default none), whose lease it reads.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
DRY=
[ "${1:-}" = --dry-run ] && { DRY=1; shift; }
VERB=fire
[ "${1:-}" = whatif ] && { VERB=whatif; shift; }
PERIOD=${1:?usage: gate_run.sh [--dry-run] PERIOD | gate_run.sh whatif PERIOD}
PIN=${PIN:-27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc}
GATE_ON=${GATE_ON:-vy-nebius-1}
PROVER_NODE=${PROVER_NODE:-vy-nebius-2}
WHATIF_ON=${WHATIF_ON:-vy-nebius-cpu-4}
ROLE=${ROLE:-QWEN25_32B_BF16} ENGINES=${ENGINES:-8} PORT=${PORT:-7951}
PLAN=${PLAN:-honest,skip:0.0625,honest,skip:0.0625}
VERIFIER_SSH=${VERIFIER_SSH:--o StrictHostKeyChecking=accept-new -i /home/research/.ssh/vy-auditor-tunnel-prover prover@81.85.1.145}
STATE=$HOME/.research/gate-runs/$PERIOD${DRY:+.dry}.env
mkdir -p "$(dirname "$STATE")"
touch "$STATE"
GIVEN_COMMIT=${COMMIT:-}
unset COMMIT
# shellcheck disable=SC1090
. "$STATE"
git -C "$ROOT" fetch -q origin main || exit 3
MAIN=$(git -C "$ROOT" rev-parse origin/main)
COMMIT=${GIVEN_COMMIT:-${COMMIT:-$MAIN}}
TREE=$COMMIT
[ "$VERB" = whatif ] && TREE=$MAIN
W=$(mktemp -d)
trap 'git -C "$ROOT" worktree remove --force "$W" > /dev/null 2>&1; rm -rf "$W"' EXIT
git -C "$ROOT" worktree add -q --detach "$W" "$TREE" || exit 3
read -r REPO REV < <(python3 -c 'import json, sys
e = json.load(open(sys.argv[1]))[sys.argv[2]]
print(e["repo"], e["revision"])' "$W/verity/core/protocols/pouw/lean/check/models.json" "$ROLE") \
  || { echo "core's pin at $TREE has no role $ROLE"; exit 3; }
SNAP=/workspace/hf/hub/models--${REPO//\//--}/snapshots/$REV
echo "period $PERIOD, COMMIT $COMMIT, $ROLE at $SNAP on $PROVER_NODE, gate key on $GATE_ON${DRY:+ (dry run)}"

record() {
  echo "$1=$2" >> "$STATE"
  printf -v "$1" %s "$2"
}

launch() {
  # NAME research-run-args...: one run from the worktree, its id recorded as NAME; with --wait, its rc returned
  local name=$1 out rc id
  shift
  out=$(cd "$ROOT" && uv run -q research run --project verity --source "$W" "$@" 2>&1)
  rc=$?
  echo "$out" | tail -n 25
  id=$(grep -o -m1 -E 'r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}' <<< "$out" | head -n 1)
  [ -n "$id" ] || { echo "$name: no run id (rc $rc)"; return 3; }
  record "$name" "$id"
  echo "$name $id rc $rc"
  return $rc
}

GATE_ENV=(--env "COMMIT=$COMMIT" --env "PERIOD=$PERIOD" --env "PIN=$PIN" ${GATE_KEY:+--env "GATE_KEY=$GATE_KEY"}
  --env "MODEL=$ROLE" --env "ENGINES=$ENGINES" --env "PORT=$PORT" --env "PROVER_NODE=$PROVER_NODE"
  ${DRY:+--env DRY=1} ${CIRCUIT:+--env "CIRCUIT=$CIRCUIT"})
gate_step() {
  # NAME VERB GB research-run-args...: gate.sh VERB on GATE_ON in GB of memory, waited for (a dry step reserves none)
  local name=$1 verb=$2 mem=(--mem-gb "$3" --mem-wait 3600)
  shift 3
  [ -z "$DRY" ] || mem=()
  launch "$name" --on "$GATE_ON" --cwd clone --send "$HERE/gate.sh" "${GATE_ENV[@]}" "${mem[@]}" "$@" \
    -- bash -lc "bash \"\$RESEARCH_RUN_DIR/inputs/gate.sh\" $verb"
}

if [ "$VERB" = whatif ]; then
  [ -n "${fire:-}" ] && [ -n "${prover:-}" ] || [ -n "${RUN_ART:-}" ] || { echo "no fire and prover runs in $STATE"; exit 3; }
  art() {
    (cd "$ROOT" && uv run -q research data show "$1" --json) \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["outputs"]["run_record"])'
  }
  if [ -z "${RUN_ART:-}" ]; then
    RUN_ART=$(art "$fire") || { echo "no artifact for $fire yet"; exit 3; }
    (cd "$ROOT" && uv run -q research data fetch "$RUN_ART" --to "$W/.fire" --path out/host-run.txt) > /dev/null || exit 3
    SUB=out/host/$(cat "$W/.fire/out/host-run.txt")
  fi
  AUDITOR_ART=${AUDITOR_ART:-$RUN_ART}
  PROVER_ART=${PROVER_ART:-$(art "$prover")} || { echo "no artifact for $prover yet"; exit 3; }
  launch whatif --on "$WHATIF_ON" --cwd clone --mem-gb 16 --timeout 30m --store-read --wait \
    --env "RUN_ART=$RUN_ART" --env "SUB=${SUB:?}" --env "AUDITOR_ART=$AUDITOR_ART" --env "PROVER_ART=$PROVER_ART" \
    --env "PERIOD=$PERIOD" --env "PIN=$PIN" --env "COMMIT=$COMMIT" -- bash benchmarks/remote_session/gate_whatif.sh
  exit $?
fi

[ -z "${fire:-}" ] || [ -n "$DRY" ] || { echo "$STATE already has fire $fire: name a new PERIOD"; exit 3; }
[ "$(grep '^COMMIT=' "$STATE" | tail -n 1)" = "COMMIT=$COMMIT" ] || record COMMIT "$COMMIT"
echo "== 1. checkpoint on $PROVER_NODE"
launch checkpoint --on "$PROVER_NODE" --cwd source --send "$HERE/checkpoint.sh" --timeout 20m --wait --env CHECK_ONLY=1 \
  --env "ROLE=$ROLE" -- bash -lc 'bash "$RESEARCH_RUN_DIR/inputs/checkpoint.sh"' || exit 3
echo "== 2. probe on $GATE_ON"
gate_step probe probe 4 --timeout 10m --wait || exit 3
echo "== 3. prover on $PROVER_NODE"
PROVER=(--on "$PROVER_NODE" --cwd source --timeout 6h --custody-ttl 12h
  --env ROLE=prover --env "PORT=$PORT" --env SERVE=node --env "PLAN=$PLAN" --env "MODEL=$SNAP"
  --env ENGINE_CPUS=0-23 --env PROVER_CPUS=24-47 --env "ENGINE_ARGS=--step-hz 2 --align-steps --arena-gib 62 --util 0.8"
  --env "NODE_ARGS=--decode-s 60 --warm-openings 2773 --wait-s 1800 --model-dir $SNAP"
  --env "VERIFIER_SSH=$VERIFIER_SSH" --env TUNNEL_WAIT_S=1800 --env "GPU_LEASE_WHO=memory-accounting-$PERIOD"
  -- /workspace/pouw/infra/bin/gpu-lease "$ENGINES" --wait --max-min 90 --mem-gb $((32 * ENGINES))
  -- bash benchmarks/remote_session/run.sh)
if [ -n "$DRY" ]; then
  echo "DRY: research run --project verity --source <worktree of $COMMIT> ${PROVER[*]}"
  PROVER_RUN=${PROVER_RUN:-none}
else
  launch prover "${PROVER[@]}" || exit 3
  PROVER_RUN=$prover
fi
echo "== 4. fire on $GATE_ON"
gate_step fire fire 16 --timeout 8h --custody-ttl 12h --store-read --env "PROVER_RUN=$PROVER_RUN" ${DRY:+--wait} || {
  r=$?
  [ -n "$DRY" ] || (cd "$ROOT" && uv run -q research cancel "$PROVER_RUN" --reason "gate_run.sh: $PERIOD's fire didn't launch")
  exit $r
}
echo "state: $STATE; once fire ends: benchmarks/remote_session/gate_run.sh whatif $PERIOD"
