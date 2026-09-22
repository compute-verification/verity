#!/usr/bin/env bash
# [E8 NIGHT 2026-09-12] Launch the one-request 32x16 SmolLM2-135M protocol run on night-e8-prove (E10's runner recipe, parametrised):
#   * REPO       = a RUN-PRIVATE copy of the shipped merged tree (cp -a /workspace/a9/veritor /workspace/a9/run-<tag>-veritor; E10's E_SOURCE_CHANGED lesson)
#   * instance   = E6's pinned protocol_instance.json for LP=32/STEPS=15 (request 7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6), population = E6's v2 file
#   * serve      = SHARED_SERVE: the PREPARATORY serve of the same workload (record_v5.serve_v5, this pod, 06:25Z) -- or a fresh serve if unset
#   * rows       = E1's producer output for the request (/workspace/e1/out/n32x16-7cc7c3bb), staged by run_night_pod.sh (E2's `tree stage`)
#   * host       = the night tree's veritor-zk-host (guest 666e937f...), SP1_PROVER=cuda
# usage (pod):  bash launch_32x16_pod.sh <tag> [phases...]     env overrides: POP PINNED_INSTANCE E1OUT SHARED_SERVE HOST SRC_TREE
set -uo pipefail
TAG=${1:?tag, e.g. n1}; shift 1
SRC_TREE=${SRC_TREE:-/workspace/a9/veritor}
export REPO=/workspace/a9/run-$TAG-veritor
if [ ! -d "$REPO" ]; then
  cp -a "$SRC_TREE" "$REPO"
  # the venv's editable veritor.pth must point INSIDE the frozen tree (run_tiny1_pod.sh's rule)
  for f in "$REPO"/.venv/lib/python3.*/site-packages/veritor.pth; do [ -f "$f" ] && sed -i "s#$SRC_TREE/src#$REPO/src#" "$f"; done
fi
export RUN_ID=${RUN_ID:-V5-N32x16-1req-bex0-inc}
export POP=${POP:-/workspace/a9/pop/32x16/population.v2.json}
export PINNED_INSTANCE=${PINNED_INSTANCE-/workspace/a9/pinned/32x16/protocol_instance.json}
export E1OUT=${E1OUT:-/workspace/e1/out/n32x16-7cc7c3bb}
export STAGED=${STAGED:-/workspace/a9/interior-values-32x16}
export SHARED_SERVE=${SHARED_SERVE-/workspace/a9/rec/serve-V5-N32x16-1req-bex0-inc}
export HOST=${HOST:-/workspace/a9/veritor-zk-host}
# [E10 0d54057b] required.json per RU = the verifier's live required layout OF THIS TREE, re-derived over an earlier record of the SAME request with
# commitments: the archived V5-32x15-bex0-inc record (request 7cc7c3bb..., the producer's --record); provenance lands in $REC/logs/interior-required-v2.json
export REQUIRED_FROM_RECORD=${REQUIRED_FROM_RECORD-/workspace/e1/rec-32x15}
POD_NAME=${POD_NAME:-night-e8-prove}; POD_ID=${POD_ID:-r0dtdsgz281ux8}; ROWS_NOTE=${ROWS_NOTE:-this pod, --only-required, --parallel 14}
REC=/workspace/a9/rec/$RUN_ID-$TAG; mkdir -p "$REC"
[ -f "$REC/LABEL.md" ] || cat > "$REC/LABEL.md" <<MD
# $RUN_ID-$TAG -- SmolLM2-135M one-request record at prompt 32 x decode 16 (LP = 32, STEPS = 15): THE LONGER-CONTEXT PROTOCOL RUN of the night program (G4: E8 with E2), $(date -u +%FT%TZ)

Model HuggingFaceTB/SmolLM2-135M @ 93efa2f0 served by vLLM d9105ea80 (cu129, batch-invariant, enforce_eager) on an RTX 4090 (RunPod SECURE
$POD_ID, $POD_NAME); request 7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6 = B0-corpus prompt 0 (32 tokens, record_v5 deterministic workload
L1-32x15 row 0), 1 prefill + 15 decode steps, greedy; hardware semantics D7 (RTX 4090).
Instance: E6's freeze serve (population_v3 on eod/e6 HEAD 3058b95a, unmodified) at LP = 32 / STEPS = 15 copied verbatim from $PINNED_INSTANCE
(instance_v5 --pinned cross-checks it against the record); PROVISIONAL until E6 blesses / re-freezes (deterministic; same digest expected).
Population: partition 77d5b63b (a9/legacy-minus-inputs/v2, N_V = 29,496,522, N_R = 1, s = 2^-14, q = 1, B_ex = 0).
Interior: E2's layout interior-layout/chunks-4096/v2 over E1's producer rows for this request ($ROWS_NOTE,
383,798,280 gates, boundary rows compared against the archived V5-32x15-bex0-inc record: 0 mismatches); required set per RU = the verifier's live
required layout of THIS tree (REQUIRED_FROM_RECORD = the archived record -> record_v5.interior_required_v2, E10 b6bcccac over E2's public
LazyInteriorLayout accessors eebe700b; the run refuses to start without it -- n2's refusal at step 4 was that fallback).
Serve: SHARED_SERVE = the preparatory record_v5.serve_v5 pass of the same workload on this pod (06:25Z; same tree lane), copied in by run_v5.sh.
Tree: $REPO = a run-private copy of the shipped merged tree (freeze.json names the hash; nobody else has the path): commit
$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('commit'), 'dirty', (d.get('dirty_patch_sha256') or '')[:16], d.get('dirty_patch_bytes'), 'B')" "$REPO/EXPORT.json") ${TREE_NOTE:-}.  E3's census over this tree
is requested the moment the hash is posted (out/night/reports/e8.md); r_s only after the verdict line.
Sequence: freeze/instance/coverage/serve first (no sample); protocol (steps 1-4, r_q, r_s) only after the census verdict; then step 5: every located
unit through E7's ONE registry over E3's PositionOracle, SP1 proofs on the 4090 under the approved guest, verify_v5 (live driver + reverify by
import), reverify_clean on this pod in a fresh venv, then T1 independently on the laptop.
ONE sample.  No redraw.  Nothing in this run is chosen from what it draws.
MD
export VERITOR_HARDWARE="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) (RunPod secure pod $(hostname), $POD_NAME $POD_ID; NIGHT 32x16 one-request $TAG)"
mkdir -p /workspace/a9/rec
{ echo "# launch $TAG $(date -u +%FT%TZ) tree=$REPO"; sha256sum "$POP" "$PINNED_INSTANCE" "$HOST" 2>/dev/null; cat "$REPO/EXPORT.json"; } | tee -a /workspace/a9/rec/launch-$TAG.txt
[ -f "$POP" ] || { echo "[launch] FATAL: no population at $POP (E6's v2 file)"; exit 3; }
[ -z "$PINNED_INSTANCE" ] || [ -f "$PINNED_INSTANCE" ] || { echo "[launch] FATAL: no pinned instance at $PINNED_INSTANCE (E6's pin) -- PINNED_INSTANCE= for a PREPARATORY generated instance"; exit 3; }
exec bash "$REPO/e2e/demo_core/night/run_night_pod.sh" "$TAG" "$@"
