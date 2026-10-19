#!/usr/bin/env bash
# The proof service's challenger on its auditor host (cwd = the source tree): `serve.py`, one session a run, its record,
# key, release log, address and summary in $RESEARCH_RUN_DIR/out, which the run publishes from this machine. It runs on a
# machine the prover doesn't control (AGENTS.md's auditor-trusted devices), never on PEER: once it listens it opens
# `ssh -R PORT` to PEER (a name in /etc/vy/peers), as benchmarks/remote_session/run.sh's verifier does, so the prover's
# client (`remote.RemoteChallenger.connect("127.0.0.1:PORT")`) connects to PEER's own 127.0.0.1:PORT.
#
#   research run --on vy-nebius-cpu-1 --project verity --source . --cwd source --declared-output 'out/*' \
#     --env PEER=vy-nebius-2 --env PORT=7811 --env SCHEME=vllm/commit \
#     --env FAMILIES='[["opening-positions",8],["opening-identities",8],["replay",8],["kernel-check",8]]' \
#     -- bash benchmarks/live_challenger/run.sh
#
# FAMILIES is the session's coin layout, [[label, count], ...] (the vLLM Commit's is `challenge.layout(runs)`); EPOCH its
# epoch (default 0); KEY the challenger's LMS key directory on this machine (default ~/.local/state/live-challenger/key,
# made at lms.DEPLOYED on first use over KEYGEN_WORKERS cores, never under out/); PY the Python (default python3: the
# challenger reads only the standard library and this tree). The verifier reads out/record.json, out/key.json and
# out/releases.jsonl from this run (`research fetch RUN`, then `remote.trusted_session`), never from the prover's.
set -euo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
exec "${PY:-python3}" benchmarks/live_challenger/serve.py --out "$O" --port "${PORT:?}" --scheme "${SCHEME:?}" \
  --families "${FAMILIES:?}" --epoch "${EPOCH:-0}" --peer "${PEER:?}" ${KEY:+--key "$KEY"} \
  ${KEYGEN_WORKERS:+--keygen-workers "$KEYGEN_WORKERS"} 2> >(tee "$O/serve.log" >&2)
